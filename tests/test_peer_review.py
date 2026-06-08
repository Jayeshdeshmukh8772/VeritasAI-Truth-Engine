"""
Tests for core/peer_review.py — PeerReviewEngine anonymization and ranking.
Covers: anonymization, ranking parse, score normalization, edge case < 2 results.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from core.result import LLMResult, LLMStatus
from core.peer_review import PeerReviewEngine


LONG_RESPONSE = (
    "This is a detailed response about the topic at hand. "
    "It contains enough words to pass the minimum threshold for testing purposes. "
    "The content is informative and comprehensive."
)


def make_result(model: str, response: str = LONG_RESPONSE) -> LLMResult:
    """Helper to create a successful LLMResult."""
    return LLMResult(model, LLMStatus.SUCCESS, response)


def make_adapter(name: str, ranking_text: str) -> MagicMock:
    """Create a mock adapter that returns a ranking response."""
    adapter = MagicMock()
    adapter.name = name
    adapter.call = AsyncMock(return_value=LLMResult(
        name, LLMStatus.SUCCESS, ranking_text
    ))
    return adapter


@pytest.mark.asyncio
async def test_fewer_than_two_results_skips_review() -> None:
    """With < 2 successful results, all peer_rank_scores should be set to 0.5."""
    single_result = make_result("ModelA")
    engine = PeerReviewEngine({})
    results = await engine.run_review("Test question", [single_result])
    assert results[0].peer_rank_score == 0.5


@pytest.mark.asyncio
async def test_peer_scores_assigned_for_two_models() -> None:
    """With 2 models, peer_rank_scores should be assigned and in [0, 1]."""
    results = [make_result("ModelA"), make_result("ModelB")]

    # Adapter A ranks B first, A second: FINAL RANKING: 1. Response B 2. Response A
    # Adapter B ranks A first, B second: FINAL RANKING: 1. Response A 2. Response B
    adapter_a = make_adapter("ModelA", "FINAL RANKING:\n1. Response B\n2. Response A")
    adapter_b = make_adapter("ModelB", "FINAL RANKING:\n1. Response A\n2. Response B")

    engine = PeerReviewEngine({"ModelA": adapter_a, "ModelB": adapter_b})
    reviewed = await engine.run_review("Test question?", results)

    for r in reviewed:
        assert 0.0 <= r.peer_rank_score <= 1.0, f"{r.model} score {r.peer_rank_score} out of range"


@pytest.mark.asyncio
async def test_peer_scores_normalized_between_zero_and_one() -> None:
    """Peer rank scores must always be in the range [0, 1]."""
    results = [make_result(f"Model{c}") for c in "ABC"]

    ranking_texts = [
        "FINAL RANKING:\n1. Response A\n2. Response B\n3. Response C",
        "FINAL RANKING:\n1. Response C\n2. Response A\n3. Response B",
        "FINAL RANKING:\n1. Response B\n2. Response C\n3. Response A",
    ]
    adapters = {}
    for i, res in enumerate(results):
        mock = make_adapter(res.model, ranking_texts[i])
        adapters[res.model] = mock

    engine = PeerReviewEngine(adapters)
    reviewed = await engine.run_review("Some question?", results)

    for r in reviewed:
        assert 0.0 <= r.peer_rank_score <= 1.0


@pytest.mark.asyncio
async def test_malformed_ranking_response_handled_gracefully() -> None:
    """Malformed ranking responses should not crash the engine."""
    results = [make_result("ModelA"), make_result("ModelB")]

    # Malformed — won't match expected pattern
    adapter_a = make_adapter("ModelA", "I think A is better than B overall.")
    adapter_b = make_adapter("ModelB", "FINAL RANKING:\n1. Response A\n2. Response B")

    engine = PeerReviewEngine({"ModelA": adapter_a, "ModelB": adapter_b})
    reviewed = await engine.run_review("Question?", results)
    # Should not raise, and all scores should be set
    assert all(hasattr(r, "peer_rank_score") for r in reviewed)


@pytest.mark.asyncio
async def test_failed_reviewer_does_not_crash() -> None:
    """If a reviewer adapter fails, the engine should handle it gracefully."""
    results = [make_result("ModelA"), make_result("ModelB")]

    failed_adapter = MagicMock()
    failed_adapter.name = "ModelA"
    failed_adapter.call = AsyncMock(return_value=LLMResult(
        "ModelA", LLMStatus.FAILED, None, "timeout", "Timed out", 1000
    ))
    good_adapter = make_adapter("ModelB", "FINAL RANKING:\n1. Response A\n2. Response B")

    engine = PeerReviewEngine({"ModelA": failed_adapter, "ModelB": good_adapter})
    reviewed = await engine.run_review("Question?", results)
    # Should complete without error
    assert len(reviewed) == 2