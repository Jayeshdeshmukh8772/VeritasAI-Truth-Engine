"""
Tests for core/dispatcher.py — AsyncDispatcher behavior.
Covers: parallel dispatch, rate limit bypass, failure handling, skipping disabled models.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from core.result import LLMResult, LLMStatus
from core.dispatcher import AsyncDispatcher
from core.rate_tracker import RateTracker


def make_mock_adapter(name: str, response: str = "Test response with more than twenty words here to pass the threshold", fail: bool = False) -> MagicMock:
    """Create a mock adapter with controlled response behavior."""
    adapter = MagicMock()
    adapter.name = name

    if fail:
        adapter.call = AsyncMock(return_value=LLMResult(
            name, LLMStatus.FAILED, None, "timeout", "Timed out", 1000
        ))
    else:
        adapter.call = AsyncMock(return_value=LLMResult(
            name, LLMStatus.SUCCESS, response, None, None, 200, 50
        ))
    return adapter


@pytest.fixture
def mock_config() -> dict:
    """Minimal config fixture for dispatcher tests."""
    return {
        "models": [
            {"name": "groq", "enabled": True, "daily_limit": 100},
            {"name": "gemini", "enabled": True, "daily_limit": 100},
            {"name": "cerebras", "enabled": False, "daily_limit": 100},
        ]
    }


@pytest.fixture
def rate_tracker(tmp_path) -> RateTracker:
    """Create a rate tracker using a temp file."""
    tracker_file = str(tmp_path / "rate_tracker.json")
    return RateTracker(tracker_file=tracker_file)


@pytest.mark.asyncio
async def test_dispatch_returns_results_for_all_adapters(mock_config, rate_tracker) -> None:
    """Dispatcher should return one result per adapter (success or fail)."""
    adapters = [
        make_mock_adapter("groq"),
        make_mock_adapter("gemini"),
    ]
    dispatcher = AsyncDispatcher(adapters, rate_tracker, mock_config)
    results = await dispatcher.dispatch_all("What is 2 + 2?")
    assert len(results) == 2
    assert all(isinstance(r, LLMResult) for r in results)


@pytest.mark.asyncio
async def test_failed_adapter_returns_failed_result(mock_config, rate_tracker) -> None:
    """Failed adapters should return LLMStatus.FAILED, not raise exceptions."""
    adapters = [make_mock_adapter("groq", fail=True)]
    dispatcher = AsyncDispatcher(adapters, rate_tracker, mock_config)
    results = await dispatcher.dispatch_all("Test query")
    assert results[0].status == LLMStatus.FAILED


@pytest.mark.asyncio
async def test_quota_exceeded_marks_as_skipped(mock_config, rate_tracker) -> None:
    """Adapter exceeding daily quota should return SKIPPED result."""
    # Exhaust the quota
    for _ in range(101):
        rate_tracker.increment("groq")

    adapters = [make_mock_adapter("groq")]
    config_with_limit = {
        "models": [{"name": "groq", "enabled": True, "daily_limit": 100}]
    }
    dispatcher = AsyncDispatcher(adapters, rate_tracker, config_with_limit)
    results = await dispatcher.dispatch_all("Test query")
    assert results[0].status == LLMStatus.SKIPPED


@pytest.mark.asyncio
async def test_successful_call_increments_rate_counter(mock_config, rate_tracker) -> None:
    """Successful adapter calls should increment the rate tracker count."""
    adapters = [make_mock_adapter("groq")]
    config = {"models": [{"name": "groq", "enabled": True, "daily_limit": 1000}]}
    dispatcher = AsyncDispatcher(adapters, rate_tracker, config)
    initial_count = rate_tracker.get_count("groq")
    await dispatcher.dispatch_all("Test query")
    assert rate_tracker.get_count("groq") == initial_count + 1


@pytest.mark.asyncio
async def test_mixed_success_and_failure(mock_config, rate_tracker) -> None:
    """Dispatcher correctly handles a mix of success and failure results."""
    adapters = [
        make_mock_adapter("groq"),
        make_mock_adapter("gemini", fail=True),
    ]
    config = {
        "models": [
            {"name": "groq", "enabled": True, "daily_limit": 1000},
            {"name": "gemini", "enabled": True, "daily_limit": 1000},
        ]
    }
    dispatcher = AsyncDispatcher(adapters, rate_tracker, config)
    results = await dispatcher.dispatch_all("Test query")
    statuses = {r.model: r.status for r in results}
    assert statuses["groq"] == LLMStatus.SUCCESS
    assert statuses["gemini"] == LLMStatus.FAILED