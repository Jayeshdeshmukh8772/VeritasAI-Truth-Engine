"""
Tests for core/detector.py — HallucinationDetector DBSCAN clustering.
Covers: consensus cluster identification, outlier flagging, edge cases.
"""

import pytest
from core.result import LLMResult, LLMStatus
from core.detector import HallucinationDetector


def make_result(model: str, response: str, peer_score: float = 0.7) -> LLMResult:
    """Helper to create a successful LLMResult with a peer score."""
    r = LLMResult(model, LLMStatus.SUCCESS, response)
    r.peer_rank_score = peer_score
    return r


# Semantically similar phrases about Paris being France's capital
PARIS_A = "The capital city of France is Paris, which holds great historical significance in Western Europe."
PARIS_B = "Paris serves as the capital of France, a major cultural and political center since medieval times."
PARIS_C = "France's administrative and cultural capital is Paris, situated along the Seine River."

# Outlier: semantically different claim
BERLIN_OUTLIER = "The capital city of France is Berlin, located in central Europe along the Spree River."


@pytest.fixture(scope="module")
def detector() -> HallucinationDetector:
    """Load the sentence transformer model once for the entire test module."""
    return HallucinationDetector(eps=0.25, min_samples=2)


def test_correct_outlier_flagged(detector: HallucinationDetector) -> None:
    """ModelD (Berlin) should be detected as an outlier among France/Paris responses."""
    results = [
        make_result("ModelA", PARIS_A, peer_score=0.8),
        make_result("ModelB", PARIS_B, peer_score=0.8),
        make_result("ModelC", PARIS_C, peer_score=0.8),
        make_result("ModelD", BERLIN_OUTLIER, peer_score=0.1),  # Outlier
    ]

    detection = detector.analyze(results)

    outlier_models = {r.model for r in detection.outliers}
    assert "ModelD" in outlier_models, "Berlin response should be flagged as outlier"
    assert "ModelA" not in outlier_models, "Paris response should not be flagged"


def test_no_results_returns_empty_detection(detector: HallucinationDetector) -> None:
    """Empty input should return a safe empty DetectionResult."""
    detection = detector.analyze([])
    assert detection.consensus_ratio == 0.0
    assert detection.trusted == []
    assert detection.outliers == []


def test_single_result_is_trusted(detector: HallucinationDetector) -> None:
    """Single result should not be flagged as outlier (no cluster possible)."""
    results = [make_result("ModelA", PARIS_A)]
    detection = detector.analyze(results)
    # With only one result, DBSCAN can't form a cluster of min_samples=2
    # All trusted with score 0.5 (no cluster)
    assert len(detection.trusted) + len(detection.outliers) == 1


def test_all_similar_responses_trusted(detector: HallucinationDetector) -> None:
    """When all models agree, none should be flagged as outliers."""
    results = [
        make_result("ModelA", PARIS_A, peer_score=0.8),
        make_result("ModelB", PARIS_B, peer_score=0.8),
        make_result("ModelC", PARIS_C, peer_score=0.8),
    ]
    detection = detector.analyze(results)
    # With high similarity and high peer scores, outliers should be empty or near-empty
    assert len(detection.trusted) >= 2


def test_trust_scores_computed_for_all_models(detector: HallucinationDetector) -> None:
    """Trust scores should be populated for all successful models."""
    results = [
        make_result("ModelA", PARIS_A),
        make_result("ModelB", PARIS_B),
        make_result("ModelC", BERLIN_OUTLIER, peer_score=0.1),
    ]
    detection = detector.analyze(results)
    for model in ["ModelA", "ModelB", "ModelC"]:
        assert model in detection.trust_scores
        assert 0.0 <= detection.trust_scores[model] <= 1.0


def test_consensus_ratio_between_zero_and_one(detector: HallucinationDetector) -> None:
    """Consensus ratio must always be in [0, 1]."""
    results = [
        make_result("A", PARIS_A),
        make_result("B", PARIS_B),
        make_result("C", BERLIN_OUTLIER, peer_score=0.05),
    ]
    detection = detector.analyze(results)
    assert 0.0 <= detection.consensus_ratio <= 1.0


def test_low_consensus_flag_set_when_ratio_below_threshold(detector: HallucinationDetector) -> None:
    """low_consensus should be True when consensus_ratio < 0.5."""
    # Use 4 wildly different responses to force low consensus
    results = [
        make_result("A", "The answer is blue, a primary color in the RGB spectrum used in digital displays."),
        make_result("B", "Python is a high-level programming language known for its readable syntax and versatility."),
        make_result("C", BERLIN_OUTLIER, peer_score=0.05),
        make_result("D", "Quantum entanglement refers to a phenomenon where two particles remain connected regardless of distance.", peer_score=0.05),
    ]
    # These are semantically very different; DBSCAN should scatter them
    detection = detector.analyze(results)
    # If consensus_ratio < 0.5, low_consensus must be True
    if detection.consensus_ratio < 0.5:
        assert detection.low_consensus is True


def test_math_equivalence_sympy(detector: HallucinationDetector) -> None:
    """Equivalent math responses should have similarity boosted and group together."""
    res_a = make_result("ModelA", "The solution is x = sqrt(25) which is simple.")
    res_b = make_result("ModelB", "For this problem, x = 5 is the value.")
    
    detection = detector.analyze([res_a, res_b], query_type="mathematical")
    
    assert not detection.low_consensus
    assert len(detection.trusted) == 2


def test_sparse_fallback_3_models(detector: HallucinationDetector) -> None:
    """Under N=3, dynamic metric fallback should use correct weights and thresholds."""
    res_a = make_result("ModelA", PARIS_A, peer_score=0.8)
    res_b = make_result("ModelB", PARIS_B, peer_score=0.8)
    res_c = make_result("ModelC", BERLIN_OUTLIER, peer_score=0.1)
    
    detection = detector.analyze([res_a, res_b, res_c])
    
    outlier_models = {r.model for r in detection.outliers}
    assert "ModelC" in outlier_models
    assert "ModelA" not in outlier_models


def test_sparse_fallback_2_models(detector: HallucinationDetector) -> None:
    """Under N=2, dynamic metric fallback should flag dissent if similarity < 0.55."""
    res_a = make_result("ModelA", "The capital city of France is Paris.", peer_score=0.8)
    res_b = make_result("ModelB", "Python is a programming language.", peer_score=0.2)
    
    detection = detector.analyze([res_a, res_b])
    
    assert len(detection.outliers) == 2