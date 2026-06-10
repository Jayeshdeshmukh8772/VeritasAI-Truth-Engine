"""
VeritasAI Engine Schema Dataclasses.
Defines static data contracts shared across all processing stages.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class LLMStatus(Enum):
    """Status enumeration for LLM call results."""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class LLMResult:
    """Result from a single LLM adapter call."""
    model: str
    status: LLMStatus
    response: Optional[str]
    error_type: Optional[str] = None
    error_msg: Optional[str] = None
    latency_ms: int = 0
    tokens_used: int = 0
    peer_rank_score: float = 0.5  # Evaluated during Stage 3
    trust_score: float = 0.0      # Evaluated during Stage 4
    is_outlier: bool = False      # Evaluated during Stage 4
    semantic_score: float = 0.5   # Evaluated during Stage 4


@dataclass
class EnhancedQuery:
    """Query after enhancement by QuestionEnhancer."""
    original: str
    enhanced: str
    query_type: str  # factual | analytical | creative | code | medical
    input_type: str = "QUESTION" # QUESTION | COMMAND | DOCUMENT | LOG | CODE | SYSTEM_OUTPUT
    enhancement_skipped: bool = False
    enhancement_reason: str = ""


@dataclass
class PeerReviewResult:
    """Result from peer review stage."""
    model: str
    raw_ranking_text: str
    parsed_ranking: list  # Ordered list of letters: ['B', 'A', 'C']
    peer_rank_score: float     # 0.0 to 1.0 normalization


@dataclass
class DetectionResult:
    """Result from hallucination detection stage."""
    trusted: list
    outliers: list
    trust_scores: dict
    consensus_ratio: float
    low_consensus: bool        # True if consensus_ratio < threshold
    high_dissent: bool = False  # True if 2 models disagree heavily
    entropy: float = 0.0
    centrality_scores: dict = field(default_factory=dict)


@dataclass
class Citation:
    """Structured citation verifying source and context."""
    url: str
    publication_date: Optional[str]
    retrieval_timestamp: str
    exact_quote: str
    supports_claim: bool
    authority_score: float
    source_family: str
    freshness_score: float

@dataclass
class FinalOutput:
    """Final synthesized output to display to user."""
    answer: Optional[str]
    consensus_ratio: float
    trust_scores: dict
    peer_rankings: dict
    hallucination_flags: list  # Populated with models flagged as outliers
    follow_up_questions: list
    low_consensus: bool
    all_results: list
    total_latency_ms: int
    high_dissent: bool = False  # True if 2 models disagree heavily
    entropy: float = 0.0
    evidence_agreement: float = 0.0
    source_freshness: float = 0.0
    authority_score: float = 0.0
    hallucination_risk: float = 0.0
    model_coverage: float = 0.0
    retrieval_authenticity: float = 0.0
    final_trust_score: float = 0.0
    citations: list = field(default_factory=list)  # List[Citation]
    outlier_reasons: dict = field(default_factory=dict)
    critical_failures: list = field(default_factory=list)