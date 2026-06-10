"""
VeritasAI Deterministic Scoring Formulas.
Replaces arbitrary LLM-generated scores with strict, testable math.
"""

from typing import List, Dict, Any

# Hardcoded backend-controlled authority map.
AUTHORITY_MAP = {
    "sec.gov": 1.00,
    "openai.com": 1.00,
    "reuters.com": 0.95,
    "bloomberg.com": 0.95,
    "bbc.com": 0.90,
    "wikipedia.org": 0.50
}

def get_authority_score(url: str) -> float:
    """Returns the authority score for a given URL based on the map."""
    url_lower = url.lower()
    for domain, score in AUTHORITY_MAP.items():
        if domain in url_lower:
            return score
        # Support wildcards like *.gov
        if domain.startswith("*.") and url_lower.endswith(domain[2:]):
            return score
    return 0.30  # Default low authority for unknown sources


def compute_retrieval_authenticity(citations: List[Any]) -> float:
    """
    100 = exact retrieved article + URL + quote
    75 = exact page retrieved but no quote
    40 = homepage only
    10 = source mentioned but not retrieved
    0 = no retrieval proof
    """
    if not citations:
        return 0.0
    
    scores = []
    for c in citations:
        # Robustly handle boolean or string for supports_claim
        supports_claim = c.supports_claim is True or str(c.supports_claim).lower() == 'true'
        url_present = bool(c.url and c.url.strip() and c.url.lower() not in ["unavailable", "none", "null", ""])
        quote_present = bool(c.exact_quote and c.exact_quote.strip() and c.exact_quote.lower() not in ["unavailable", "none", "null", ""])

        if url_present and quote_present and supports_claim:
            scores.append(100.0)
        elif url_present and not quote_present:
            # Check if it's just a homepage
            if c.url.count("/") <= 3: # e.g. https://domain.com/
                scores.append(40.0)
            else:
                scores.append(75.0)
        elif not url_present and c.source_family and c.source_family.lower() not in ["unavailable", "none", "null", ""]:
            scores.append(10.0)
        else:
            scores.append(0.0)
    
    return sum(scores) / len(scores)


def compute_freshness(citations: List[Any], query_intent_date: str = "today") -> float:
    """
    Official source + no date -> heavy freshness penalty
    News source + no date -> reject citation (score 0)
    """
    if not citations:
        return 0.0
    
    scores = []
    for c in citations:
        if not c.publication_date:
            auth = get_authority_score(c.url)
            if auth >= 0.95: # Official
                scores.append(0.30)
            else:
                scores.append(0.0)
        else:
            # Basic decay simulation. In production, parse date strings and compare.
            scores.append(1.0) 
            
    return sum(scores) / len(scores)


def get_confidence_bin(score: float) -> str:
    """Returns a descriptive bin avoiding false precision."""
    val = score * 100
    if val >= 95:
        return "Very High (95-99)"
    elif val >= 85:
        return "High (85-94)"
    elif val >= 70:
        return "Medium (70-84)"
    else:
        return "Low (<70)"


def compute_final_trust(
    consensus_ratio: float, 
    model_coverage: float, 
    evidence_agreement: float, 
    authority_score: float, 
    freshness_score: float, 
    retrieval_authenticity: float,
    critical_failures: List[str]
) -> float:
    """
    0.15 Model Consensus
    0.10 Model Coverage
    0.35 Evidence Agreement
    0.20 Authority Score
    0.15 Source Freshness
    0.05 Retrieval Authenticity
    """
    base_trust = (
        (0.15 * consensus_ratio) +
        (0.10 * model_coverage) +
        (0.35 * evidence_agreement) +
        (0.20 * authority_score) +
        (0.15 * freshness_score) +
        (0.05 * retrieval_authenticity)
    )
    
    # Cap trust score if any critical failures exist
    if critical_failures:
        base_trust = min(base_trust, 0.50)
        
    return base_trust
