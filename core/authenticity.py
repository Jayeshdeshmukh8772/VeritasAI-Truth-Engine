"""
VeritasAI Source Authenticity Layer.
Checks source independence and verifies numerical ground truth.
"""

from typing import List, Dict, Any

class SourceAuthenticityChecker:
    
    def __init__(self):
        pass
        
    def detect_source_families(self, sources: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Maps URLs to a source_family_id to detect syndication.
        E.g., MSN quoting Reuters should have the same family ID as Reuters.
        """
        families = {}
        for idx, src in enumerate(sources):
            # Simplistic proxy: in real production, cross-check text overlap or metadata tags.
            domain = src.get("url", "").split("/")[2] if "//" in src.get("url", "") else src.get("url", "")
            families[src.get("url", "")] = f"family_{domain}"
        return families

    def validate_ground_truth_numbers(self, context_text: str, extracted_claims: List[Dict]) -> List[str]:
        """
        Checks if the statistical numbers mentioned in claims actually exist in the raw context.
        """
        failures = []
        for claim in extracted_claims:
            for num in claim.get("numbers", []):
                if str(num) not in context_text:
                    failures.append(f"Fabricated statistic: {num}")
        return failures
