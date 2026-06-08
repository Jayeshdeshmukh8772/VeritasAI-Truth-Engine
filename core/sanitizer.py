"""
VeritasAI Input Sanitization Engine.
Cleans and validates user input before dispatching to LLM providers.
Strips PII, enforces length caps, and detects query intent type.
"""

import re
from typing import Optional


class InputSanitizer:
    """Sanitizes user input: removes PII, enforces char limits, detects query type."""

    # PII detection patterns
    EMAIL_PATTERN: str = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    PHONE_PATTERN: str = r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'
    SSN_PATTERN: str = r'\b(?!000|666|9\d\d)\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b'
    CREDIT_CARD_PATTERN: str = r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'

    # Query type detection keyword maps
    QUERY_TYPE_MAP: dict = {
        "code": ["code", "function", "bug", "python", "javascript", "error", "debug",
                 "algorithm", "class", "method", "variable", "compile", "runtime", "syntax"],
        "medical": ["symptom", "doctor", "treatment", "diagnosis", "disease", "medication",
                    "surgery", "health", "clinical", "patient", "therapy", "condition"],
        "creative": ["poem", "story", "fiction", "write", "creative", "novel", "essay",
                     "plot", "character", "narrative", "screenplay", "lyrics"],
        "analytical": ["compare", "analyze", "analyse", "evaluate", "assess", "explain why",
                       "pros and cons", "difference between", "impact of", "cause of"],
    }

    def __init__(self, max_chars: int = 2000) -> None:
        """
        Initialize the sanitizer with a character limit.

        Args:
            max_chars: Maximum allowed characters in input query (default: 2000)
        """
        self.max_chars = max_chars

    def sanitize(self, text: str) -> str:
        """
        Clean user input: strip whitespace, enforce char cap, remove PII.

        Args:
            text: Raw user input string

        Returns:
            Sanitized string ready for LLM dispatch, or empty string if input invalid
        """
        if not text or not text.strip():
            return ""

        # Strip leading/trailing whitespace
        text = text.strip()

        # Enforce character limit
        if len(text) > self.max_chars:
            text = text[: self.max_chars]

        # Redact PII in sequence
        text = re.sub(self.EMAIL_PATTERN, "[EMAIL]", text)
        text = re.sub(self.PHONE_PATTERN, "[PHONE]", text)
        text = re.sub(self.SSN_PATTERN, "[SSN]", text)
        text = re.sub(self.CREDIT_CARD_PATTERN, "[CARD]", text)

        return text.strip()

    def detect_query_type(self, text: str) -> str:
        """
        Infer the intent type of a query from keyword signals.

        Args:
            text: Cleaned query string

        Returns:
            One of: 'code' | 'medical' | 'creative' | 'analytical' | 'factual'
        """
        text_lower = text.lower()
        for q_type, keywords in self.QUERY_TYPE_MAP.items():
            if any(kw in text_lower for kw in keywords):
                return q_type
        return "factual"

    def validate(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Validate input before processing. Returns (is_valid, error_message).

        Args:
            text: Raw user input

        Returns:
            Tuple of (True, None) if valid, or (False, reason) if invalid
        """
        if not text or not text.strip():
            return False, "Query cannot be empty."
        if len(text.strip()) < 3:
            return False, "Query is too short. Please provide more context."
        return True, None