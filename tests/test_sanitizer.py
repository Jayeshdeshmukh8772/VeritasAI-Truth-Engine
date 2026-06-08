"""
Tests for core/sanitizer.py — InputSanitizer edge cases.
Covers: empty input, PII removal, length capping, query type detection, validation.
"""

import pytest
from core.sanitizer import InputSanitizer


@pytest.fixture
def sanitizer() -> InputSanitizer:
    """Create a default sanitizer instance for testing."""
    return InputSanitizer(max_chars=2000)


def test_empty_string_returns_empty(sanitizer: InputSanitizer) -> None:
    """Empty string input should return empty string."""
    assert sanitizer.sanitize("") == ""
    assert sanitizer.sanitize("   ") == ""


def test_whitespace_stripping(sanitizer: InputSanitizer) -> None:
    """Leading/trailing whitespace should be stripped."""
    result = sanitizer.sanitize("  hello world  ")
    assert result == "hello world"


def test_max_chars_enforcement() -> None:
    """Input longer than max_chars should be truncated."""
    s = InputSanitizer(max_chars=100)
    long_input = "a" * 200
    result = s.sanitize(long_input)
    assert len(result) <= 100


def test_email_pii_removed(sanitizer: InputSanitizer) -> None:
    """Email addresses should be replaced with [EMAIL]."""
    text = "Contact me at john.doe@example.com for more info."
    result = sanitizer.sanitize(text)
    assert "john.doe@example.com" not in result
    assert "[EMAIL]" in result


def test_phone_pii_removed(sanitizer: InputSanitizer) -> None:
    """Phone numbers should be replaced with [PHONE]."""
    text = "Call me at 555-123-4567 or +1 (800) 555-0199"
    result = sanitizer.sanitize(text)
    assert "555-123-4567" not in result
    assert "[PHONE]" in result


def test_normal_text_preserved(sanitizer: InputSanitizer) -> None:
    """Normal text without PII should pass through unchanged."""
    text = "What is the difference between machine learning and deep learning?"
    result = sanitizer.sanitize(text)
    assert result == text


def test_detect_query_type_code(sanitizer: InputSanitizer) -> None:
    """Code-related queries should return 'code'."""
    assert sanitizer.detect_query_type("Write a Python function to sort a list") == "code"
    assert sanitizer.detect_query_type("Fix this bug in my JavaScript code") == "code"


def test_detect_query_type_medical(sanitizer: InputSanitizer) -> None:
    """Medical queries should return 'medical'."""
    assert sanitizer.detect_query_type("What are symptoms of diabetes?") == "medical"


def test_detect_query_type_creative(sanitizer: InputSanitizer) -> None:
    """Creative queries should return 'creative'."""
    assert sanitizer.detect_query_type("Write a poem about the ocean") == "creative"


def test_detect_query_type_analytical(sanitizer: InputSanitizer) -> None:
    """Analytical queries should return 'analytical'."""
    assert sanitizer.detect_query_type("Compare Docker vs Kubernetes") == "analytical"


def test_detect_query_type_default_factual(sanitizer: InputSanitizer) -> None:
    """Queries with no recognized keywords should default to 'factual'."""
    assert sanitizer.detect_query_type("What is the capital of France?") == "factual"


def test_validate_empty_returns_invalid(sanitizer: InputSanitizer) -> None:
    """Empty input should fail validation."""
    is_valid, msg = sanitizer.validate("")
    assert not is_valid
    assert msg is not None


def test_validate_very_short_returns_invalid(sanitizer: InputSanitizer) -> None:
    """Very short input (< 3 chars) should fail validation."""
    is_valid, msg = sanitizer.validate("Hi")
    assert not is_valid


def test_validate_normal_returns_valid(sanitizer: InputSanitizer) -> None:
    """Normal query should pass validation."""
    is_valid, msg = sanitizer.validate("What is quantum computing?")
    assert is_valid
    assert msg is None