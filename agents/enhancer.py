"""
VeritasAI Prompt Optimization Engine.
Transforms raw user queries into structured, context-rich prompts
optimized for multi-LLM dispatch. Uses Groq's fast llama-3.1-8b-instant model.
"""

from core.adapter import LLMAdapter
from core.result import EnhancedQuery, LLMStatus
from core.sanitizer import InputSanitizer


ENHANCER_SYSTEM_PROMPT = (
    "You are an expert prompt engineer. Rewrite the user's question to be maximally specific, "
    "detailed, and clear for a large language model. Preserve the user's intent exactly. "
    "Add relevant context, specify the type of answer expected, and clarify any ambiguities. "
    "Return only the rewritten question, nothing else."
)


class QuestionEnhancer:
    """Enhances raw user queries into optimized prompts for LLM dispatch."""

    def __init__(self, groq_adapter: LLMAdapter, sanitizer: InputSanitizer) -> None:
        """
        Initialize the question enhancer.

        Args:
            groq_adapter: A fast Groq adapter instance (llama-3.1-8b-instant preferred)
            sanitizer: InputSanitizer instance for cleaning raw input
        """
        self.adapter = groq_adapter
        self.sanitizer = sanitizer

    async def enhance(self, raw_query: str) -> EnhancedQuery:
        """
        Sanitize, validate, and enhance the user's raw query.

        Args:
            raw_query: The raw string from the Streamlit text input

        Returns:
            EnhancedQuery dataclass with original, enhanced text, and detected query type
        """
        # Step 1: Sanitize input (returns str, not tuple)
        cleaned_text = self.sanitizer.sanitize(raw_query)
        if not cleaned_text:
            return EnhancedQuery(original=raw_query, enhanced=raw_query, query_type="factual")

        # Step 2: Detect query type from cleaned text
        query_type = self.sanitizer.detect_query_type(cleaned_text)

        # Step 3: Build enhancement prompt (system + user query combined)
        composed_prompt = f"{ENHANCER_SYSTEM_PROMPT}\n\nUser Question:\n\"{cleaned_text}\""

        # Step 4: Call Groq for enhancement (fast model)
        result = await self.adapter.call(composed_prompt)

        # Step 5: Use enhanced text if successful, otherwise fall back to cleaned input
        enhanced_text = cleaned_text
        if result.status == LLMStatus.SUCCESS and result.response:
            candidate = result.response.strip()
            # Sanity check: enhanced text should be at least as long as original (roughly)
            if len(candidate) >= len(cleaned_text) * 0.5:
                enhanced_text = candidate

        return EnhancedQuery(
            original=cleaned_text,
            enhanced=enhanced_text,
            query_type=query_type,
        )