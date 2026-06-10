"""
VeritasAI Prompt Optimization Engine.
Transforms raw user queries into structured, context-rich prompts
optimized for multi-LLM dispatch. Uses Groq's fast llama-3.1-8b-instant model.
"""

from core.adapter import LLMAdapter
from core.result import EnhancedQuery, LLMStatus
from core.sanitizer import InputSanitizer


ENHANCER_SYSTEM_PROMPT = (
    "Role: Senior Prompt Engineer & Information Architect.\n"
    "Task: Rewrite the following user query into a \"Perfect Prompt\" for an LLM ensemble.\n\n"
    "Guidelines:\n"
    "1. Disambiguate: Identify and clarify vague terms.\n"
    "2. Contextualize: Add necessary background or domain-specific terminology.\n"
    "3. Constraint: Specify that the answer should be objective, data-driven, and cite sources if possible.\n"
    "4. Temporal Preservation: NEVER corrupt or hardcode dates. If the user asks for 'today' or 'current', preserve that temporal intent exactly. Do not convert it into a stale past date (e.g., changing 'today' to '2023').\n"
    "5. Formatting: Ensure the output is a singular, direct instruction.\n\n"
    "Constraint: You must output ONLY the rewritten prompt inside the XML tags provided. No preamble.\n\n"
    "Output Schema:\n"
    "<enhanced_query>\n"
    "[Your optimized prompt here]\n"
    "</enhanced_query>"
)

CLASSIFIER_SYSTEM_PROMPT = (
    "Role: Input Classification Engine.\n"
    "Task: Classify the user's input into exactly one of the following strict categories:\n"
    "1. QUESTION: A factual, analytical, or general question directed at you.\n"
    "2. COMMAND: An instruction to perform an action.\n"
    "3. DOCUMENT: A pasted article, essay, or block of prose text without a clear question.\n"
    "4. LOG: A server log, stack trace, error message, or diagnostic output.\n"
    "5. CODE: A snippet of programming code.\n"
    "6. SYSTEM_OUTPUT: Pasted UI output, system metrics, or structured tool outputs.\n\n"
    "Output Schema: Output ONLY the classification category string inside <category> tags. No preamble.\n"
    "<category>\n"
    "[CATEGORY]\n"
    "</category>"
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

        # Step 3: Classify input type
        classifier_prompt = f"{CLASSIFIER_SYSTEM_PROMPT}\n\nUser Input:\n\"{cleaned_text}\""
        classifier_result = await self.adapter.call(classifier_prompt)
        input_type = "QUESTION"
        if classifier_result.status == LLMStatus.SUCCESS and classifier_result.response:
            import re
            match = re.search(r'<category>(.*?)</category>', classifier_result.response, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip().upper()
                if extracted in ["QUESTION", "COMMAND", "DOCUMENT", "LOG", "CODE", "SYSTEM_OUTPUT"]:
                    input_type = extracted

        # Step 4: Skip enhancement if input is non-query
        if input_type in ["DOCUMENT", "LOG", "CODE", "SYSTEM_OUTPUT"]:
            return EnhancedQuery(
                original=cleaned_text,
                enhanced=cleaned_text,
                query_type=query_type,
                input_type=input_type,
                enhancement_skipped=True,
                enhancement_reason=f"Input classified as {input_type}"
            )

        # Step 5: Build enhancement prompt (system + user query combined)
        composed_prompt = f"{ENHANCER_SYSTEM_PROMPT}\n\nUser Question:\n\"{cleaned_text}\""

        # Step 6: Call Groq for enhancement (fast model)
        result = await self.adapter.call(composed_prompt)

        # Step 7: Use enhanced text if successful, otherwise fall back to cleaned input
        enhanced_text = cleaned_text
        if result.status == LLMStatus.SUCCESS and result.response:
            candidate = result.response.strip()
            # Extract content from <enhanced_query> tags using re.DOTALL
            import re
            match = re.search(r'<enhanced_query>(.*?)</enhanced_query>', candidate, re.DOTALL | re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
            # Sanity check: enhanced text should be at least as long as original (roughly)
            if len(candidate) >= len(cleaned_text) * 0.5:
                enhanced_text = candidate

        return EnhancedQuery(
            original=cleaned_text,
            enhanced=enhanced_text,
            query_type=query_type,
            input_type=input_type,
            enhancement_skipped=False,
            enhancement_reason=""
        )