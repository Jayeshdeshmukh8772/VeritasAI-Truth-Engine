"""
VeritasAI Synthesis Combiner Agent.
Consolidates verified model responses into one authoritative final answer.
Also generates follow-up question suggestions.
"""

from typing import List, Tuple
from core.adapter import LLMAdapter
from core.result import LLMResult, LLMStatus
from core.combiner_logic import (
    build_synthesis_prompt,
    parse_synthesis_output,
)


class ResultCombiner:
    """
    Synthesis agent that combines trusted model responses into a final answer
    and generates 3 follow-up question suggestions.
    """

    def __init__(self, fallback_adapters: List[LLMAdapter]) -> None:
        """
        Initialize the combiner with ordered fallback adapters.

        Args:
            fallback_adapters: List of adapters ordered by synthesis priority
                               (fastest/most trusted first; Groq is recommended first)
        """
        self.adapters = fallback_adapters

    async def synthesize(self, original_question: str, trusted_results: List[LLMResult]) -> str:
        """
        Legacy single-synthesis call for backward compatibility.
        """
        answer, _ = await self.synthesize_with_followups(original_question, trusted_results)
        return answer

    async def synthesize_with_followups(
        self, original_question: str, trusted_results: List[LLMResult]
    ) -> Tuple[str, List[str]]:
        """
        Synthesize answer AND generate follow-up question suggestions in a single step.

        Args:
            original_question: The enhanced query string
            trusted_results: LLMResults from models in the consensus cluster

        Returns:
            Tuple of (synthesized_answer: str, follow_up_questions: List[str])
        """
        default_followups = [
            "Can you provide more details on this topic?",
            "What are the practical implications of this?",
            "Are there any notable exceptions or edge cases?",
        ]

        if not trusted_results:
            return (
                "The verification engine could not reach a consensus. Please review the individual model responses below.",
                default_followups
            )

        # If only one trusted result, return it directly
        if len(trusted_results) == 1:
            best_res = trusted_results[0].response or "No response available."
            return best_res, default_followups

        synthesis_prompt = build_synthesis_prompt(original_question, trusted_results)

        for adapter in self.adapters:
            result = await adapter.call(synthesis_prompt)
            if result.status == LLMStatus.SUCCESS and result.response:
                answer, follow_ups = parse_synthesis_output(result.response)
                if answer:
                    if not follow_ups:
                        follow_ups = default_followups
                    return answer, follow_ups[:3]

        # Last resort: return the response with highest trust score
        best = max(trusted_results, key=lambda r: r.trust_score)
        fallback_answer = best.response or trusted_results[0].response or "Synthesis failed."
        return fallback_answer, default_followups