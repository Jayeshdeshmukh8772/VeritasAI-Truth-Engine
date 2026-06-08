"""
VeritasAI Synthesis Combiner Agent.
Consolidates verified model responses into one authoritative final answer.
Also generates follow-up question suggestions via a second LLM call.
"""

from typing import List, Tuple
from core.adapter import LLMAdapter
from core.result import LLMResult, LLMStatus
from core.combiner_logic import (
    build_synthesis_prompt,
    build_followup_prompt,
    parse_followup_questions,
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
        Synthesize trusted results into one coherent answer.

        Args:
            original_question: The enhanced query string
            trusted_results: LLMResults from models in the consensus cluster

        Returns:
            Synthesized answer string (or best single response if synthesis fails)
        """
        if not trusted_results:
            return "The verification engine could not reach a consensus. Please review the individual model responses below."

        # If only one trusted result, return it directly
        if len(trusted_results) == 1:
            return trusted_results[0].response or "No response available."

        synthesis_prompt = build_synthesis_prompt(original_question, trusted_results)

        for adapter in self.adapters:
            result = await adapter.call(synthesis_prompt)
            if result.status == LLMStatus.SUCCESS and result.response:
                return result.response.strip()

        # Last resort: return the response with highest trust score
        best = max(trusted_results, key=lambda r: r.trust_score)
        return best.response or trusted_results[0].response or "Synthesis failed."

    async def synthesize_with_followups(
        self, original_question: str, trusted_results: List[LLMResult]
    ) -> Tuple[str, List[str]]:
        """
        Synthesize answer AND generate follow-up question suggestions.

        Args:
            original_question: The enhanced query string
            trusted_results: LLMResults from models in the consensus cluster

        Returns:
            Tuple of (synthesized_answer: str, follow_up_questions: List[str])
        """
        # Stage 1: Synthesis
        answer = await self.synthesize(original_question, trusted_results)

        # Stage 2: Follow-up question generation
        follow_ups: List[str] = []
        if answer and len(answer) > 50:
            followup_prompt = build_followup_prompt(original_question, answer)
            for adapter in self.adapters:
                fu_result = await adapter.call(followup_prompt)
                if fu_result.status == LLMStatus.SUCCESS and fu_result.response:
                    follow_ups = parse_followup_questions(fu_result.response)
                    if follow_ups:
                        break

        # Default follow-ups if generation failed
        if not follow_ups:
            follow_ups = [
                "Can you provide more details on this topic?",
                "What are the practical implications of this?",
                "Are there any notable exceptions or edge cases?",
            ]

        return answer, follow_ups