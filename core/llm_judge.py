"""
VeritasAI LLM Judge factuality and correctness scoring.
Queries a capable model (usually Groq Llama 3.3 70B) to critique all candidate responses
against the user query and retrieved grounding context in a single call to save latency and token costs.
"""

import re
import json
import asyncio
from typing import List, Dict, Optional
from core.result import LLMResult, LLMStatus
from core.adapter import LLMAdapter

JUDGE_SYSTEM_PROMPT = (
    "Role: Senior Factual Verification Judge & Hallucination Auditor.\n"
    "Task: Evaluate a set of candidate LLM responses against the provided Grounding Context and User Query.\n\n"
    "Guidelines:\n"
    "1. Rate each candidate response from 0.0 (contains severe contradictions, false assumptions, or hallucinations) "
    "to 1.0 (completely accurate, fully grounded in the context, and free of contradictions).\n"
    "2. Base your evaluation strictly on the facts presented in the Grounding Context. If the context does not support "
    "a claim, it should be penalized.\n"
    "3. Output your evaluation strictly as a valid JSON object matching the schema below. No other text or explanation.\n\n"
    "Output Schema:\n"
    "{\n"
    "  \"Response A\": 0.85,\n"
    "  \"Response B\": 0.40\n"
    "}"
)


class LLMJudge:
    """Evaluates candidate responses against grounding context using a supervisor LLM."""

    def __init__(self, judge_adapter: LLMAdapter) -> None:
        self.adapter = judge_adapter

    async def evaluate_responses(self, query: str, context: str, results: List[LLMResult]) -> Dict[str, float]:
        """
        Evaluate candidate responses against grounding context.
        
        Returns:
            Dict mapping model names to factuality scores [0.0 - 1.0]
        """
        # Default scores to neutral
        default_scores = {r.model: 0.5 for r in results}
        if not results:
            return {}
        if not context:
            # If no grounding context is available, judge is neutral
            return default_scores

        # Map models to anonymous labels
        model_map = {}
        candidate_blocks = []
        
        for idx, res in enumerate(results):
            label = f"Response {chr(65 + idx)}"  # Response A, Response B...
            model_map[label] = res.model
            candidate_blocks.append(f"### {label}\n{res.response}\n")

        candidates_str = "\n".join(candidate_blocks)

        composed_prompt = (
            f"{JUDGE_SYSTEM_PROMPT}\n\n"
            f"[GROUNDING CONTEXT]:\n{context}\n\n"
            f"[USER QUERY]:\n{query}\n\n"
            f"[CANDIDATE RESPONSES]:\n{candidates_str}\n\n"
            f"Output JSON:"
        )

        try:
            # Execute judge evaluation via adapter
            judge_res = await self.adapter.call(composed_prompt)
            if judge_res.status == LLMStatus.SUCCESS and judge_res.response:
                text = judge_res.response.strip()
                
                # Extract JSON block
                json_match = re.search(r'\{.*?\}', text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    
                    scores = {}
                    for label, score in parsed.items():
                        model_name = model_map.get(label)
                        if model_name:
                            scores[model_name] = round(float(score), 3)
                            
                    # Merge any missing models with default score
                    for res in results:
                        if res.model not in scores:
                            scores[res.model] = 0.5
                            
                    return scores
        except Exception:
            pass

        return default_scores
