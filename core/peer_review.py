"""
VeritasAI Social Signal Validator.
Anonymizes model answers and handles cross-critique blind peer evaluations.
"""

import asyncio
import re
import random
from typing import List, Dict
from core.adapter import LLMAdapter
from core.result import LLMResult, LLMStatus, PeerReviewResult

class PeerReviewEngine:
    def __init__(self, adapter_map: Dict[str, LLMAdapter]):
        self.adapter_map = adapter_map
        self.ranking_regex = re.compile(r'(?:FINAL RANKING:.*?)(?:\bResponse\s+([A-Z]))', re.DOTALL | re.IGNORECASE)

    async def run_review(self, question: str, successful_results: List[LLMResult]) -> List[LLMResult]:
        """Runs parallel zero-knowledge critique iterations across all operational models."""
        n = len(successful_results)
        if n < 2:
            # Fallback when response counts are insufficient to support relative peer matrices
            for res in successful_results:
                res.peer_rank_score = 0.5
            return successful_results

        # Generate unique letters to ensure complete blind evaluation isolation
        labels = [chr(65 + i) for i in range(n)]
        shuffled_indices = list(range(n))
        random.shuffle(shuffled_indices)

        # Build cross-reference maps to reconstruct identity lineages post-execution
        label_to_index = {}
        rendered_payload = ""
        for position, idx in enumerate(shuffled_indices):
            lbl = labels[position]
            label_to_index[lbl] = idx
            rendered_payload += f"\nResponse {lbl}:\n\"\"\"{successful_results[idx].response}\"\"\"\n"

        review_prompt = (
            f"You are evaluating AI responses. The question was:\n\"{question}\"\n"
            f"Below are {n} responses labeled A through {labels[-1]}:\n{rendered_payload}\n"
            "Evaluate each response for accuracy, completeness, and clarity.\n"
            "Respond ONLY with the following format nothing else:\n"
            "FINAL RANKING:\n"
            "1. Response [letter]\n"
            "2. Response [letter]\n"
            "(Best first, worst last. Every letter must appear exactly once.)"
        )

        tasks = []
        active_reviewers: List[str] = []

        for res in successful_results:
            adapter = self.adapter_map.get(res.model)
            if adapter:
                active_reviewers.append(res.model)
                tasks.append(adapter.call(review_prompt))

        review_outputs = await asyncio.gather(*tasks)
        
        # Initialize tabular accumulator matrices to track raw numerical placements
        rank_accumulators = {i: [] for i in range(n)}

        for idx, review_res in enumerate(review_outputs):
            if review_res.status != LLMStatus.SUCCESS or not review_res.response:
                continue

            extracted_letters = re.findall(r'Response\s+([A-Z])', review_res.response, re.IGNORECASE)
            
            # Filter structural malformations or incomplete ordinal extractions
            if len(set(extracted_letters)) == n and all(lbl in label_to_index for lbl in extracted_letters):
                for placement, lbl in enumerate(extracted_letters):
                    target_model_index = label_to_index[lbl]
                    rank_accumulators[target_model_index].append(placement)

        # Normalize relative positions into linear 0.0 - 1.0 confidence ranges
        for i, res in enumerate(successful_results):
            placements = rank_accumulators[i]
            if placements:
                avg_placement = sum(placements) / len(placements)
                # Formula scales inverse distance limits: 0 is worst, n-1 is best
                res.peer_rank_score = round(1.0 - (avg_placement / (n - 1)), 3) if n > 1 else 0.5
            else:
                res.peer_rank_score = 0.5

        return successful_results