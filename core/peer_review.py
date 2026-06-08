"""
VeritasAI Social Signal Validator.
Anonymizes model answers and handles cross-critique blind peer evaluations.
"""

import asyncio
import re
import random
from typing import List, Dict
from core.adapter import LLMAdapter
from core.result import LLMResult, LLMStatus


class PeerReviewEngine:
    def __init__(self, adapter_map: Dict[str, LLMAdapter]):
        self.adapter_map = adapter_map

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
            rendered_payload += f"\nModel {lbl}:\n\"\"\"{successful_results[idx].response}\"\"\"\n"

        review_prompt = (
            f"Role: Objective Forensic Auditor.\n"
            f"Task: Evaluate the following candidate answers for the query: \"{question}\"\n\n"
            f"Evaluation Rubric:\n"
            f"1. Accuracy: Are the facts verifiable? \n"
            f"2. Completeness: Does it address all parts of the query?\n"
            f"3. Hallucination Check: Does it invent facts or cite non-existent data?\n\n"
            f"Candidates:\n{rendered_payload}\n"
            f"Instructions: \n"
            f"- Perform a brief internal critique for each model.\n"
            f"- Rank models from most trustworthy to least.\n"
            f"- Use model identifiers (e.g., Model A, Model B).\n\n"
            f"Output Schema:\n"
            f"<analysis>\n"
            f"[Brief 1-sentence critique per model]\n"
            f"</analysis>\n\n"
            f"<ranking>\n"
            f"[Model Name, Model Name, Model Name]\n"
            f"</ranking>"
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

            # 1. Try to extract content inside <ranking>...</ranking>
            ranking_content = ""
            ranking_match = re.search(r'<ranking>(.*?)</ranking>', review_res.response, re.DOTALL | re.IGNORECASE)
            if ranking_match:
                ranking_content = ranking_match.group(1).strip()
            else:
                # Fallback: find any bracketed list like [Model A, Model B]
                bracket_match = re.search(r'\[(.*?)\]', review_res.response, re.DOTALL)
                if bracket_match:
                    ranking_content = bracket_match.group(1).strip()

            # 2. Extract model letters (e.g. Model A -> A) from ranking content
            if ranking_content:
                extracted_letters = re.findall(r'(?:Model\s+)?([A-Z])', ranking_content, re.IGNORECASE)
                extracted_letters = [lbl.upper() for lbl in extracted_letters]
            else:
                # Fallback: search the entire response for Model A or Response A
                extracted_letters = re.findall(r'(?:Model|Response)\s+([A-Z])', review_res.response, re.IGNORECASE)
                extracted_letters = [lbl.upper() for lbl in extracted_letters]

            # Filter out labels that are not in our dynamic labels
            valid_extracted = [lbl for lbl in extracted_letters if lbl in labels]
            
            # Deduplicate while preserving order
            seen = set()
            unique_extracted = []
            for lbl in valid_extracted:
                if lbl not in seen:
                    seen.add(lbl)
                    unique_extracted.append(lbl)

            if len(unique_extracted) == n:
                for placement, lbl in enumerate(unique_extracted):
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