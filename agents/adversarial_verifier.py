"""
VeritasAI Adversarial Verifier and Reconstructor.
Implements the GAN-style Verification Node.
Stress-tests the initial consensus output to isolate weaknesses or assumptions,
then reconstructs a bulletproof final output.
"""

import json
from typing import Tuple, Optional, Dict, Any
from core.adapter import LLMAdapter
from core.result import LLMStatus

VERIFIER_PROMPT = (
    "Role: Senior Factual Adversarial Auditor and Truth Engine Verifier.\n"
    "Task: Rigorously audit the following candidate answer against the Original Query, Enhanced Query, and Grounding Context.\n"
    "You must extract exact quotes. Do not fabricate data. Refuse to cite sources without a direct supporting quote.\n\n"
    "Original Query:\n"
    "{original_query}\n\n"
    "Enhanced Query:\n"
    "{enhanced_query}\n\n"
    "Grounding Context:\n"
    "{context}\n\n"
    "Candidate Answer to Audit:\n"
    "{answer}\n\n"
    "Instructions:\n"
    "1. Intent Verification: Check if the Enhanced Query corrupted the Original Query's temporal intent (e.g. changing 'today' to a past date). If so, log a critical failure: 'temporal_corruption'.\n"
    "2. Proof of Retrieval: For every claim, extract the exact supporting quote from the Grounding Context. If no context exists, explicitly state 'NO EVIDENCE FOUND'. Do NOT invent numbers or timestamps. If a publication date or retrieval timestamp is not explicitly provided in the context, output exactly \"Unavailable\". NEVER fabricate or invent placeholder dates.\n"
    "3. Output a structured JSON response enclosed in <metrics> tags with your evaluation. Format strictly as JSON inside the tag.\n"
    "<metrics>\n"
    "{\n"
    "  \"critique\": \"Your strict textual critique here.\",\n"
    "  \"evidence_count\": 0,\n"
    "  \"critical_failures\": [],\n"
    "  \"citations\": [\n"
    "    {\n"
    "      \"url\": \"...\",\n"
    "      \"publication_date\": \"...\",\n"
    "      \"retrieval_timestamp\": \"...\",\n"
    "      \"exact_quote\": \"...\",\n"
    "      \"supports_claim\": true,\n"
    "      \"source_family\": \"...\"\n"
    "    }\n"
    "  ]\n"
    "}\n"
    "</metrics>\n\n"
    "Output:"
)

RECONSTRUCTOR_PROMPT = (
    "Role: Senior Consensus Synthesis Architect.\n"
    "Task: Reconstruct the consensus answer by addressing the critiques. Remove any ungrounded assumptions.\n\n"
    "Original Answer:\n"
    "{answer}\n\n"
    "Auditor Critique:\n"
    "{critique}\n\n"
    "Output the finalized markdown answer."
)

class AdversarialVerifier:
    def __init__(self, verifier_adapter: LLMAdapter) -> None:
        self.adapter = verifier_adapter

    async def verify_and_rebuild(self, original_query: str, enhanced_query: str, context: str, initial_answer: str) -> Tuple[str, str, Dict[str, Any]]:
        metrics_dict = {
            "evidence_count": 0,
            "critical_failures": [],
            "citations": []
        }
        
        if not initial_answer:
            return initial_answer, "No initial answer to verify.", metrics_dict

        prompt = VERIFIER_PROMPT.format(
            original_query=original_query,
            enhanced_query=enhanced_query,
            context=context,
            answer=initial_answer
        )
        res = await self.adapter.call(prompt)
        
        if res.status != LLMStatus.SUCCESS or not res.response:
            return initial_answer, "Auditor call failed.", metrics_dict

        output_text = res.response.strip()
        
        # Parse <metrics>
        import re
        match = re.search(r'<metrics>(.*?)</metrics>', output_text, re.DOTALL | re.IGNORECASE)
        critique_points = "NO_CRITIQUE"
        if match:
            try:
                metrics_json = json.loads(match.group(1).strip())
                critique_points = metrics_json.get("critique", "NO_CRITIQUE")
                metrics_dict["evidence_count"] = metrics_json.get("evidence_count", 0)
                metrics_dict["critical_failures"] = metrics_json.get("critical_failures", [])
                metrics_dict["citations"] = metrics_json.get("citations", [])
            except Exception:
                critique_points = output_text

        # Hard Refusal Rule: No Evidence -> No Claim
        if metrics_dict["evidence_count"] == 0:
            if "no_evidence" not in metrics_dict["critical_failures"]:
                metrics_dict["critical_failures"].append("no_evidence")
            return "⚠️ **Insufficient Evidence.** The truth engine could not find verifiable, authentic sources to support this claim.", "Hard refusal triggered: evidence_count == 0.", metrics_dict

        # Check for temporal corruption directly
        if "temporal_corruption" in metrics_dict["critical_failures"]:
            critique_points += "\nCRITICAL: The query optimizer corrupted the temporal intent."

        # Reconstruct if critiques exist
        if "NO_CRITIQUE" not in critique_points and len(critique_points) > 10:
            rebuild_prompt = RECONSTRUCTOR_PROMPT.format(answer=initial_answer, critique=critique_points)
            rebuild_res = await self.adapter.call(rebuild_prompt)
            if rebuild_res.status == LLMStatus.SUCCESS and rebuild_res.response:
                return rebuild_res.response.strip(), f"⚠️ Auditor Flagged Issues:\n{critique_points}", metrics_dict

        return initial_answer, "Auditor verified successfully.", metrics_dict
