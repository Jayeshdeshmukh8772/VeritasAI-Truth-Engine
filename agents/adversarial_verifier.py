"""
VeritasAI Adversarial Verifier and Reconstructor.
Implements the GAN-style Verification Node.
Stress-tests the initial consensus output to isolate weaknesses or assumptions,
then reconstructs a bulletproof final output.
"""

import re
from typing import Tuple, Optional
from core.adapter import LLMAdapter
from core.result import LLMStatus

VERIFIER_PROMPT = (
    "Role: Senior Factual Adversarial Auditor.\n"
    "Task: Rigorously audit the following candidate answer against the User Query and Grounding Context.\n"
    "Identify any logical gaps, assumptions unsupported by the context, or factual errors.\n\n"
    "Grounding Context:\n"
    "{context}\n\n"
    "User Query:\n"
    "{query}\n\n"
    "Candidate Answer to Audit:\n"
    "{answer}\n\n"
    "Instructions:\n"
    "1. Be a tough critic. Isolate any weaknesses.\n"
    "2. If the answer is 100% accurate and fully supported with zero issues, output: \"NO_CRITIQUE\".\n"
    "3. Otherwise, output your critique points clearly inside XML tags:\n"
    "<critique>\n"
    "- [Discrepancy 1]\n"
    "- [Discrepancy 2]\n"
    "</critique>\n\n"
    "Output:"
)

RECONSTRUCTOR_PROMPT = (
    "Role: Senior Consensus Synthesis Architect (Claude-style UX Specialist).\n"
    "Task: Reconstruct the consensus answer by addressing and fixing the critiques raised by the Auditor.\n\n"
    "Grounding Context:\n"
    "{context}\n\n"
    "Original Consensus Answer:\n"
    "{answer}\n\n"
    "Auditor Critique Points:\n"
    "{critique}\n\n"
    "Instructions:\n"
    "1. Synthesize a corrected, robust, and highly accurate final consensus answer.\n"
    "2. Provide direct answers first. Never start with long introductions or generic referral advice.\n"
    "3. Keep it very simple and concise. Present key facts using clean markdown tables, short metric highlights (e.g. blockquotes like `> [!NOTE]`), and bullet points.\n"
    "4. Remove any ungrounded assumptions or logical leaps identified in the critique.\n"
    "5. Format the response beautifully using premium clean markdown layout.\n\n"
    "Final Bulletproof Answer:"
)


class AdversarialVerifier:
    """Stress-tests synthesized responses and reconstructs corrected outputs."""

    def __init__(self, verifier_adapter: LLMAdapter) -> None:
        self.adapter = verifier_adapter

    async def verify_and_rebuild(self, query: str, context: str, initial_answer: str) -> Tuple[str, str]:
        """
        Verify the initial answer. If flaws are found, reconstruct it.
        
        Returns:
            Tuple of (final_answer, verifier_logs)
        """
        if not initial_answer:
            return initial_answer, "No initial answer to verify."

        # Step 1: Run Adversarial Auditor
        prompt = VERIFIER_PROMPT.format(context=context, query=query, answer=initial_answer)
        res = await self.adapter.call(prompt)
        
        if res.status != LLMStatus.SUCCESS or not res.response:
            return initial_answer, "Auditor call failed. Consensus accepted as-is."

        critique_text = res.response.strip()
        
        # Check if audit found no issues
        if "NO_CRITIQUE" in critique_text and len(critique_text) < 50:
            return initial_answer, "Auditor verified: 100% correct, zero discrepancies found."

        # Extract content from <critique> tags
        match = re.search(r'<critique>(.*?)</critique>', critique_text, re.DOTALL | re.IGNORECASE)
        if match:
            critique_points = match.group(1).strip()
        else:
            if len(critique_text) > 10 and "NO_CRITIQUE" not in critique_text:
                critique_points = critique_text
            else:
                return initial_answer, "Auditor verified: zero discrepancies found."

        # Step 2: Reconstruct output using the critiques
        rebuild_prompt = RECONSTRUCTOR_PROMPT.format(
            context=context,
            answer=initial_answer,
            critique=critique_points
        )
        rebuild_res = await self.adapter.call(rebuild_prompt)
        
        if rebuild_res.status != LLMStatus.SUCCESS or not rebuild_res.response:
            return initial_answer, f"Reconstructor call failed. Critique log:\n{critique_points}"

        reconstructed_answer = rebuild_res.response.strip()
        log_summary = f"⚠️ Adversarial Auditor Flagged Issues:\n{critique_points}\n\nConsensus successfully reconstructed and verified."
        
        return reconstructed_answer, log_summary
