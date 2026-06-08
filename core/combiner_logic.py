"""
VeritasAI Synthesis Logic Engine.
Builds synthesis prompts and follow-up question requests for the combiner stage.
"""

import re
from typing import List, Tuple
from core.result import LLMResult


def build_synthesis_prompt(question: str, trusted_results: List[LLMResult]) -> str:
    """
    Build the synthesis prompt that combines trusted model responses into one answer and
    generates 3 follow-up questions in a single step using XML tagging.
    """
    response_block = "\n".join(
        f"Candidate Answer:\n\"\"\"{r.response}\"\"\"\n" for r in trusted_results if r.response
    )
    return (
        f"Role: Chief Editor & Fact-Checker.\n"
        f"Task: Synthesize the provided trusted candidate answers into a single authoritative \"Veritas Response\" for the query: \"{question}\"\n\n"
        f"Synthesis Protocol:\n"
        f"1. Consensus Rule: If the majority of models agree on a fact, prioritize it.\n"
        f"2. Conflict Protocol: If numerical data or specific facts contradict, state the discrepancy (e.g., \"Sources vary between X and Y\").\n"
        f"3. Tone: Professional, encyclopedic, and neutral.\n\n"
        f"Trusted Input Data:\n{response_block}\n\n"
        f"Output Schema:\n"
        f"<final_answer>\n"
        f"[The synthesized response]\n"
        f"</final_answer>\n\n"
        f"<followup_questions>\n"
        f"- [Question 1]\n"
        f"- [Question 2]\n"
        f"- [Question 3]\n"
        f"</followup_questions>"
    )


def parse_synthesis_output(raw_text: str) -> Tuple[str, List[str]]:
    """
    Parse the synthesized answer and follow-up questions from the XML-delimited response.
    """
    # Extract the main answer
    answer_match = re.search(r'<final_answer>(.*?)</final_answer>', raw_text, re.DOTALL | re.IGNORECASE)
    answer = answer_match.group(1).strip() if answer_match else ""
    
    # Extract follow-ups
    followup_match = re.search(r'<followup_questions>(.*?)</followup_questions>', raw_text, re.DOTALL | re.IGNORECASE)
    questions = []
    if followup_match:
        for line in followup_match.group(1).strip().split('\n'):
            line = line.strip()
            # Strip leading bullets "-", "*", or numbers like "1."
            line_cleaned = re.sub(r'^[-*\d\.\)\s]+', '', line).strip()
            if line_cleaned:
                questions.append(line_cleaned)
    
    return answer, questions


def build_followup_prompt(question: str, answer: str) -> str:
    """Legacy helper (retained for safety)."""
    return ""


def parse_followup_questions(raw_text: str) -> List[str]:
    """Legacy helper (retained for safety)."""
    return []


def should_synthesize(trusted_count: int, total_count: int, consensus_threshold: float = 0.5) -> bool:
    """
    Determine whether the synthesis stage should run or be skipped (low consensus).
    """
    if total_count == 0 or trusted_count == 0:
        return False
    ratio = trusted_count / total_count
    return ratio >= consensus_threshold
