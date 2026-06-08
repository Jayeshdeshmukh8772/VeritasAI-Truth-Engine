"""
VeritasAI Synthesis Logic Engine.
Builds synthesis prompts and follow-up question requests for the combiner stage.
Handles the low-consensus fallback path where synthesis is skipped.
"""

from typing import List
from core.result import LLMResult


def build_synthesis_prompt(question: str, trusted_results: List[LLMResult]) -> str:
    """
    Build the synthesis prompt that combines trusted model responses into one answer.

    Args:
        question: The original or enhanced question string
        trusted_results: List of LLMResult objects with status=SUCCESS and not flagged as outliers

    Returns:
        Complete synthesis prompt string ready to send to the combiner LLM
    """
    response_block = "\n".join(
        f"- {r.response}" for r in trusted_results if r.response
    )
    return (
        f"You are a synthesis agent. Below are {len(trusted_results)} responses from different "
        f"AI models that broadly agree on the answer to this question:\n\n"
        f"Question: {question}\n\n"
        f"Responses to synthesize:\n{response_block}\n\n"
        "Combine them into one accurate, complete, and well-structured answer. "
        "Remove redundancy. Preserve important nuances from each response. "
        "Do not add information not present in the responses above. "
        "Write clearly and concisely. Use markdown formatting where helpful."
    )


def build_followup_prompt(question: str, answer: str) -> str:
    """
    Build the follow-up question generation prompt.

    Args:
        question: The original question the user asked
        answer: The synthesized answer produced by the combiner

    Returns:
        Prompt string requesting exactly 3 follow-up questions
    """
    return (
        f"Given this question and its answer, generate exactly 3 thoughtful follow-up questions "
        f"a curious user might ask next.\n\n"
        f"Original Question: {question}\n\n"
        f"Answer Summary: {answer[:500]}...\n\n"
        "Return ONLY the 3 questions as a numbered list:\n"
        "1. [question]\n"
        "2. [question]\n"
        "3. [question]\n"
        "Nothing else."
    )


def parse_followup_questions(raw_text: str) -> List[str]:
    """
    Parse follow-up questions from the LLM's raw numbered list response.

    Args:
        raw_text: The raw string output from the LLM containing numbered questions

    Returns:
        List of up to 3 cleaned question strings
    """
    import re
    lines = raw_text.strip().split("\n")
    questions: List[str] = []
    pattern = re.compile(r"^\s*\d+[\.\)]\s*(.+)$")
    for line in lines:
        match = pattern.match(line)
        if match:
            q = match.group(1).strip()
            if q:
                questions.append(q)
        if len(questions) >= 3:
            break
    return questions


def should_synthesize(trusted_count: int, total_count: int, consensus_threshold: float = 0.5) -> bool:
    """
    Determine whether the synthesis stage should run or be skipped (low consensus).

    Args:
        trusted_count: Number of trusted (non-outlier) results
        total_count: Total number of successful model results
        consensus_threshold: Ratio below which synthesis is skipped

    Returns:
        True if synthesis should proceed, False if low-consensus path should be taken
    """
    if total_count == 0 or trusted_count == 0:
        return False
    ratio = trusted_count / total_count
    return ratio >= consensus_threshold
