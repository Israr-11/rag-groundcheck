"""Completeness scorer.

Answers the question: "did the answer address every part of a
multi-part question, or did it quietly skip half of it?" A faithful,
relevant answer can still be incomplete -- e.g. answering "What's the
refund window?" while ignoring "...and who pays return shipping?" in
the same question.

Same dependency-free lexical approach as faithfulness.py and
relevancy.py -- see faithfulness.py's docstring for the tradeoffs.
Here we split the question into parts, then measure keyword overlap
between each part and the answer.
"""

import re

from ragsanity.scorers.faithfulness import _tokenize

_WH_WORDS = ("what", "who", "when", "where", "why", "how", "which", "whose", "whom")
_CONJUNCTION_RE = re.compile(r"\band\b|\bor\b", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[?.!])\s+")


def _split_question_parts(question: str) -> list[str]:
    """Split a question into its constituent parts.

    Handles two common multi-part shapes:
      - Separate sentences: "What's the refund window? Who pays shipping?"
      - Joined by and/or: "What's the refund window and who pays shipping?"

    The and/or split only triggers when the sentence has 2+ question
    words, to avoid wrongly splitting things like "black and white"
    inside an otherwise single-part question.
    """
    sentences = [s.strip() for s in _SENTENCE_RE.split(question.strip()) if s.strip()]
    parts = []
    for sentence in sentences:
        wh_count = sum(
            1 for w in _WH_WORDS if re.search(rf"\b{w}\b", sentence, re.IGNORECASE)
        )
        if wh_count >= 2:
            sub_parts = _CONJUNCTION_RE.split(sentence)
            parts.extend(p.strip() for p in sub_parts if p.strip())
        else:
            parts.append(sentence)
    return parts if parts else [question.strip()]


def score_completeness(question: str, answer: str, contexts: list[str] | None = None) -> tuple[float, dict]:
    """Score how completely `answer` addresses every part of `question`.

    `contexts` is accepted for API symmetry with the other scorers and
    for future use (e.g. checking whether an unanswered part even has
    supporting context) but isn't required for the current heuristic.

    Returns:
        (score, details) where score is in [0, 1].
    """
    if not question or not question.strip():
        return 0.0, {"reason": "empty question"}
    if not answer or not answer.strip():
        return 0.0, {"reason": "empty answer"}

    parts = _split_question_parts(question)
    answer_words = _tokenize(answer)

    if len(parts) <= 1:
        return 1.0, {"reason": "single-part question", "parts": []}

    if not answer_words:
        return 0.0, {"reason": "answer has no meaningful keywords", "parts": []}

    breakdown = []
    coverages = []
    for part in parts:
        part_words = _tokenize(part)
        if not part_words:
            continue
        overlap = part_words & answer_words
        coverage = len(overlap) / len(part_words)
        coverages.append(coverage)
        breakdown.append({
            "part": part,
            "coverage": round(coverage, 2),
            "addressed": coverage >= 0.4,
        })

    if not coverages:
        return 1.0, {"reason": "no substantive parts to check", "parts": breakdown}

    score = sum(coverages) / len(coverages)
    return round(score, 4), {"parts": breakdown}
