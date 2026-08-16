"""Relevancy scorer.

Answers the question: "does the answer actually address what was asked,
or is it grounded-but-off-topic?" A faithful answer can still be
irrelevant (e.g. correctly quoting the context but answering a
different question than the one asked).

Same dependency-free lexical approach as faithfulness.py -- see that
module's docstring for the tradeoffs. Here we measure keyword overlap
between the question and the answer, plus a light penalty for
non-answers ("I don't know", "not mentioned", etc.) since those are
common RAG failure modes worth surfacing.
"""

from groundcheck.scorers.faithfulness import _tokenize

_NON_ANSWER_PHRASES = (
    "i don't know",
    "i do not know",
    "cannot answer",
    "can't answer",
    "not mentioned",
    "not provided",
    "no information",
    "not available in the context",
    "unable to answer",
)


def score_relevancy(question: str, answer: str) -> tuple[float, dict]:
    """Score how well `answer` addresses `question`.

    Returns:
        (score, details) where score is in [0, 1].
    """
    if not question or not question.strip():
        return 0.0, {"reason": "empty question"}
    if not answer or not answer.strip():
        return 0.0, {"reason": "empty answer"}

    lowered_answer = answer.lower()
    for phrase in _NON_ANSWER_PHRASES:
        if phrase in lowered_answer:
            return 0.0, {"reason": f"answer appears to be a non-answer ('{phrase}')"}

    question_words = _tokenize(question)
    answer_words = _tokenize(answer)

    if not question_words:
        return 0.0, {"reason": "question has no meaningful keywords"}
    if not answer_words:
        return 0.0, {"reason": "answer has no meaningful keywords"}

    overlap = question_words & answer_words
    # How much of the question's "topic" shows up in the answer.
    coverage = len(overlap) / len(question_words)

    details = {
        "question_keywords": sorted(question_words),
        "matched_keywords": sorted(overlap),
        "coverage": round(coverage, 2),
    }
    return round(min(coverage, 1.0), 4), details
