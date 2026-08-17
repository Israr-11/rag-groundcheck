"""Faithfulness scorer.

Answers the question: "is the answer actually grounded in the retrieved
context, or did the model make stuff up?"

Implementation note: to keep ragsanity dependency-free, this uses a
lexical overlap heuristic rather than an LLM judge or embeddings model.
It splits the answer into sentences and checks how much of each
sentence's meaningful vocabulary is present in the combined context.
This is fast, deterministic, and needs no API key -- but it is a proxy
for faithfulness, not a perfect measure. Paraphrased or inferred claims
that don't share vocabulary with the context can score lower than they
"should." Swap in an LLM-based scorer later if you need more nuance.
"""

import re

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on",
    "for", "with", "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "at", "by", "from",
    "you", "your", "i", "we", "they", "he", "she", "them", "his", "her",
    "their", "our", "not", "no", "do", "does", "did", "so", "than", "then",
    "there", "here", "can", "will", "would", "should", "could", "may",
    "might", "must", "have", "has", "had", "which", "who", "whom", "what",
    "when", "where", "why", "how", "all", "any", "each", "into", "about",
}

_WORD_RE = re.compile(r"[a-zA-Z0-9']+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _tokenize(text: str) -> set[str]:
    """Lowercase, strip stopwords, return the set of meaningful words."""
    words = _WORD_RE.findall(text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def score_faithfulness(answer: str, contexts: list[str]) -> tuple[float, dict]:
    """Score how grounded `answer` is in `contexts`.

    Returns:
        (score, details) where score is in [0, 1] and details contains
        the per-sentence breakdown so callers can see *why* something
        scored low.
    """
    if not answer or not answer.strip():
        return 0.0, {"reason": "empty answer", "sentences": []}

    context_vocab = _tokenize(" ".join(contexts))
    if not context_vocab:
        return 0.0, {"reason": "no context provided", "sentences": []}

    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0, {"reason": "no sentences found", "sentences": []}

    breakdown = []
    sentence_scores = []
    for sentence in sentences:
        sentence_words = _tokenize(sentence)
        if not sentence_words:
            # SENTENCE HAS NO MEANINGFUL CONTENT WORDS (e.g. "Yes.") --
            # NEUTRAL, DON'T PENALIZE OR REWARD IT.
            continue
        supported = sentence_words & context_vocab
        ratio = len(supported) / len(sentence_words)
        sentence_scores.append(ratio)
        breakdown.append({
            "sentence": sentence,
            "support_ratio": round(ratio, 2),
            "grounded": ratio >= 0.5,
        })

    if not sentence_scores:
        return 1.0, {"reason": "no substantive claims to check", "sentences": breakdown}

    score = sum(sentence_scores) / len(sentence_scores)
    return round(score, 4), {"sentences": breakdown}
