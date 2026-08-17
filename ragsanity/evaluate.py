"""The main entry point: evaluate()."""

from ragsanity.models.result import EvalResult
from ragsanity.scorers.faithfulness import score_faithfulness
from ragsanity.scorers.relevancy import score_relevancy


def evaluate(
    question: str,
    contexts: list[str],
    answer: str,
    *,
    metrics: list[str] | None = None,
) -> EvalResult:
    """Evaluate a single RAG (question, contexts, answer) triple.

    Args:
        question: The user's original question.
        contexts: The chunks retrieved by your retriever, as a list of
            strings (in whatever order your pipeline returned them).
        answer: The answer your LLM generated using those contexts.
        metrics: Which scorers to run. Defaults to all available
            metrics: ["faithfulness", "relevancy"]. Pass a subset to
            skip scorers you don't need (e.g. metrics=["relevancy"]
            skips the faithfulness check).

    Returns:
        An EvalResult with the requested scores populated. Scores for
        metrics not requested are left as None.

    Example:
        >>> from ragsanity import evaluate
        >>> result = evaluate(
        ...     question="What is the refund policy?",
        ...     contexts=["You can return items within 30 days of purchase."],
        ...     answer="The refund policy allows returns within 30 days.",
        ... )
        >>> result.faithfulness > 0.5
        True
    """
    if not isinstance(contexts, list):
        raise TypeError("contexts must be a list of strings")

    available_metrics = {"faithfulness", "relevancy"}
    selected = set(metrics) if metrics is not None else available_metrics
    unknown = selected - available_metrics
    if unknown:
        raise ValueError(
            f"Unknown metric(s): {sorted(unknown)}. "
            f"Available metrics: {sorted(available_metrics)}"
        )

    result = EvalResult(question=question, answer=answer, contexts=contexts)

    if "faithfulness" in selected:
        result.faithfulness, result.faithfulness_details = score_faithfulness(
            answer, contexts
        )

    if "relevancy" in selected:
        result.relevancy, result.relevancy_details = score_relevancy(
            question, answer
        )

    return result
