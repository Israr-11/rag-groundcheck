"""Result object returned by evaluate()."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalResult:
    """Holds the scores and inputs for a single RAG evaluation.

    Attributes:
        question: The original user question.
        answer: The LLM-generated answer being evaluated.
        contexts: The retrieved chunks used to generate the answer.
        faithfulness: 0-1 score for how grounded the answer is in the
            provided contexts. None if the scorer was skipped.
        relevancy: 0-1 score for how well the answer addresses the
            question. None if the scorer was skipped.
        faithfulness_details: Optional breakdown/explanation from the
            faithfulness scorer (e.g. unsupported sentences).
        relevancy_details: Optional breakdown/explanation from the
            relevancy scorer.
    """

    question: str
    answer: str
    contexts: list[str]
    faithfulness: Optional[float] = None
    relevancy: Optional[float] = None
    completeness: Optional[float] = None
    faithfulness_details: dict = field(default_factory=dict)
    relevancy_details: dict = field(default_factory=dict)
    completeness_details: dict = field(default_factory=dict)

    @property
    def overall(self) -> Optional[float]:
        """Average of all computed scores. None if nothing was scored."""
        scores = [s for s in (self.faithfulness, self.relevancy, self.completeness) if s is not None]
        if not scores:
            return None
        return sum(scores) / len(scores)

    def to_dict(self) -> dict:
        """Serialize the result to a plain dict (e.g. for JSON logging)."""
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "faithfulness": self.faithfulness,
            "relevancy": self.relevancy,
            "completeness": self.completeness,
            "overall": self.overall,
            "faithfulness_details": self.faithfulness_details,
            "relevancy_details": self.relevancy_details,
            "completeness_details": self.completeness_details,
        }

    def summary(self) -> str:
        """Human-readable summary for print(result.summary())."""
        lines = ["ragsanity result", "-" * 16]
        lines.append(f"Question:     {self.question}")
        if self.faithfulness is not None:
            lines.append(f"Faithfulness: {self.faithfulness:.2f}")
        if self.relevancy is not None:
            lines.append(f"Relevancy:    {self.relevancy:.2f}")
        if self.completeness is not None:
            lines.append(f"Completeness: {self.completeness:.2f}")
        if self.overall is not None:
            lines.append(f"Overall:      {self.overall:.2f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        f = f"{self.faithfulness:.2f}" if self.faithfulness is not None else "None"
        r = f"{self.relevancy:.2f}" if self.relevancy is not None else "None"
        c = f"{self.completeness:.2f}" if self.completeness is not None else "None"
        return f"EvalResult(faithfulness={f}, relevancy={r}, completeness={c})"
