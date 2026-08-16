"""groundcheck: lightweight evaluation for RAG pipelines.

    from groundcheck import evaluate

    result = evaluate(
        question="What is the refund policy?",
        contexts=["You can return items within 30 days..."],
        answer="The refund policy allows returns within 30 days.",
    )
    print(result.summary())
"""

from groundcheck.evaluate import evaluate
from groundcheck.models.result import EvalResult

__version__ = "0.1.0"
__all__ = ["evaluate", "EvalResult"]
