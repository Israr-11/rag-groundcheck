"""Minimal example: evaluate one RAG response.

Run with:
    python examples/basic_usage.py
"""

from ragsanity import evaluate

result = evaluate(
    question="What is the refund policy?",
    contexts=[
        "Our return policy allows customers to return items within 30 "
        "days of purchase for a full refund. Items must be unused and "
        "in original packaging."
    ],
    answer="The refund policy allows returns within 30 days of purchase.",
)

print(result.summary())
print()
print("Faithfulness detail:", result.faithfulness_details)
print("Relevancy detail:", result.relevancy_details)

# Example of a bad answer, for contrast.
bad_result = evaluate(
    question="What is the refund policy?",
    contexts=[
        "Our return policy allows customers to return items within 30 "
        "days of purchase for a full refund."
    ],
    answer="Our headquarters are located in San Francisco.",
)

print()
print(bad_result.summary())
