# ragsanity

A sanity check for RAG (Retrieval-Augmented Generation) pipelines. One
function call, no setup.

```bash
pip install ragsanity
```

```python
from ragsanity import evaluate

result = evaluate(
    question="What is the refund policy?",
    contexts=["You can return items within 30 days of purchase for a full refund."],
    answer="The refund policy allows returns within 30 days of purchase.",
)

print(result.summary())
```

```
ragsanity result
------------------
Question:     What is the refund policy?
Faithfulness: 0.62
Relevancy:    1.00
Overall:      0.81
```

That's the whole setup. No API key, no config file, no model download, no
signup. Two lines to install and import, one function call to get a score.

## What it catches

`ragsanity` scores three failure modes independently, because they need
different fixes:

- **Faithfulness** — is the answer grounded in the retrieved context, or did
  the model add things that aren't there?
- **Relevancy** — does the answer actually address the question, or is it
  correct-but-off-topic?
- **Completeness** — for multi-part questions, did the answer cover every
  part, or did it quietly skip half of it?

```python
from ragsanity import evaluate

result = evaluate(
    question="What is the refund window and who pays for return shipping?",
    contexts=["Refunds are given within 30 days. The customer pays return shipping."],
    answer="The refund window is 30 days.",
)
print(result.summary())
```

```
ragsanity result
----------------
Question:     What is the refund window and who pays for return shipping?
Faithfulness: 0.50
Relevancy:    0.40
Completeness: 0.50
Overall:      0.47

```

`result.completeness_details["parts"]` tells you exactly which part got
skipped — here, "who pays for return shipping?" — instead of just a low
number with no explanation.

Here's faithfulness and relevancy catching a fully hallucinated, off-topic
answer, back to back:

```python
from ragsanity import evaluate

context = ["Our return policy allows customers to return items within 30 "
           "days of purchase for a full refund. Items must be unused and "
           "in original packaging."]

# A grounded, on-topic answer
good = evaluate(
    question="What is the refund policy?",
    contexts=context,
    answer="The refund policy allows returns within 30 days of purchase.",
)
print(good.faithfulness, good.relevancy)   # 0.62  1.0

# A hallucinated, off-topic answer
bad = evaluate(
    question="What is the refund policy?",
    contexts=context,
    answer="Our headquarters are located in San Francisco.",
)
print(bad.faithfulness, bad.relevancy)     # 0.0   0.0
```

No fixture, no mock, no API key required to reproduce that — copy it into a
`.py` file and run it right now.

## Why zero dependencies matters here

Most RAG-eval tooling needs an LLM API key for judge-based scoring, or pulls
in ML libraries (embedding models, BERT-based scorers) for semantic
comparison. Both are the right call when you need that level of nuance — but
they mean a real setup cost before you get your first number: API keys,
`.env` files, multi-GB model downloads, dependency conflicts in an existing
project.

`ragsanity` is deliberately the opposite tradeoff. It uses lexical overlap
— checking how much of an answer's meaningful vocabulary is grounded in the
context, and how much of a question's vocabulary the answer addresses — pure
Python, standard library only. That makes it:

- **Instant to install** — no extras, no optional dependency tree
- **Instant to run** — no network calls, no model to warm up, no cost
- **Safe to drop into CI** — nothing to leak, nothing to rate-limit
- **Honest about its limits** — see below

The tradeoff is real and worth naming: this is a proxy, not a semantic
judge. A well-paraphrased answer that shares little vocabulary with the
source can score lower than it "should." Use `ragsanity` as a fast first
pass — the thing you run on every commit or every response before reaching
for something heavier — not as your only evaluation layer.

## API

### `evaluate(question, contexts, answer, *, metrics=None)`

- `question` — `str`, the user's question.
- `contexts` — `list[str]`, the chunks your retriever returned.
- `answer` — `str`, the LLM's generated answer.
- `metrics` — optional `list[str]`, subset of
  `["faithfulness", "relevancy", "completeness"]` to run. Defaults to all
  three.

Returns an `EvalResult`.

### `EvalResult`

| Attribute | Type | Description |
|---|---|---|
| `faithfulness` | `float \| None` | 0-1 groundedness score |
| `relevancy` | `float \| None` | 0-1 question-alignment score |
| `completeness` | `float \| None` | 0-1 multi-part question coverage score |
| `overall` | `float \| None` | average of computed scores (property) |
| `faithfulness_details` | `dict` | per-sentence breakdown, for debugging |
| `relevancy_details` | `dict` | matched/unmatched keywords, for debugging |
| `completeness_details` | `dict` | per-part coverage breakdown, for debugging |
| `.summary()` | `str` | pretty-printed report |
| `.to_dict()` | `dict` | JSON-serializable version of the result |

## CLI

No Python needed for one-off checks:

```bash
ragsanity run examples.json
ragsanity run examples.json --json   # machine-readable output
```

`examples.json` can be a single object or a list of objects:

```json
{
  "question": "What is the refund policy?",
  "contexts": ["You can return items within 30 days of purchase."],
  "answer": "The refund policy allows returns within 30 days."
}
```

## Roadmap

- [x] `completeness` scorer — did the answer cover every part of a
      multi-part question? (v0.2)
- [ ] Optional LLM-judge backend for teams that want semantic (not just
      lexical) scoring, opt-in via an extra so the core install stays
      dependency-free
- [ ] Batch evaluation helpers with aggregate stats (mean, pass-rate at a
      threshold, etc.)

## Contributing

Issues and PRs welcome.

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
