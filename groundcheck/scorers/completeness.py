"""Completeness scorer -- planned for v0.2.

Will check whether the answer covers *all* parts of a multi-part
question (e.g. "What's the refund window and who pays return
shipping?" only half-answered). Not implemented yet; not wired into
evaluate() or EvalResult. Tracked here so the module layout is stable
for when it lands.
"""


def score_completeness(question: str, answer: str, contexts: list[str]) -> tuple[float, dict]:
    raise NotImplementedError("completeness scoring ships in groundcheck v0.2")
