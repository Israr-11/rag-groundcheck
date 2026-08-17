from ragsanity import EvalResult, evaluate


def test_evaluate_returns_eval_result():
    result = evaluate(
        question="What is the refund policy?",
        contexts=["You can return items within 30 days of purchase."],
        answer="The refund policy allows returns within 30 days.",
    )
    assert isinstance(result, EvalResult)
    assert result.faithfulness is not None
    assert result.relevancy is not None


def test_evaluate_good_answer_scores_high():
    result = evaluate(
        question="What is the refund policy?",
        contexts=["You can return items within 30 days of purchase for a full refund."],
        answer="The refund policy allows returns within 30 days of purchase.",
    )
    # Lexical-overlap scoring: paraphrased words ("policy", "allows")
    # won't all match the context verbatim, so we check "clearly grounded"
    # rather than near-perfect.
    assert result.faithfulness > 0.5
    assert result.relevancy > 0.7


def test_evaluate_off_topic_answer_scores_low():
    result = evaluate(
        question="What is the refund policy?",
        contexts=["You can return items within 30 days of purchase."],
        answer="Our headquarters are located in San Francisco.",
    )
    assert result.relevancy < 0.3


def test_evaluate_metrics_filter():
    result = evaluate(
        question="What is the refund policy?",
        contexts=["You can return items within 30 days of purchase."],
        answer="The refund policy allows returns within 30 days.",
        metrics=["relevancy"],
    )
    assert result.relevancy is not None
    assert result.faithfulness is None
    assert result.completeness is None


def test_evaluate_runs_completeness_by_default():
    result = evaluate(
        question="What is the refund window and who pays for return shipping?",
        contexts=["Refunds are given within 30 days. The customer pays return shipping."],
        answer="The refund window is 30 days.",
    )
    assert result.completeness is not None
    assert result.completeness < 1.0


def test_evaluate_rejects_unknown_metric():
    try:
        evaluate(
            question="Q",
            contexts=["C"],
            answer="A",
            metrics=["not_a_real_metric"],
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_evaluate_rejects_non_list_contexts():
    try:
        evaluate(question="Q", contexts="not a list", answer="A")
        assert False, "expected TypeError"
    except TypeError:
        pass


def test_result_overall_average():
    result = EvalResult(
        question="Q", answer="A", contexts=["C"], faithfulness=0.8, relevancy=0.6
    )
    assert result.overall == 0.7


def test_result_summary_contains_scores():
    result = EvalResult(
        question="Q", answer="A", contexts=["C"], faithfulness=0.8, relevancy=0.6
    )
    summary = result.summary()
    assert "0.80" in summary
    assert "0.60" in summary
