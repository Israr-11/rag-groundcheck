from groundcheck.scorers.faithfulness import score_faithfulness


def test_grounded_answer_scores_high():
    contexts = ["The Eiffel Tower is located in Paris, France, and was completed in 1889."]
    answer = "The Eiffel Tower is in Paris and was completed in 1889."
    score, details = score_faithfulness(answer, contexts)
    assert score > 0.7
    assert details["sentences"][0]["grounded"] is True


def test_fabricated_answer_scores_low():
    contexts = ["The Eiffel Tower is located in Paris, France."]
    answer = "The Eiffel Tower was built by aliens in the year 3000."
    score, _ = score_faithfulness(answer, contexts)
    assert score < 0.5


def test_empty_answer_scores_zero():
    score, details = score_faithfulness("", ["some context"])
    assert score == 0.0
    assert "reason" in details


def test_empty_context_scores_zero():
    score, details = score_faithfulness("some answer", [])
    assert score == 0.0
    assert "reason" in details


def test_multi_sentence_answer_averages_sentences():
    contexts = ["Cats are mammals. Cats sleep up to 16 hours a day."]
    answer = "Cats are mammals. Cats can fly to the moon."
    score, details = score_faithfulness(answer, contexts)
    assert len(details["sentences"]) == 2
    assert details["sentences"][0]["grounded"] is True
    assert details["sentences"][1]["grounded"] is False
