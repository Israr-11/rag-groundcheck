from ragsanity.scorers.completeness import score_completeness


def test_single_part_question_scores_complete():
    score, details = score_completeness(
        "What is the refund policy?", "Returns within 30 days."
    )
    assert score == 1.0
    assert details["reason"] == "single-part question"


def test_multi_part_question_fully_answered():
    question = "What is the refund window and who pays for return shipping?"
    answer = "The refund window is 30 days, and the customer pays for return shipping."
    score, details = score_completeness(question, answer)
    assert score == 1.0
    assert all(p["addressed"] for p in details["parts"])


def test_multi_part_question_half_answered():
    question = "What is the refund window and who pays for return shipping?"
    answer = "The refund window is 30 days."
    score, details = score_completeness(question, answer)
    assert 0.3 < score < 0.7
    addressed_flags = [p["addressed"] for p in details["parts"]]
    assert True in addressed_flags
    assert False in addressed_flags


def test_multi_sentence_question_split():
    question = "What is the refund window? Who pays for return shipping?"
    answer = "The refund window is 30 days."
    score, details = score_completeness(question, answer)
    assert len(details["parts"]) == 2
    assert score < 1.0


def test_empty_question_scores_zero():
    score, details = score_completeness("", "some answer")
    assert score == 0.0
    assert "reason" in details


def test_empty_answer_scores_zero():
    score, details = score_completeness("some question", "")
    assert score == 0.0
    assert "reason" in details
