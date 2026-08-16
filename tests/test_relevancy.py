from groundcheck.scorers.relevancy import score_relevancy


def test_relevant_answer_scores_high():
    question = "What is the refund policy?"
    answer = "The refund policy gives you 30 days to return an item."
    score, _ = score_relevancy(question, answer)
    assert score > 0.7


def test_off_topic_answer_scores_low():
    question = "What is the refund policy?"
    answer = "Our headquarters are located in San Francisco."
    score, _ = score_relevancy(question, answer)
    assert score < 0.3


def test_non_answer_scores_zero():
    question = "What is the refund policy?"
    answer = "I don't know."
    score, details = score_relevancy(question, answer)
    assert score == 0.0
    assert "non-answer" in details["reason"]


def test_empty_question_scores_zero():
    score, details = score_relevancy("", "some answer")
    assert score == 0.0
    assert "reason" in details


def test_empty_answer_scores_zero():
    score, details = score_relevancy("some question", "")
    assert score == 0.0
    assert "reason" in details
