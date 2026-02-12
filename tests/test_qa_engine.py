from qa_engine import should_use_fallback, FALLBACK_ANSWER


def test_should_use_fallback_when_no_docs():
    assert should_use_fallback(scored_docs=[], confidence=0.9, threshold=0.5) is True


def test_should_use_fallback_when_confidence_low():
    docs = [("doc", 0.1)]
    assert should_use_fallback(scored_docs=docs, confidence=0.4, threshold=0.5) is True


def test_should_not_use_fallback_when_confident():
    docs = [("doc", 0.1)]
    assert should_use_fallback(scored_docs=docs, confidence=0.7, threshold=0.5) is False


def test_fallback_answer_constant():
    assert "I don't know" in FALLBACK_ANSWER

