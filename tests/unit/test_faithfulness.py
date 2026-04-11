import json
import pytest
from unittest.mock import patch, MagicMock
from app.eval.faithfulness import compute_faithfulness


MOCK_FULL_SUPPORT = json.dumps({
    "claims": ["X is true", "Y happened"],
    "supported": [True, True]
})

MOCK_PARTIAL_SUPPORT = json.dumps({
    "claims": ["X is true", "Y happened"],
    "supported": [True, False]
})

MOCK_NO_CLAIMS = json.dumps({
    "claims": [],
    "supported": []
})

MOCK_MALFORMED = "not json at all"

MOCK_MARKDOWN_WRAPPED = '```json\n{"claims": ["X"], "supported": [true]}\n```'


def _mock_llm(response: str):
    m = MagicMock()
    m.generate.return_value = response
    return m


def test_faithfulness_full_support():
    with patch("app.eval.faithfulness.LLMClient", return_value=_mock_llm(MOCK_FULL_SUPPORT)):
        result = compute_faithfulness("Some answer.", ["Some context."])
    assert result["score"] == pytest.approx(1.0)
    assert result["claims"] == ["X is true", "Y happened"]
    assert result["supported"] == [True, True]


def test_faithfulness_partial_support():
    with patch("app.eval.faithfulness.LLMClient", return_value=_mock_llm(MOCK_PARTIAL_SUPPORT)):
        result = compute_faithfulness("Some answer.", ["Some context."])
    assert result["score"] == pytest.approx(0.5)


def test_faithfulness_no_claims_returns_neutral():
    with patch("app.eval.faithfulness.LLMClient", return_value=_mock_llm(MOCK_NO_CLAIMS)):
        result = compute_faithfulness("OK.", ["Some context."])
    assert result["score"] == pytest.approx(1.0)
    assert result["claims"] == []


def test_faithfulness_malformed_response_returns_zero():
    with patch("app.eval.faithfulness.LLMClient", return_value=_mock_llm(MOCK_MALFORMED)):
        result = compute_faithfulness("Some answer.", ["Some context."])
    assert result["score"] == pytest.approx(0.0)
    assert result["claims"] == []


def test_faithfulness_strips_markdown_codeblock():
    with patch("app.eval.faithfulness.LLMClient", return_value=_mock_llm(MOCK_MARKDOWN_WRAPPED)):
        result = compute_faithfulness("X is true.", ["X is true."])
    assert result["score"] == pytest.approx(1.0)
