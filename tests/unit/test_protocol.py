import pytest
from harness.protocol import RAGEvaluable, validate_evaluable


class MockRAG:
    """Minimal compliant implementation."""

    def run(self, question: str, contexts_override: list[str] | None = None) -> dict:
        return {
            "answer": f"Answer to: {question}",
            "contexts": contexts_override or ["mock context"],
            "retrieved_doc_ids": [],
        }


class LangChainRAG:
    """Simulate a LangChain chain — doesn't implement protocol directly."""

    def invoke(self, inputs: dict) -> dict:
        return {"result": "answer", "source_documents": []}


def test_mock_rag_satisfies_protocol():
    rag = MockRAG()
    assert isinstance(rag, RAGEvaluable)


def test_langchain_rag_does_not_satisfy_protocol():
    rag = LangChainRAG()
    assert not isinstance(rag, RAGEvaluable)


def test_validate_evaluable_passes():
    rag = MockRAG()
    validate_evaluable(rag)  # should not raise


def test_validate_evaluable_raises_for_noncompliant():
    rag = LangChainRAG()
    with pytest.raises(TypeError, match="does not implement RAGEvaluable"):
        validate_evaluable(rag)


def test_mock_rag_returns_required_keys():
    rag = MockRAG()
    result = rag.run("What is RAG?")
    assert "answer" in result
    assert "contexts" in result
    assert isinstance(result["answer"], str)
    assert isinstance(result["contexts"], list)
