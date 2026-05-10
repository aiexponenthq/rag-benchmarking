from unittest.mock import MagicMock, patch

from app.sdk.client import RagEval


def test_sdk_evaluate_list_of_dicts():
    """Basic SDK usage — evaluate a list of samples."""
    client = RagEval(api_url="http://localhost:5001", api_key="test")
    samples = [
        {
            "question": "What is RAG?",
            "contexts": ["RAG is Retrieval-Augmented Generation."],
            "answer": "RAG combines retrieval with generation.",
        }
    ]
    with patch("app.sdk.client.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "scores": {"faithfulness": 0.9},
            "run_id": "run-001",
        }
        mock_post.return_value.raise_for_status = MagicMock()
        report = client.evaluate(samples, metrics=["faithfulness"])
    assert report["scores"]["faithfulness"] == 0.9


def test_sdk_from_langchain_style():
    """SDK from_langchain adapter converts chain output to EvalSample dict."""

    class FakeDoc:
        page_content = "RAG is Retrieval-Augmented Generation."
        metadata = {"id": "doc-1"}

    chain_output = {
        "query": "What is RAG?",
        "result": "RAG is a technique combining retrieval with generation.",
        "source_documents": [FakeDoc()],
    }
    sample = RagEval.from_langchain(chain_output)
    assert sample["question"] == "What is RAG?"
    assert len(sample["contexts"]) == 1
    assert sample["contexts"][0] == "RAG is Retrieval-Augmented Generation."
    assert sample["retrieved_doc_ids"] == ["doc-1"]


def test_sdk_from_llamaindex_style():
    """SDK from_llamaindex adapter converts query engine response."""

    class FakeNode:
        text = "RAG stands for Retrieval-Augmented Generation."
        node_id = "node-abc"

        def get_content(self):
            return self.text

    class FakeResponse:
        source_nodes = [FakeNode()]

        def __str__(self):
            return "RAG is Retrieval-Augmented Generation."

    sample = RagEval.from_llamaindex(FakeResponse(), "What is RAG?")
    assert sample["question"] == "What is RAG?"
    assert len(sample["contexts"]) == 1
    assert "node-abc" in sample["retrieved_doc_ids"]


def test_sdk_list_runs():
    """list_runs makes GET /v1/runs."""
    client = RagEval(api_url="http://localhost:5001", api_key="test")
    with patch("app.sdk.client.requests.get") as mock_get:
        mock_get.return_value.json.return_value = [{"run_id": "r1", "n_samples": 5}]
        mock_get.return_value.raise_for_status = MagicMock()
        runs = client.list_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == "r1"
