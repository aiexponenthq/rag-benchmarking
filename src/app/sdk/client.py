from __future__ import annotations

from typing import Any

import requests


class RagEval:
    """
    Python SDK for the rag-benchmarking evaluation harness.

    Quick start::

        client = RagEval(api_url="http://localhost:5001", api_key="your-key")
        report = client.evaluate(samples, metrics=["faithfulness", "answer_relevancy"])
        print(report["scores"])

    LangChain integration::

        chain_output = my_chain.invoke({"query": question})
        sample = RagEval.from_langchain(chain_output)
        report = client.evaluate([sample])

    LlamaIndex integration::

        response = engine.query(question)
        sample = RagEval.from_llamaindex(response, question)
        report = client.evaluate([sample])
    """

    def __init__(
        self,
        api_url: str = "http://localhost:5001",
        api_key: str = "",
    ) -> None:
        self._url = api_url.rstrip("/")
        self._headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json",
        }

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        samples: list[dict[str, Any]],
        metrics: list[str] | None = None,
        metric_group: str | None = None,
        timeout: int = 120,
    ) -> dict:
        """
        Evaluate a list of RAG samples.

        Each sample should be a dict with at minimum:
          - "question": str
          - "contexts": list[str]
          - "answer": str

        Optional fields:
          - "ground_truth": str  (enables context_precision, context_recall)
          - "retrieved_doc_ids": list[str]  (enables Precision@K, Recall@K)
          - "relevant_doc_ids": list[str]
        """
        payload: dict[str, Any] = {"samples": samples}
        if metrics:
            payload["metrics"] = metrics
        if metric_group:
            payload["metric_group"] = metric_group

        resp = requests.post(
            f"{self._url}/v1/evaluate",
            json=payload,
            headers=self._headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def evaluate_agent(
        self,
        trace: dict[str, Any],
        metrics: list[str] | None = None,
        timeout: int = 120,
    ) -> dict:
        """
        Evaluate an agentic RAG trace.

        trace should be a dict matching AgentTrace schema:
          - "question": str
          - "final_answer": str
          - "tool_calls": list[ToolCall dict]
          - "reasoning_steps": list[ReasoningStep dict]
          - "retrieved_chunks": list[RetrievedChunk dict]
        """
        payload: dict[str, Any] = {"trace": trace}
        if metrics:
            payload["metrics"] = metrics

        resp = requests.post(
            f"{self._url}/v1/evaluate/agent",
            json=payload,
            headers=self._headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def list_runs(self, timeout: int = 30) -> list[dict]:
        """Return a list of previous benchmark runs."""
        resp = requests.get(
            f"{self._url}/v1/runs",
            headers=self._headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def compare_runs(
        self, run_ids: list[str], timeout: int = 30
    ) -> dict:
        """Compare metrics across multiple named runs."""
        resp = requests.post(
            f"{self._url}/v1/runs/compare",
            json=run_ids,
            headers=self._headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # ── Framework adapters ─────────────────────────────────────────────────────

    @staticmethod
    def from_langchain(chain_output: dict[str, Any]) -> dict[str, Any]:
        """
        Convert a LangChain RetrievalQA output dict to an evaluation sample.

        Handles the common output format::

            {
                "query": str,  # or "question"
                "result": str,  # or "answer"
                "source_documents": [Document, ...]
            }
        """
        question = chain_output.get("query") or chain_output.get("question", "")
        answer = chain_output.get("result") or chain_output.get("answer", "")
        docs = chain_output.get("source_documents", [])

        contexts = [
            getattr(doc, "page_content", str(doc)) for doc in docs
        ]
        retrieved_doc_ids = [
            getattr(doc, "metadata", {}).get("id", "") for doc in docs
        ]

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "retrieved_doc_ids": retrieved_doc_ids,
        }

    @staticmethod
    def from_llamaindex(response: Any, question: str) -> dict[str, Any]:
        """
        Convert a LlamaIndex query engine response to an evaluation sample.

        Works with Response objects that have:
          - .source_nodes: list of NodeWithScore or TextNode
          - str(response): the generated answer
        """
        source_nodes = getattr(response, "source_nodes", [])

        contexts = []
        for node in source_nodes:
            text = getattr(node, "text", None)
            if text is None:
                get_content = getattr(node, "get_content", None)
                text = get_content() if callable(get_content) else str(node)
            contexts.append(text)

        retrieved_doc_ids = [
            getattr(node, "node_id", "") for node in source_nodes
        ]

        return {
            "question": question,
            "answer": str(response),
            "contexts": contexts,
            "retrieved_doc_ids": retrieved_doc_ids,
        }
