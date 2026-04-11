from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RAGEvaluable(Protocol):
    """
    The only interface a RAG system must satisfy to use this harness.
    Implement `run()` in a wrapper class around your existing RAG system.

    Example — wrapping a LangChain chain:

        class MyLangChainRAG:
            def __init__(self):
                self.chain = RetrievalQA.from_chain_type(...)

            def run(self, question: str, contexts_override=None) -> dict:
                result = self.chain.invoke({"query": question})
                return {
                    "answer": result["result"],
                    "contexts": [d.page_content for d in result["source_documents"]],
                    "retrieved_doc_ids": [d.metadata.get("id", "") for d in result["source_documents"]],
                }

    Example — wrapping a LlamaIndex query engine:

        class MyLlamaIndexRAG:
            def __init__(self):
                self.engine = index.as_query_engine()

            def run(self, question: str, contexts_override=None) -> dict:
                response = self.engine.query(question)
                return {
                    "answer": str(response),
                    "contexts": [n.text for n in response.source_nodes],
                    "retrieved_doc_ids": [n.node_id for n in response.source_nodes],
                }
    """

    def run(
        self,
        question: str,
        contexts_override: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run a single query through the RAG system.

        Returns a dict with:
          - "answer"             str        (required)
          - "contexts"           list[str]  (required — retrieved text chunks)
          - "retrieved_doc_ids"  list[str]  (optional — enables Precision@K, Recall@K)
        """
        ...


def validate_evaluable(obj: Any) -> None:
    """
    Raise TypeError with a helpful message if obj does not satisfy RAGEvaluable.
    Call this at the start of an evaluation run to give the user a clear error.
    """
    if not isinstance(obj, RAGEvaluable):
        raise TypeError(
            f"{type(obj).__name__} does not implement RAGEvaluable. "
            "Your class must have a `run(question: str, ...) -> dict` method "
            "returning at least {'answer': str, 'contexts': list[str]}. "
            "See harness/protocol.py for examples."
        )
