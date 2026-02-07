from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_query_rerank_flag(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    client = TestClient(app)

    # Mock retrieval to return out-of-order scores
    import app.retrieval.service as svc

    def fake_retrieve(query: str, top_k: int = 5):  # type: ignore[no-untyped-def]
        return [
            {"text": "c1", "source_id": "s1", "chunk_index": 0, "score": 0.1},
            {"text": "c2", "source_id": "s2", "chunk_index": 1, "score": 0.9},
        ]

    monkeypatch.setattr(svc, "retrieve_top_chunks", fake_retrieve)

    # Mock reranker to invert order deterministically
    import app.retrieval.reranker as rr

    class FakeReranker:
        def rerank(self, query, chunks, top_k):  # type: ignore[no-untyped-def]
            return list(reversed(chunks))[:top_k]

    monkeypatch.setattr(rr, "CrossEncoderReranker", lambda: FakeReranker())

    # Mock LLM
    import app.llm.client as llm

    class FakeLLM:
        def generate(self, system_prompt: str, user_prompt: str) -> str:  # type: ignore[no-untyped-def]
            return "ok"

    monkeypatch.setattr(llm, "LLMClient", lambda: FakeLLM())

    resp = client.post("/v1/query", json={"query": "q", "top_k": 1, "rerank": True})
    assert resp.status_code == 200
    data = resp.json()
    # With reversed order, top should be 'c2'
    assert data["citations"][0]["text"] == "c2"
