from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_query_validation() -> None:
    client = TestClient(app)
    resp = client.post("/v1/query", json={"query": "", "top_k": 3})
    assert resp.status_code == 422


def test_query_smoke(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    client = TestClient(app)

    def fake_retrieve_top_chunks(query: str, top_k: int = 5):  # type: ignore[no-untyped-def]
        return [
            {"text": "answer chunk", "source_id": "s.txt", "chunk_index": 0, "score": 0.9},
        ]

    # Monkeypatch the service function used by the router
    import app.retrieval.service as svc

    monkeypatch.setattr(svc, "retrieve_top_chunks", fake_retrieve_top_chunks)

    # Also patch LLM client to avoid external calls
    import app.llm.client as llm

    class FakeLLM:
        def generate(self, system_prompt: str, user_prompt: str) -> str:  # type: ignore[no-untyped-def]
            # Echo back the user_prompt's trailing stub to simulate a completion
            return user_prompt

    monkeypatch.setattr(llm, "LLMClient", lambda: FakeLLM())

    resp = client.post("/v1/query", json={"query": "what is rag?", "top_k": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer chunk" in data["answer"]
    assert len(data["citations"]) == 1
    assert "timings_ms" in data
    assert "tokens" in data
    assert data["citations"][0]["source_id"] == "s.txt"
