from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_evaluate_smoke(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    client = TestClient(app)

    # Monkeypatch run_evaluation to avoid heavy imports during test
    import app.eval.ragas_runner as rr

    def fake_run(samples, metrics=None):  # type: ignore[no-untyped-def]
        return {"metrics": {"faithfulness": 0.9}}

    monkeypatch.setattr(rr, "run_evaluation", fake_run)

    payload = {
        "samples": [
            {
                "question": "What is RAG?",
                "contexts": ["RAG retrieves documents and generates answers."],
                "answer": "RAG combines retrieval and generation.",
                "ground_truths": ["RAG combines retrieval and generation."],
            }
        ]
    }
    resp = client.post("/v1/evaluate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data and "metrics" in data["result"]
    assert data["result"]["metrics"]["faithfulness"] == 0.9
