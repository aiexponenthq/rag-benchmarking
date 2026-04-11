from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_evaluate_smoke(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    client = TestClient(app)

    # Patch run_evaluation where EvaluationRunner imports it
    import harness.runner as runner_mod

    def fake_run(samples, metrics=None):  # type: ignore[no-untyped-def]
        return {
            "metrics": {"faithfulness": 0.9},
            "per_sample": {},
            "skipped_metrics": [],
            "skip_reason": None,
        }

    monkeypatch.setattr(runner_mod, "run_evaluation", fake_run)

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
    # Response shape: metrics at top level
    assert "metrics" in data
    assert data["metrics"]["faithfulness"] == 0.9
