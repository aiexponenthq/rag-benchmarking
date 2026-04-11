import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True, scope="module")
def _set_e2e_env(monkeypatch_module):
    """Isolate env vars for the e2e module; restore on teardown."""
    monkeypatch_module.setenv("API_KEY", "test-key")
    monkeypatch_module.setenv("ENFORCE_API_KEY", "true")
    monkeypatch_module.setenv("LLM_PROVIDER", "echo")
    # Clear settings cache so the patched env is picked up
    from app.config.settings import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="module")
def monkeypatch_module(request):
    """Module-scoped monkeypatch fixture."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def client(_set_e2e_env):
    from app.main import app
    return TestClient(app)


HEADERS = {"X-API-Key": "test-key"}


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_agent_eval_endpoint_source_attribution(client):
    """Test agent evaluation with the deterministic source_attribution_accuracy metric."""
    payload = {
        "trace": {
            "question": "What is EU AI Act Article 53?",
            "final_answer": "Article 53 covers GPAI model transparency requirements.",
            "tool_calls": [
                {
                    "tool_name": "retrieve",
                    "tool_input": {"query": "EU AI Act Article 53"},
                    "tool_output": "Article 53 requires GPAI providers to publish documentation.",
                    "step_index": 0,
                }
            ],
            "reasoning_steps": [],
            "retrieved_chunks": [],
        },
        "metrics": ["source_attribution_accuracy"],
    }
    resp = client.post("/v1/evaluate/agent", json=payload, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "scores" in data
    assert "source_attribution_accuracy" in data["scores"]
    assert "trace_id" in data


def test_runs_list_endpoint(client):
    resp = client.get("/v1/runs", headers=HEADERS)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
