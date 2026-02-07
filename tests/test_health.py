from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint_structure() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "model" in data and "providers" in data["model"]
    assert "vectordb" in data and "provider" in data["vectordb"]
