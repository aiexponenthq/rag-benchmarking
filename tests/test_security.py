import os
from unittest import mock

from fastapi.testclient import TestClient

from app.main import app


def test_api_key_security_open_mode():
    """Test that API is open when API_KEY is not set"""
    with mock.patch.dict(os.environ, {}, clear=True):
        # We need to reload settings because they are lru_cached
        from app.config.settings import get_settings

        get_settings.cache_clear()

        client = TestClient(app)
        # We expect 422 because body is missing, NOT 403
        resp = client.post("/v1/query", json={})
        assert resp.status_code == 422


def test_api_key_security_enforced_mode():
    """Test that API requires key when configured"""
    with mock.patch.dict(os.environ, {"API_KEY": "secret-123"}, clear=True):
        from app.config.settings import get_settings

        get_settings.cache_clear()

        client = TestClient(app)

        # 1. No key -> 403
        resp = client.post("/v1/query", json={"query": "hi"})
        assert resp.status_code == 403

        # 2. Wrong key -> 403
        resp = client.post("/v1/query", json={"query": "hi"}, headers={"X-API-Key": "wrong"})
        assert resp.status_code == 403

        # 3. Correct key -> 200 (or 500 if services down, but passed auth)
        # We mock the engine to avoid 500
        # 3. Correct key -> 200
        # Use dependency_overrides to mock the engine
        from app.api.query import get_rag_engine
        from app.engine.rag_engine import RAGResult

        mock_engine = mock.Mock()
        mock_engine.query.return_value = RAGResult(answer="ok", citations=[], timings={})

        app.dependency_overrides[get_rag_engine] = lambda: mock_engine

        try:
            resp = client.post(
                "/v1/query", json={"query": "hi"}, headers={"X-API-Key": "secret-123"}
            )
            assert resp.status_code == 200
        finally:
            app.dependency_overrides = {}
