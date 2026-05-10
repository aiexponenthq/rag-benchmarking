"""
Shared pytest configuration.

Problem: pydantic-settings reads .env directly, so os.environ patches
don't prevent ENFORCE_API_KEY=true from being loaded in tests.

Solution: Override AppSettings model_config to not read .env during tests,
and reset the lru_cache before each test so patches take effect.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def disable_api_key_enforcement(monkeypatch):
    """
    Disable API key enforcement for all tests by default.
    Tests that explicitly test enforcement (test_security.py) override
    this via their own mock.patch contexts.
    """
    monkeypatch.setenv("ENFORCE_API_KEY", "false")
    monkeypatch.setenv("API_KEY", "")

    # Clear the lru_cache so AppSettings re-reads from environment
    from app.config.settings import get_settings

    get_settings.cache_clear()

    # Patch AppSettings to not read .env file during tests
    from pydantic_settings import SettingsConfigDict

    from app.config import settings as settings_module

    original_config = settings_module.AppSettings.model_config
    settings_module.AppSettings.model_config = SettingsConfigDict(
        env_file=None,  # Do not read .env during tests
        env_file_encoding="utf-8",
        extra="ignore",
    )

    yield

    # Restore original config and clear cache after each test
    settings_module.AppSettings.model_config = original_config
    get_settings.cache_clear()
