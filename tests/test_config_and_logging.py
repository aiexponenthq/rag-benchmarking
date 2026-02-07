from __future__ import annotations

import logging

from app.config.settings import get_settings
from app.logging.json_logger import configure_json_logging


def test_settings_loads_defaults() -> None:
    settings = get_settings()
    assert settings.app_env == "dev"
    assert settings.log_level in {"INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"}


def test_json_logger_configures_root_logger() -> None:
    logger = configure_json_logging("INFO")
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.INFO
