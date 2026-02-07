from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

from pythonjsonlogger import jsonlogger

trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)


class JsonFormatter(jsonlogger.JsonFormatter):
    """JSON formatter that enforces the required logging schema."""

    def add_fields(
        self, log_record: dict[str, Any], record: logging.LogRecord, message_dict: dict[str, Any]
    ) -> None:  # noqa: D401
        super().add_fields(log_record, record, message_dict)
        log_record.setdefault("timestamp", self.formatTime(record, self.datefmt))
        log_record.setdefault("level", record.levelname)
        log_record.setdefault("logger", record.name)
        log_record.setdefault("message", record.getMessage())
        log_record.setdefault("file", record.filename)
        log_record.setdefault("line", record.lineno)
        log_record.setdefault("function", record.funcName)
        log_record.setdefault("trace_id", trace_id_var.get())


def configure_json_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger with JSON formatting.

    Parameters
    ----------
    level: str
        Logging level string (e.g., INFO, DEBUG).

    Returns
    -------
    logging.Logger
        Configured root logger.
    """

    handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level.upper())
    root_logger.propagate = False
    return root_logger
