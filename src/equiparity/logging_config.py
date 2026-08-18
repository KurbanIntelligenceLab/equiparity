"""Logging configuration. Called only at entrypoints; library code never configures handlers.

Final-result runs use JSON logs; interactive runs use a
plain human-readable format.
"""

from __future__ import annotations

import json
import logging
from typing import Any


class _JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter for final-result runs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(*, level: int = logging.INFO, json_logs: bool = False) -> None:
    """Configure the root logger. Call once, at the entrypoint only.

    Args:
        level: Root log level.
        json_logs: Emit JSON records (use for final-result runs) instead of plain text.
    """
    handler = logging.StreamHandler()
    if json_logs:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
