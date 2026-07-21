"""
Structured JSON logger for the multi-agent pipeline.

Captures: tool selected, execution time, prompt snippets, tokens, errors, ReAct trace.
Format: JSON Lines (one JSON object per log entry).
"""
from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from settings import settings


class _JSONFormatter(logging.Formatter):
    """Formats log records as JSON objects for structured log analysis."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = traceback.format_exception(*record.exc_info)

        # Include any extra fields attached to the record
        for key in ("tool", "query", "tokens", "latency_ms", "step"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val

        return json.dumps(payload, default=str)


def setup_logging() -> None:
    """
    Configure root logger with:
    - Console handler (human-readable for development)
    - File handler (JSON Lines for structured analysis)
    """
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(log_level)

    # ── Console handler (plain text) ──────────────────────────────────────────
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(log_level)
        console.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(console)

    # ── File handler (JSON Lines) ──────────────────────────────────────────────
    log_path = settings.log_file_path
    if not any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(log_path)
        for h in root.handlers
    ):
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(_JSONFormatter())
        root.addHandler(file_handler)

    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Call setup_logging() once at startup first."""
    return logging.getLogger(name)
