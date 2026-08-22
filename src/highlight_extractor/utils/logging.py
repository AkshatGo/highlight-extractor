"""Structured JSON logging and per-stage timing capture."""

import json
import logging
import sys
import time
from typing import Any, Dict, Optional


class StageTimer:
    """Context manager that records wall-clock time for a pipeline stage."""

    def __init__(self, stage_name: str, job_id: Optional[str] = None):
        self.stage_name = stage_name
        self.job_id = job_id
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage_name,
            "elapsed_s": round(self.elapsed, 3),
            "job_id": self.job_id,
        }


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
        }
        # Include extra fields attached to the record
        for key in ("stage", "job_id", "elapsed_s"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        return json.dumps(log_entry)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure structured JSON logging to stderr.

    Returns the root logger for the highlight_extractor namespace.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JSONFormatter())

    logger = logging.getLogger("highlight_extractor")
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the highlight_extractor namespace."""
    return logging.getLogger(f"highlight_extractor.{name}")
