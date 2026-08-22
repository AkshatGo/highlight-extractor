"""Shared helpers: audio I/O, logging, config loading, QC."""

from highlight_extractor.utils.audio import load_and_normalize, validate_format, validate_duration
from highlight_extractor.utils.config import load_scoring_weights
from highlight_extractor.utils.logging import setup_logging, get_logger

__all__ = [
    "load_and_normalize",
    "validate_format",
    "validate_duration",
    "load_scoring_weights",
    "setup_logging",
    "get_logger",
]
