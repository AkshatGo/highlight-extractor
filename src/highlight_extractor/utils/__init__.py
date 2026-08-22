"""Shared helpers: audio I/O, logging, config loading, QC."""

from highlight_extractor.utils.audio import load_and_normalize, validate_duration, validate_format
from highlight_extractor.utils.config import load_scoring_weights
from highlight_extractor.utils.logging import get_logger, setup_logging

__all__ = [
    "get_logger",
    "load_and_normalize",
    "load_scoring_weights",
    "setup_logging",
    "validate_duration",
    "validate_format",
]
