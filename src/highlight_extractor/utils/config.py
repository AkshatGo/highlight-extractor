"""Configuration loading from YAML files."""

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_WEIGHTS_PATH = Path(__file__).parents[3] / "config" / "scoring_weights.yaml"


def load_scoring_weights(path: str | Path | None = None) -> dict[str, Any]:
    """Load scoring weights from YAML.

    Args:
        path: Path to weights YAML. Defaults to config/scoring_weights.yaml
              relative to the repo root.

    Returns:
        Dict with weight keys and float values.
    """
    if path is None:
        path = _DEFAULT_WEIGHTS_PATH
    with open(path) as f:
        weights: dict[str, Any] = yaml.safe_load(f)
    return weights


def merge_weights(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge a partial overrides dict into base weights, returning a new dict."""
    merged = dict(base)
    merged.update(overrides)
    return merged
