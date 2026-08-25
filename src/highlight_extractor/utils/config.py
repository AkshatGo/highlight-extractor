"""Configuration loading from YAML files."""

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_WEIGHTS_PATH = Path(__file__).parents[3] / "config" / "scoring_weights.yaml"
_DEFAULT_KEYWORDS_PATH = Path(__file__).parents[3] / "config" / "keyword_presets.yaml"


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


def load_keyword_preset(preset_name: str = "default", path: str | Path | None = None) -> list[str]:
    """Load a keyword preset from config/keyword_presets.yaml.

    Args:
        preset_name: Name of the preset (e.g. 'default', 'tech', 'comedy').
        path: Path to keywords YAML. Defaults to config/keyword_presets.yaml.

    Returns:
        List of keyword strings for the preset.
    """
    if path is None:
        path = _DEFAULT_KEYWORDS_PATH
    if not Path(path).exists():
        return []
    with open(path) as f:
        presets: dict[str, list[str]] = yaml.safe_load(f)
    if preset_name not in presets:
        return []
    return presets[preset_name]


def list_keyword_presets(path: str | Path | None = None) -> list[str]:
    """List available keyword preset names."""
    if path is None:
        path = _DEFAULT_KEYWORDS_PATH
    if not Path(path).exists():
        return []
    with open(path) as f:
        presets: dict[str, Any] = yaml.safe_load(f)
    return list(presets.keys())
