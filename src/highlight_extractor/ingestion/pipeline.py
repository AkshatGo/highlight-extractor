"""Ingestion stage: validate, transcode, QC."""

from pathlib import Path
from typing import Tuple

import numpy as np

from highlight_extractor.utils.audio import (
    load_and_normalize,
    validate_duration,
    validate_format,
    TARGET_SR,
)
from highlight_extractor.utils.qc import QCResult, run_qc


def run_ingestion(path: str | Path) -> Tuple[np.ndarray, int, float, QCResult]:
    """Run the full ingestion stage.

    Steps:
        1. Validate file format.
        2. Validate duration.
        3. Load, transcode to 16 kHz mono.
        4. Run QC checks.

    Returns:
        (audio_array, sample_rate, duration_s, qc_result)
    """
    path = Path(path)
    validate_format(path)
    duration_s = validate_duration(path)
    audio, sr = load_and_normalize(path)
    qc = run_qc(audio, sr, duration_s)

    return audio, sr, duration_s, qc
