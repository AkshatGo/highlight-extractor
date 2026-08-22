"""Shared audio I/O helpers — writing normalized audio to disk."""

import soundfile as sf
import numpy as np


def write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """Write a float32 audio array to a WAV file.

    Args:
        path: Output file path.
        audio: Audio array (float32, mono).
        sr: Sample rate.
    """
    sf.write(path, audio, sr, subtype="PCM_16")
