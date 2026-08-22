"""Audio I/O helpers using librosa and pydub."""

import io
from pathlib import Path

import librosa
import numpy as np
from pydub import AudioSegment

TARGET_SR: int = 16000
SUPPORTED_FORMATS: tuple[str, ...] = (".wav", ".mp3", ".m4a", ".flac")
MAX_DURATION_S: float = 4 * 3600  # 4 hours default


def load_and_normalize(path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load audio, convert to mono, resample to target_sr.

    Returns:
        (audio_array: float32, sample_rate: int)
    """
    audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return audio, sr


def transcode_to_wav(path: str | Path, target_sr: int = TARGET_SR) -> bytes:
    """Transcode any supported format to 16kHz mono WAV bytes."""
    seg = AudioSegment.from_file(str(path))
    seg = seg.set_frame_rate(target_sr).set_channels(1)
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    return buf.getvalue()


def get_audio_duration(path: str | Path) -> float:
    """Return audio duration in seconds."""
    seg = AudioSegment.from_file(str(path))
    return len(seg) / 1000.0


def validate_format(path: str | Path) -> str:
    """Check that file extension is supported. Returns the extension."""
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{ext}'. Supported: {', '.join(SUPPORTED_FORMATS)}")
    return ext


def validate_duration(path: str | Path, max_s: float = MAX_DURATION_S) -> float:
    """Check that audio does not exceed max_duration_s. Returns duration_s."""
    duration = get_audio_duration(path)
    if duration > max_s:
        raise ValueError(f"Audio duration {duration:.1f}s exceeds maximum {max_s:.1f}s")
    return duration
