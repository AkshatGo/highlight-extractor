"""Transcription stage: word-level timestamps via faster-whisper."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Word:
    text: str
    start_s: float
    end_s: float
    confidence: float


@dataclass
class TranscriptionResult:
    words: List[Word] = field(default_factory=list)
    language: Optional[str] = None
    model_version: str = ""


def _create_model(device: str = "auto", compute_type: str = "float16"):
    """Create a WhisperModel with the given device/compute_type."""
    from faster_whisper import WhisperModel
    return WhisperModel("base", device=device, compute_type=compute_type)


def _build_result(segments, info, model_version: str) -> TranscriptionResult:
    """Build a TranscriptionResult from faster-whisper output."""
    result = TranscriptionResult(
        language=info.language,
        model_version=model_version,
    )
    for seg in segments:
        for w in seg.words:
            result.words.append(
                Word(text=w.word.strip(), start_s=w.start, end_s=w.end, confidence=w.probability)
            )
    return result


def _transcribe_with_fallback(audio_path: str | Path) -> TranscriptionResult:
    """Try GPU transcription, fall back to CPU on CUDA errors."""
    try:
        model = _create_model(device="auto", compute_type="float16")
        segments, info = model.transcribe(str(audio_path), word_timestamps=True)
        return _build_result(segments, info, "faster-whisper-base-gpu")
    except (RuntimeError, OSError) as e:
        logger.warning("GPU transcription failed (%s), falling back to CPU (int8)", e)
        model = _create_model(device="cpu", compute_type="int8")
        segments, info = model.transcribe(str(audio_path), word_timestamps=True)
        return _build_result(segments, info, "faster-whisper-base-cpu")


def run_transcription(audio_path: str | Path) -> TranscriptionResult:
    """Run Whisper transcription on a normalized audio file.

    Uses faster-whisper; returns word-level timestamps with confidence scores.
    Automatically falls back from GPU (float16) to CPU (int8) on CUDA errors.

    Args:
        audio_path: Path to a 16 kHz mono WAV file.

    Returns:
        TranscriptionResult with word list, language, model version.
    """
    return _transcribe_with_fallback(audio_path)
