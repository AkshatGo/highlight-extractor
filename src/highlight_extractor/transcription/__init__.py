"""Transcription stage: word-level timestamps via faster-whisper."""

from highlight_extractor.transcription.pipeline import (
    TranscriptionResult,
    Word,
    run_transcription,
)

__all__ = ["TranscriptionResult", "Word", "run_transcription"]
