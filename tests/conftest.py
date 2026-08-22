"""Shared fixtures for tests."""

import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from highlight_extractor.alignment import AlignedSegment
from highlight_extractor.diarization.pipeline import (
    DiarizationResult,
    SpeakerTurn,
)
from highlight_extractor.transcription.pipeline import (
    TranscriptionResult,
    Word,
)


def _make_word(text: str, start_s: float, end_s: float, confidence: float = 0.95) -> Word:
    """Helper to create a Word with default confidence."""
    return Word(text=text, start_s=start_s, end_s=end_s, confidence=confidence)


@pytest.fixture
def sample_transcription():
    """Synthetic 2-speaker transcription with word-level timestamps."""
    return TranscriptionResult(
        words=[
            Word(text="Hello", start_s=0.0, end_s=0.3, confidence=0.95),
            Word(text="and", start_s=0.3, end_s=0.4, confidence=0.90),
            Word(text="welcome", start_s=0.4, end_s=0.7, confidence=0.97),
            Word(text="everyone", start_s=0.7, end_s=0.9, confidence=0.92),
            Word(text="Thanks", start_s=1.0, end_s=1.2, confidence=0.88),
            Word(text="for", start_s=1.2, end_s=1.3, confidence=0.85),
            Word(text="having", start_s=1.3, end_s=1.5, confidence=0.94),
            Word(text="me", start_s=1.5, end_s=1.6, confidence=0.91),
            Word(text="Great", start_s=2.0, end_s=2.2, confidence=0.96),
            Word(text="to", start_s=2.2, end_s=2.3, confidence=0.93),
            Word(text="be", start_s=2.3, end_s=2.4, confidence=0.95),
            Word(text="here", start_s=2.4, end_s=2.5, confidence=0.89),
        ],
        language="en",
        model_version="faster-whisper-test",
    )


@pytest.fixture
def sample_diarization():
    """Synthetic diarization with 2 speakers."""
    return DiarizationResult(
        turns=[
            SpeakerTurn(start_s=0.0, end_s=0.95, speaker="SPEAKER_00"),
            SpeakerTurn(start_s=1.0, end_s=1.7, speaker="SPEAKER_01"),
            SpeakerTurn(start_s=2.0, end_s=2.6, speaker="SPEAKER_00"),
        ],
        num_speakers=2,
        model_version="pyannote-test",
    )


@pytest.fixture
def sample_aligned_segments():
    """Pre-built aligned segments for scoring tests.

    Includes words for feature extraction (ASR confidence, speech rate).
    """
    return [
        AlignedSegment(
            start_s=0.0,
            end_s=2.0,
            speaker="SPEAKER_00",
            text="This is amazing news",
            words=[
                _make_word("This", 0.0, 0.3),
                _make_word("is", 0.3, 0.4),
                _make_word("amazing", 0.4, 0.9),
                _make_word("news", 0.9, 1.3),
            ],
            crosstalk=False,
        ),
        AlignedSegment(
            start_s=3.0,
            end_s=5.0,
            speaker="SPEAKER_01",
            text="Wait seriously incredible",
            words=[
                _make_word("Wait", 3.0, 3.3),
                _make_word("seriously", 3.3, 3.9),
                _make_word("incredible", 3.9, 4.5),
            ],
            crosstalk=True,
        ),
        AlignedSegment(
            start_s=6.0,
            end_s=8.0,
            speaker="SPEAKER_00",
            text="I cannot believe this",
            words=[
                _make_word("I", 6.0, 6.1),
                _make_word("cannot", 6.1, 6.4),
                _make_word("believe", 6.4, 6.8),
                _make_word("this", 6.8, 7.1),
            ],
            crosstalk=False,
        ),
    ]
