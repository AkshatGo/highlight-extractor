"""Tests for the alignment module — highest-value test suite, pure Python.

These tests use synthetic fabricated ASR word lists and fabricated RTTM-style
diarization turns. No ML dependency, fast and deterministic.
"""

import pytest
from highlight_extractor.transcription.pipeline import TranscriptionResult, Word
from highlight_extractor.diarization.pipeline import DiarizationResult, SpeakerTurn
from highlight_extractor.alignment import run_alignment


def test_simple_two_speaker_alignment():
    """Two speakers alternating cleanly, no overlap, no pauses."""
    transcription = TranscriptionResult(
        words=[
            Word(text="Hello", start_s=0.0, end_s=0.3, confidence=0.95),
            Word(text="world", start_s=0.3, end_s=0.5, confidence=0.93),
            Word(text="Hi", start_s=0.6, end_s=0.7, confidence=0.90),
            Word(text="there", start_s=0.7, end_s=0.9, confidence=0.92),
            Word(text="How", start_s=1.0, end_s=1.1, confidence=0.94),
            Word(text="are", start_s=1.1, end_s=1.2, confidence=0.91),
            Word(text="you", start_s=1.2, end_s=1.3, confidence=0.89),
        ],
        language="en",
        model_version="test",
    )
    diarization = DiarizationResult(
        turns=[
            SpeakerTurn(start_s=0.0, end_s=0.55, speaker="SPEAKER_00"),
            SpeakerTurn(start_s=0.6, end_s=0.95, speaker="SPEAKER_01"),
            SpeakerTurn(start_s=1.0, end_s=1.35, speaker="SPEAKER_00"),
        ],
        num_speakers=2,
        model_version="test",
    )

    result = run_alignment(transcription, diarization)
    assert len(result.segments) == 3
    assert result.segments[0].speaker == "SPEAKER_00"
    assert result.segments[0].text == "Hello world"
    assert result.segments[1].speaker == "SPEAKER_01"
    assert result.segments[1].text == "Hi there"
    assert result.segments[2].speaker == "SPEAKER_00"
    assert result.segments[2].text == "How are you"


def test_long_pause_splits_monologue():
    """Same speaker with >700ms pause should split into two segments."""
    transcription = TranscriptionResult(
        words=[
            Word(text="First", start_s=0.0, end_s=0.3, confidence=0.95),
            Word(text="part", start_s=0.3, end_s=0.5, confidence=0.93),
            Word(text="Second", start_s=1.5, end_s=1.7, confidence=0.90),
            Word(text="part", start_s=1.7, end_s=1.9, confidence=0.92),
        ],
        language="en",
        model_version="test",
    )
    diarization = DiarizationResult(
        turns=[
            SpeakerTurn(start_s=0.0, end_s=2.0, speaker="SPEAKER_00"),
        ],
        num_speakers=1,
        model_version="test",
    )

    result = run_alignment(transcription, diarization, min_pause_s=0.7)
    # 0.5 to 1.5 gap = 1.0s > 0.7s → split
    assert len(result.segments) == 2
    assert result.segments[0].text == "First part"
    assert result.segments[1].text == "Second part"


def test_no_pause_keeps_combined():
    """Same speaker with small gap should stay in one segment."""
    transcription = TranscriptionResult(
        words=[
            Word(text="First", start_s=0.0, end_s=0.3, confidence=0.95),
            Word(text="part", start_s=0.3, end_s=0.5, confidence=0.93),
            Word(text="Second", start_s=0.6, end_s=0.7, confidence=0.90),
        ],
        language="en",
        model_version="test",
    )
    diarization = DiarizationResult(
        turns=[
            SpeakerTurn(start_s=0.0, end_s=0.8, speaker="SPEAKER_00"),
        ],
        num_speakers=1,
        model_version="test",
    )

    result = run_alignment(transcription, diarization, min_pause_s=0.7)
    # 0.5 to 0.6 gap = 0.1s < 0.7s → no split
    assert len(result.segments) == 1
    assert result.segments[0].text == "First part Second"


def test_short_segments_not_discarded():
    """Very short segments should still produce output (scoring handles length)."""
    transcription = TranscriptionResult(
        words=[
            Word(text="Hi", start_s=0.0, end_s=0.1, confidence=0.95),
        ],
        language="en",
        model_version="test",
    )
    diarization = DiarizationResult(
        turns=[
            SpeakerTurn(start_s=0.0, end_s=0.15, speaker="SPEAKER_00"),
        ],
        num_speakers=1,
        model_version="test",
    )
    result = run_alignment(transcription, diarization)
    assert len(result.segments) == 1
    assert result.segments[0].text == "Hi"


def test_empty_transcription():
    """No words should produce empty result."""
    transcription = TranscriptionResult(words=[], model_version="test")
    diarization = DiarizationResult(
        turns=[SpeakerTurn(start_s=0.0, end_s=1.0, speaker="SPEAKER_00")],
        num_speakers=1,
        model_version="test",
    )
    result = run_alignment(transcription, diarization)
    assert len(result.segments) == 0


def test_no_diarization_fallback():
    """No diarization turns → single-speaker fallback."""
    transcription = TranscriptionResult(
        words=[
            Word(text="Hello", start_s=0.0, end_s=0.3, confidence=0.95),
        ],
        language="en",
        model_version="test",
    )
    diarization = DiarizationResult(turns=[], num_speakers=0, model_version="test")
    result = run_alignment(transcription, diarization)
    assert len(result.segments) == 1
    assert result.segments[0].speaker == "SPEAKER_00"
