"""Tests for scoring/segment.py — candidate segment derivation.

Pure Python, no ML models, fast and deterministic.
"""

import pytest
from highlight_extractor.alignment import AlignedSegment
from highlight_extractor.scoring.segment import (
    derive_candidate_segments,
    _split_on_pauses,
    _clamp_and_merge,
    _split_long_segment,
)


def _make_word(text, start, end, confidence=0.95):
    """Create a mock Word-like object for testing."""
    from highlight_extractor.transcription.pipeline import Word
    return Word(text=text, start_s=start, end_s=end, confidence=confidence)


def _make_seg(start, end, speaker="SPEAKER_00", words=None, crosstalk=False):
    """Create an AlignedSegment with words."""
    if words is None:
        words = [_make_word("word", start + 0.1, end - 0.1)]
    text = " ".join(w.text for w in words)
    return AlignedSegment(
        start_s=start, end_s=end, speaker=speaker,
        text=text, words=words, crosstalk=crosstalk,
    )


class TestSplitOnPauses:
    def test_no_pauses_returns_single(self):
        words = [_make_word("a", 0, 0.2), _make_word("b", 0.25, 0.5)]
        seg = _make_seg(0, 0.5, words=words)
        result = _split_on_pauses(seg, min_pause_s=0.7)
        assert len(result) == 1

    def test_long_pause_splits(self):
        words = [
            _make_word("a", 0, 0.2),
            _make_word("b", 0.25, 0.4),
            # 1.0s gap
            _make_word("c", 1.5, 1.7),
            _make_word("d", 1.75, 1.9),
        ]
        seg = _make_seg(0, 1.9, words=words)
        result = _split_on_pauses(seg, min_pause_s=0.7)
        assert len(result) == 2
        assert result[0].text == "a b"
        assert result[1].text == "c d"

    def test_empty_words_returns_empty(self):
        seg = _make_seg(0, 1.0, words=[])
        result = _split_on_pauses(seg, min_pause_s=0.7)
        assert len(result) == 0


class TestClampAndMerge:
    def test_short_segments_merged(self):
        segs = [
            _make_seg(0, 5, words=[_make_word("a", 0, 5)]),
            _make_seg(5, 8, words=[_make_word("b", 5, 8)]),
        ]
        result = _clamp_and_merge(segs, min_clip_s=12.0, max_clip_s=90.0)
        # Both < 12s and same speaker → merged
        assert len(result) == 1
        assert result[0].start_s == 0
        assert result[0].end_s == 8

    def test_different_speakers_not_merged(self):
        segs = [
            _make_seg(0, 5, speaker="SPEAKER_00", words=[_make_word("a", 0, 5)]),
            _make_seg(5, 8, speaker="SPEAKER_01", words=[_make_word("b", 5, 8)]),
        ]
        result = _clamp_and_merge(segs, min_clip_s=12.0, max_clip_s=90.0)
        assert len(result) == 2

    def test_empty_input(self):
        result = _clamp_and_merge([], min_clip_s=12.0, max_clip_s=90.0)
        assert len(result) == 0


class TestDeriveCandidateSegments:
    def test_empty_input(self):
        result = derive_candidate_segments([], min_clip_s=12.0, max_clip_s=90.0)
        assert result == []

    def test_segments_with_words_are_kept(self):
        segs = [
            _make_seg(0, 30, words=[_make_word("hello", 0, 30)]),
        ]
        result = derive_candidate_segments(segs, min_clip_s=12.0, max_clip_s=90.0)
        assert len(result) >= 1
        assert all(len(s.words) > 0 for s in result)

    def test_zero_speech_segments_discarded(self):
        segs = [
            AlignedSegment(start_s=0, end_s=10, speaker="SPEAKER_00", text="", words=[]),
        ]
        result = derive_candidate_segments(segs, min_clip_s=12.0, max_clip_s=90.0)
        assert len(result) == 0


class TestSplitLongSegment:
    def test_short_segment_with_small_gaps_not_split(self):
        seg = _make_seg(0, 10, words=[
            _make_word("a", 0, 3),
            _make_word("b", 3.1, 6),
            _make_word("c", 6.1, 10),
        ])
        result = _split_long_segment(seg, max_clip_s=90.0)
        # Small gaps (< 0.3s) → no split at a sentence boundary
        assert len(result) == 1

    def test_long_segment_split(self):
        # Words with gaps > 0.3s between them to trigger splitting
        words = []
        for i in range(10):
            words.append(_make_word(f"w{i}", i * 10, i * 10 + 8))  # 2s gaps between words
        seg = _make_seg(0, 100, words=words)
        result = _split_long_segment(seg, max_clip_s=50.0)
        assert len(result) >= 2
