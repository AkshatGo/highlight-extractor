"""Tests for scoring/ranking — pure Python, synthetic features, no ML models."""

import pytest
from highlight_extractor.scoring.features import SegmentFeatures
from highlight_extractor.scoring.rank import (
    compute_scores,
    temporal_nms,
    _compute_overlap,
    rank_highlights,
)
from highlight_extractor.utils.config import load_scoring_weights


def _make_features(count: int) -> list:
    """Create synthetic features with known values."""
    base_weights = load_scoring_weights()
    features = []
    for i in range(count):
        features.append(SegmentFeatures(
            segment_id=i,
            start_s=float(i * 20),
            end_s=float(i * 20 + 15),
            speaker=f"SPEAKER_{i % 3:02d}",
            sentiment_delta=0.5 if i % 2 == 0 else 0.1,
            sentiment_extremity=0.8 if i % 2 == 0 else 0.2,
            energy_zscore=1.0 if i % 2 == 0 else -0.5,
            pitch_variance=0.3 if i % 2 == 0 else 0.05,
            speech_rate_delta=0.2 if i % 3 == 0 else -0.1,
            keyword_density=0.05 if i % 2 == 0 else 0.0,
            crosstalk_flag=(i % 3 == 1),
            asr_confidence=0.95 if i != 0 else 0.3,  # first segment has low conf
        ))
    return features


class TestComputeScores:
    def test_basic_scoring(self):
        features = _make_features(5)
        weights = load_scoring_weights()
        highlights = compute_scores(features, weights)
        assert len(highlights) == 5
        # All should have a score
        assert all(h.score != 0.0 for h in highlights)
        # Sorted descending
        for i in range(len(highlights) - 1):
            assert highlights[i].score >= highlights[i + 1].score

    def test_low_confidence_flag(self):
        features = _make_features(3)
        weights = load_scoring_weights()
        highlights = compute_scores(features, weights)
        # First segment has asr_confidence=0.3, below 0.5 threshold
        low_conf_hl = [h for h in highlights if h.low_confidence]
        assert len(low_conf_hl) == 1

    def test_crosstalk_bonus(self):
        """Segments with crosstalk should get a bonus."""
        features = _make_features(3)
        weights = load_scoring_weights()
        highlights = compute_scores(features, weights)
        # Find crosstalk segments (id % 3 == 1, so index 1)
        crosstalk_h = [h for h in highlights if any("crosstalk" in r for r in h.reasons)]
        assert len(crosstalk_h) >= 1


class TestTemporalNMS:
    def test_no_overlap_passes_all(self):
        """Segments far apart should all be accepted."""
        features = _make_features(3)
        weights = load_scoring_weights()
        scored = compute_scores(features, weights)
        result = temporal_nms(scored, top_n=10, overlap_threshold=0.2)
        assert len(result) == 3

    def test_complete_overlap_deduplicates(self):
        """Identical segments should be collapsed."""
        features = _make_features(5)
        weights = load_scoring_weights()
        scored = compute_scores(features, weights)
        # Force two highlights to the same time range
        scored[0].start_s = 10
        scored[0].end_s = 20
        scored[1].start_s = 12
        scored[1].end_s = 18  # 80% overlap with scored[0]

        result = temporal_nms(scored, top_n=10, overlap_threshold=0.2)
        assert len(result) < 5  # at least one was suppressed

    def test_top_n_limits_output(self):
        features = _make_features(20)
        weights = load_scoring_weights()
        scored = compute_scores(features, weights)
        result = temporal_nms(scored, top_n=5, overlap_threshold=0.2)
        assert len(result) <= 5


class TestComputeOverlap:
    def test_no_overlap(self):
        hl = lambda s, e: type("HL", (), {"start_s": s, "end_s": e, "speaker": "", "score": 0.0, "reasons": [], "transcript_excerpt": "", "low_confidence": False})
        # We use _compute_overlap as a standalone
        result = _compute_overlap(hl(0, 5), hl(10, 15))
        assert result == 0.0

    def test_partial_overlap(self):
        hl = lambda s, e: type("HL", (), {"start_s": s, "end_s": e, "speaker": "", "score": 0.0, "reasons": [], "transcript_excerpt": "", "low_confidence": False})
        result = _compute_overlap(hl(0, 10), hl(5, 15))
        # overlap 5s, min_duration 10s → 0.5
        assert result == 0.5

    def test_contained(self):
        hl = lambda s, e: type("HL", (), {"start_s": s, "end_s": e, "speaker": "", "score": 0.0, "reasons": [], "transcript_excerpt": "", "low_confidence": False})
        result = _compute_overlap(hl(0, 20), hl(5, 15))
        # overlap 10s, min_duration 10s → 1.0
        assert result == 1.0


class TestComputeHighlights:
    def test_full_rank_pipeline(self):
        features = _make_features(10)
        weights = load_scoring_weights()
        results = rank_highlights(features, weights, top_n=3)
        assert len(results) <= 3
        assert all(h.score != 0.0 for h in results)
        assert all(isinstance(h.score, float) for h in results)
        assert all(isinstance(h.reasons, list) for h in results)
