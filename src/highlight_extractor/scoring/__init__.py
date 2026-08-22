"""Scoring stage: feature extraction, composite ranking, and non-max suppression."""

from highlight_extractor.scoring.features import SegmentFeatures, extract_features
from highlight_extractor.scoring.rank import Highlight, compute_scores, rank_highlights, temporal_nms
from highlight_extractor.scoring.segment import derive_candidate_segments

__all__ = [
    "Highlight",
    "SegmentFeatures",
    "compute_scores",
    "derive_candidate_segments",
    "extract_features",
    "rank_highlights",
    "temporal_nms",
]
