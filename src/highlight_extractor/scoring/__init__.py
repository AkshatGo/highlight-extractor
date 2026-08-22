"""Scoring stage: feature extraction, composite ranking, and non-max suppression."""

from highlight_extractor.scoring.features import SegmentFeatures, extract_features
from highlight_extractor.scoring.rank import Highlight, rank_highlights, compute_scores, temporal_nms
from highlight_extractor.scoring.segment import derive_candidate_segments

__all__ = [
    "SegmentFeatures",
    "extract_features",
    "Highlight",
    "rank_highlights",
    "compute_scores",
    "temporal_nms",
    "derive_candidate_segments",
]
