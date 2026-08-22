"""Ranking: composite scoring, temporal non-max suppression, output formatting."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from highlight_extractor.scoring.features import SegmentFeatures


@dataclass
class Highlight:
    """Output contract per highlight — matches the scoring design doc exactly."""

    start_s: float
    end_s: float
    speaker: str
    score: float
    reasons: list[str] = field(default_factory=list)
    transcript_excerpt: str = ""
    low_confidence: bool = False


def compute_scores(
    features: list[SegmentFeatures],
    weights: dict[str, Any],
    transcripts: dict[int, str] | None = None,
) -> list[Highlight]:
    """Compute composite scores and return ranked Highlights.

    Steps:
        1. Build feature matrix and z-score per episode.
        2. Apply weighted linear composite formula.
        3. Apply ASR-confidence gate.
        4. Return Highlights sorted by score descending.

    Args:
        features: Extracted features per segment.
        weights: Scoring weights from config/scoring_weights.yaml.
        transcripts: Optional mapping of segment_id → transcript text.

    Returns:
        List of Highlights sorted by score descending (before NMS).
    """
    if not features:
        return []

    # Build feature matrix for z-scoring
    feature_keys = [
        "sentiment_delta",
        "sentiment_extremity",
        "energy_zscore",
        "pitch_variance",
        "speech_rate_delta",
        "keyword_density",
    ]
    matrix = []
    for f in features:
        row = [getattr(f, k, 0.0) or 0.0 for k in feature_keys]
        matrix.append(row)
    matrix = np.array(matrix, dtype=float)

    if matrix.shape[1] > 1:
        # Z-score each column (handle constant cols by keeping 0)
        col_mean = np.mean(matrix, axis=0)
        col_std = np.std(matrix, axis=0)
        col_std[col_std == 0] = 1.0
        matrix_z = (matrix - col_mean) / col_std
    else:
        matrix_z = matrix

    # Load weight values
    w_sent_delta = weights.get("sentiment_delta", 0.25)
    w_sent_extrem = weights.get("sentiment_extremity", 0.20)
    w_energy = weights.get("energy_zscore", 0.15)
    w_pitch = weights.get("pitch_variance", 0.10)
    w_speech = weights.get("speech_rate_delta", 0.10)
    w_keyword = weights.get("keyword_density", 0.10)
    w_crosstalk = weights.get("crosstalk_bonus", 0.10)
    w_confidence_penalty = weights.get("asr_confidence_penalty", 0.15)
    confidence_threshold = weights.get("asr_confidence_threshold", 0.5)

    highlights: list[Highlight] = []
    reason_threshold = 1.0  # z > 1.0 contributes a reason string

    for i, f in enumerate(features):
        z_row = matrix_z[i]
        score = (
            w_sent_delta * z_row[0]
            + w_sent_extrem * z_row[1]
            + w_energy * z_row[2]
            + w_pitch * z_row[3]
            + w_speech * z_row[4]
            + w_keyword * z_row[5]
        )

        if f.crosstalk_flag:
            score += w_crosstalk

        low_conf = False
        if f.asr_confidence < confidence_threshold:
            penalty = w_confidence_penalty * (confidence_threshold - f.asr_confidence)
            score -= penalty
            low_conf = True

        # Build reasons list
        reasons = []
        reason_labels = [
            ("sentiment_delta", "strong sentiment swing"),
            ("sentiment_extremity", "strong sentiment"),
            ("energy_zscore", "high energy"),
            ("pitch_variance", "peak pitch variance"),
            ("speech_rate_delta", "speech rate shift"),
            ("keyword_density", "keyword match"),
        ]
        for idx, (_key, label) in enumerate(reason_labels):
            if idx < len(z_row) and abs(z_row[idx]) > reason_threshold:
                reasons.append(label)
        if f.crosstalk_flag:
            reasons.append("crosstalk")

        excerpt = ""
        if transcripts and i in transcripts:
            excerpt = transcripts[i]

        hl = Highlight(
            start_s=f.start_s,
            end_s=f.end_s,
            speaker=f.speaker,
            score=float(score),
            reasons=reasons,
            transcript_excerpt=excerpt,
            low_confidence=low_conf,
        )
        highlights.append(hl)

    # Sort by score descending
    highlights.sort(key=lambda h: h.score, reverse=True)
    return highlights


def temporal_nms(
    highlights: list[Highlight],
    top_n: int = 15,
    overlap_threshold: float = 0.20,
) -> list[Highlight]:
    """Temporal non-max suppression: greedily select diverse top-N.

    Args:
        highlights: Pre-sorted list (descending score) from compute_scores.
        top_n: Maximum highlights to return.
        overlap_threshold: Maximum allowed overlap fraction with accepted
                          candidates (default 20%).

    Returns:
        Up to top_n non-overlapping highlights.
    """
    accepted: list[Highlight] = []

    for hl in highlights:
        if len(accepted) >= top_n:
            break

        # Check overlap with all accepted highlights
        has_overlap = False
        for acc in accepted:
            overlap = _compute_overlap(hl, acc)
            if overlap > overlap_threshold:
                has_overlap = True
                break

        if not has_overlap:
            accepted.append(hl)

    return accepted


def _compute_overlap(a: Highlight, b: Highlight) -> float:
    """Compute temporal overlap fraction: (intersection) / min(a_dur, b_dur)."""
    a_dur = a.end_s - a.start_s
    b_dur = b.end_s - b.start_s
    if a_dur <= 0 or b_dur <= 0:
        return 0.0

    inter_start = max(a.start_s, b.start_s)
    inter_end = min(a.end_s, b.end_s)
    intersection = max(0.0, inter_end - inter_start)
    min_dur = min(a_dur, b_dur)
    return intersection / min_dur


def rank_highlights(
    features: list[SegmentFeatures],
    weights: dict[str, Any],
    top_n: int = 15,
    overlap_threshold: float = 0.20,
    transcripts: dict[int, str] | None = None,
) -> list[Highlight]:
    """Full ranking pipeline: compute scores → NMS → return.

    Args:
        features: Extracted features per candidate segment.
        weights: Scoring weights dict.
        top_n: Number of highlights to return.
        overlap_threshold: NMS overlap threshold.
        transcripts: Optional segment_id → text mapping.

    Returns:
        List of top-N diverse Highlights.
    """
    scored = compute_scores(features, weights, transcripts=transcripts)
    return temporal_nms(scored, top_n=top_n, overlap_threshold=overlap_threshold)
