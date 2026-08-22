"""Feature extraction per candidate segment.

All features are z-scored per episode before combination
(handled in rank.py after extraction).
"""

from dataclasses import dataclass

import numpy as np

from highlight_extractor.alignment import AlignedSegment

# Default keyword list for density scoring
_DEFAULT_KEYWORDS = [
    "but",
    "wow",
    "amazing",
    "wait",
    "seriously",
    "incredible",
    "unbelievable",
    "shocking",
    "hilarious",
    "exactly",
    "right",
    "absolutely",
    "definitely",
    "literally",
    "honestly",
    "problem",
    "solution",
    "secret",
    "truth",
    "mistake",
]


@dataclass
class SegmentFeatures:
    """Per-segment feature vector before z-scoring and combination."""

    segment_id: int
    start_s: float
    end_s: float
    speaker: str
    sentiment_delta: float = 0.0
    sentiment_extremity: float = 0.0
    energy_zscore: float = 0.0
    pitch_variance: float = 0.0
    speech_rate_delta: float = 0.0
    keyword_density: float = 0.0
    crosstalk_flag: bool = False
    asr_confidence: float = 0.0


def extract_features(
    segments: list[AlignedSegment],
    audio: np.ndarray,
    sr: int,
    keywords: list[str] | None = None,
) -> list[SegmentFeatures]:
    """Extract all feature signals from aligned segments.

    Args:
        segments: Aligned segments from alignment.py.
        audio: Raw audio array (16 kHz mono).
        sr: Sample rate.
        keywords: Keyword list for density scoring. Defaults to _DEFAULT_KEYWORDS.

    Returns:
        List of SegmentFeatures, one per segment.
    """
    if keywords is None:
        keywords = _DEFAULT_KEYWORDS

    features: list[SegmentFeatures] = []

    # Compute rolling stats per speaker (for energy z-score, speech rate)
    speaker_stats = _compute_speaker_stats(segments, audio, sr)

    for idx, seg in enumerate(segments):
        f = SegmentFeatures(
            segment_id=idx,
            start_s=seg.start_s,
            end_s=seg.end_s,
            speaker=seg.speaker,
        )

        # Extract audio slice
        start_idx = int(seg.start_s * sr)
        end_idx = int(seg.end_s * sr)
        seg_audio = audio[start_idx:end_idx] if end_idx <= len(audio) else audio[start_idx:]

        if len(seg_audio) == 0:
            features.append(f)
            continue

        # --- Sentiment ---
        pos, neg = _compute_sentiment(seg.text)
        # sentiment_delta: max swing from neutral (0.5)
        f.sentiment_delta = abs(pos - 0.5)
        # sentiment_extremity: absolute sentiment peak (0-1)
        f.sentiment_extremity = max(pos, neg)

        # --- Energy (RMS) ---
        rms = _compute_rms(seg_audio)
        stats = speaker_stats.get(seg.speaker)
        if stats and stats["rms_std"] > 0:
            f.energy_zscore = (rms - stats["rms_mean"]) / stats["rms_std"]

        # --- Pitch variance ---
        f.pitch_variance = _compute_pitch_variance(seg_audio, sr)

        # --- Speech rate delta ---
        dur_s = seg.end_s - seg.start_s
        if dur_s > 0:
            wps = len(seg.words) / dur_s
            avg_wps = stats.get("avg_wps", wps) if stats else wps
            f.speech_rate_delta = wps - avg_wps if avg_wps > 0 else 0.0

        # --- Keyword density ---
        f.keyword_density = _keyword_density(seg.text.lower(), keywords)

        # --- Crosstalk flag ---
        f.crosstalk_flag = seg.crosstalk

        # --- ASR confidence ---
        if seg.words:
            f.asr_confidence = float(np.mean([w.confidence for w in seg.words]))

        features.append(f)

    return features


def _compute_speaker_stats(
    segments: list[AlignedSegment],
    audio: np.ndarray,
    sr: int,
) -> dict[str, dict[str, float]]:
    """Compute per-speaker rolling means for energy (RMS) and speech rate."""
    stats: dict[str, dict[str, float]] = {}

    speaker_rms_values: dict[str, list[float]] = {}
    speaker_word_counts: dict[str, list[float]] = {}

    for seg in segments:
        spk = seg.speaker
        dur = seg.end_s - seg.start_s

        # Speech rate
        if dur > 0:
            wps = len(seg.words) / dur
            speaker_word_counts.setdefault(spk, []).append(wps)

        # RMS energy per segment
        start_idx = int(seg.start_s * sr)
        end_idx = min(int(seg.end_s * sr), len(audio))
        seg_audio = audio[start_idx:end_idx]
        if len(seg_audio) > 0:
            rms = float(np.sqrt(np.mean(seg_audio**2)))
            speaker_rms_values.setdefault(spk, []).append(rms)

    for spk in set(list(speaker_word_counts.keys()) + list(speaker_rms_values.keys())):
        rates = speaker_word_counts.get(spk, [0.0])
        rms_vals = speaker_rms_values.get(spk, [0.0])
        rms_mean = float(np.mean(rms_vals))
        rms_std = float(np.std(rms_vals))
        stats[spk] = {
            "avg_wps": float(np.mean(rates)),
            "rms_mean": rms_mean,
            "rms_std": rms_std if rms_std > 1e-10 else 1.0,
        }

    return stats


def _compute_sentiment(text: str) -> tuple:
    """Compute positive and negative sentiment scores using transformers pipeline.

    Returns (positive_score, negative_score).
    """
    from transformers import pipeline

    # Lazily load the sentiment pipeline (cached after first call)
    if not hasattr(_compute_sentiment, "_pipe"):
        _compute_sentiment._pipe = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )

    if not text.strip():
        return 0.5, 0.5

    result = _compute_sentiment._pipe(text[:512])[0]  # Truncate long text
    label = result["label"]
    score = result["score"]
    if label == "POSITIVE":
        return score, 1.0 - score
    else:
        return 1.0 - score, score


def _compute_rms(audio_segment: np.ndarray) -> float:
    """Compute RMS energy of an audio segment."""
    if len(audio_segment) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio_segment**2)))


def _compute_pitch_variance(audio_segment: np.ndarray, sr: int) -> float:
    """Compute F0 variance via librosa pyin as a proxy for emphasis."""
    import librosa

    if len(audio_segment) < sr // 4:  # Too short for pitch tracking
        return 0.0

    f0, _, _ = librosa.pyin(
        audio_segment.astype(float),
        sr=sr,
        fmin=65,
        fmax=2093,
    )
    f0_clean = f0[~np.isnan(f0)]
    if len(f0_clean) < 2:
        return 0.0
    return float(np.var(f0_clean))


def _keyword_density(text: str, keywords: list[str]) -> float:
    """Count keyword hits normalized by text length."""
    if not text:
        return 0.0
    tokens = text.split()
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in keywords or t.rstrip(".,!?") in keywords)
    return hits / len(tokens)
