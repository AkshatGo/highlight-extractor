#!/usr/bin/env python3
"""Evaluate highlight extraction quality against a labeled eval set.

Computes precision@K for the scoring pipeline using synthetic data
and the eval ground truth in benchmarks/eval_set/eval_data.json.

Usage:
    python scripts/evaluate.py
"""

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from highlight_extractor.alignment import run_alignment
from highlight_extractor.diarization.pipeline import DiarizationResult, SpeakerTurn
from highlight_extractor.scoring.features import extract_features
from highlight_extractor.scoring.rank import rank_highlights
from highlight_extractor.scoring.segment import derive_candidate_segments
from highlight_extractor.transcription.pipeline import TranscriptionResult, Word
from highlight_extractor.utils.config import load_scoring_weights


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------

def _mock_sentiment(text: str) -> tuple:
    lower = text.lower()
    if any(w in lower for w in ["incredible", "amazing", "wow", "unbelievable", "hilarious"]):
        return (0.92, 0.08)
    if any(w in lower for w in ["cannot", "walked", "problem", "mistake", "shocking"]):
        return (0.15, 0.85)
    if any(w in lower for w in ["secret", "truth", "honestly"]):
        return (0.7, 0.3)
    return (0.55, 0.45)


def _make_episode(duration_s: int, num_speakers: int):
    """Generate a full mock episode with words, diarization, and audio."""
    avg_wps = 3.0
    word_templates = [
        "welcome", "everyone", "to", "the", "show", "today", "we", "have",
        "an", "incredible", "story", "this", "is", "absolutely", "amazing",
        "and", "the", "truth", "is", "nobody", "expected", "it", "wait",
        "seriously", "how", "did", "that", "even", "happen", "i", "cannot",
        "believe", "they", "just", "walked", "out", "oh", "wow", "that",
        "is", "unbelievable", "the", "whole", "thing", "was", "hilarious",
        "exactly", "right", "and", "the", "secret", "behind", "it", "is",
        "that", "they", "had", "been", "planning", "this", "for", "months",
        "it", "was", "a", "huge", "mistake", "on", "their", "part",
        "honestly", "but", "sometimes", "the", "best", "things", "come",
        "from", "unexpected", "places", "so", "you", "are", "telling", "me",
        "the", "entire", "board", "resigned", "that", "is", "shocking",
        "this", "changes", "everything", "wow", "just", "wow",
    ]
    words = []
    t = 0.0
    idx = 0
    while t < duration_s:
        w = word_templates[idx % len(word_templates)]
        dur = 1.0 / avg_wps
        words.append(Word(text=w, start_s=t, end_s=min(t + dur, duration_s), confidence=0.92))
        t += dur
        idx += 1

    turns = []
    turn_dur = duration_s / (num_speakers * 3)
    t = 0.0
    spk = 0
    while t < duration_s:
        end = min(t + turn_dur, duration_s)
        turns.append(SpeakerTurn(start_s=t, end_s=end, speaker=f"SPEAKER_{spk:02d}"))
        t = end + 0.1
        spk = (spk + 1) % num_speakers

    sr = 16000
    time_arr = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 200 * time_arr)
    audio += 0.1 * np.sin(2 * np.pi * 400 * time_arr)
    audio += np.random.randn(len(audio)).astype(np.float32) * 0.01
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8

    transcription = TranscriptionResult(words=words, language="en", model_version="mock")
    diarization = DiarizationResult(turns=turns, num_speakers=num_speakers, model_version="mock")

    return transcription, diarization, audio, sr


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _overlap_fraction(predicted, ground_truth):
    """Fraction of ground truth covered by at least one predicted segment."""
    if not ground_truth:
        return 1.0
    covered = 0
    for gt in ground_truth:
        for pred in predicted:
            inter_start = max(pred["start_s"], gt["start_s"])
            inter_end = min(pred["end_s"], gt["end_s"])
            intersection = max(0, inter_end - inter_start)
            gt_dur = gt["end_s"] - gt["start_s"]
            if gt_dur > 0 and intersection / gt_dur > 0.5:
                covered += 1
                break
    return covered / len(ground_truth)


def evaluate(eval_data_path: str = None):
    """Run evaluation across all episodes in the eval set."""
    if eval_data_path is None:
        eval_data_path = Path(__file__).resolve().parents[1] / "benchmarks" / "eval_set" / "eval_data.json"

    with open(eval_data_path) as f:
        episodes = json.load(f)

    weights = load_scoring_weights()
    results = []

    print(f"\n  Evaluating {len(episodes)} episodes...")
    print(f"  {'Episode':<25} {'K':>3} {'P@5':>6} {'P@10':>6} {'Recall':>7} {'Time':>7}")
    print(f"  {'-'*25} {'---':>3} {'----':>6} {'-----':>6} {'------':>7} {'----':>7}")

    for ep in episodes:
        ep_id = ep["episode_id"]
        gt = ep["ground_truth_highlights"]
        duration_s = ep["duration_s"]
        num_speakers = ep["num_speakers"]

        t0 = time.perf_counter()
        transcription, diarization, audio, sr = _make_episode(duration_s, num_speakers)
        aligned = run_alignment(transcription, diarization)
        candidates = derive_candidate_segments(aligned.segments)

        with patch("highlight_extractor.scoring.features._compute_sentiment", _mock_sentiment):
            features = extract_features(candidates, audio, sr)

        highlights = rank_highlights(features, weights=weights, top_n=10)
        elapsed = time.perf_counter() - t0

        # Convert highlights to dicts for overlap computation
        pred_all = [{"start_s": h.start_s, "end_s": h.end_s} for h in highlights]
        pred_5 = pred_all[:5]
        pred_10 = pred_all[:10]

        p5 = _overlap_fraction(pred_5, gt)
        p10 = _overlap_fraction(pred_10, gt)
        recall = _overlap_fraction(gt, pred_all)  # How many GT are covered

        results.append({
            "episode_id": ep_id,
            "precision_at_5": round(p5, 3),
            "precision_at_10": round(p10, 3),
            "recall": round(recall, 3),
            "num_highlights": len(highlights),
            "time_s": round(elapsed, 2),
        })

        print(f"  {ep_id:<25} {len(highlights):>3} {p5:>6.1%} {p10:>6.1%} {recall:>7.1%} {elapsed:>6.1f}s")

    # Aggregate
    avg_p5 = np.mean([r["precision_at_5"] for r in results])
    avg_p10 = np.mean([r["precision_at_10"] for r in results])
    avg_recall = np.mean([r["recall"] for r in results])

    print(f"\n  {'AVERAGE':<25} {'':>3} {avg_p5:>6.1%} {avg_p10:>6.1%} {avg_recall:>7.1%}")

    # Write results
    out_path = Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "eval_baseline.json"
    output = {
        "metric": "precision_recall_baseline",
        "num_episodes": len(episodes),
        "avg_precision_at_5": round(float(avg_p5), 3),
        "avg_precision_at_10": round(float(avg_p10), 3),
        "avg_recall": round(float(avg_recall), 3),
        "episodes": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results written to {out_path}")

    return output


if __name__ == "__main__":
    print("=" * 60)
    print("  Highlight Extraction — Evaluation Baseline")
    print("=" * 60)
    evaluate()
    print("=" * 60)
