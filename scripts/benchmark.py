#!/usr/bin/env python3
"""Benchmark pipeline stages across audio lengths and speaker counts.

Produces benchmarks/results.csv with timing data for each stage.

Usage:
    python scripts/benchmark.py [--runs 3]
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from highlight_extractor.alignment import run_alignment
from highlight_extractor.diarization.pipeline import DiarizationResult, SpeakerTurn, run_diarization
from highlight_extractor.scoring.features import extract_features
from highlight_extractor.scoring.rank import rank_highlights
from highlight_extractor.scoring.segment import derive_candidate_segments
from highlight_extractor.transcription.pipeline import TranscriptionResult, Word
from highlight_extractor.utils.config import load_scoring_weights


# ---------------------------------------------------------------------------
# Mock factories — simulate ML model output at various scales
# ---------------------------------------------------------------------------

def _make_words(duration_s: float, avg_wps: float = 3.0) -> list[Word]:
    """Generate simulated words spanning the given duration."""
    words = []
    t = 0.0
    word_list = [
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
    ]
    idx = 0
    while t < duration_s:
        w = word_list[idx % len(word_list)]
        dur = 1.0 / avg_wps
        words.append(Word(text=w, start_s=t, end_s=min(t + dur, duration_s), confidence=0.92))
        t += dur
        idx += 1
    return words


def _make_diarization(duration_s: float, num_speakers: int = 2) -> DiarizationResult:
    """Generate simulated speaker turns."""
    turns = []
    turn_dur = duration_s / (num_speakers * 3)
    t = 0.0
    spk = 0
    while t < duration_s:
        end = min(t + turn_dur, duration_s)
        turns.append(SpeakerTurn(start_s=t, end_s=end, speaker=f"SPEAKER_{spk:02d}"))
        t = end + 0.1
        spk = (spk + 1) % num_speakers
    return DiarizationResult(turns=turns, num_speakers=num_speakers, model_version="mock")


def _mock_sentiment(text: str) -> tuple:
    """Mock sentiment for benchmarking (no transformers dependency)."""
    lower = text.lower()
    if any(w in lower for w in ["incredible", "amazing", "wow", "unbelievable"]):
        return (0.92, 0.08)
    if any(w in lower for w in ["cannot", "walked", "problem", "mistake"]):
        return (0.15, 0.85)
    return (0.55, 0.45)


def _make_audio(duration_s: int, sr: int = 16000) -> np.ndarray:
    """Generate simulated audio."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 200 * t)
    audio += 0.1 * np.sin(2 * np.pi * 400 * t)
    audio += np.random.randn(len(audio)).astype(np.float32) * 0.01
    return audio / (np.max(np.abs(audio)) + 1e-8) * 0.8


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _bench_stage(name: str, fn, runs: int) -> float:
    """Run fn() `runs` times, return median time in ms."""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return sorted(times)[len(times) // 2]


def run_benchmark(runs: int = 3):
    """Run benchmark across configurations and write results.csv."""
    configs = [
        {"duration_s": 60, "num_speakers": 2, "label": "1m-2spk"},
        {"duration_s": 180, "num_speakers": 2, "label": "3m-2spk"},
        {"duration_s": 300, "num_speakers": 2, "label": "5m-2spk"},
        {"duration_s": 300, "num_speakers": 4, "label": "5m-4spk"},
        {"duration_s": 600, "num_speakers": 2, "label": "10m-2spk"},
    ]

    results = []
    weights = load_scoring_weights()

    for cfg in configs:
        print(f"\n  Benchmarking {cfg['label']} ({cfg['duration_s']}s, {cfg['num_speakers']} speakers)...")

        # Pre-generate data
        words = _make_words(cfg["duration_s"])
        diarization = _make_diarization(cfg["duration_s"], cfg["num_speakers"])
        audio = _make_audio(cfg["duration_s"])
        transcription = TranscriptionResult(words=words, language="en", model_version="mock")

        # 1. Alignment
        t_align = _bench_stage("alignment", lambda: run_alignment(transcription, diarization), runs)
        aligned = run_alignment(transcription, diarization)

        # 2. Segmentation
        t_seg = _bench_stage("segmentation", lambda: derive_candidate_segments(aligned.segments), runs)
        candidates = derive_candidate_segments(aligned.segments)

        # 3. Feature extraction (mocked sentiment)
        with patch("highlight_extractor.scoring.features._compute_sentiment", _mock_sentiment):
            t_feat = _bench_stage("feature_extraction", lambda: extract_features(candidates, audio, 16000), runs)
            features = extract_features(candidates, audio, 16000)

        # 4. Scoring + ranking
        transcript_map = {i: s.text for i, s in enumerate(candidates)}
        t_score = _bench_stage("scoring", lambda: rank_highlights(features, weights=weights, top_n=15, transcripts=transcript_map), runs)

        total = t_align + t_seg + t_feat + t_score
        row = {
            "config": cfg["label"],
            "duration_s": cfg["duration_s"],
            "num_speakers": cfg["num_speakers"],
            "num_words": len(words),
            "num_segments": len(candidates),
            "alignment_ms": round(t_align, 1),
            "segmentation_ms": round(t_seg, 1),
            "feature_extraction_ms": round(t_feat, 1),
            "scoring_ms": round(t_score, 1),
            "total_ms": round(total, 1),
            "runs": runs,
        }
        results.append(row)
        print(f"    alignment={t_align:.0f}ms  segments={t_seg:.0f}ms  features={t_feat:.0f}ms  scoring={t_score:.0f}ms  total={total:.0f}ms")

    # Write CSV
    csv_path = Path(__file__).resolve().parents[1] / "benchmarks" / "results.csv"
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Results written to {csv_path}")

    # Write JSON for programmatic access
    json_path = csv_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  JSON copy at {json_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark highlight extraction pipeline")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per stage (median reported)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Highlight Extraction Pipeline Benchmark")
    print("=" * 60)
    run_benchmark(runs=args.runs)
    print("=" * 60)
