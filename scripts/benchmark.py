#!/usr/bin/env python3
"""Benchmark the pipeline across audio lengths and speaker counts.

Usage:
    python scripts/benchmark.py [--gpu] [--output benchmarks/results.csv]
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from highlight_extractor.ingestion.pipeline import run_ingestion
from highlight_extractor.transcription.pipeline import run_transcription
from highlight_extractor.diarization.pipeline import run_diarization
from highlight_extractor.alignment import run_alignment
from highlight_extractor.scoring.features import extract_features
from highlight_extractor.utils.config import load_scoring_weights


BENCHMARK_MATRIX = [
    ("10min_2spk", "benchmarks/eval_set/10min_2spk.wav", 2),
    ("30min_2spk", "benchmarks/eval_set/30min_2spk.wav", 2),
    ("90min_2spk", "benchmarks/eval_set/90min_2spk.wav", 2),
    ("90min_4spk", "benchmarks/eval_set/90min_4spk.wav", 4),
    ("3hr_2spk", "benchmarks/eval_set/3hr_2spk.wav", 2),
]


def run_benchmark(audio_path: str) -> dict:
    """Run pipeline and return timing/memory stats."""
    import tracemalloc

    tracemalloc.start()
    t_start = time.time()

    stg_times = {}
    t0 = time.time()
    audio, sr, duration_s, qc = run_ingestion(audio_path)
    stg_times["ingestion"] = time.time() - t0

    import soundfile as sf
    norm_path = "/tmp/benchmark_normalized.wav"
    sf.write(norm_path, audio, sr, subtype="PCM_16")

    t0 = time.time()
    transcription = run_transcription(norm_path)
    stg_times["transcription"] = time.time() - t0

    t0 = time.time()
    diarization = run_diarization(norm_path)
    stg_times["diarization"] = time.time() - t0

    t0 = time.time()
    aligned = run_alignment(transcription, diarization)
    stg_times["alignment"] = time.time() - t0

    t0 = time.time()
    features = extract_features(aligned.segments, audio, sr)
    stg_times["features"] = time.time() - t0

    total = time.time() - t_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    realtime_factor = total / duration_s if duration_s > 0 else 0.0

    return {
        "duration_s": duration_s,
        "total_s": round(total, 2),
        "real_time_factor": round(realtime_factor, 3),
        "peak_memory_mb": round(peak / (1024 * 1024), 1),
        **{f"stage_{k}_s": round(v, 2) for k, v in stg_times.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark highlight extraction pipeline")
    parser.add_argument("--gpu", action="store_true", help="Flag to note GPU was used")
    parser.add_argument("--output", type=str, default="benchmarks/results.csv")
    args = parser.parse_args()

    results = []
    for label, path, speakers in BENCHMARK_MATRIX:
        p = Path(path)
        if not p.exists():
            print(f"SKIP: {label} — file not found at {path}", file=sys.stderr)
            continue
        print(f"Benchmarking {label} ({p.stat().st_size / 1024 / 1024:.1f} MB)...", file=sys.stderr)
        stats = run_benchmark(str(p))
        stats["label"] = label
        stats["speakers"] = speakers
        stats["gpu"] = args.gpu
        results.append(stats)
        print(f"  Total: {stats['total_s']}s, RT factor: {stats['real_time_factor']}x, "
              f"Peak mem: {stats['peak_memory_mb']} MB", file=sys.stderr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if results:
        fieldnames = list(results[0].keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Results written to {out_path}", file=sys.stderr)
    else:
        print("No results (no benchmark files found)", file=sys.stderr)


if __name__ == "__main__":
    main()
