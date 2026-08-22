#!/usr/bin/env python3
"""CLI script to run the full pipeline on a single audio file.

Usage:
    python scripts/run_pipeline.py path/to/episode.mp3 [--top_n 15]
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from highlight_extractor.ingestion.pipeline import run_ingestion
from highlight_extractor.transcription.pipeline import run_transcription
from highlight_extractor.diarization.pipeline import run_diarization
from highlight_extractor.alignment import run_alignment
from highlight_extractor.scoring.features import extract_features
from highlight_extractor.scoring.rank import rank_highlights
from highlight_extractor.utils.config import load_scoring_weights


def main():
    parser = argparse.ArgumentParser(description="Run highlight extraction pipeline on a file")
    parser.add_argument("audio_file", type=str, help="Path to audio file (wav/mp3/m4a/flac)")
    parser.add_argument("--top_n", type=int, default=15, help="Number of highlights to return")
    parser.add_argument("--output", type=str, default=None, help="Output path for highlights.json")
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()

    print(f"[Pipeline] Ingestion...", file=sys.stderr)
    t1 = time.time()
    audio, sr, duration_s, qc = run_ingestion(audio_path)
    print(f"  duration={duration_s:.1f}s, sr={sr}, qc_pass={qc.pass_qc}", file=sys.stderr)
    if qc.quality_warning:
        print(f"  warning: {qc.quality_warning}", file=sys.stderr)

    # Save normalized audio
    import soundfile as sf
    norm_path = audio_path.with_suffix(".normalized.wav")
    sf.write(str(norm_path), audio, sr, subtype="PCM_16")
    print(f"  normalized audio saved to {norm_path}", file=sys.stderr)

    print(f"[Pipeline] Transcription...", file=sys.stderr)
    t2 = time.time()
    transcription = run_transcription(norm_path)
    print(f"  language={transcription.language}, words={len(transcription.words)}, "
          f"model={transcription.model_version}", file=sys.stderr)

    print(f"[Pipeline] Diarization...", file=sys.stderr)
    t3 = time.time()
    diarization = run_diarization(norm_path)
    print(f"  speakers={diarization.num_speakers}, turns={len(diarization.turns)}", file=sys.stderr)

    print(f"[Pipeline] Alignment...", file=sys.stderr)
    t4 = time.time()
    aligned = run_alignment(transcription, diarization)
    print(f"  segments={len(aligned.segments)}", file=sys.stderr)

    print(f"[Pipeline] Feature extraction...", file=sys.stderr)
    t5 = time.time()
    features = extract_features(aligned.segments, audio, sr)
    print(f"  features computed for {len(features)} segments", file=sys.stderr)

    print(f"[Pipeline] Scoring + ranking...", file=sys.stderr)
    weights = load_scoring_weights()
    transcript_map = {s.segment_id: s.text for s in aligned.segments}
    highlights = rank_highlights(features, weights=weights, top_n=args.top_n, transcripts=transcript_map)

    t6 = time.time()

    print(f"\nPipeline timing:", file=sys.stderr)
    print(f"  Ingestion:       {t2-t1:.1f}s", file=sys.stderr)
    print(f"  Transcription:   {t3-t2:.1f}s", file=sys.stderr)
    print(f"  Diarization:     {t4-t3:.1f}s", file=sys.stderr)
    print(f"  Alignment:       {t5-t4:.1f}s", file=sys.stderr)
    print(f"  Scoring:         {t6-t5:.1f}s", file=sys.stderr)
    print(f"  Total:           {t6-t0:.1f}s", file=sys.stderr)

    # Output
    output = [
        {
            "start_s": h.start_s,
            "end_s": h.end_s,
            "speaker": h.speaker,
            "score": round(h.score, 3),
            "reasons": h.reasons,
            "transcript_excerpt": h.transcript_excerpt,
            "low_confidence": h.low_confidence,
        }
        for h in highlights
    ]

    output_path = args.output or (audio_path.parent / "highlights.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n{len(highlights)} highlights written to {output_path}", file=sys.stderr)

    # Also print aligned transcript to stdout
    print("\n--- Aligned Transcript ---")
    for seg in aligned.segments[:20]:  # first 20 segments preview
        print(f"[{seg.start_s:7.1f}-{seg.end_s:7.1f}] {seg.speaker}: {seg.text}")


if __name__ == "__main__":
    main()
