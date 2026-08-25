#!/usr/bin/env python3
"""Full pipeline demo — mocks ML models so you can see
scoring working end-to-end without GPU/torch."""

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from highlight_extractor.transcription.pipeline import TranscriptionResult, Word
from highlight_extractor.diarization.pipeline import DiarizationResult, SpeakerTurn
from highlight_extractor.alignment import run_alignment
from highlight_extractor.scoring.segment import derive_candidate_segments
from highlight_extractor.scoring.features import extract_features
from highlight_extractor.scoring.rank import rank_highlights
from highlight_extractor.utils.config import load_scoring_weights


def _mock_sentiment(text: str) -> tuple:
    """Mock sentiment — returns (pos, neg) without transformers."""
    lower = text.lower()
    if any(w in lower for w in ["incredible", "amazing", "wow", "unbelievable", "hilarious"]):
        return (0.92, 0.08)
    if any(w in lower for w in ["cannot", "walked out", "problem", "mistake", "shocking"]):
        return (0.15, 0.85)
    if any(w in lower for w in ["secret", "truth", "honestly"]):
        return (0.7, 0.3)
    return (0.55, 0.45)


# ------------------------------------------------------------------
# Simulated transcription — a 3-minute podcast conversation
# ------------------------------------------------------------------
def make_words():
    """Two speakers having a 3-min conversation."""
    return [
        # Speaker A: Intro (0-8s)
        Word("Welcome", 0.0, 0.4, 0.95), Word("everyone", 0.5, 1.0, 0.93),
        Word("to", 1.0, 1.1, 0.91), Word("the", 1.1, 1.2, 0.92), Word("show", 1.2, 1.5, 0.94),
        Word("Today", 2.0, 2.3, 0.90), Word("we", 2.3, 2.4, 0.88),
        Word("have", 2.4, 2.6, 0.91), Word("an", 2.6, 2.7, 0.89),
        Word("incredible", 2.7, 3.3, 0.96), Word("story", 3.3, 3.6, 0.93),
        Word("This", 4.0, 4.2, 0.91), Word("is", 4.2, 4.3, 0.90),
        Word("absolutely", 4.3, 4.8, 0.94), Word("amazing", 4.8, 5.3, 0.95),
        Word("and", 5.5, 5.6, 0.88), Word("the", 5.6, 5.7, 0.89),
        Word("truth", 5.7, 6.0, 0.92), Word("is", 6.0, 6.1, 0.90),
        Word("nobody", 6.1, 6.4, 0.88), Word("expected", 6.4, 6.8, 0.91),
        Word("it", 6.8, 6.9, 0.87),
        # Speaker B: Reaction (10-22s)
        Word("Wait", 10.0, 10.3, 0.87), Word("seriously", 10.3, 10.9, 0.92),
        Word("this", 11.0, 11.2, 0.90), Word("is", 11.2, 11.3, 0.88),
        Word("amazing", 11.3, 11.8, 0.95),
        Word("How", 12.5, 12.7, 0.91), Word("did", 12.7, 12.8, 0.90),
        Word("that", 12.8, 13.0, 0.92), Word("even", 13.0, 13.2, 0.89),
        Word("happen", 13.2, 13.6, 0.93),
        Word("I", 14.0, 14.1, 0.91), Word("cannot", 14.1, 14.4, 0.93),
        Word("believe", 14.4, 14.8, 0.94), Word("they", 14.8, 15.0, 0.92),
        Word("just", 15.0, 15.2, 0.90), Word("walked", 15.2, 15.5, 0.88),
        Word("out", 15.5, 15.7, 0.85),
        Word("Oh", 16.5, 16.7, 0.93), Word("wow", 16.7, 17.0, 0.96),
        Word("that", 17.0, 17.2, 0.91), Word("is", 17.2, 17.3, 0.89),
        Word("unbelievable", 17.3, 18.0, 0.97),
        Word("The", 18.5, 18.6, 0.90), Word("whole", 18.6, 18.8, 0.91),
        Word("thing", 18.8, 19.0, 0.89), Word("was", 19.0, 19.1, 0.88),
        Word("hilarious", 19.1, 19.7, 0.95),
        # Speaker A: Deep dive (24-38s)
        Word("Exactly", 24.0, 24.5, 0.94), Word("right", 24.5, 24.8, 0.92),
        Word("and", 25.0, 25.1, 0.88), Word("the", 25.1, 25.2, 0.89),
        Word("secret", 25.2, 25.6, 0.93), Word("behind", 25.6, 25.9, 0.91),
        Word("it", 25.9, 26.0, 0.87),
        Word("is", 26.5, 26.6, 0.90), Word("that", 26.6, 26.8, 0.92),
        Word("they", 26.8, 27.0, 0.88), Word("had", 27.0, 27.1, 0.89),
        Word("been", 27.1, 27.3, 0.87), Word("planning", 27.3, 27.7, 0.91),
        Word("this", 27.7, 27.8, 0.90), Word("for", 27.8, 27.9, 0.88),
        Word("months", 27.9, 28.3, 0.93),
        Word("It", 29.0, 29.1, 0.91), Word("was", 29.1, 29.2, 0.88),
        Word("a", 29.2, 29.3, 0.87), Word("huge", 29.3, 29.6, 0.92),
        Word("mistake", 29.6, 30.1, 0.94), Word("on", 30.3, 30.4, 0.89),
        Word("their", 30.4, 30.6, 0.90), Word("part", 30.6, 30.9, 0.91),
        Word("honestly", 30.9, 31.4, 0.95),
        Word("But", 32.0, 32.2, 0.88), Word("sometimes", 32.2, 32.6, 0.91),
        Word("the", 32.6, 32.7, 0.89), Word("best", 32.7, 33.0, 0.93),
        Word("things", 33.0, 33.3, 0.90), Word("come", 33.3, 33.5, 0.91),
        Word("from", 33.5, 33.6, 0.88), Word("unexpected", 33.6, 34.1, 0.94),
        Word("places", 34.1, 34.5, 0.92),
        # Speaker B: Shocking reveal (38-52s)
        Word("So", 38.0, 38.1, 0.90), Word("you", 38.1, 38.2, 0.89),
        Word("are", 38.2, 38.3, 0.88), Word("telling", 38.3, 38.6, 0.91),
        Word("me", 38.6, 38.7, 0.87),
        Word("the", 39.0, 39.1, 0.90), Word("entire", 39.1, 39.4, 0.92),
        Word("board", 39.4, 39.7, 0.91), Word("resigned", 39.7, 40.2, 0.93),
        Word("That", 41.0, 41.2, 0.88), Word("is", 41.2, 41.3, 0.87),
        Word("shocking", 41.3, 41.8, 0.95),
        Word("I", 42.5, 42.6, 0.91), Word("never", 42.6, 42.8, 0.90),
        Word("saw", 42.8, 43.0, 0.89), Word("that", 43.0, 43.2, 0.91),
        Word("coming", 43.2, 43.5, 0.92),
        Word("This", 44.0, 44.2, 0.90), Word("changes", 44.2, 44.6, 0.93),
        Word("everything", 44.6, 45.1, 0.96),
        Word("Wow", 45.5, 45.8, 0.97),
        Word("just", 46.0, 46.2, 0.88), Word("wow", 46.2, 46.5, 0.96),
        # Speaker A: Closing (54-68s)
        Word("So", 54.0, 54.1, 0.90), Word("that", 54.1, 54.3, 0.92),
        Word("is", 54.3, 54.4, 0.88), Word("the", 54.4, 54.5, 0.89),
        Word("story", 54.5, 54.8, 0.93),
        Word("Incredible", 55.5, 56.1, 0.96), Word("stuff", 56.1, 56.4, 0.91),
        Word("right", 56.4, 56.6, 0.88), Word("there", 56.6, 56.9, 0.90),
        Word("Absolutely", 57.5, 58.1, 0.94), Word("definitely", 58.1, 58.6, 0.95),
        Word("one", 58.8, 59.0, 0.91), Word("for", 59.0, 59.1, 0.88),
        Word("the", 59.1, 59.2, 0.89), Word("books", 59.2, 59.5, 0.92),
        Word("Thanks", 60.0, 60.4, 0.93), Word("for", 60.4, 60.5, 0.88),
        Word("listening", 60.5, 61.0, 0.94),
        Word("And", 61.5, 61.7, 0.88), Word("remember", 61.7, 62.1, 0.91),
        Word("to", 62.1, 62.2, 0.89), Word("subscribe", 62.2, 62.7, 0.93),
        Word("See", 63.0, 63.2, 0.90), Word("you", 63.2, 63.3, 0.88),
        Word("next", 63.3, 63.5, 0.91), Word("time", 63.5, 63.8, 0.92),
    ]


def make_diarization():
    """Simulated speaker turns."""
    return DiarizationResult(
        turns=[
            SpeakerTurn(0.0, 9.0, "SPEAKER_00"),    # A: intro
            SpeakerTurn(10.0, 23.0, "SPEAKER_01"),   # B: reaction
            SpeakerTurn(24.0, 37.0, "SPEAKER_00"),   # A: deep dive
            SpeakerTurn(38.0, 53.0, "SPEAKER_01"),   # B: shock
            SpeakerTurn(54.0, 68.0, "SPEAKER_00"),   # A: closing
        ],
        num_speakers=2,
        model_version="pyannote-3.1-demo",
    )


# ------------------------------------------------------------------
print("🎯 Highlight Extraction Pipeline Demo")
print("=" * 60)

print("\n[1/5] Transcription... (simulated)")
words = make_words()
transcription = TranscriptionResult(words=words, language="en", model_version="whisper-base-demo")
print(f"       {len(words)} words transcribed")

print("[2/5] Diarization... (simulated)")
diarization = make_diarization()
print(f"       {diarization.num_speakers} speakers, {len(diarization.turns)} turns")

print("[3/5] Alignment...")
t0 = time.time()
aligned = run_alignment(transcription, diarization)
dt = time.time() - t0
print(f"       {len(aligned.segments)} segments in {dt:.3f}s")
for seg in aligned.segments:
    print(f"       [{seg.start_s:5.1f}-{seg.end_s:5.1f}] {seg.speaker}: {seg.text}")

print("[4/5] Feature extraction...")
sr = 16000
duration_s = 70.0
t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
# Simulate varied energy per segment
audio = np.zeros(len(t), dtype=np.float32)
for seg in aligned.segments:
    mask = (t >= seg.start_s) & (t < seg.end_s)
    freq = 200 + hash(seg.speaker) % 200
    amp = 0.3 + 0.2 * (hash(seg.text) % 100) / 100.0
    audio[mask] = amp * np.sin(2 * np.pi * freq * t[mask])
    audio[mask] *= 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t[mask])
audio += np.random.randn(len(audio)).astype(np.float32) * 0.01
audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8

candidates = derive_candidate_segments(aligned.segments, min_clip_s=5.0, max_clip_s=90.0)
print(f"       {len(candidates)} candidate segments")

with patch("highlight_extractor.scoring.features._compute_sentiment", _mock_sentiment):
    features = extract_features(candidates, audio, sr)
print(f"       Features extracted for {len(features)} segments")
for f in features:
    print(f"         [{f.start_s:5.1f}-{f.end_s:5.1f}] {f.speaker} "
          f"sent={f.sentiment_delta:.2f} energy={f.energy_zscore:.2f} "
          f"keywords={f.keyword_density:.2f}")

print("[5/5] Scoring & ranking...")
weights = load_scoring_weights()
transcript_map = {i: s.text for i, s in enumerate(candidates)}
highlights = rank_highlights(features, weights=weights, top_n=5, transcripts=transcript_map)

# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------
print()
print("=" * 60)
print(f"  TOP {len(highlights)} HIGHLIGHTS")
print("=" * 60)
for i, h in enumerate(highlights, 1):
    dur = h.end_s - h.start_s
    print()
    print(f"  #{i}  Score: {h.score:.3f}")
    print(f"      Time:    {h.start_s:.1f}s – {h.end_s:.1f}s ({dur:.1f}s)")
    print(f"      Speaker: {h.speaker}")
    print(f'      Text:    "{h.transcript_excerpt}"')
    if h.reasons:
        print(f"      Reasons: {', '.join(h.reasons)}")
    if h.low_confidence:
        print("      ⚠  Low confidence")
print()
print("=" * 60)

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
out_path = "/tmp/highlights_demo.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"  Saved to {out_path}")
print("=" * 60)
