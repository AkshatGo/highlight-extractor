# Testing & Benchmarking Plan

## Unit / integration test strategy by stage

| Stage | What to test | How (no real ML models in CI) |
|-------|--------------|-------------------------------|
| **Ingestion** | Format validation, transcode correctness (16 kHz mono WAV output), QC thresholds (SNR floor, clipping detection, duration cap) | Small few-second fixture files (wav, mp3, flac, corrupt file) — test accept/reject/transcode behavior without calling any model |
| **Alignment** | Word-to-speaker attribution, boundary-word handling (words overlapping turn boundaries), overlapping turns, gaps between turns, empty turns | **Pure unit tests** with synthetic fabricated word lists (JSON) + fabricated RTTM-style diarization turns. No ML dependency, fast and deterministic. **This is the highest-value test suite in the project.** |
| **Feature extraction** | Sentiment scores on known text, RMS energy on known waveforms, pitch on known tones, keyword density on known phrases, speech rate on known timestamps | Synthetic inputs with known expected outputs — e.g. a 1-second sine wave tests RMS and pitch; a sentence with explicit keywords tests density; a text with positive words tests sentiment |
| **Scoring / ranking** | Composite score math (weighted sum + gate/penalty logic), z-scoring correctness, temporal non-max suppression (candidates with known overlaps), output contract fields | Synthetic candidates with known feature values and overlaps — test that score formula matches expected output, NMS correctly rejects >20% overlap, output JSON schema validation |
| **API** | Job state transitions (QUEUED → DONE, QUEUED → FAILED from each stage), error propagation (stage failure produces correct error body), input validation (bad file type, oversized file, missing file) | FastAPI `TestClient` with the pipeline execution mocked at each stage boundary — no real models needed |
| **End-to-end** | Full pipeline on a real ~1 minute audio sample + known speakers | Marked `slow` / `gpu` — run nightly, not on every PR (real models are heavy) |

**Rule of thumb:** Alignment and scoring stages should have **zero
dependency** on actual ML models in their test suite. That's the whole point
of the architectural decoupling — these stages operate on structured data
that can be synthesized trivially.

## Runtime benchmarking

Benchmark matrix (run via `scripts/benchmark.py`, results written to
`benchmarks/results.csv`):

| Audio length | Speaker count | What to measure |
|--------------|---------------|-----------------|
| 10 min | 2 spk | Total wall time, time per stage, peak memory, realtime factor |
| 30 min | 2 spk | Same as above |
| **90 min** | **2 spk** | **Primary target (typical podcast)** |
| 90 min | 4+ spk | Diarization cost scales with speaker count |
| 3 hr | 2 spk | Stress test / upper bound |

For each cell record: total wall time (s), per-stage wall time (s), peak
memory (MB), realtime factor (processing_time / audio_duration).

Run on both GPU and CPU-only hardware — the PRD latency requirement
specifies both targets (N1: <0.5× realtime GPU, <2× realtime CPU).

A regression here (e.g. an accuracy-improving but 3×-slower model upgrade)
is a real product tradeoff worth tracking over time, not just a curiosity.

## Quality evaluation (ranking correctness)

Since there's no off-the-shelf "highlight" label, build a small internal
eval set:

1. **Pick 15–20 diverse episodes** (different speaker counts, audio quality,
   genres — interview vs panel vs solo monologue).
2. **Have 1–2 human editors independently mark timestamps** they'd consider
   highlight-worthy, **without seeing model output first** (to avoid anchoring).
3. **Compute precision@10 and recall@N** against those marks.
4. **Re-run whenever scoring weights change materially.**

This eval set is a **first-class deliverable** stored in
`benchmarks/eval_set/` with a `README.md` on provenance (who labeled it,
when), not an afterthought.

## What "done" looks like for Phase 3/4 exit

- [ ] All stage unit tests green in CI on every PR.
- [ ] End-to-end test green nightly.
- [ ] Benchmark matrix populated with real numbers in `benchmarks/results.csv`.
- [ ] Eval set precision@10 baseline recorded.
- [ ] Failure modes doc has at least one real reproduction per documented
      failure mode (not just theoretical description).
