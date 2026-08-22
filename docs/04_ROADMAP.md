# Roadmap — Phased Build Plan

## Phase 0 — Spike (1–2 days)

**Goal:** Prove Whisper and pyannote actually work acceptably on real sample
audio before designing further around them.

- Download 3–5 podcast episodes (varying speaker counts, audio quality).
- Run faster-whisper transcription and pyannote diarization separately via
  their Python APIs (no wrapper yet).
- Evaluate ASR quality by ear, check diarization separation quality.
- Note any integration friction (model download times, GPU memory, format
  incompatibilities).

**Deliverable:** Short internal note (not code) documenting model quality
observations, approximate runtime, and any gotchas.

**Exit criteria:** ASR quality acceptable by ear; diarization correctly
separates speakers on clean stereo (2-spk) audio.

---

## Phase 1 — Core pipeline as a script (3–5 days)

**Goal:** Ingestion, transcription wrapper, diarization wrapper, alignment
— wired together as a CLI script. No service yet.

- Implement `src/highlight_extractor/ingestion/` — format validation,
  transcode to 16 kHz mono WAV, basic QC (SNR, clipping, duration check).
- Implement `src/highlight_extractor/transcription/` — wrapper around
  faster-whisper returning a structured word list with timestamps and
  confidence scores.
- Implement `src/highlight_extractor/diarization/` — wrapper around pyannote
  returning RTTM-format speaker turns.
- Implement `src/highlight_extractor/alignment.py` — merge transcript +
  diarization by timestamp overlap into aligned, speaker-attributed segments.
- Wire into `scripts/run_pipeline.py`.

**Deliverable:** `python scripts/run_pipeline.py file.mp3` prints aligned,
speaker-attributed transcript segments to stdout.

**Exit criteria:** Correct on Phase 0's test samples — words attributed to
the right speaker, boundaries reasonable.

---

## Phase 2 — Scoring (3–4 days)

**Goal:** Feature extraction, composite scoring, non-max suppression, output
as highlights.json.

- Implement feature extraction per candidate segment (see
  [docs/03_SCORING_DESIGN.md](03_SCORING_DESIGN.md)):
  sentiment (transformers), energy (Librosa RMS), pitch (Librosa pyin),
  speech rate, keyword density, crosstalk flag aggregation, ASR confidence
  aggregation.
- Implement z-scoring per episode.
- Implement composite score with weights from `config/scoring_weights.yaml`.
- Implement temporal non-max suppression.
- Extend `scripts/run_pipeline.py` to output `highlights.json`.

**Deliverable:** Running the pipeline script produces `highlights.json`
matching the scoring design's output contract.

**Exit criteria:** Highlights are subjectively plausible on manual review of
a few episodes. Formal eval deferred to Phase 4.

---

## Phase 3 — Service hardening (4–6 days)

**Goal:** Harden from a working script into a tested service.

- Wrap pipeline in FastAPI + async job model (`POST /v1/jobs`, polling,
  results).
- Implement in-process job queue + worker (design for Redis/Celery
  swap-in via `JobBackend` interface).
- Implement artifact persistence between stages so rescoring doesn't
  re-run ASR/diarization (design for S3-compatible swap-in via
  `ArtifactBackend` interface).
- Add structured logging + per-stage timing capture.
- Add input validation, timeouts, defined error states per stage.
- Write unit tests per stage (especially alignment — highest value).

**Deliverable:** Running `uvicorn` and submitting a job over HTTP returns
results; a bad file fails with clear stage + reason, not a stack trace.

**Exit criteria:** Can submit a job, poll status, get results. Bad files
fail gracefully with structured errors.

---

## Phase 4 — Benchmarking & failure-mode documentation (2–3 days)

**Goal:** Real performance numbers and documented failure reproductions.

- Run runtime benchmarks across audio lengths (10 min–3 hr) and speaker
  counts (2–4+) using `scripts/benchmark.py`.
- Build small labeled eval set (15–20 episodes, human-marked highlights)
  in `benchmarks/eval_set/`.
- Compute precision@10 baseline.
- Reproduce each known failure mode (overlapping speech, bad audio, long
  files) and document with actual outputs in
  [docs/06_FAILURE_MODES.md](06_FAILURE_MODES.md).

**Deliverable:** `benchmarks/results.csv` with real numbers, eval set
precision@10 baseline recorded, failure modes doc has real reproductions.

**Exit criteria:** Testing doc and failure modes doc contain real data,
not placeholders.

---

## Phase 5 — v1.1+ backlog (not blocking)

- **Learned scoring model:** logistic regression over features, trained
  from editor accept/reject feedback on highlights.
- **Per-show configurable keyword lists** instead of a single global list.
- **Overlapping-speech-aware ASR:** source separation pre-pass before
  transcription.
- **Multi-language support:** Whisper supports many languages out of box;
  need to extend sentiment pipeline and keyword lists.
- **Webhook-based job completion** instead of polling.

---

## Suggested team shape

~3 weeks end-to-end for 1–2 engineers. A single part-time engineer can do
the 3-day spike then iterate, but Phase 1 (ASR/diarization/alignment) and
Phase 2 (scoring) can be split across two engineers working in parallel
after Phase 0 completes.
