# Failure Modes — Operational Runbook

Documenting these explicitly, rather than letting them surface as confusing
support tickets, is a deliverable in its own right.

---

## 1. Overlapping speech / crosstalk

**Symptom:**
- Garbled or merged ASR text where two speakers talk simultaneously.
- Diarization ambiguity — who is "speaking" during the overlap window.

**Why it happens:**
- Single-channel ASR (Whisper) isn't built for simultaneous speakers.
  It transcribes the dominant signal and often hallucinates a merged
  utterance.
- pyannote's overlap detection is a known weak point even in SOTA
  diarization models.

**Handling:**
- Overlap regions are flagged via `crosstalk_flag` rather than discarded,
  because genuine crosstalk is often a real highlight signal (laughter,
  reactions, interruptions).
- ASR confidence for these segments is downweighted per the scoring
  design's confidence gate (`asr_confidence_threshold: 0.5`).
- Output segments are annotated with `crosstalk_flag: true` so editors
  know text may be unreliable even when audio-energy signal is good.
- **Source separation as v2 mitigation** (not blocking v1).

---

## 2. Low-quality / noisy audio

**Symptom:**
- High word error rate in transcription.
- Diarization speaker-count errors (over-splitting one speaker into many,
  or collapsing multiple speakers into one).
- Unreliable RMS-based energy features (noise floor swamps speaker energy).

**Why it happens:**
- Field recordings, phone calls, remote interviews over VoIP —
  degraded audio is common in podcast production.

**Handling:**
- Ingestion QC computes basic SNR and clipping stats up front.
- Jobs below a quality floor complete but carry a `quality_warning` string
  field rather than silently producing low-trust rankings.
- ASR-confidence gating downweights unreliable segments.
- **Audio enhancement / denoising is NOT attempted in v1** — adds another
  model + failure surface. Tracked in backlog for v1.1.

---

## 3. Speaker count misestimation

**Symptom:**
- pyannote's automatic speaker count estimation over-splits (10+ speakers
  on a 2-person interview due to mic-level variation) or under-splits
  (collapses a 4-person panel into 2).

**Why it happens:**
- Roundtable/panel formats with similar-acoustic speakers or varying mic
  distances challenge pyannote's internal clustering.

**Handling:**
- API supports an optional `expected_num_speakers` hint passed through to
  diarization when the producer knows the format.
- When unset, speaker labels are treated as opaque IDs (`SPEAKER_00`,
  `SPEAKER_01`, …) with no correctness assumption, surfaced in response
  metadata as `num_speakers_detected`.
- Editors see the detected count and can override via a re-submission with
  a hint.

---

## 4. Very long files (multi-hour)

**Symptom:**
- Memory pressure on the worker (especially during diarization).
- Timeout risk for stages that process the full file in one pass.
- Job may accept, then fail mid-way after significant compute.

**Why it happens:**
- Some shows record 3–6 hour sessions (daily news wrap-ups, live events).

**Handling:**
- Ingestion enforces a configurable max duration (default 4 hours). Files
  exceeding this are rejected early with a clear error rather than accepted
  and failed midway.
- Long files are chunked for transcription (Whisper handles this via its
  own windowing internally).
- **Diarization runs on the FULL file** for turn consistency — this is the
  dominant cost driver for long files. Referenced in the benchmark matrix.

---

## 5. Silence / non-speech-heavy audio

**Symptom:**
- Empty or near-empty transcript segments.
- Diarization produces noise-like turns over music/silence.
- Spuriously low-energy candidates scored as "boring" when they're just
  non-speech.

**Why it happens:**
- Musical interludes, long pauses, applause breaks, ad segments.

**Handling:**
- Ingestion QC runs a lightweight VAD (voice activity detection) pass.
- Non-speech regions are excluded from candidate segmentation entirely
  rather than scored as spuriously low-energy candidates.

---

## 6. Model / version drift

**Symptom:**
- Upgrading Whisper or pyannote silently changes transcript/diarization
  output, which changes scoring outcomes with no code change.
- Previously working episodes produce different (worse) highlights after
  a model upgrade.

**Why it happens:**
- Model releases (faster-whisper, pyannote) improve accuracy on their
  benchmarks but change behavior at boundaries.

**Handling:**
- Artifacts are stored keyed by model version (artifact path includes
  model name + version hash).
- A model version bump is a deliberate logged event.
- Old jobs' artifacts remain reproducible against the model version that
  produced them — re-scoring with the same cached features gives the same
  result.

---

## 7. Escalation guidance (for whoever's on call)

- A job stuck in a non-terminal state for >2× its stage's p95 benchmark
  time should be treated as **stuck, not slow** — check worker logs for
  that stage.
- Repeated failures at the **same stage** across unrelated files likely
  indicate a **model/dependency regression** (check recent deploys), not
  a data issue.
- Repeated failures concentrated on files from **one source / show**
  likely indicate a **source-specific data quality issue** (mic setup,
  encoding), not a service bug.
