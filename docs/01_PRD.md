# Product Requirements Document — Highlight Extraction Service

## Problem

Podcast and long-form-talk editors manually scrub through hours of audio to
find the 10–30 moments worth clipping for social, trailers, or show notes.
At scale — a producer with a 20-episode catalog spends days just *listening*.
An automated highlight extraction service can surface a ranked shortlist of
candidate moments, letting the editor review and select in minutes instead of
hours.

## Users

- **Primary:** Podcast editors and social-media clippers. They upload the raw
  episode, get back a shortlist of timestamps with scores and reasons, and
  export the ones they want.
- **Secondary:** Internal tooling that auto-generates trailer rough-cuts or
  timestamped show notes from the scored segments.

## Non-goals (v1)

- **Not a video editor** — returns timestamps and metadata, not rendered clips.
- **Not real-time / streaming** — a 90-minute file takes minutes to process.
- **Not abstractive summarization** — no episode summary or transcript
  summarization; only segment ranking.
- **Not speaker identification** — only diarization (telling speakers apart
  without naming them). Speakers are labeled SPEAKER_00, SPEAKER_01, etc.

## Core user story

> An editor uploads a 90-minute podcast episode (MP3, 128 kbps) through a
> simple form or API call. The service returns a job ID immediately. After
> 3–7 minutes (GPU) the job completes, and the editor fetches a ranked
> shortlist of 15 segments (15–90s each). Each segment shows a score,
> speaker label, contributing reasons (e.g. "strong sentiment swing +
> crosstalk"), and a transcript excerpt. The editor scans the list, picks
> the 3–5 they want, and exports them — without ever scrubbing the full
> recording.

## Functional requirements

| ID   | Requirement |
|------|-------------|
| F1   | Accept audio upload (wav/mp3/m4a/flac) and return a job ID immediately (async pattern) |
| F2   | Transcribe audio with word-level timestamps using faster-whisper |
| F3   | Diarize speakers (pyannote.audio) and attribute transcript words to speakers via timestamp overlap |
| F4   | Compute per-segment features: sentiment delta/extremity, RMS energy, pitch variance, speech rate delta, keyword density, crosstalk flag, ASR confidence |
| F5   | Score segments with a weighted composite function and apply temporal non-max suppression to select a diverse top-N |
| F6   | Return top-N non-overlapping windows with per-highlight score and contributing reasons |
| F7   | Support configurable clip-length bounds (min_clip_s / max_clip_s) and top-N count |
| F8   | Provide HTTP polling API for job status and results |
| F9   | Persist intermediate artifacts (transcript, diarization, features) so rescoring does not re-run ASR/diarization |

## Non-functional requirements

| ID   | Requirement |
|------|-------------|
| N1   | Latency: <0.5× realtime on GPU, <2× realtime on CPU fallback (for a 90-min file: <45 min GPU, <3 hr CPU — acceptable for async) |
| N2   | Graceful degradation on bad audio: jobs below quality threshold complete but carry a quality_warning field rather than failing silently |
| N3   | Horizontal scalability: workers are stateless, job queue and artifact store are accessed through thin interfaces |
| N4   | Independent testability/cacheability: every stage can be run and tested in isolation; artifacts from expensive stages (ASR, diarization) are cached so scoring-only iterations skip them |
| N5   | Observability: structured JSON logging per stage, per-stage timing metrics recorded for every job |

## Success metrics

- **Precision@10 (eval):** Of the top-10 returned highlights, what fraction
  overlap with human-marked timestamps from a small labeled eval set.
  Target TBD after baseline.
- **Latency (p95):** Wall-clock processing time for a 90-min / 2-spk file
  on reference GPU hardware.
- **Job completion rate:** Fraction of submitted jobs that reach DONE
  without manual intervention (target >95%).

## Open questions

- Multi-language support beyond English — deferred to v1.1.
- Whether keyword lists should be per-show configurable or global — need
  product input, not blocking v1.
- Whether editors want a web UI or are comfortable with API-first — v1
  is API-only; UI is a product decision.
