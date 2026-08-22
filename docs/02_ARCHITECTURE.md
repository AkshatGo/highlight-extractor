# Architecture — Highlight Extraction Service

## Design principles

1. **Pipeline of independently cacheable stages.** ASR and diarization are
   expensive (minutes of GPU time) and change slowly (model upgrades). Scoring
   is cheap (milliseconds) and is the thing that gets tuned constantly. Artifacts
   from expensive stages are cached so a scoring-only re-run costs near-zero time.
2. **Async job API, not blocking request.** Processing takes minutes for a
   90-minute file. Clients submit and poll — no synchronous HTTP endpoint.
3. **Stateless workers + stateful store.** Workers hold no durable state.
   Job queue, artifact store, and job metadata DB are external. Scaling is
   adding more worker replicas.
4. **Fail loud with documented degradation.** Known failure modes produce
   structured error fields and quality_warning flags rather than silently
   producing low-trust output.

## High-level data flow

```
audio file
    │
    ▼
┌──────────────────────────────────────────────┐
│  Ingestion                                    │
│  (validate format, transcode to 16kHz mono    │  ──► audio_normalized.wav
│   WAV, QC: SNR/clipping/duration check)       │
└──────────────────────────────────────────────┘
    │
    ├──────────────────┬─────────────────────────┘
    ▼                  ▼
┌──────────────┐  ┌─────────────────┐
│ Transcription │  │ Diarization      │
│ (faster-     │  │ (pyannote.audio, │
│  whisper,    │  │  speaker turns    │
│  word-level  │  │  + overlap flags) │
│  timestamps) │  │                  │
│              │  │                  │
│ transcript   │  │ diarization.rttm │
│ .json        │  │                  │
└──────┬───────┘  └────────┬─────────┘
       │                   │
       └─────────┬─────────┘
                 ▼
┌──────────────────────────────────────────────┐
│  Alignment                                    │
│  (merge transcript words → speaker turns     │  ──► aligned_segments.json
│   via timestamp overlap, handle boundaries   │
│   and overlapping speech)                     │
└──────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────┐
│  Feature Extraction                           │
│  (sentiment, energy RMS, pitch variance,     │  ──► features.json
│   speech rate delta, keyword density,         │
│   crosstalk flag, ASR confidence)             │
└──────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────┐
│  Scoring & Ranking                            │
│  (composite score, temporal non-max           │  ──► highlights.json
│   suppression, clip-length shaping)           │
└──────────────────────────────────────────────┘
                 │
                 ▼
         FastAPI result store
```

## Component responsibilities

| Component | Key libs | Responsibility |
|-----------|----------|----------------|
| `ingestion/` | `pydub`, `librosa` | Validate format, transcode to 16 kHz mono WAV, compute QC stats (SNR, clipping, duration), reject oversized files |
| `transcription/` | `faster-whisper` | Run ASR with word-level timestamps; return structured word list with confidence scores |
| `diarization/` | `pyannote.audio` | Run speaker diarization; return RTTM-format speaker turns with overlap regions |
| `alignment/` | — (pure Python) | Merge transcript words → speaker turns by timestamp overlap; handle boundary words, overlapping speech attribution, candidate segmentation |
| `scoring/` | `transformers` (sentiment), `librosa` (energy/pitch) | Extract per-segment features; compute composite score; run non-max suppression to select diverse top-N |
| `api/` | `FastAPI`, `pydantic` | HTTP endpoints, job model, request validation, result serialization |
| Job queue + worker | (v1: in-process `asyncio.Queue`, target: Redis + Celery) | Accept jobs, dispatch to pipeline, persist results |
| Artifact store | (v1: local filesystem, target: S3-compatible) | Read/write intermediate artifacts keyed by job_id + stage + model version |
| Job/status DB | (v1: SQLite, target: Postgres) | Persist job state, stage_history timestamps, final results |

## Why these boundaries specifically

**Transcription and diarization are decoupled** rather than using a combined
"diarized ASR" pipeline. They have different failure modes (ASR degrades on
noise, diarization on overlapping speech), different upgrade cadences (Whisper
and pyannote release independently), and different hardware profiles (ASR is
GPU-heavy, diarization can run CPU-only on short files). Decoupling lets us
upgrade or swap either without touching the other.

**Alignment is its own stage** rather than baked into ASR or diarization output
generation. Merging word-level timestamps with speaker-turn boundaries is a
well-known bug surface: boundary-word attribution errors, overlapping-speech
handling, gaps between speaker turns. Extracting this into a pure-Python stage
with no ML dependency makes it the highest-value unit-test target in the system
— synthetic fixture tests cover the edge cases deterministically.

**Feature extraction and scoring are split** even though both are "cheap."
Features (sentiment, energy, pitch, etc.) are mostly stable — the computation
depends on the raw audio and transcript, which rarely changes. Scoring weights
are the thing that gets tuned constantly (weekly or per-show). Splitting means
a scoring-weight change never touches feature-computation code, and re-scoring
reuses cached features.

## Job state machine

```
         ┌─────────────────────────────────────────────┐
         │                                             │
         ▼                                             │
   QUEUED ──► INGESTING ──► TRANSCRIBING ──────┐      │
                              │                  │      │
                              ├── (parallel) ────┤      │
                              │                  │      │
                         DIARIZING ─────────────┘      │
                                       │                │
                                       ▼                │
                                  ALIGNING              │
                                       │                │
                                       ▼                │
                              EXTRACTING_FEATURES       │
                                       │                │
                                       ▼                │
                                   SCORING              │
                                       │                │
                                       ▼                │
                                    DONE ◄──────────────┘
                                       │
                                       ▼
                                   FAILED
          (any stage → FAILED carries stage + error)
```

Every transition is persisted with a timestamp (`stage_history` array:
`{stage, started_at, ended_at}`). This gives per-stage latency metrics for
free across every job.

## Deployment shape (target, not day-one)

| Component | v1 / dev | Target prod |
|-----------|----------|-------------|
| API service | Single `uvicorn` process | Stateless, N replicas behind load balancer |
| Worker pool | In-process worker thread | GPU-backed worker pool (1+N replicas) |
| Job queue | `asyncio.Queue` (in-process) | Redis / Celery |
| Job metadata DB | SQLite | Postgres |
| Artifact store | Local filesystem | S3-compatible (MinIO / AWS S3) |

The queue and artifact store are accessed through thin Python interfaces
(`JobBackend`, `ArtifactBackend`). Moving from the v1 shape to the target
shape is a config change — swap the backend class — not a code rewrite.
