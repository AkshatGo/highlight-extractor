# highlight-extractor

Automatically extract highlight moments from long-form talk/podcast audio
using speaker diarization and sentiment/energy scoring — so a human editor
can review a shortlist instead of scrubbing through hours of recordings.

**Stack:** Python, faster-whisper, pyannote.audio, Librosa, FastAPI.

---

## 🎯 What it does

1. **Ingests** audio files (WAV, MP3, M4A, FLAC) up to 4 hours
2. **Transcribes** with word-level timestamps using faster-whisper
3. **Diarizes** speakers with pyannote.audio
4. **Aligns** transcript to speaker turns
5. **Scores** each segment on sentiment, energy, pitch, speech rate, keywords
6. **Ranks** and deduplicates to surface the top highlight moments
7. **Returns** a structured JSON response via REST API

## 🚀 Quick start

### Option 1: Docker (recommended)

```bash
# Copy and configure environment
cp .env.example .env
# Edit .env — set HF_TOKEN for diarization (see Setup below)

# Start the service
docker compose up

# Submit audio
curl -X POST http://localhost:8000/v1/jobs \
  -F "file=@episode.mp3" \
  -F "top_n=10"
```

### Option 2: Local development

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run pipeline on a single file (CLI)
python scripts/run_pipeline.py path/to/episode.mp3

# Start API server
uvicorn highlight_extractor.api.app:app --host 0.0.0.0 --port 8000

# Run demo (no ML models needed)
python scripts/demo_pipeline.py
```

### Option 3: Run benchmarks

```bash
# Time each pipeline stage across audio lengths
python scripts/benchmark.py --runs 3

# Evaluate precision@10 against labeled data
python scripts/evaluate.py
```

---

## ⚙️ Setup: model access (required before first run)

1. Copy `.env.example` to `.env`.
2. **Accept the gated pyannote models** on HuggingFace (free account needed):
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - plus the segmentation/embedding models it links to from that page
3. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read scope is enough) and set `HF_TOKEN` in `.env`.
4. Model weights download automatically on first run (~200 MB total).

No `HF_TOKEN` is needed for transcription only; diarization will fail without it.

---

## 📡 API Reference

### Submit a job

```
POST /v1/jobs
Content-Type: multipart/form-data

file:        (required) Audio file
top_n:       (optional) Number of highlights to return (default: 15)
min_clip_s:  (optional) Minimum clip length in seconds (default: 12)
max_clip_s:  (optional) Maximum clip length in seconds (default: 90)
expected_num_speakers: (optional) Speaker count hint
keyword_preset: (optional) Keyword preset name (default, tech, comedy, news, true_crime, interview)
webhook_url: (optional) URL to POST to on job completion
```

**Response:** `202 Accepted`
```json
{
  "job_id": "abc-123",
  "status": "QUEUED",
  "created_at": "2026-08-26T12:00:00Z",
  "keyword_preset": "tech",
  "webhook_url": "https://example.com/hook"
}
```

### Poll job status

```
GET /v1/jobs/{job_id}
```

### Get highlights

```
GET /v1/jobs/{job_id}/highlights
```

**Response:** `200 OK`
```json
{
  "job_id": "abc-123",
  "audio_duration_s": 3600.0,
  "num_speakers_detected": 2,
  "highlights": [
    {
      "start_s": 120.5,
      "end_s": 135.2,
      "speaker": "SPEAKER_00",
      "score": 0.847,
      "reasons": ["high energy", "strong sentiment swing"],
      "transcript_excerpt": "This is absolutely incredible...",
      "low_confidence": false
    }
  ]
}
```

### Re-score with different parameters

```
POST /v1/jobs/{job_id}/rescore
Content-Type: application/json

{
  "top_n": 5,
  "weights_override": {"energy_zscore": 0.3},
  "keywords": ["breakthrough", "game changer"]
}
```

### List keyword presets

```
GET /v1/presets
```

**Response:** `200 OK`
```json
{
  "keyword_presets": ["default", "tech", "comedy", "news", "true_crime", "interview"]
}
```

### Health & readiness

```
GET /healthz    → 200 OK
GET /readyz     → 200 OK or 503
```

---

## 🏗️ Architecture

```
src/highlight_extractor/
├── ingestion/         # Validate, transcode, QC incoming audio
├── transcription/     # faster-whisper wrapper (word timestamps)
├── diarization/       # pyannote.audio speaker diarization
├── alignment.py       # Merge transcript + diarization by timestamp
├── scoring/           # Feature extraction + composite scoring + NMS
│   ├── features.py    # Sentiment, energy, pitch, keywords
│   ├── rank.py        # Composite scoring + temporal NMS
│   └── segment.py     # Candidate segment derivation
├── api/               # FastAPI app, job model, endpoints
└── utils/             # Audio I/O, config, logging, settings
```

### Scoring features

| Feature | Weight | Description |
|---------|--------|-------------|
| `sentiment_delta` | 0.25 | Max swing from neutral sentiment |
| `sentiment_extremity` | 0.20 | Absolute sentiment peak |
| `energy_zscore` | 0.15 | RMS energy relative to speaker average |
| `pitch_variance` | 0.10 | F0 variance (emphasis proxy) |
| `speech_rate_delta` | 0.10 | Words-per-second deviation from mean |
| `keyword_density` | 0.10 | Matched keyword frequency |
| `crosstalk_bonus` | 0.10 | Bonus for overlapping speech |
| `asr_confidence_penalty` | 0.15 | Penalty for low-confidence ASR |

Weights are configurable via `config/scoring_weights.yaml` (runtime reload, no redeploy).

---

## 🧪 Testing

```bash
# Run all tests
pytest -v

# Fast unit tests only (CI suite)
pytest -v -m "not slow"

# Slow end-to-end tests
pytest -v -m "slow"

# Lint
ruff check src/ tests/

# Type check
mypy src/ --ignore-missing-imports
```

**Current status:** 79 tests passing, 0 lint errors.

---

## 📊 Benchmarks

Run `python scripts/benchmark.py` to generate timing data across audio lengths:

| Duration | Speakers | Alignment | Features | Scoring | Total |
|----------|----------|-----------|----------|---------|-------|
| 1 min | 2 | ~2ms | ~15ms | ~1ms | ~20ms |
| 5 min | 2 | ~5ms | ~40ms | ~2ms | ~50ms |
| 10 min | 2 | ~8ms | ~75ms | ~3ms | ~90ms |

Note: Excludes ML model inference (transcription/diarization) which dominates in production.

---

## 📁 Project layout

```
├── src/highlight_extractor/   # Main package
├── tests/                     # 79 tests (unit + integration)
├── scripts/
│   ├── run_pipeline.py        # CLI pipeline runner
│   ├── demo_pipeline.py       # Demo with mocked ML models
│   ├── benchmark.py           # Performance benchmarks
│   └── evaluate.py            # Precision@10 evaluation
├── benchmarks/
│   ├── eval_set/              # Labeled evaluation data
│   └── results/               # Benchmark + eval results
├── config/
│   ├── scoring_weights.yaml   # Tunable scoring weights
│   └── keyword_presets.yaml   # Per-show keyword presets
├── docs/                      # Design docs, roadmap, runbook
├── Dockerfile                 # Multi-stage production build
├── docker-compose.yml         # Local dev + production
└── deploy/                    # Production docker-compose
```

---

## 📚 Documentation

| Doc | What it covers |
|-----|----------------|
| [docs/01_PRD.md](docs/01_PRD.md) | Product requirements |
| [docs/02_ARCHITECTURE.md](docs/02_ARCHITECTURE.md) | System design |
| [docs/03_SCORING_DESIGN.md](docs/03_SCORING_DESIGN.md) | Scoring deep-dive |
| [docs/04_ROADMAP.md](docs/04_ROADMAP.md) | Phased build plan |
| [docs/05_TESTING_AND_BENCHMARKING.md](docs/05_TESTING_AND_BENCHMARKING.md) | Test strategy |
| [docs/06_FAILURE_MODES.md](docs/06_FAILURE_MODES.md) | Operational runbook |
| [docs/07_API_SPEC.md](docs/07_API_SPEC.md) | HTTP API specification |

---

## 🎬 Status

**All phases complete.** Full pipeline with:
- ✅ Ingestion, transcription, diarization, alignment
- ✅ Feature extraction + composite scoring + NMS
- ✅ FastAPI async job API with rescore support
- ✅ Per-show keyword presets (tech, comedy, news, true_crime, interview)
- ✅ Multi-language support (Whisper + configurable models)
- ✅ Webhook-based job completion notifications
- ✅ 79 tests, benchmarks, evaluation baseline
- ✅ Docker + CI/CD deployment ready
- ✅ Configurable via environment variables + .env files
