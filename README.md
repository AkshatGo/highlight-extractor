# Highlight Extractor

**Find the best moments in any podcast, interview, or talk — automatically.**

[![Live Demo](https://img.shields.io/badge/LIVE%20DEMO-Click%20Here-blue)](https://highlight-extractor-thtb.onrender.com)

---

## The Problem

You have a 2-hour podcast or interview. You need to find the 3-5 best moments to post on social media. Right now, someone has to listen to the whole thing and manually pick clips. That takes hours.

## What This Does

Upload the audio → It listens to the whole thing → It finds the best moments automatically → You get a list of clips ranked by "highlight quality."

**Live Demo:** https://highlight-extractor-thtb.onrender.com

## How It Picks Highlights

- 🎤 **Emotional peaks** — Finds when speakers get excited, surprised, or passionate
- 🔊 **Energy shifts** — Spots when voices get louder or more animated
- 💬 **Keyword matches** — Catches "incredible," "shocking," "game changer," etc.
- 👥 **Speaker reactions** — Identifies moments where one speaker reacts strongly to another
- 🎯 **Context-aware** — Different keyword sets for tech, comedy, news, true crime, interviews

## Who It's For

- **Podcast editors** who need to create social media clips
- **Newsrooms** cutting highlights from long interviews
- **Content creators** repurposing long-form content
- **Any team** processing audio/video at scale

## What You Get

- Ranked list of the best moments
- Exact timestamps (start/end times)
- Transcript excerpts
- Reason why each moment was flagged
- All in a web app — drag, drop, done

---

## Quick Start

### Option 1: Use the live app (no setup)

Go to **https://highlight-extractor-thtb.onrender.com** and upload an audio file.

### Option 2: Docker (recommended for production)

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

### Option 3: Local development

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

---

## Setup: Model Access (required before first run)

1. Copy `.env.example` to `.env`.
2. **Accept the gated pyannote models** on HuggingFace (free account needed):
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - plus the segmentation/embedding models it links to from that page
3. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read scope is enough) and set `HF_TOKEN` in `.env`.
4. Model weights download automatically on first run (~200 MB total).

No `HF_TOKEN` is needed for transcription only; diarization will fail without it.

---

## Features

| Feature | Description |
|---------|-------------|
| **Drag-and-drop upload** | Web UI — no technical skills needed |
| **Real-time progress** | Watch each stage complete |
| **6 keyword presets** | Tech, comedy, news, true crime, interview, default |
| **Custom keywords** | Add your own highlight triggers |
| **Rescore** | Re-run scoring with different weights |
| **Webhooks** | Get notified when processing completes |
| **Multi-language** | Supports any language Whisper understands |
| **4-hour files** | Handles long podcasts and interviews |
| **Mobile ready** | Works on phones and tablets |

---

## API Reference

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
  "created_at": "2026-08-26T12:00:00Z"
}
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
      "transcript_excerpt": "This is absolutely incredible..."
    }
  ]
}
```

### List keyword presets

```
GET /v1/presets
```

### Health check

```
GET /healthz    → 200 OK
GET /readyz     → 200 OK or 503
```

Full API docs: **https://highlight-extractor-thtb.onrender.com/docs**

---

## Architecture

```
src/highlight_extractor/
├── ingestion/         # Validate, transcode, QC incoming audio
├── transcription/     # faster-whisper wrapper (word timestamps)
├── diarization/       # pyannote.audio speaker diarization
├── alignment.py       # Merge transcript + diarization by timestamp
├── scoring/           # Feature extraction + composite scoring + NMS
├── api/               # FastAPI app, job model, endpoints
├── utils/             # Audio I/O, config, logging, settings
└── static/            # Web frontend (HTML/CSS/JS)
```

### Scoring Features

| Feature | Weight | What It Measures |
|---------|--------|------------------|
| Sentiment | 25% | Emotional intensity |
| Energy | 15% | Voice volume relative to speaker average |
| Keywords | 10% | Matched highlight trigger words |
| Pitch | 10% | Voice emphasis (higher = more emphatic) |
| Speech Rate | 10% | Speed changes (faster = more excited) |
| Crosstalk | 10% | Overlapping speech (often a highlight) |
| Confidence | 15% | Transcription reliability |

---

## Tech Stack

- **Backend:** Python, FastAPI, faster-whisper, pyannote.audio
- **Frontend:** HTML, CSS, JavaScript
- **ML:** Whisper (transcription), pyannote (speaker detection), transformers (sentiment)
- **Deployment:** Docker, Render
- **Testing:** 79 tests (pytest)

---

## Status

**Production ready.** Deployed at https://highlight-extractor-thtb.onrender.com

- ✅ Web UI with drag-and-drop upload
- ✅ Full pipeline: transcription → diarization → alignment → scoring
- ✅ 6 keyword presets (tech, comedy, news, true crime, interview)
- ✅ Real-time progress tracking
- ✅ Mobile responsive design
- ✅ API with documentation
- ✅ 79 tests passing
- ✅ Docker deployment
- ✅ Free tier hosting on Render
