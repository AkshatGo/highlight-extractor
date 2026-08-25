# highlight-extractor

Automatically extract highlight moments from long-form talk/podcast audio
using speaker diarization and sentiment/energy scoring — so a human editor
can review a shortlist instead of scrubbing through hours of recordings.

Stack: Python, faster-whisper, pyannote.audio, Librosa, FastAPI.

## Start here

| Doc | What it covers |
|-----|----------------|
| [docs/01_PRD.md](docs/01_PRD.md) | Product requirements — problem, users, functional & non-functional requirements, success metrics |
| [docs/02_ARCHITECTURE.md](docs/02_ARCHITECTURE.md) | System design — pipeline stages, data flow, component responsibilities, job state machine, deployment shape |
| [docs/03_SCORING_DESIGN.md](docs/03_SCORING_DESIGN.md) | Scoring function deep-dive — feature set, composite score formula, non-max suppression, output contract, evaluation loop |
| [docs/04_ROADMAP.md](docs/04_ROADMAP.md) | Phased build plan (Phase 0–5), team shape recommendation |
| [docs/05_TESTING_AND_BENCHMARKING.md](docs/05_TESTING_AND_BENCHMARKING.md) | Test strategy by stage, runtime benchmark matrix, quality eval set, exit criteria |
| [docs/06_FAILURE_MODES.md](docs/06_FAILURE_MODES.md) | Operational runbook — overlapping speech, noisy audio, speaker-count errors, very long files, escalation guidance |
| [docs/07_API_SPEC.md](docs/07_API_SPEC.md) | HTTP API specification — job submission, polling, results, rescore, error model |

## Repo layout

```
src/highlight_extractor/
├── ingestion/         # Validate, transcode, and QC incoming audio files
├── transcription/     # Wrapper around faster-whisper (word-level timestamps)
├── diarization/       # Wrapper around pyannote.audio speaker diarization
├── alignment.py       # Merge transcript + diarization by timestamp overlap
├── scoring/           # Feature extraction + composite scoring + non-max suppression
├── api/               # FastAPI app, job model, endpoint handlers
└── utils/             # Shared helpers (audio I/O, logging, config loading)
tests/                 # Unit, integration, and end-to-end tests (pytest)
benchmarks/            # Runtime benchmarks, eval set, results CSV
scripts/               # run_pipeline.py (single-file CLI), benchmark.py
config/
└── scoring_weights.yaml  # Tunable scoring weights (not in code)
```

## Setup: model access (required before first run)

1. Copy `.env.example` to `.env`.
2. **Accept the gated pyannote models** on HuggingFace (free account needed):
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - plus the segmentation/embedding models it links to from that page
3. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read scope is enough) and set `HF_TOKEN` in `.env`.
4. Model weights download automatically on first run (Whisper ~150 MB for `base`, pyannote ~50 MB). Set `WHISPER_MODEL=small` or `medium` in `.env` for better accuracy at the cost of speed.

No `HF_TOKEN` is needed for transcription only; diarization will fail without it.

## Quickstart

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run pipeline on a single file (prints aligned transcript + highlights.json)
python scripts/run_pipeline.py path/to/episode.mp3

# Start API server
uvicorn src.highlight_extractor.api.app:app --host 0.0.0.0 --port 8000

# Run tests
pytest -v -m "not slow"      # Fast unit tests (CI suite)
pytest -v -m "slow"           # Slow end-to-end tests (nightly)
```

## Status

**Phase 3 complete — service hardening done.** All pipeline stages (ingestion,
transcription, diarization, alignment, feature extraction, scoring) are
implemented with a FastAPI async job API. 35 unit tests pass. See
[docs/04_ROADMAP.md](docs/04_ROADMAP.md) for the full build plan.

Next: Phase 4 (benchmarking with real audio data, eval set, failure-mode
reproductions).
