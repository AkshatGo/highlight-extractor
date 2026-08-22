# Highlight Extractor — Project Dependency Graph

## Module Dependency Map (Internal)

```
highlight_extractor/
│
├── utils/
│   ├── audio.py ──────────────► librosa, pydub
│   ├── audio_io.py ───────────► soundfile, numpy
│   ├── config.py ─────────────► yaml
│   ├── logging.py ────────────► (stdlib only)
│   └── qc.py ─────────────────► numpy, librosa
│
├── ingestion/
│   └── pipeline.py ───────────► utils/audio.py, utils/qc.py
│
├── transcription/
│   └── pipeline.py ───────────► faster_whisper (external)
│
├── diarization/
│   └── pipeline.py ───────────► pyannote.audio (external)
│
├── alignment.py ──────────────► transcription/pipeline.py (Word, TranscriptionResult)
│                              ► diarization/pipeline.py (SpeakerTurn, DiarizationResult)
│
├── scoring/
│   ├── features.py ───────────► alignment.py (AlignedSegment)
│   │                        ──► transformers (sentiment pipeline)
│   │                        ──► librosa (pitch via pyin)
│   │                        ──► numpy, scipy
│   ├── rank.py ───────────────► scoring/features.py (SegmentFeatures)
│   │                        ──► numpy
│   └── segment.py ────────────► alignment.py (AlignedSegment)
│
├── api/
│   ├── models.py ─────────────► pydantic
│   ├── job_manager.py ────────► alignment.py, ingestion/pipeline.py
│   │                        ──► transcription/pipeline.py, diarization/pipeline.py
│   │                        ──► scoring/features.py, scoring/rank.py
│   │                        ──► scoring/segment.py (derive_candidate_segments)
│   │                        ──► utils/config.py, api/models.py
│   └── app.py ────────────────► api/job_manager.py, api/models.py
│                              ► scoring/rank.py, utils/config.py
│
└── scripts/
    ├── run_pipeline.py ───────► ALL pipeline stages (CLI orchestrator)
    └── benchmark.py ──────────► ALL pipeline stages (timing/memory benchmarks)
```

## Pipeline Data Flow

```
Audio File (wav/mp3/m4a/flac)
    │
    ▼
┌─────────────────────────────────────────┐
│  INGESTING                              │
│  ingestion/pipeline.py                  │
│  → validate_format(path)                │
│  → validate_duration(path)              │
│  → load_and_normalize(path) → audio[], sr
│  → run_qc(audio, sr, dur) → QCResult   │
│                                         │
│  Output: audio (np.ndarray), sr, dur, qc│
│  Artifact: qc.json                      │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌───────────────────┐  ┌──────────────────────┐
│  TRANSCRIBING     │  │  DIARIZING           │
│  transcription/   │  │  diarization/        │
│  pipeline.py      │  │  pipeline.py         │
│                   │  │                      │
│  → WhisperModel   │  │  → Pipeline.from_    │
│  → model.transcribe│ │    pretrained(...)   │
│  → Word[] with    │  │  → itertracks()      │
│    timestamps     │  │  → SpeakerTurn[]     │
│                   │  │                      │
│  Output:          │  │  Output:             │
│  TranscriptionResult│ DiarizationResult    │
│  Artifact:        │  │  Artifact:           │
│  transcript.json  │  │  diarization.json    │
└────────┬──────────┘  └──────────┬──────────┘
         │                        │
         └──────────┬─────────────┘
                    ▼
┌─────────────────────────────────────────┐
│  ALIGNING                               │
│  alignment.py                           │
│                                         │
│  → _assign_word(word, turns)            │
│  → Group consecutive same-speaker words  │
│  → Split on pauses ≥ 700ms             │
│                                         │
│  Output: AlignmentResult                │
│  Artifact: aligned_segments.json        │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  EXTRACTING_FEATURES                    │
│  scoring/features.py                    │
│  scoring/segment.py (NEW)               │
│                                         │
│  → derive_candidate_segments()          │
│    (split on pauses, clamp, merge)      │
│  → extract_features(candidates, audio)  │
│    → _compute_sentiment(text)           │
│    → _compute_rms(audio_slice)          │
│    → _compute_pitch_variance(audio)     │
│    → _keyword_density(text, keywords)   │
│    → _compute_speaker_stats()           │
│                                         │
│  Output: SegmentFeatures[]              │
│  Artifact: features.json                │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  SCORING                                │
│  scoring/rank.py                        │
│  config/scoring_weights.yaml            │
│                                         │
│  → compute_scores(features, weights)    │
│    → z-score per episode                │
│    → weighted composite formula         │
│    → ASR confidence gate                │
│  → temporal_nms(scored, top_n=15)       │
│    → greedy non-overlapping selection   │
│                                         │
│  Output: Highlight[]                    │
│  Artifact: highlights.json              │
└──────────────────┬──────────────────────┘
                   │
                   ▼
              DONE / FAILED
```

## API Endpoint Map

```
POST /v1/jobs                    → create_job()
  multipart/form-data            → _save_upload()
  → manager.submit()            → JobRecord (QUEUED)
  → threading.Thread(run_pipeline)
  → 202 {job_id, status, created_at}

GET  /v1/jobs/{job_id}           → get_job_status()
  → 200 {job_id, status, stage_history, quality_warning}

GET  /v1/jobs/{job_id}/highlights → get_highlights()
  → 200 {job_id, audio_duration_s, num_speakers_detected, highlights[]}

GET  /v1/jobs/{job_id}/transcript → get_transcript()
  → 200 {job_id, segments[]}

POST /v1/jobs/{job_id}/rescore   → rescore_job()
  → Reuses cached features from source job
  → 202 {job_id, source_job_id, status}
```

## Job State Machine

```
QUEUED → INGESTING → TRANSCRIBING → DIARIZING → ALIGNING
  → EXTRACTING_FEATURES → SCORING → DONE
         │                                        │
         └──────────────► FAILED ◄────────────────┘
                       (any stage)
```

## External Dependencies

```
faster-whisper ≥1.0.0    → ASR (word-level timestamps)
pyannote.audio ≥3.0.0    → Speaker diarization
librosa ≥0.10.0          → Audio I/O, RMS energy, pitch (pyin)
numpy <2.0.0             → Numerical computing
scipy ≥1.10.0            → Statistical functions
transformers ≥4.30.0     → Sentiment analysis (distilbert)
torch ≥2.0.0             → ML runtime
fastapi ≥0.100.0         → HTTP API framework
uvicorn ≥0.20.0          → ASGI server
pydantic ≥2.0.0          → Data validation
pydub ≥0.25.0            → Audio format transcoding
pyyaml ≥6.0              → Config file parsing
soundfile                → WAV file I/O
```

## Test Coverage Map

```
tests/
├── test_alignment.py       ─── 6 tests (pure Python, no ML)
│   ├── simple_two_speaker_alignment
│   ├── long_pause_splits_monologue
│   ├── no_pause_keeps_combined
│   ├── short_segments_not_discarded
│   ├── empty_transcription
│   └── no_diarization_fallback
│
├── test_api.py             ─── 3 tests (mocked pipeline)
│   ├── create_job_no_file
│   ├── get_nonexistent_job
│   └── get_highlights_before_done
│
├── test_ingestion.py       ─── 5 tests (no ML)
│   ├── supported_formats
│   ├── unsupported_format_raises
│   ├── run_qc_clean_audio
│   ├── run_qc_clipped_audio
│   └── run_qc_silence
│
├── test_ranking.py         ─── 10 tests (pure Python, no ML)
│   ├── basic_scoring
│   ├── low_confidence_flag
│   ├── crosstalk_bonus
│   ├── no_overlap_passes_all
│   ├── complete_overlap_deduplicates
│   ├── top_n_limits_output
│   ├── no_overlap (3 tests)
│   ├── partial_overlap
│   ├── contained
│   └── full_rank_pipeline
│
├── test_segmentation.py    ─── 10 tests (pure Python, no ML)
│   ├── split_on_pauses (3 tests)
│   ├── clamp_and_merge (3 tests)
│   ├── derive_candidate_segments (3 tests)
│   └── split_long_segment (2 tests)
│
└── conftest.py             ─── Shared fixtures (synthetic data)

Total: 34 fast tests, 1 slow e2e test (requires GPU/models)
```

## Config Files

```
config/scoring_weights.yaml
├── sentiment_delta: 0.25
├── sentiment_extremity: 0.20
├── energy_zscore: 0.15
├── pitch_variance: 0.10
├── speech_rate_delta: 0.10
├── keyword_density: 0.10
├── crosstalk_bonus: 0.10
├── asr_confidence_penalty: 0.15
└── asr_confidence_threshold: 0.5
```
