# API Specification — Highlight Extraction Service

All endpoints are under `/v1` and follow the async job pattern described
in the architecture doc: submit → poll → fetch results.

---

### `POST /v1/jobs`

Submit audio for highlight extraction.

**Request:** `multipart/form-data`

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `file` | binary file | — | yes | Audio file (wav/mp3/m4a/flac) |
| `top_n` | int | 15 | no | Number of highlights to return |
| `min_clip_s` | float | 12.0 | no | Minimum clip length in seconds |
| `max_clip_s` | float | 90.0 | no | Maximum clip length in seconds |
| `expected_num_speakers` | int | — | no | Diarization hint; unset = auto-detect |

**Response:** `202 Accepted`

```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "QUEUED",
  "created_at": "2026-08-21T12:00:00Z"
}
```

---

### `GET /v1/jobs/{job_id}`

Poll job status.

**Response:** `200 OK` (in-progress example)

```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "TRANSCRIBING",
  "stage_history": [
    {"stage": "QUEUED", "started_at": "2026-08-21T12:00:00Z", "ended_at": "2026-08-21T12:00:01Z"},
    {"stage": "INGESTING", "started_at": "2026-08-21T12:00:01Z", "ended_at": "2026-08-21T12:00:05Z"},
    {"stage": "TRANSCRIBING", "started_at": "2026-08-21T12:00:05Z", "ended_at": null}
  ],
  "quality_warning": null
}
```

**Response:** `200 OK` (FAILED example)

```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "FAILED",
  "failed_stage": "INGESTING",
  "error": {
    "code": "unsupported_audio_format",
    "message": "File extension '.wma' is not supported. Accepted: wav, mp3, m4a, flac."
  },
  "stage_history": [
    {"stage": "QUEUED", "started_at": "2026-08-21T12:00:00Z", "ended_at": "2026-08-21T12:00:01Z"},
    {"stage": "INGESTING", "started_at": "2026-08-21T12:00:01Z", "ended_at": "2026-08-21T12:00:02Z"}
  ],
  "quality_warning": null
}
```

`status` is one of: `QUEUED` | `INGESTING` | `TRANSCRIBING` | `DIARIZING` |
`ALIGNING` | `EXTRACTING_FEATURES` | `SCORING` | `DONE` | `FAILED`.

---

### `GET /v1/jobs/{job_id}/highlights`

Fetch results (available when status is `DONE`).

**Response:** `200 OK`

```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "audio_duration_s": 5400.0,
  "num_speakers_detected": 3,
  "quality_warning": null,
  "highlights": [
    {
      "start_s": 1243.5,
      "end_s": 1298.0,
      "speaker": "SPEAKER_01",
      "score": 3.27,
      "reasons": [
        "strong sentiment swing",
        "peak pitch variance",
        "crosstalk"
      ],
      "transcript_excerpt": "Wait, wait — are you seriously telling me they just... walked out?",
      "low_confidence": false
    }
  ]
}
```

---

### `GET /v1/jobs/{job_id}/transcript`

Fetch the full aligned, speaker-attributed transcript. Primarily for
debugging and re-scoring tooling rather than typical end-user use.

**Response:** `200 OK`

```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "segments": [
    {
      "start_s": 0.0,
      "end_s": 4.2,
      "speaker": "SPEAKER_00",
      "text": "Welcome back to the show everyone.",
      "words": [
        {"text": "Welcome", "start_s": 0.0, "end_s": 0.4, "confidence": 0.98},
        {"text": "back", "start_s": 0.5, "end_s": 0.7, "confidence": 0.97}
      ]
    }
  ]
}
```

---

### `POST /v1/jobs/{job_id}/rescore`

Re-run only the scoring stage with different parameters, reusing cached
transcript, diarization, and feature artifacts. The new job starts at
`SCORING`, skipping earlier stages.

**Request:** `application/json`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `top_n` | int | same as original | Number of highlights |
| `min_clip_s` | float | same as original | Minimum clip length |
| `max_clip_s` | float | same as original | Maximum clip length |
| `weights_override` | object | null | Partial YAML-weight overrides (`{"sentiment_delta": 0.35}`) |

**Response:** `202 Accepted`

```json
{
  "job_id": "e82b4f1a-77cc-4372-a567-0e02b2c3d480",
  "source_job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "SCORING",
  "created_at": "2026-08-21T12:10:00Z"
}
```

---

## Error model

All 4xx/5xx responses use a consistent error envelope:

```json
{
  "error": {
    "code": "invalid_audio_format",
    "message": "File extension '.wma' is not supported. Accepted: wav, mp3, m4a, flac."
  }
}
```

Error codes are stable strings (not just HTTP status codes) so client code
can branch on them. Codes map to the failure taxonomy in the failure-modes
doc:

| HTTP status | Error code | Description |
|-------------|------------|-------------|
| 400 | `invalid_audio_format` | File extension not in allowed list |
| 400 | `file_too_large` | File exceeds max duration (default 4 hr) |
| 400 | `invalid_parameters` | top_n / min_clip_s / max_clip_s out of range |
| 404 | `job_not_found` | job_id does not exist |
| 409 | `job_not_done` | Highlights/transcript requested before DONE |
| 422 | `missing_file` | No file uploaded |
| 500 | `internal_error` | Unexpected worker failure (rare) |
