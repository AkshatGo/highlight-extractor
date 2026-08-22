"""FastAPI application with async job endpoints."""

import os
import tempfile
import threading
import atexit
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from highlight_extractor.api.models import (
    ErrorBody,
    ErrorResponse,
    HighlightsResponse,
    JobResponse,
    JobStatus,
    RescoreRequest,
    RescoreResponse,
    TranscriptResponse,
)
from highlight_extractor.api.job_manager import JobManager, ArtifactStore
from highlight_extractor.scoring.rank import rank_highlights
from highlight_extractor.utils.config import load_scoring_weights, merge_weights

app = FastAPI(title="Highlight Extraction Service", version="0.1.0")

# Global job manager (in-process; thread-safe for this use pattern)
store = ArtifactStore()
manager = JobManager(artifact_store=store)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB
_uploaded_temp_files: set[str] = set()  # Track temp files for cleanup


def _cleanup_temp_files():
    """Remove all temp files created by uploads."""
    for path in _uploaded_temp_files:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass
    _uploaded_temp_files.clear()


atexit.register(_cleanup_temp_files)


def _save_upload(file: UploadFile) -> str:
    """Save uploaded file to a temp path and return the path."""
    ext = Path(file.filename or "upload.wav").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=ErrorBody(
                code="invalid_audio_format",
                message=f"Extension '{ext}' not supported. Accepted: {', '.join(SUPPORTED_EXTENSIONS)}",
            ).model_dump(),
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    content = file.file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=ErrorBody(
                code="file_too_large",
                message=f"File exceeds maximum size of {MAX_UPLOAD_SIZE // (1024*1024)} MB",
            ).model_dump(),
        )
    tmp.write(content)
    tmp.close()
    _uploaded_temp_files.add(tmp.name)
    return tmp.name


@app.post("/v1/jobs", status_code=202)
async def create_job(
    file: UploadFile = File(...),
    top_n: Optional[int] = Form(15),
    min_clip_s: Optional[float] = Form(12.0),
    max_clip_s: Optional[float] = Form(90.0),
    expected_num_speakers: Optional[int] = Form(None),
):
    """Submit audio for highlight extraction."""
    audio_path = _save_upload(file)
    job_id = manager.submit(
        audio_path=audio_path,
        top_n=top_n if top_n is not None else 15,
        min_clip_s=min_clip_s if min_clip_s is not None else 12.0,
        max_clip_s=max_clip_s if max_clip_s is not None else 90.0,
        expected_num_speakers=expected_num_speakers,
    )

    # Launch pipeline in background thread
    thread = threading.Thread(target=manager.run_pipeline, args=(job_id,), daemon=True)
    thread.start()

    record = manager.get_job(job_id)
    return record.to_response()


@app.get("/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Poll job status."""
    record = manager.get_job(job_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorBody(code="job_not_found", message=f"No job with id '{job_id}'").model_dump(),
        )
    return record.to_response()


@app.get("/v1/jobs/{job_id}/highlights")
async def get_highlights(job_id: str):
    """Fetch scored highlights (available when DONE)."""
    record = manager.get_job(job_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorBody(code="job_not_found", message=f"No job with id '{job_id}'").model_dump(),
        )

    if record.status != JobStatus.DONE:
        raise HTTPException(
            status_code=409,
            detail=ErrorBody(
                code="job_not_done",
                message=f"Job status is '{record.status.value}', not DONE",
            ).model_dump(),
        )

    highlights = manager.get_highlights(job_id)
    diarization = record._diarization

    return HighlightsResponse(
        job_id=job_id,
        audio_duration_s=record._duration_s or 0.0,
        num_speakers_detected=diarization.num_speakers if diarization else 0,
        quality_warning=record.quality_warning,
        highlights=highlights or [],
    )


@app.get("/v1/jobs/{job_id}/transcript")
async def get_transcript(job_id: str):
    """Fetch the full aligned transcript (debugging/re-scoring tooling)."""
    record = manager.get_job(job_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorBody(code="job_not_found", message=f"No job with id '{job_id}'").model_dump(),
        )

    if record.status != JobStatus.DONE and record.status != JobStatus.FAILED:
        raise HTTPException(
            status_code=409,
            detail=ErrorBody(
                code="job_not_done",
                message=f"Job status is '{record.status.value}', not DONE",
            ).model_dump(),
        )

    segments = manager.get_transcript(job_id)
    return TranscriptResponse(job_id=job_id, segments=segments or [])


@app.post("/v1/jobs/{job_id}/rescore", status_code=202)
async def rescore_job(job_id: str, body: RescoreRequest):
    """Re-run only the scoring stage with different parameters, reusing cached artifacts."""
    record = manager.get_job(job_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorBody(code="job_not_found", message=f"No job with id '{job_id}'").model_dump(),
        )

    if record._features is None:
        raise HTTPException(
            status_code=409,
            detail=ErrorBody(
                code="job_not_ready",
                message="Source job has no cached features (not past EXTRACTING_FEATURES stage)",
            ).model_dump(),
        )

    # Create a new job
    new_job_id = manager.submit(
        audio_path=record.audio_path,
        top_n=body.top_n if body.top_n is not None else record.top_n,
        min_clip_s=body.min_clip_s if body.min_clip_s is not None else record.min_clip_s,
        max_clip_s=body.max_clip_s if body.max_clip_s is not None else record.max_clip_s,
    )
    new_record = manager.get_job(new_job_id)
    new_record._audio = record._audio
    new_record._sr = record._sr
    new_record._duration_s = record._duration_s
    new_record._transcription = record._transcription
    new_record._diarization = record._diarization
    new_record._aligned = record._aligned
    new_record._features = record._features

    # Record the skipped stages (cached artifacts reused)
    for stage_name in ["INGESTING", "TRANSCRIBING", "DIARIZING", "ALIGNING", "EXTRACTING_FEATURES"]:
        new_record.transition_to(JobStatus(stage_name))

    # Override weights if provided
    weights = load_scoring_weights()
    if body.weights_override:
        weights = merge_weights(weights, body.weights_override)

    # Re-run scoring
    new_record.transition_to(JobStatus.SCORING)
    try:
        transcript_map = {
            s.segment_id: s.text
            for s in (record._aligned.segments if record._aligned else [])
        }
        highlights = rank_highlights(
            record._features,
            weights=weights,
            top_n=new_record.top_n,
            transcripts=transcript_map,
        )
        new_record._highlights = highlights
        new_record.transition_to(JobStatus.DONE)
    except Exception as e:
        new_record.fail("SCORING", "internal_error", str(e))

    return RescoreResponse(
        job_id=new_job_id,
        source_job_id=job_id,
        status=new_record.status.value,
        created_at=new_record.created_at,
    )
