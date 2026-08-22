"""FastAPI application with async job endpoints, health check, CORS, and structured logging."""

import atexit
import logging
import os
import signal
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from highlight_extractor.api.job_manager import ArtifactStore, JobManager
from highlight_extractor.api.models import (
    ErrorBody,
    HighlightsResponse,
    JobStatus,
    RescoreRequest,
    RescoreResponse,
    TranscriptResponse,
)
from highlight_extractor.scoring.rank import rank_highlights
from highlight_extractor.utils.config import load_scoring_weights, merge_weights
from highlight_extractor.utils.logging import get_logger, setup_logging
from highlight_extractor.utils.settings import Settings, load_settings

logger = get_logger("api")

# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown hooks
# ---------------------------------------------------------------------------

_temp_files: set = set()


def _cleanup_temp_files():
    """Remove all temp files created by uploads."""
    for path in list(_temp_files):
        try:
            if os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass
    _temp_files.clear()


def _handle_shutdown(signum, frame):
    """Graceful shutdown handler for SIGTERM/SIGINT."""
    logger.info("shutdown_signal_received", extra={"signal": signum})
    _cleanup_temp_files()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: setup on startup, cleanup on shutdown."""
    # Startup
    settings = load_settings()
    setup_logging(getattr(logging, settings.log_level, logging.INFO))
    logger.info(
        "app_starting",
        extra={"host": settings.host, "port": settings.port, "workers": settings.workers},
    )

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    # Register atexit as fallback
    atexit.register(_cleanup_temp_files)

    yield

    # Shutdown
    logger.info("app_shutting_down")
    _cleanup_temp_files()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional settings override. Loaded from env if None.

    Returns:
        Configured FastAPI instance.
    """
    if settings is None:
        settings = load_settings()

    _app = FastAPI(
        title="Highlight Extraction Service",
        version="0.1.0",
        description="Automated highlight extraction from long-form podcast/talk audio.",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # --- CORS ---
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
    )

    # --- Global exception handler ---
    @_app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("unhandled_exception", extra={"path": request.url.path, "error": str(exc)})
        return JSONResponse(
            status_code=500,
            content=ErrorBody(
                code="internal_error",
                message="An unexpected error occurred. Please try again.",
            ).model_dump(),
        )

    return _app


# ---------------------------------------------------------------------------
# Default app instance (for uvicorn direct invocation)
# ---------------------------------------------------------------------------

_settings = load_settings()
app = create_app(_settings)

# Global job manager (in-process; thread-safe for this use pattern)
store = ArtifactStore(base_path=_settings.artifact_store_path)
manager = JobManager(artifact_store=store)

SUPPORTED_EXTENSIONS = _settings.supported_extensions
MAX_UPLOAD_SIZE = _settings.max_upload_size_bytes
_uploaded_temp_files: set = set()  # Track temp files for cleanup


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
                message=f"Extension '{ext}' not supported. Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            ).model_dump(),
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    content = file.file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=ErrorBody(
                code="file_too_large",
                message=f"File exceeds maximum size of {MAX_UPLOAD_SIZE // (1024 * 1024)} MB",
            ).model_dump(),
        )
    tmp.write(content)
    tmp.close()
    _uploaded_temp_files.add(tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/healthz", tags=["ops"])
async def health_check():
    """Health check endpoint for load balancers and orchestrators.

    Returns 200 OK when the service is running and accepting requests.
    """
    return {"status": "ok", "version": "0.1.0"}


@app.get("/readyz", tags=["ops"])
async def readiness_check():
    """Readiness probe: confirms the job manager and artifact store are functional.

    Returns 200 when ready, 503 when not.
    """
    try:
        store_ok = store.base.exists()
        return {"status": "ready" if store_ok else "not_ready", "artifact_store": str(store.base)}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "not_ready", "error": str(e)})


@app.post("/v1/jobs", status_code=202)
async def create_job(
    file: UploadFile = File(...),
    top_n: int | None = Form(15),
    min_clip_s: float | None = Form(12.0),
    max_clip_s: float | None = Form(90.0),
    expected_num_speakers: int | None = Form(None),
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

    logger.info("job_submitted", extra={"job_id": job_id, "filename": file.filename})

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
        transcript_map = {s.segment_id: s.text for s in (record._aligned.segments if record._aligned else [])}
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
