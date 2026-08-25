"""Pydantic models for the HTTP API."""

from __future__ import annotations

from dataclasses import field
from enum import Enum

from pydantic import BaseModel


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    INGESTING = "INGESTING"
    TRANSCRIBING = "TRANSCRIBING"
    DIARIZING = "DIARIZING"
    ALIGNING = "ALIGNING"
    EXTRACTING_FEATURES = "EXTRACTING_FEATURES"
    SCORING = "SCORING"
    DONE = "DONE"
    FAILED = "FAILED"


class StageEntry(BaseModel):
    stage: str
    started_at: str | None = None
    ended_at: str | None = None


class ErrorBody(BaseModel):
    code: str
    message: str


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str | None = None
    stage_history: list[StageEntry] = field(default_factory=list)
    quality_warning: str | None = None
    failed_stage: str | None = None
    error: ErrorBody | None = None
    webhook_url: str | None = None
    keyword_preset: str | None = None


class HighlightItem(BaseModel):
    start_s: float
    end_s: float
    speaker: str
    score: float
    reasons: list[str] = []
    transcript_excerpt: str = ""
    low_confidence: bool = False


class HighlightsResponse(BaseModel):
    job_id: str
    audio_duration_s: float
    num_speakers_detected: int
    quality_warning: str | None = None
    highlights: list[HighlightItem] = []


class TranscriptWord(BaseModel):
    text: str
    start_s: float
    end_s: float
    confidence: float


class TranscriptSegment(BaseModel):
    start_s: float
    end_s: float
    speaker: str
    text: str
    words: list[TranscriptWord] = []


class TranscriptResponse(BaseModel):
    job_id: str
    segments: list[TranscriptSegment] = []


class RescoreRequest(BaseModel):
    top_n: int | None = None
    min_clip_s: float | None = None
    max_clip_s: float | None = None
    weights_override: dict[str, float] | None = None
    keywords: list[str] | None = None


class PresetsResponse(BaseModel):
    keyword_presets: list[str] = []


class WebhookPayload(BaseModel):
    """Payload sent to webhook_url on job completion or failure."""
    event: str  # "job.completed" or "job.failed"
    job_id: str
    status: str
    highlights_url: str | None = None
    error: ErrorBody | None = None


class RescoreResponse(BaseModel):
    job_id: str
    source_job_id: str
    status: str
    created_at: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
