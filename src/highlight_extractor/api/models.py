"""Pydantic models for the HTTP API."""

from __future__ import annotations

from dataclasses import field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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
    started_at: Optional[str] = None
    ended_at: Optional[str] = None


class ErrorBody(BaseModel):
    code: str
    message: str


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: Optional[str] = None
    stage_history: List[StageEntry] = field(default_factory=list)
    quality_warning: Optional[str] = None
    failed_stage: Optional[str] = None
    error: Optional[ErrorBody] = None


class HighlightItem(BaseModel):
    start_s: float
    end_s: float
    speaker: str
    score: float
    reasons: List[str] = []
    transcript_excerpt: str = ""
    low_confidence: bool = False


class HighlightsResponse(BaseModel):
    job_id: str
    audio_duration_s: float
    num_speakers_detected: int
    quality_warning: Optional[str] = None
    highlights: List[HighlightItem] = []


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
    words: List[TranscriptWord] = []


class TranscriptResponse(BaseModel):
    job_id: str
    segments: List[TranscriptSegment] = []


class RescoreRequest(BaseModel):
    top_n: Optional[int] = None
    min_clip_s: Optional[float] = None
    max_clip_s: Optional[float] = None
    weights_override: Optional[Dict[str, float]] = None


class RescoreResponse(BaseModel):
    job_id: str
    source_job_id: str
    status: str
    created_at: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorBody
