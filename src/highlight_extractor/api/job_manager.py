"""In-process job manager: queue, artifact store, and pipeline orchestrator."""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from highlight_extractor.alignment import AlignedSegment, AlignmentResult, run_alignment
from highlight_extractor.scoring.segment import derive_candidate_segments
from highlight_extractor.api.models import (
    ErrorBody,
    HighlightItem,
    JobResponse,
    JobStatus,
    StageEntry,
    TranscriptSegment,
    TranscriptWord,
)
from highlight_extractor.diarization.pipeline import DiarizationResult, run_diarization
from highlight_extractor.ingestion.pipeline import run_ingestion
from highlight_extractor.scoring.features import SegmentFeatures, extract_features
from highlight_extractor.scoring.rank import Highlight, rank_highlights
from highlight_extractor.transcription.pipeline import TranscriptionResult, run_transcription
from highlight_extractor.utils.config import load_scoring_weights, merge_weights


class JobRecord:
    """In-memory job record (production would use Postgres)."""

    def __init__(
        self,
        job_id: str,
        audio_path: str,
        top_n: int = 15,
        min_clip_s: float = 12.0,
        max_clip_s: float = 90.0,
        expected_num_speakers: Optional[int] = None,
    ):
        self.job_id = job_id
        self.audio_path = audio_path
        self.top_n = top_n
        self.min_clip_s = min_clip_s
        self.max_clip_s = max_clip_s
        self.expected_num_speakers = expected_num_speakers
        self.status = JobStatus.QUEUED
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.stage_history: List[StageEntry] = [StageEntry(stage="QUEUED", started_at=self.created_at)]
        self.quality_warning: Optional[str] = None
        self.failed_stage: Optional[str] = None
        self.error: Optional[ErrorBody] = None

        # Cached artifacts
        self._audio: Optional[np.ndarray] = None
        self._sr: Optional[int] = None
        self._duration_s: Optional[float] = None
        self._transcription: Optional[TranscriptionResult] = None
        self._diarization: Optional[DiarizationResult] = None
        self._aligned: Optional[AlignmentResult] = None
        self._features: Optional[List[SegmentFeatures]] = None
        self._highlights: Optional[List[Highlight]] = None

    def transition_to(self, status: JobStatus):
        """Record state transition with timestamp."""
        now = datetime.now(timezone.utc).isoformat()
        # End previous stage
        if self.stage_history:
            self.stage_history[-1].ended_at = now
        self.status = status
        self.stage_history.append(StageEntry(stage=status.value, started_at=now))

    def fail(self, stage: str, code: str, message: str):
        """Mark job as failed with structured error."""
        self.failed_stage = stage
        self.error = ErrorBody(code=code, message=message)
        self.transition_to(JobStatus.FAILED)

    def to_response(self) -> JobResponse:
        return JobResponse(
            job_id=self.job_id,
            status=self.status,
            created_at=self.created_at,
            stage_history=self.stage_history,
            quality_warning=self.quality_warning,
            failed_stage=self.failed_stage,
            error=self.error,
        )


class ArtifactStore:
    """Local-filesystem artifact store (swap for S3 in production)."""

    def __init__(self, base_path: str = "/tmp/highlight_artifacts"):
        self.base = Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)

    def _job_dir(self, job_id: str) -> Path:
        p = self.base / job_id
        p.mkdir(exist_ok=True)
        return p

    def save_json(self, job_id: str, name: str, data: Any):
        path = self._job_dir(job_id) / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_json(self, job_id: str, name: str) -> Optional[Any]:
        path = self._job_dir(job_id) / f"{name}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)


class JobManager:
    """Orchestrates the pipeline for a single job. Thread-safe for in-process use."""

    def __init__(self, artifact_store: Optional[ArtifactStore] = None):
        self.store = artifact_store or ArtifactStore()
        self._jobs: Dict[str, JobRecord] = {}

    def submit(
        self,
        audio_path: str,
        top_n: int = 15,
        min_clip_s: float = 12.0,
        max_clip_s: float = 90.0,
        expected_num_speakers: Optional[int] = None,
    ) -> str:
        job_id = str(uuid.uuid4())
        record = JobRecord(
            job_id=job_id,
            audio_path=audio_path,
            top_n=top_n,
            min_clip_s=min_clip_s,
            max_clip_s=max_clip_s,
            expected_num_speakers=expected_num_speakers,
        )
        self._jobs[job_id] = record
        return job_id

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    def run_pipeline(self, job_id: str):
        """Execute the full pipeline for a job. Intended to run in a worker thread."""
        record = self.get_job(job_id)
        if record is None:
            return

        try:
            # --- INGESTING ---
            record.transition_to(JobStatus.INGESTING)
            audio, sr, duration_s, qc = run_ingestion(record.audio_path)
            record._audio = audio
            record._sr = sr
            record._duration_s = duration_s
            if qc.quality_warning:
                record.quality_warning = qc.quality_warning
            self.store.save_json(job_id, "qc", {
                "snr_db": qc.snr_db,
                "clipping_fraction": qc.clipping_fraction,
                "duration_s": qc.duration_s,
                "quality_warning": qc.quality_warning,
            })

            # Save normalized audio for reuse
            import soundfile as sf
            norm_path = str(self.store._job_dir(job_id) / "audio_normalized.wav")
            sf.write(norm_path, audio, sr, subtype="PCM_16")

            # --- TRANSCRIBING ---
            record.transition_to(JobStatus.TRANSCRIBING)
            transcription = run_transcription(norm_path)
            record._transcription = transcription
            self.store.save_json(job_id, "transcript", {
                "words": [
                    {"text": w.text, "start_s": w.start_s, "end_s": w.end_s, "confidence": w.confidence}
                    for w in transcription.words
                ],
                "language": transcription.language,
                "model_version": transcription.model_version,
            })

            # --- DIARIZING ---
            record.transition_to(JobStatus.DIARIZING)
            diarization = run_diarization(norm_path, expected_num_speakers=record.expected_num_speakers)
            record._diarization = diarization
            self.store.save_json(job_id, "diarization", {
                "turns": [
                    {"start_s": t.start_s, "end_s": t.end_s, "speaker": t.speaker, "overlap": t.overlap}
                    for t in diarization.turns
                ],
                "num_speakers": diarization.num_speakers,
                "model_version": diarization.model_version,
            })

            # --- ALIGNING ---
            record.transition_to(JobStatus.ALIGNING)
            aligned = run_alignment(transcription, diarization)
            record._aligned = aligned
            self.store.save_json(job_id, "aligned_segments", {
                "segments": [
                    {
                        "start_s": s.start_s, "end_s": s.end_s, "speaker": s.speaker,
                        "text": s.text, "crosstalk": s.crosstalk,
                    }
                    for s in aligned.segments
                ],
            })

            # --- EXTRACTING FEATURES ---
            record.transition_to(JobStatus.EXTRACTING_FEATURES)
            candidates = derive_candidate_segments(
                aligned.segments,
                min_clip_s=record.min_clip_s,
                max_clip_s=record.max_clip_s,
            )
            features = extract_features(candidates, audio, sr)
            record._features = features
            self.store.save_json(job_id, "features", [
                {
                    "segment_id": f.segment_id, "start_s": f.start_s, "end_s": f.end_s,
                    "speaker": f.speaker, "sentiment_delta": f.sentiment_delta,
                    "sentiment_extremity": f.sentiment_extremity,
                    "energy_zscore": f.energy_zscore, "pitch_variance": f.pitch_variance,
                    "speech_rate_delta": f.speech_rate_delta,
                    "keyword_density": f.keyword_density,
                    "crosstalk_flag": f.crosstalk_flag, "asr_confidence": f.asr_confidence,
                }
                for f in features
            ])

            # --- SCORING ---
            record.transition_to(JobStatus.SCORING)
            weights = load_scoring_weights()
            transcript_map = {s.segment_id: s.text for s in candidates}
            highlights = rank_highlights(
                features,
                weights=weights,
                top_n=record.top_n,
                transcripts=transcript_map,
            )
            record._highlights = highlights
            self.store.save_json(job_id, "highlights", [
                {
                    "start_s": h.start_s, "end_s": h.end_s, "speaker": h.speaker,
                    "score": h.score, "reasons": h.reasons,
                    "transcript_excerpt": h.transcript_excerpt,
                    "low_confidence": h.low_confidence,
                }
                for h in highlights
            ])

            # --- DONE ---
            record.transition_to(JobStatus.DONE)

        except Exception as e:
            current_stage = record.status.value
            record.fail(current_stage, "internal_error", str(e))

    def get_highlights(self, job_id: str) -> Optional[List[HighlightItem]]:
        """Return scored highlights as Pydantic models."""
        record = self.get_job(job_id)
        if record is None or record._highlights is None:
            return None
        return [
            HighlightItem(
                start_s=h.start_s,
                end_s=h.end_s,
                speaker=h.speaker,
                score=h.score,
                reasons=h.reasons,
                transcript_excerpt=h.transcript_excerpt,
                low_confidence=h.low_confidence,
            )
            for h in record._highlights
        ]

    def get_transcript(self, job_id: str) -> Optional[List[TranscriptSegment]]:
        """Return aligned transcript segments."""
        record = self.get_job(job_id)
        if record is None or record._aligned is None:
            return None
        return [
            TranscriptSegment(
                start_s=s.start_s,
                end_s=s.end_s,
                speaker=s.speaker,
                text=s.text,
                words=[
                    TranscriptWord(text=w.text, start_s=w.start_s, end_s=w.end_s, confidence=w.confidence)
                    for w in s.words
                ],
            )
            for s in record._aligned.segments
        ]
