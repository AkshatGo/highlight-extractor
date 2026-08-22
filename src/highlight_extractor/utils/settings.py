"""Environment-based application settings.

All configuration is driven by environment variables with sane defaults.
Production deployments set these via .env, Docker env, or process manager.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class Settings:
    """Immutable application settings loaded from environment."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "INFO"
    reload: bool = False

    # CORS
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["*"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])

    # Upload limits
    max_upload_size_mb: int = 500
    supported_extensions: frozenset = frozenset({".wav", ".mp3", ".m4a", ".flac"})

    # Audio limits
    max_audio_duration_s: float = 4 * 3600  # 4 hours

    # Pipeline defaults
    default_top_n: int = 15
    default_min_clip_s: float = 12.0
    default_max_clip_s: float = 90.0

    # Artifact store
    artifact_store_path: str = "/tmp/highlight_artifacts"

    # Scoring
    scoring_weights_path: Optional[str] = None  # None = default config path

    # Models (for pre-download / caching)
    whisper_model: str = "base"
    diarization_model: str = "pyannote/speaker-diarization-3.1"

    # Worker
    pipeline_timeout_s: int = 7200  # 2 hours max per job

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


def load_settings() -> Settings:
    """Load settings from environment variables with defaults.

    Environment variable mapping:
        HOST              → host (default: 0.0.0.0)
        PORT              → port (default: 8000)
        WORKERS           → workers (default: 1)
        LOG_LEVEL         → log_level (default: INFO)
        RELOAD            → reload (default: false)
        CORS_ORIGINS      → comma-separated origins (default: *)
        CORS_METHODS      → comma-separated methods (default: *)
        CORS_HEADERS      → comma-separated headers (default: *)
        MAX_UPLOAD_MB     → max_upload_size_mb (default: 500)
        MAX_AUDIO_DURATION_S → max_audio_duration_s (default: 14400)
        ARTIFACT_STORE    → artifact_store_path (default: /tmp/highlight_artifacts)
        SCORING_WEIGHTS   → scoring_weights_path (default: None → config/scoring_weights.yaml)
        WHISPER_MODEL     → whisper_model (default: base)
        DIARIZATION_MODEL → diarization_model (default: pyannote/speaker-diarization-3.1)
        PIPELINE_TIMEOUT  → pipeline_timeout_s (default: 7200)
    """
    def _list_env(key: str, default: str) -> List[str]:
        val = os.environ.get(key, "")
        if not val or val.strip() == "*":
            return [item.strip() for item in default.split(",")]
        return [item.strip() for item in val.split(",") if item.strip()]

    return Settings(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        workers=int(os.environ.get("WORKERS", "1")),
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        reload=os.environ.get("RELOAD", "false").lower() in ("true", "1", "yes"),
        cors_origins=_list_env("CORS_ORIGINS", "*"),
        cors_methods=_list_env("CORS_METHODS", "*"),
        cors_headers=_list_env("CORS_HEADERS", "*"),
        max_upload_size_mb=int(os.environ.get("MAX_UPLOAD_MB", "500")),
        max_audio_duration_s=float(os.environ.get("MAX_AUDIO_DURATION_S", "14400")),
        artifact_store_path=os.environ.get("ARTIFACT_STORE", "/tmp/highlight_artifacts"),
        scoring_weights_path=os.environ.get("SCORING_WEIGHTS"),
        whisper_model=os.environ.get("WHISPER_MODEL", "base"),
        diarization_model=os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1"),
        pipeline_timeout_s=int(os.environ.get("PIPELINE_TIMEOUT", "7200")),
    )
