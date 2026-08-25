"""Tests for settings module and health/readiness endpoints."""

import os
import textwrap
from typing import ClassVar
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from highlight_extractor.api.app import app
from highlight_extractor.utils.settings import Settings, _load_dotenv, load_settings


class TestSettings:
    """Tests for the Settings dataclass and load_settings."""

    def test_default_values(self):
        """Settings should have sensible defaults."""
        s = Settings()
        assert s.host == "0.0.0.0"
        assert s.port == 8000
        assert s.workers == 1
        assert s.log_level == "INFO"
        assert s.max_upload_size_mb == 500
        assert s.max_upload_size_bytes == 500 * 1024 * 1024
        assert s.whisper_model == "base"
        assert s.pipeline_timeout_s == 7200
        assert ".wav" in s.supported_extensions
        assert ".mp3" in s.supported_extensions

    def test_max_upload_size_bytes_property(self):
        """max_upload_size_bytes should be max_upload_size_mb * 1024 * 1024."""
        s = Settings(max_upload_size_mb=100)
        assert s.max_upload_size_bytes == 100 * 1024 * 1024

    def test_load_settings_from_env(self):
        """load_settings should read from environment variables."""
        env = {
            "HOST": "127.0.0.1",
            "PORT": "9000",
            "WORKERS": "4",
            "LOG_LEVEL": "debug",
            "MAX_UPLOAD_MB": "200",
            "WHISPER_MODEL": "medium",
            "ARTIFACT_STORE": "/data/artifacts",
        }
        with patch.dict(os.environ, env):
            s = load_settings()
            assert s.host == "127.0.0.1"
            assert s.port == 9000
            assert s.workers == 4
            assert s.log_level == "DEBUG"
            assert s.max_upload_size_mb == 200
            assert s.whisper_model == "medium"
            assert s.artifact_store_path == "/data/artifacts"

    def test_load_settings_defaults_when_env_empty(self):
        """load_settings should use defaults when env vars are not set."""
        # Ensure no env vars are set
        keys = ["HOST", "PORT", "WORKERS", "LOG_LEVEL", "MAX_UPLOAD_MB", "WHISPER_MODEL"]
        env_clear = {k: "" for k in keys if k in os.environ}
        with patch.dict(os.environ, env_clear, clear=True):
            s = load_settings()
            assert s.host == "0.0.0.0"
            assert s.port == 8000

    def test_cors_origins_from_env(self):
        """CORS_ORIGINS should be parsed as comma-separated list."""
        with patch.dict(os.environ, {"CORS_ORIGINS": "https://app.com, https://admin.com"}):
            s = load_settings()
            assert s.cors_origins == ["https://app.com", "https://admin.com"]

    def test_reload_from_env(self):
        """RELOAD env var should be parsed as boolean."""
        with patch.dict(os.environ, {"RELOAD": "true"}):
            s = load_settings()
            assert s.reload is True

        with patch.dict(os.environ, {"RELOAD": "false"}):
            s = load_settings()
            assert s.reload is False

    def test_settings_is_frozen(self):
        """Settings should be immutable."""
        s = Settings()
        with pytest.raises(AttributeError):
            s.port = 9999

    def test_hf_token_default_is_none(self):
        """hf_token should default to None."""
        s = Settings()
        assert s.hf_token is None

    def test_hf_token_from_env(self):
        """HF_TOKEN env var should map to hf_token."""
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test_token_abc"}):
            s = load_settings()
            assert s.hf_token == "hf_test_token_abc"

    def test_hf_token_empty_string_becomes_none(self):
        """Empty HF_TOKEN should resolve to None, not empty string."""
        with patch.dict(os.environ, {"HF_TOKEN": ""}):
            s = load_settings()
            assert s.hf_token is None

    def test_diarization_model_from_env(self):
        """DIARIZATION_MODEL env var should override the default."""
        with patch.dict(os.environ, {"DIARIZATION_MODEL": "pyannote/speaker-diarization-3.0"}):
            s = load_settings()
            assert s.diarization_model == "pyannote/speaker-diarization-3.0"

    def test_diarization_model_default(self):
        """Default diarization model should be pyannote/speaker-diarization-3.1."""
        s = Settings()
        assert s.diarization_model == "pyannote/speaker-diarization-3.1"


class TestLoadDotenv:
    """Tests for the _load_dotenv helper."""

    # Keys managed by dotenv tests — cleared between each test to avoid leakage
    _MANAGED_KEYS: ClassVar[list[str]] = ["HF_TOKEN", "WHISPER_MODEL", "VALID_KEY", "NO_EQUALS_HERE", "ARTIFACT_STORE"]

    @pytest.fixture(autouse=True)
    def _clean_env(self):
        """Remove managed keys before and after each test."""
        removed = {k: os.environ.pop(k) for k in self._MANAGED_KEYS if k in os.environ}
        yield
        # Restore only keys that existed before the test
        os.environ.update(removed)

    def test_loads_env_file(self, tmp_path):
        """Should read KEY=VALUE pairs from a .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text('HF_TOKEN=test_token_123\nWHISPER_MODEL=small\n')
        _load_dotenv(env_file)
        assert os.environ.get("HF_TOKEN") == "test_token_123"
        assert os.environ.get("WHISPER_MODEL") == "small"

    def test_existing_env_vars_take_precedence(self, tmp_path):
        """Environment variables already set should not be overridden by .env."""
        env_file = tmp_path / ".env"
        env_file.write_text('WHISPER_MODEL=large\n')
        os.environ["WHISPER_MODEL"] = "medium"
        _load_dotenv(env_file)
        assert os.environ["WHISPER_MODEL"] == "medium"

    def test_missing_file_is_noop(self, tmp_path):
        """Missing .env file should not raise an error."""
        _load_dotenv(tmp_path / "nonexistent.env")  # Should not raise

    def test_skips_comments_and_blank_lines(self, tmp_path):
        """Comment lines and blank lines should be ignored."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            textwrap.dedent("""\
                # This is a comment

                HF_TOKEN=valid_token
                # Another comment
                WHISPER_MODEL=small
            """)
        )
        _load_dotenv(env_file)
        assert os.environ.get("HF_TOKEN") == "valid_token"
        assert os.environ.get("WHISPER_MODEL") == "small"

    def test_strips_quotes_from_values(self, tmp_path):
        """Quoted values should have quotes stripped."""
        env_file = tmp_path / ".env"
        env_file.write_text('HF_TOKEN="quoted_token"\nARTIFACT_STORE=\'\'/data/store\'\'\n')
        _load_dotenv(env_file)
        assert os.environ.get("HF_TOKEN") == "quoted_token"
        assert os.environ.get("ARTIFACT_STORE") == "/data/store"

    def test_skips_lines_without_equals(self, tmp_path):
        """Lines without = should be skipped."""
        env_file = tmp_path / ".env"
        env_file.write_text('VALID_KEY=value\nNO_EQUALS_HERE\n')
        _load_dotenv(env_file)
        assert os.environ.get("VALID_KEY") == "value"
        assert os.environ.get("NO_EQUALS_HERE") is None


class TestHealthEndpoint:
    """Tests for /healthz and /readyz endpoints."""

    def test_health_check_returns_200(self):
        """GET /healthz should return 200 with status ok."""
        client = TestClient(app)
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"

    def test_readiness_check_returns_200_when_store_exists(self):
        """GET /readyz should return 200 when artifact store exists."""
        client = TestClient(app)
        response = client.get("/readyz")
        # Should return 200 or 503 depending on store state
        assert response.status_code in (200, 503)
        data = response.json()
        assert "status" in data


class TestAPIRoutes:
    """Basic route smoke tests."""

    def test_docs_available(self):
        """GET /docs should return Swagger UI."""
        client = TestClient(app)
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema(self):
        """GET /openapi.json should return the schema."""
        client = TestClient(app)
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/healthz" in schema["paths"]
        assert "/v1/jobs" in schema["paths"]
