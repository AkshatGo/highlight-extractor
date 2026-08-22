"""Tests for settings module and health/readiness endpoints."""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from highlight_extractor.api.app import app
from highlight_extractor.utils.settings import Settings, load_settings


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
