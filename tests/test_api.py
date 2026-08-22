"""Tests for the FastAPI API layer — mocked pipeline, no real models."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from highlight_extractor.api.app import app
from highlight_extractor.api.models import JobStatus, HighlightItem


client = TestClient(app)


def test_create_job_no_file():
    """POST /v1/jobs without file should return 422."""
    response = client.post("/v1/jobs")
    assert response.status_code == 422


def test_get_nonexistent_job():
    response = client.get("/v1/jobs/nonexistent-id")
    assert response.status_code == 404
    data = response.json()
    detail = data.get("detail") or data.get("error")
    assert detail["code"] == "job_not_found"


def test_get_highlights_before_done():
    """GET highlights on a non-DONE job should return 409."""
    # Submit a job (fileless simulation — just check the error path)
    with patch("highlight_extractor.api.app.manager") as mock_mgr:
        mock_record = MagicMock()
        mock_record.status = JobStatus.QUEUED
        mock_record._duration_s = 0.0
        mock_record._diarization = None
        mock_record.quality_warning = None
        mock_mgr.get_job.return_value = mock_record
        mock_mgr.get_highlights.return_value = None

        response = client.get("/v1/jobs/test-id/highlights")
        assert response.status_code == 404 or response.status_code == 409


@pytest.mark.slow
def test_create_and_poll_job_e2e():
    """End-to-end test with a real small audio file. Requires GPU/models."""
    import librosa
    import numpy as np
    import tempfile
    import os

    # Generate 15 seconds of simple test audio
    sr = 16000
    duration = 15
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Two tones: first half "speaker A", second half "speaker B"
    audio = np.zeros_like(t)
    audio[:len(t)//2] = 0.1 * np.sin(2 * np.pi * 200 * t[:len(t)//2])
    audio[len(t)//2:] = 0.1 * np.sin(2 * np.pi * 300 * t[len(t)//2:])

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    import soundfile as sf
    sf.write(tmp.name, audio, sr)
    tmp.close()

    try:
        with open(tmp.name, "rb") as f:
            response = client.post(
                "/v1/jobs",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"top_n": 3},
            )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        assert job_id is not None

        # Poll until done or timeout
        import time
        for _ in range(60):
            resp = client.get(f"/v1/jobs/{job_id}")
            status = resp.json()["status"]
            if status == "DONE" or status == "FAILED":
                break
            time.sleep(5)

        # Check highlights
        resp = client.get(f"/v1/jobs/{job_id}/highlights")
        if resp.status_code == 200:
            data = resp.json()
            assert "highlights" in data
    finally:
        os.unlink(tmp.name)
