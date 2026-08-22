"""Tests for the ingestion module."""

import pytest
import numpy as np
from pathlib import Path
from highlight_extractor.utils.audio import validate_format, validate_duration
from highlight_extractor.utils.qc import QCResult, run_qc


class TestValidateFormat:
    def test_supported_formats(self):
        for ext in [".wav", ".mp3", ".m4a", ".flac"]:
            p = Path(f"audio{ext}")
            result = validate_format(p)
            assert result == ext

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            validate_format(Path("audio.wma"))


class TestQCFunctions:
    def test_run_qc_clean_audio(self):
        sr = 16000
        duration = 2.0
        # Generate speech-like audio: moderate amplitude sine + low noise
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        audio += np.random.randn(len(audio)).astype(np.float32) * 0.001
        result = run_qc(audio, sr, duration)
        assert isinstance(result, QCResult)
        assert result.pass_qc is True
        # Moderate amplitude should not trigger clipping or low-SNR warnings
        assert result.quality_warning is None

    def test_run_qc_clipped_audio(self):
        sr = 16000
        duration = 1.0
        # Create signal with lots of clipping
        audio = np.ones(int(sr * duration), dtype=np.float32) * 0.999
        result = run_qc(audio, sr, duration)
        assert result.pass_qc is True
        # Should have a warning about clipping
        assert result.quality_warning is not None
        assert "clipping" in result.quality_warning.lower()

    def test_run_qc_silence(self):
        sr = 16000
        duration = 1.0
        audio = np.zeros(int(sr * duration), dtype=np.float32)
        result = run_qc(audio, sr, duration)
        assert result.pass_qc is True
        assert result.snr_db is None or result.snr_db == 0.0 or result.snr_db == float('inf')
