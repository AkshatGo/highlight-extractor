"""Integration test: run the demo pipeline and verify it produces valid output."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
class TestDemoPipeline:
    """End-to-end test of the demo pipeline script."""

    def test_demo_pipeline_runs_and_produces_json(self, tmp_path):
        """The demo pipeline should exit 0 and produce valid highlights JSON."""
        script = Path(__file__).resolve().parents[1] / "scripts" / "demo_pipeline.py"
        assert script.exists(), f"Demo script not found at {script}"

        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(tmp_path),
        )

        assert result.returncode == 0, (
            f"Demo pipeline exited with code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

        # The script writes to /tmp/highlights_demo.json
        out_path = Path("/tmp/highlights_demo.json")
        assert out_path.exists(), "Demo pipeline did not produce highlights_demo.json"

        data = json.loads(out_path.read_text())
        assert isinstance(data, list), "Output should be a JSON array"
        assert len(data) > 0, "Output should contain at least one highlight"

        # Validate structure of each highlight item
        for item in data:
            assert "start_s" in item
            assert "end_s" in item
            assert "speaker" in item
            assert "score" in item
            assert isinstance(item["score"], (int, float))
            assert item["start_s"] < item["end_s"]

    def test_demo_pipeline_output_ranked(self, tmp_path):
        """Highlights should be ranked by score in descending order."""
        script = Path(__file__).resolve().parents[1] / "scripts" / "demo_pipeline.py"

        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(tmp_path),
        )

        assert result.returncode == 0

        data = json.loads(Path("/tmp/highlights_demo.json").read_text())
        scores = [item["score"] for item in data]
        assert scores == sorted(scores, reverse=True), (
            f"Highlights should be sorted by score descending, got {scores}"
        )

    def test_demo_pipeline_stdout_has_summary(self, tmp_path):
        """The demo should print a TOP N summary to stdout."""
        script = Path(__file__).resolve().parents[1] / "scripts" / "demo_pipeline.py"

        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(tmp_path),
        )

        assert result.returncode == 0
        assert "TOP" in result.stdout, "Expected 'TOP N HIGHLIGHTS' in stdout"
        assert "Score:" in result.stdout, "Expected 'Score:' in stdout output"
