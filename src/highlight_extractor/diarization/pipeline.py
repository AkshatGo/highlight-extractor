"""Diarization stage: speaker turn detection via pyannote.audio."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SpeakerTurn:
    start_s: float
    end_s: float
    speaker: str
    overlap: bool = False


@dataclass
class DiarizationResult:
    turns: list[SpeakerTurn] = field(default_factory=list)
    num_speakers: int = 0
    model_version: str = ""


def run_diarization(
    audio_path: str | Path,
    expected_num_speakers: int | None = None,
) -> DiarizationResult:
    """Run speaker diarization on normalized audio.

    Uses pyannote.audio pipeline; returns speaker turns with overlap flags.

    Args:
        audio_path: Path to a 16 kHz mono WAV file.
        expected_num_speakers: Optional hint to constrain speaker count.

    Returns:
        DiarizationResult with speaker turns, count, and model version.
    """
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=None,
    )

    if expected_num_speakers is not None:
        pipeline.instantiate(
            {
                "clustering": {
                    "method": "centroid",
                    "min_components": expected_num_speakers,
                    "max_components": expected_num_speakers,
                }
            }
        )

    diarization: Annotation = pipeline(str(audio_path))

    result = DiarizationResult(
        model_version="pyannote-speaker-diarization-3.1",
    )

    seen_speakers: set = set()
    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
        result.turns.append(
            SpeakerTurn(
                start_s=turn.start,
                end_s=turn.end,
                speaker=str(speaker_label),
                overlap=False,
            )
        )
        seen_speakers.add(str(speaker_label))

    result.num_speakers = len(seen_speakers)
    return result
