"""Alignment stage: merge transcript words with speaker turns by timestamp overlap.

This is the highest-value unit-test target in the project — it is pure Python
with no ML dependency and edge cases (boundary words, overlapping turns, gaps).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from highlight_extractor.transcription.pipeline import TranscriptionResult, Word
from highlight_extractor.diarization.pipeline import DiarizationResult, SpeakerTurn


@dataclass
class AlignedSegment:
    start_s: float
    end_s: float
    speaker: str
    text: str
    words: List[Word] = field(default_factory=list)
    crosstalk: bool = False


@dataclass
class AlignmentResult:
    segments: List[AlignedSegment] = field(default_factory=list)


def _word_overlap(word: Word, turn: SpeakerTurn) -> float:
    """Compute overlap duration between a word and a speaker turn."""
    start = max(word.start_s, turn.start_s)
    end = min(word.end_s, turn.end_s)
    return max(0.0, end - start)


def _assign_word(word: Word, turns: List[SpeakerTurn]) -> Optional[SpeakerTurn]:
    """Assign a word to the speaker turn with the most overlap.

    Returns None if no turn overlaps (shouldn't happen with good alignment).
    """
    best_turn: Optional[SpeakerTurn] = None
    best_overlap = 0.0
    for turn in turns:
        overlap = _word_overlap(word, turn)
        if overlap > best_overlap:
            best_overlap = overlap
            best_turn = turn
    return best_turn if best_overlap > 0 else None


def _word_midpoint(w: Word) -> float:
    return (w.start_s + w.end_s) / 2.0


def run_alignment(
    transcription: TranscriptionResult,
    diarization: DiarizationResult,
    min_pause_s: float = 0.7,
) -> AlignmentResult:
    """Merge transcription words into speaker-attributed segments.

    Strategy:
        1. Assign each word to the speaker turn with the most overlap.
        2. Group consecutive words assigned to the same speaker.
        3. Split groups on pauses >= min_pause_s within a single speaker.
        4. Build AlignedSegment objects.

    Args:
        transcription: Output from run_transcription.
        diarization: Output from run_diarization.
        min_pause_s: Minimum silence gap to split a monologue (default 700ms).

    Returns:
        AlignmentResult with speaker-attributed text segments.
    """
    if not transcription.words:
        return AlignmentResult()

    if not diarization.turns:
        # No diarization — fall back to single-speaker
        turns = [SpeakerTurn(
            start_s=transcription.words[0].start_s,
            end_s=transcription.words[-1].end_s,
            speaker="SPEAKER_00",
        )]
    else:
        turns = diarization.turns

    # Assign each word to a speaker
    word_speaker: List[Tuple[Word, str]] = []
    for w in transcription.words:
        assigned = _assign_word(w, turns)
        speaker = assigned.speaker if assigned else "SPEAKER_00"
        word_speaker.append((w, speaker))

    # Group consecutive same-speaker words, splitting on long pauses
    segments: List[AlignedSegment] = []
    current_words: List[Word] = []
    current_speaker: Optional[str] = None

    def _flush():
        nonlocal current_words, current_speaker
        if not current_words:
            return
        text = " ".join(w.text for w in current_words)
        seg = AlignedSegment(
            start_s=current_words[0].start_s,
            end_s=current_words[-1].end_s,
            speaker=current_speaker or "SPEAKER_00",
            text=text,
            words=list(current_words),
            crosstalk=False,
        )
        segments.append(seg)
        current_words = []
        current_speaker = None

    for w, spk in word_speaker:
        if current_speaker is None:
            current_speaker = spk
            current_words.append(w)
        elif spk != current_speaker:
            _flush()
            current_speaker = spk
            current_words.append(w)
        else:
            # Same speaker — check pause
            if current_words:
                gap = w.start_s - current_words[-1].end_s
                if gap >= min_pause_s:
                    _flush()
                    current_speaker = spk
            current_words.append(w)

    _flush()

    return AlignmentResult(segments=segments)
