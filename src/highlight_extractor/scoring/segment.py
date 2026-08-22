"""Candidate segment derivation for scoring.

Segmentation before scoring:
1. Start with diarized speaker turns (from alignment output).
2. Split turns on ASR-detected pauses (silence >700ms between words).
3. Clamp to min_clip_s / max_clip_s.
4. Merge short segments with adjacent neighbors.
5. Split long segments at nearest sentence-boundary pause.
6. Discard zero-speech segments.
"""

from highlight_extractor.alignment import AlignedSegment


def derive_candidate_segments(
    aligned_segments: list[AlignedSegment],
    min_clip_s: float = 12.0,
    max_clip_s: float = 90.0,
    min_pause_s: float = 0.7,
) -> list[AlignedSegment]:
    """Derive candidate segments from aligned segments for scoring.

    Steps:
        1. Split long aligned segments on internal pauses.
        2. Clamp to [min_clip_s, max_clip_s].
        3. Merge segments shorter than min_clip_s with adjacent neighbors.
        4. Split segments longer than max_clip_s at sentence-boundary pauses.
        5. Discard segments with no words.

    Args:
        aligned_segments: Output from alignment.py.
        min_clip_s: Minimum candidate clip length in seconds.
        max_clip_s: Maximum candidate clip length in seconds.
        min_pause_s: Minimum internal pause to trigger a split (seconds).

    Returns:
        List of AlignedSegment candidates suitable for feature extraction.
    """
    if not aligned_segments:
        return []

    # Step 1: Split segments on internal pauses
    split_segments: list[AlignedSegment] = []
    for seg in aligned_segments:
        if not seg.words:
            continue
        split_segments.extend(_split_on_pauses(seg, min_pause_s))

    # Step 2-3: Clamp and merge
    candidates = _clamp_and_merge(split_segments, min_clip_s, max_clip_s)

    # Step 4: Split any remaining over-length segments
    final: list[AlignedSegment] = []
    for seg in candidates:
        dur = seg.end_s - seg.start_s
        if dur > max_clip_s:
            final.extend(_split_long_segment(seg, max_clip_s))
        elif dur >= min_clip_s:
            final.append(seg)
        # Segments shorter than min_clip_s after merge are discarded

    # Step 5: Discard zero-speech segments
    return [seg for seg in final if seg.words]


def _split_on_pauses(segment: AlignedSegment, min_pause_s: float) -> list[AlignedSegment]:
    """Split a segment on internal pauses >= min_pause_s."""
    if not segment.words:
        return []

    result: list[AlignedSegment] = []
    current_words = [segment.words[0]]

    for i in range(1, len(segment.words)):
        gap = segment.words[i].start_s - segment.words[i - 1].end_s
        if gap >= min_pause_s:
            # Flush current words
            result.append(_make_segment(segment, current_words))
            current_words = [segment.words[i]]
        else:
            current_words.append(segment.words[i])

    if current_words:
        result.append(_make_segment(segment, current_words))

    return result


def _make_segment(original: AlignedSegment, words: list) -> AlignedSegment:
    """Create a new AlignedSegment from a subset of words."""
    text = " ".join(w.text for w in words)
    return AlignedSegment(
        start_s=words[0].start_s,
        end_s=words[-1].end_s,
        speaker=original.speaker,
        text=text,
        words=list(words),
        crosstalk=original.crosstalk,
    )


def _clamp_and_merge(
    segments: list[AlignedSegment],
    min_clip_s: float,
    max_clip_s: float,
) -> list[AlignedSegment]:
    """Clamp segments to [min_clip_s, max_clip_s] and merge short ones."""
    if not segments:
        return []

    # First, clamp long segments
    clamped: list[AlignedSegment] = []
    for seg in segments:
        dur = seg.end_s - seg.start_s
        if dur > max_clip_s:
            # Truncate to max_clip_s (split will happen later)
            clamped.append(
                AlignedSegment(
                    start_s=seg.start_s,
                    end_s=seg.start_s + max_clip_s,
                    speaker=seg.speaker,
                    text=seg.text[: int(len(seg.text) * max_clip_s / dur)],
                    words=seg.words[: int(len(seg.words) * max_clip_s / dur)] or seg.words[:1],
                    crosstalk=seg.crosstalk,
                )
            )
        else:
            clamped.append(seg)

    # Then merge short segments with neighbors
    merged: list[AlignedSegment] = [clamped[0]]
    for seg in clamped[1:]:
        prev = merged[-1]
        if (prev.end_s - prev.start_s) < min_clip_s and prev.speaker == seg.speaker:
            # Merge
            merged[-1] = AlignedSegment(
                start_s=prev.start_s,
                end_s=seg.end_s,
                speaker=prev.speaker,
                text=prev.text + " " + seg.text,
                words=prev.words + seg.words,
                crosstalk=prev.crosstalk or seg.crosstalk,
            )
        else:
            merged.append(seg)

    return merged


def _split_long_segment(segment: AlignedSegment, max_clip_s: float) -> list[AlignedSegment]:
    """Split a segment longer than max_clip_s at the best internal pause.

    Only splits if a suitable internal pause (>0.3s) is found near the midpoint.
    If no suitable pause exists, returns the original segment unsplit.
    """
    if not segment.words:
        return []

    target_dur = max_clip_s
    best_split_idx = None
    best_gap = float("inf")

    # Find the nearest suitable internal pause to the midpoint
    mid_time = segment.start_s + target_dur / 2
    for i in range(1, len(segment.words)):
        gap = segment.words[i].start_s - segment.words[i - 1].end_s
        word_time = (segment.words[i].start_s + segment.words[i].end_s) / 2
        if abs(word_time - mid_time) < best_gap and gap > 0.3:
            best_gap = abs(word_time - mid_time)
            best_split_idx = i

    if best_split_idx is None:
        # No suitable split point found — return as-is
        return [segment]

    words_a = segment.words[:best_split_idx]
    words_b = segment.words[best_split_idx:]

    result: list[AlignedSegment] = []
    if words_a:
        result.append(_make_segment(segment, words_a))
    if words_b:
        result.append(_make_segment(segment, words_b))

    return result
