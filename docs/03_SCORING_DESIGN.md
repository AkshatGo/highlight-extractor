# Scoring Design — Highlight Worthiness

There is no ground-truth model for "is this a highlight," so every input to
the score must be inspectable and every weight tunable without a redeploy.
This scoring function will be iterated on constantly — it is the core
heuristic of the service.

## Segmentation before scoring

Scoring never runs on arbitrary windows. Candidate segments are derived as
follows:

1. Start with diarized speaker turns (from `alignment.py` output).
2. Split turns further on ASR-detected pauses (silence >700 ms between words
   within a single speaker turn), so long monologues don't become one giant
   candidate.
3. Clamp each candidate to `min_clip_s` (default 12 s) and `max_clip_s`
   (default 90 s). Segments shorter than the minimum are merged with an
   adjacent neighbor. Segments longer than the maximum are split at the
   nearest sentence-boundary pause within the window.
4. Discard segments with zero speech content (non-speech regions were
   already excluded by VAD at ingestion time, but this is a safety filter).

## Feature set

All features are **z-scored per episode** before combination, so scoring
self-calibrates per episode/show/mic setup rather than needing global
calibration.

| Feature | Signal (what it captures and why) | Source |
|---------|-----------------------------------|--------|
| `sentiment_delta` | Max sentiment swing within segment vs rolling 60 s baseline. A big swing = emotional moment (laughter, anger, surprise). | HuggingFace `transformers` sentiment pipeline |
| `sentiment_extremity` | \|sentiment\| at peak. Both very positive AND very negative count — argument, laughter, and emotional agreement all matter. | Same as above |
| `energy_zscore` | RMS energy vs **that speaker's own** rolling mean/std (not global). Prevents loud speakers from dominating the ranking. | Librosa `rms` |
| `pitch_variance` | F0 variance — proxy for excitement/emphasis. Higher variance = more animated speech. | Librosa `pyin` |
| `speech_rate_delta` | Words/sec vs that speaker's episode-average rate. Faster/slower = departure from baseline = notable. | Transcript timestamps |
| `keyword_density` | Hits per segment length against a configurable keyword/phrase list (e.g. "but", "wow", "amazing", "wait", names). | Regex matches on transcript text |
| `crosstalk_flag` | Boolean: overlap detected in this segment. Genuine reaction signal (laughter, interruption) **and** ASR-confidence risk. | Diarization overlap regions |
| `asr_confidence` | Mean word-level confidence from Whisper for words in this segment. Gate, not just a feature — below threshold the segment is heavily downweighted. | Whisper output |

## Composite score

```
score = w1 * sentiment_delta
      + w2 * sentiment_extremity
      + w3 * energy_zscore
      + w4 * pitch_variance
      + w5 * speech_rate_delta
      + w6 * keyword_density
      + bonus(crosstalk_flag)
      - penalty(low_asr_confidence)
```

Where:

- `bonus(crosstalk_flag)` = `w7` if crosstalk is detected, else 0.
- `penalty(low_asr_confidence)` = `w8 * (threshold - mean_asr_confidence)`
  when `mean_asr_confidence < threshold`, else 0.

Weights live in `config/scoring_weights.yaml`, not code:

```yaml
# config/scoring_weights.yaml
sentiment_delta: 0.25
sentiment_extremity: 0.20
energy_zscore: 0.15
pitch_variance: 0.10
speech_rate_delta: 0.10
keyword_density: 0.10
crosstalk_bonus: 0.10
asr_confidence_penalty: 0.15
asr_confidence_threshold: 0.5
```

`asr_confidence` acts as a **gate**: below 0.5 the segment is heavily
downweighted **and** flagged `low_confidence: true` in output. This lets
editors distinguish between a low score from a genuinely dull moment vs a
low score from bad ASR.

## Selecting the final top-N

Raw top-N by score tends to cluster around the same moment (three
overlapping candidates around one laugh). We apply **temporal non-max
suppression**:

1. Sort all candidates by score descending.
2. Greedily accept the highest-scored candidate.
3. Skip any candidate whose temporal overlap with an already-accepted
   candidate exceeds a configurable threshold (default 20%).
4. Continue until N candidates are accepted (default N=15).

This guarantees a spread of distinct moments across the episode.

## Output contract per highlight

```json
{
  "start_s": 1243.5,
  "end_s": 1298.0,
  "speaker": "SPEAKER_01",
  "score": 3.27,
  "reasons": [
    "strong sentiment swing",
    "peak pitch variance",
    "crosstalk"
  ],
  "transcript_excerpt": "Wait, wait — are you seriously telling me they just... walked out?",
  "low_confidence": false
}
```

`reasons` = the top contributing score components whose z-scored value
exceeds a configurable threshold (default >1.0). This makes ranking legible
to a human editor instead of a black-box number.

## Evaluation loop

Maintain a small labeled eval set: 15–20 episodes with human-marked
"actually a highlight" timestamps. Whenever scoring weights change:

1. Re-score the eval set episodes (reuses cached features — fast).
2. Compute precision@10 and recall@N against the human marks.
3. Review regressions before shipping the weight change.

This is a **tunable ranking heuristic with human-in-the-loop eval for v1**,
not a trained classifier. A learned model (logistic regression over these
same features, later something more sophisticated) is a natural v2 once
enough labeled data accumulates from editor accept/reject feedback. See
[docs/04_ROADMAP.md](04_ROADMAP.md) Phase 5 for the tracking item.
