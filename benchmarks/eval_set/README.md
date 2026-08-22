# Eval Set — Highlight Extraction Quality Evaluation

## Purpose

This directory stores labeled audio files and human-marked highlight timestamps
for computing precision@10 and recall@N against model output.

## How to use

1. Place podcast audio files (wav/mp3/m4a/flac) in this directory.
2. Create a corresponding `*.labels.json` file with human-marked timestamps:

```json
{
  "episode_id": "example_episode",
  "labeled_by": "editor_name",
  "labeled_at": "2026-08-22",
  "highlights": [
    {"start_s": 1243.5, "end_s": 1298.0, "reason": "emotional moment"},
    {"start_s": 3200.0, "end_s": 3260.0, "reason": "funny exchange"}
  ]
}
```

3. Run the pipeline on the audio and compare model output against labels.
4. Compute precision@10 and recall@N.

## Provenance

| File | Speakers | Duration | Labeled by | Date |
|------|----------|----------|------------|------|
| (add entries here as files are added) | | | | |

## Guidelines

- **Label independently:** Editors should mark timestamps they consider
  highlight-worthy *without* seeing model output first (to avoid anchoring bias).
- **Diverse episodes:** Aim for 15-20 episodes covering different formats:
  interview (2-speaker), panel (4+ speakers), solo monologue, varying audio quality.
- **Consistent format:** Each `.labels.json` file follows the schema above.
- **Record provenance:** Track who labeled each file and when.
