<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Arena

## Overview

The Arena is a platform for blind human evaluation of model outputs on tasks that are difficult to evaluate automatically. Evaluators are shown inputs (e.g. a video clip, a prompt) alongside two anonymized model outputs and asked to pick a winner. The system supports arbitrary combinations of input and output types — images, videos, text, and structured JSON — making it applicable to a wide range of visual generation and understanding tasks.

## S3 Storage Layout

All arena data lives under `s3://processed_data/arenas/`. Each arena is a folder containing a `manifest.json` and an `assets/` directory:

```
arenas/
  {arena_name}/
    manifest.json
    assets/
      {item_id}/
        input_{input_name}.{ext}         # e.g. input_video.mp4, input_prompt.txt
        {model_name}_{output_name}.{ext}  # e.g. modelA_video.mp4, modelB_response.txt
```

- **Item**: A single evaluation instance (one set of inputs). Each item has a unique `item_id`.
- **Inputs**: Shared context shown to the evaluator. Named per `manifest.inputs`.
- **Outputs**: Per-model results. File naming follows `{model}_{output_name}{ext}` where the extension is determined by type (`video` → `.mp4`, `image` → `.jpg`, `text` → `.txt`, `json` → `.json`).
- Text and JSON outputs are small enough to be fetched inline; video and image outputs are streamed via signed asset URLs.

## Manifest Format

Each arena's `manifest.json` defines the task, models, and I/O schema:

```json
{
  "name": "video-gen-arena",
  "display_name": "Video Generation Arena",
  "description": "Compare video generation models on driving scenes",
  "owners": ["alice", "bob"],
  "instructions": "Select the more realistic and temporally consistent video.",
  "models": ["model_a", "model_b", "model_c"],
  "items": ["clip_001", "clip_002", "clip_003"],
  "inputs": [
    {"name": "video", "type": "video", "label": "Input Video"}
  ],
  "outputs": [
    {"name": "video", "type": "video", "label": "Generated Video"},
    {"name": "metrics", "type": "json", "label": "Metrics"}
  ],
  "elo_config": {
    "k_factor": 32,
    "initial_rating": 1500
  }
}
```

| Field | Description |
|---|---|
| `owners` | Usernames who can publish/unpublish and refresh the manifest |
| `inputs` | Array of `{name, type, label}` defining per-item context shown to evaluators |
| `outputs` | Array of `{name, type, label}` defining per-model results to compare |
| `items` | List of item IDs corresponding to subfolders in `assets/` |
| `models` | List of model names; output files are prefixed with these |
| `elo_config` | Optional ELO tuning (k-factor, initial rating) |

Supported types for both inputs and outputs: `video`, `image`, `text`, `json`.

## Local Storage

Arena state is stored in a local SQLite database configured via the `arena_db` key in the YAML config. It contains three tables:

- **`arena_manifests`**: Cached manifests synced from S3, with publish state and last-updated timestamp.
- **`arena_votes`**: All submitted votes with match ID, models, winner, user, and timestamp.
- **`arena_elo`**: Current ELO ratings, match counts, wins, losses, and ties per model per arena.

The database is created automatically on first launch.

## Features

- **Blind evaluation**: Model identities are hidden during voting and revealed only after a vote is submitted.
- **Graded votes**: Five-point scale (A much better, A better, Tie, B better, B much better) plus "Both are bad" and Skip options. Keyboard shortcuts (1-5, 0, S) for fast annotation.
- **ELO ratings**: Live ELO leaderboard updated after each vote. Graded votes use weighted scores (strong win = 1.0/0.0, weak win = 0.75/0.25, tie = 0.5). "Both bad" penalizes both models.
- **Bootstrap confidence intervals**: 95% CIs computed via 1000-iteration bootstrap resampling of the vote history.
- **ELO history chart**: Replay of ratings over time, rendered as a per-model line chart.
- **Balanced sampling**: Match pair selection is weighted by coverage (under-sampled pairs prioritized), ELO proximity (close matches are more informative), and a provisional boost for models with few votes.
- **Match review**: Click any row in the Recent Votes table to review a past match with the original inputs, outputs, and winning vote highlighted.
- **Live updates**: New models and items can be added to a running arena by updating the `manifest.json` and uploading new assets to S3. Owners and admins can then click "Refresh Manifest" in the UI to pull the latest manifest into the server. Existing votes and ELO ratings are preserved; new models start at the initial rating and the balanced sampler automatically prioritizes them.
- **Ownership and publishing**: Arenas start as unpublished drafts visible only to owners and admins. Owners can publish to make them visible to all users.
- **SQLite-cached manifests**: Manifests are synced from S3 periodically (60s cooldown) and cached in SQLite, eliminating S3 round-trips on every request.
- **CSV export**: Full vote history exportable as CSV from the leaderboard view.

## Arena-Only Mode

The server can run in a lightweight mode that only loads the users DB and arena store, skipping all heavy search/annotation stores. Use the arena-only config:

```bash
ARENA_DEBUG=1 python scripts/launch_server.py \
  config/wheel_launch_arena_only_config.yaml
```

The config specifies just `users_db`, `arena_db`, and `server` settings. All other store loading is skipped. The `/arena` and `/login` pages work normally; other pages (search, annotations, predictions) are non-functional in this mode.