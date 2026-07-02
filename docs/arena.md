<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Arena

## Overview

The Arena is a platform for blind human evaluation of model outputs on tasks that are difficult to evaluate automatically. Evaluators are shown inputs (e.g. a video clip, a prompt) alongside two anonymized model outputs and asked to pick a winner. The system supports arbitrary combinations of input and output types (images, videos, text, and structured JSON), making it applicable to a wide range of visual generation and understanding tasks.

## S3 Storage Layout

Production arenas live under `s3://processed_data/arenas_prod/`; a parallel `arenas_dev/` prefix is used for staging. The server scans `arenas_prod/` by default and `arenas_dev/` when the `ARENA_DEV` environment variable is set.

Arenas can be organized into nested folders (e.g. by project). Any folder containing a `manifest.json` is treated as an arena; folders without one are recursed into. The arena's name in the URL and DB is the full relative path:

```
arenas_prod/
  {project}/{sub_project}/{arena_name}/    # nested: name = "{project}/{sub_project}/{arena_name}"
    manifest.json
    assets/
      {item_id}/
        meta.json                            # optional: per-input time ranges for video clipping
        input_{input_name}.{ext}             # e.g. input_video.mp4, input_prompt.txt
        {model_name}_{output_name}.{ext}     # e.g. modelA_video.mp4, modelB_response.txt
  {arena_name}/                              # flat arenas (no parent folder) are still supported
    manifest.json
    assets/
      ...
```

- **Item**: A single evaluation instance (one set of inputs). Each item has a unique `item_id`.
- **Inputs**: Shared context shown to the evaluator. Named per `manifest.inputs`.
- **Outputs**: Per-model results. File naming follows `{model}_{output_name}{ext}` where the extension is determined by type (`video` → `.mp4`, `image` → `.jpg`, `text` → `.txt`, `json` → `.json`).
- Text and JSON outputs are small enough to be fetched inline; video and image outputs are streamed via signed asset URLs.
- **`meta.json`** (optional): `{ "input_<name>": { "start_time": <sec>, "end_time": <sec> } }` to play a sub-range of a video input. Useful when the same source clip serves multiple items.

## Manifest Format

Each arena's `manifest.json` defines the task, models, and I/O schema:

```json
{
  "name": "video-gen-arena",
  "display_name": "Video Generation Arena",
  "description": "Compare video generation models on driving scenes",
  "owners": ["alice", "bob"],
  "instructions": "Select the more realistic and temporally consistent video.",
  "criteria": [
    {"name": "realism",     "description": "Lighting, textures, and motion look natural", "mode": "preference"},
    {"name": "consistency", "description": "No flicker, pop-in, or sudden geometry changes", "mode": "preference"},
    {"name": "traffic_lights", "description": "Are traffic lights handled correctly?", "mode": "passfail"}
  ],
  "models": ["model_a", "model_b", "model_c"],
  "items": ["clip_001", "clip_002", "clip_003"],
  "item_labels": {
    "clip_001": "highway,night",
    "clip_002": "urban,rain"
  },
  "inputs": [
    {"name": "video", "type": "video", "label": "Input Video"}
  ],
  "outputs": [
    {"name": "video", "type": "video", "label": "Generated Video"},
    {"name": "metrics", "type": "json", "label": "Metrics"}
  ],
  "sync_all_videos": false,
  "rating_config": {
    "initial_rating": 1500,
    "initial_rd": 350,
    "initial_volatility": 0.02,
    "tau": 0.5
  }
}
```

| Field | Description |
|---|---|
| `owners` | Usernames who can publish/unpublish and refresh the manifest |
| `instructions` | Shown to evaluators. Used as the prompt when `criteria` is omitted (single-criterion mode). |
| `criteria` | Optional list of `{name, description, mode}`. Each criterion produces an independent A/B vote and its own rating leaderboard. `mode` is `"preference"` (A++/A+/Tie/B+/B++/Both Bad) or `"passfail"` (Only A/Both Good/Only B/Both Bad). Omit to fall back to a single `"overall"` criterion. |
| `inputs` | Array of `{name, type, label}` defining per-item context shown to evaluators |
| `outputs` | Array of `{name, type, label}` defining per-model results to compare |
| `items` | List of item IDs corresponding to subfolders in `assets/` |
| `item_labels` | Optional `{item_id: "csv,labels"}`. Enables label-based filtering and per-label rating slices in the analytics tab. |
| `models` | List of model names; output files are prefixed with these |
| `sync_all_videos` | If `true`, all videos on the comparison page play in lockstep (input + output A + output B). Default `false` syncs only within each group. Useful when output videos continue from an input clip (e.g. I2V conditioning). |
| `rating_config` | Optional Glicko-2 tuning: `initial_rating` (default 1500), `initial_rd` (default 350), `initial_volatility` (default 0.02 — kept low because arena models are static artifacts), `tau` (default 0.5). Defaults live in `arena_store.py` as `DEFAULT_*` constants. See below. |

Supported types for both inputs and outputs: `video`, `image`, `text`, `json`.

## Local Storage

Arena state is stored in a local SQLite database configured via the `arena_db` key in the YAML config. It contains three tables:

- **`arena_manifests`**: Cached manifests synced from S3, with publish state and last-updated timestamp.
- **`arena_votes`**: All submitted votes with match ID, models, winner, user, criterion, and timestamp. Multi-criteria votes share a match ID; uniqueness is `(match_id, criterion)`.
- **`arena_elo`**: Current Glicko-2 state per model per arena per criterion: `rating`, `rating_deviation`, `volatility`, `matches`, `wins`, `losses`, `ties`.

The database is created automatically on first launch. Manifests are mirrored from the DB into an in-memory dict at startup so access checks are zero-cost on the request path; a background daemon refreshes from S3 every 5 minutes.

## Features

- **Blind evaluation**: Model identities are hidden during voting and revealed only after a vote is submitted.
- **Multi-criteria evaluation**: Each arena can define multiple criteria, evaluated independently. Each criterion gets its own rating leaderboard; an aggregate (probability-space average across criteria) is also shown.
- **Two voting modes per criterion**:
  - `preference` (default): 5-point scale (A++ / A+ / Tie / B+ / B++) plus Both Bad and Skip. Keyboard shortcuts 1-5, 0, S.
  - `passfail`: Binary correctness (Only A / Both Good / Only B / Both Bad). Same underlying vote values, different button labels, appropriate for objective criteria like "are traffic lights correct?".
- **Glicko-2 ratings**: Each (model, criterion) tracks three numbers — rating, rating deviation (RD, uncertainty), and volatility (rate of drift). Ratings are displayed on the ELO-compatible scale (around 1500). Every vote is applied as its own rating period, so the chart shows a smooth per-vote trajectory. Because arena models are static artifacts (they don't drift like a chess player's skill), `initial_volatility` defaults to 0.02 rather than Glickman's classic 0.06 — this lets RDs shrink further with heavy voting. Graded votes use weighted scores (strong win = 1.0/0.0, weak win = 0.75/0.25, tie = 0.5). "Both bad" penalizes both models. New models start with high RD and converge quickly; settled models have low RD and move slowly.
- **Closed-form confidence intervals**: 95% CIs are `rating ± 1.96 · RD`, computed in constant time. No bootstrap.
- **Rating history chart**: Replay of ratings over time, rendered as a per-model line chart.
- **Folder tree sidebar**: Arenas are grouped by their nested S3 path. Top-level folders are styled as project tiles with an arena count badge; nested folders are collapsible. Active arena's containing folders auto-expand.
- **Item labels**: Per-data-point CSV labels in the manifest enable label-based filtering: slice the leaderboard and analytics by `highway`, `night`, etc.
- **Analytics tab** (per arena):
  - **Annotators table**: per-annotator alignment (Kendall's τ vs. consensus), influence (avg rating shift per vote when removed), and rank shifts.
  - **Data table**: per-item alignment, consensus Δ (does removing this item improve or worsen annotator agreement?), influence, and rank shifts. Sorts to surface noisy / influential items.
  - **Pairwise annotator agreement matrix**: Kendall's τ between every pair of annotators' solo rating rankings, with no item overlap required.
  - **Filtered leaderboard**: live rating replay with delta-rating and delta-rank columns showing the effect of excluding selected annotators, items, criteria, or labels.
- **Information-gain sampling**: Match pair selection weights each candidate by expected variance reduction under one Glicko-2 update, summed across criteria. Coverage, rating-proximity, and provisional-boost heuristics collapse into this single closed-form quantity: settled pairs have low RD (low gain), competitive pairs have E≈0.5 (high gain), new models have high RD (high gain). No hyperparameters.
- **Match review**: Click any row in the Recent Votes table to review a past match with the original inputs, outputs, and winning vote highlighted. Image inputs/outputs can be clicked to open a full-screen lightbox.
- **Live updates**: New models and items can be added to a running arena by updating the `manifest.json` and uploading new assets to S3. Owners and admins can then click "Refresh Manifest" in the UI to pull the latest manifest into the server. Existing votes and ratings are preserved; new models start with high RD and the sampler naturally prioritizes them.
- **Ownership and publishing**: Arenas start as unpublished drafts visible only to owners and admins. Owners can publish to make them visible to all users.
- **In-memory manifest mirror**: Manifests are loaded into memory at startup and refreshed by a 5-minute background sync. Access checks on the request path never hit SQLite or S3.
- **VLM judge**: Owners can run an automated batch of VLM-judged votes against any arena (one vote per criterion per match). Requires `NV_INFERENCE_API_KEY` in the server environment.
- **JSON vote export**: Full vote history exportable as JSON from the leaderboard view.
