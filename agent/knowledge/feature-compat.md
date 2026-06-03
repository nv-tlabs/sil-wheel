<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Feature × Data Source Compatibility Matrix

**Why this matters**: SIL Wheel features are NOT uniformly available across data sources. The most expensive silent failures of 2026 came from agents composing `classifier_select=X data_source=Y` or `search_speed=X data_source=Y` where the feature `X` is not indexed on source `Y` — these return `0` quickly (caption / classifier) or slowly (trajectory predicate, ~50–150 s) with **no error** and the agent concludes "no clips" when the right answer is "wrong source".

This page is the single source of truth. Cross-check before composing.

## Matrix

Cell legend:
- `✓` indexed and confirmed working
- `?` plausibly indexed but not empirically verified — probe before relying
- `✗` known to silently return 0 (or near-0) — **the SDK warns when you try**
- `~` partial / a known subset only

| Feature                              | MADS | MADS-1M | AV V1 train | AV V1 validation | AV V2 train | AV V2 validation | AV V2.2. train | celsius2_l3_55k | celsius2_sauron_15k_osm | celsius2_l3_wf10k_720p | celsius2_sauron_nohighway_7k | ncore-lidar-model-static-full | Waymo train | Waymo test | OpenDV-YouTube | MultiCountry-800K | NVIQ | OGameData | Physical AI | Golden Dataset v2 | HCM Dataset | HCM Dataset v2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Caption FTS5** (`caption_search`)  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ? | ✓ | ✓ | ✓ | ✓ | ? | ✓ | ✓ | ? | ? | ? |
| **Cosmos sim (clip↔clip)** (`semantic_search_by_clip`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ? | ✓ | ✓ | ✓ | ✓ | ? | ? | ✓ | ? | ? | ? |
| **Cosmos sim (text↔clip)** (`semantic_search_by_text`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ? | ✓ | ✓ | ✓ | ✓ | ? | ? | ✓ | ? | ? | ? |
| **CLIP visual-text** (`visual_search_by_text`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ? | ~ | ~ | ~ | ~ | ? | ? | ✓ | ? | ? | ? |
| **Trajectory predicates** (`search_speed`, `trajectory_pattern`) — **SDK warns ✗** | **✗** | **✗** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ? | ? | ? | ? | ✓ | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | ? | ? | ? |
| **Trajectory shape sim** (`trajectory_search_by_clip`) | ~ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ? | ? | ? | ? | ✓ | ✓ | ? | ✓ | ? | ? | ? | ? | ? | ? |
| **Classifier inference (varies per label)** (`classifier_search`) — **SDK warns when known-missing** | ~ | ~ | ✓ | ✓ | ✓ | ✓ | ✓ | ~ | ~ | ~ | ~ | ? | ~ | ~ | ✗ | ✗ | ? | ✗ | ~ | ? | ? | ? |
| **Annotation labels** (`annotation_search`) | ~ | ~ | ✓ | ✓ | ✓ | ✓ | ✓ | ~ | ~ | ~ | ~ | ? | ✗ | ✗ | ✗ | ✗ | ? | ✗ | ✓ | ? | ? | ? |
| **World-model objects (perception)** (`world_model_search`) | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ? | ✓ | ✓ | ✗ | ~ | ? | ✗ | ✓ | ? | ? | ? |
| **Numeric metric filter** (`numeric_filter_search`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ~ | ~ | ~ | ~ | ? | ~ | ~ | ✗ | ✗ | ? | ✗ | ✓ | ? | ? | ? |
| **Country filter** (`country_search`)            | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ? | ✓ | ✓ | ✓ | ✓ | ? | ? | ✓ | ? | ? | ? |
| **InstantNuRec reconstruction** (`reconstruct`)  | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Comments** (`comment_search`)                  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **VLM Judge validate** (`vlm_judge_validate_search`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Clip duration (matters for time-window queries)

| Source       | clip duration | speed/curvature samples | positions samples |
|--------------|---------------|-------------------------|-------------------|
| MADS-1M      | **10 s**      | 351 (~35 Hz)            | 36 (~3.6 Hz)      |
| MADS         | 10 s          | 351 (~35 Hz)            | 36 (~3.6 Hz)      |
| AV V1 / V2 / V2.2 train+val | **20 s** | 605 (~30 Hz) | 61 (~3.0 Hz) |
| celsius2_*   | (verify)      | (verify)                | (verify)          |
| Waymo        | (verify)      | (verify)                | (verify)          |

This is invisible to most searches but matters for trajectory analysis: a 10 s window at 5 m/s captures ~50 m of path — too short for a closed-loop turn at typical urban speeds. AV V1/V2's 20 s window has been the only source where we've observed true 360° trajectories (e.g. parking-lot donuts).

## Asymmetric classifier coverage — important caveat

Even within the AV-style data sources, **classifier inference is per-label**. Verified examples:

| Classifier                       | MADS-1M | AV V1 train | AV V2 train | All sources |
|----------------------------------|--------:|------------:|------------:|------------:|
| `Change lane to the right`       | **0**   | 1,500,454   | 1,425,104   | 1,800,150   |
| `Change lane to the left`        | 266,963 | 1,834,202   | 3,966,262   | 5,080,074   |

That is, the **same classifier family** can be inferenced on MADS-1M for one direction and not the other. There is no documented reason — this is operational state, not design. **Always probe first**:

```python
coverage = client.get_classifier_coverage("Change lane to the right")
# {"MADS": 0, "MADS-1M": 0, "AV V2 train": 1425104, ...}
```

The SDK now maintains a small in-code `_CLASSIFIER_KNOWN_MISSING` map (in `sil_wheel_agent/wheel_client.py`) for the most common silent zeros and will warn on `client.search(classifier_select=X, data_source=Y)` when the combination is in the known-missing set. To extend that map with new entries, run `get_classifier_coverage(label)` and submit a one-line PR.

## Server caps

These are silent caps that affect the matrix above:

| Endpoint                                    | Cap     | How to detect | Workaround |
|---------------------------------------------|---------|---------------|------------|
| `search(n=N)`                               | 20      | Warning fires for `n>20`. | Use `search_all_pages(max_results=N)`. |
| `caption_search_any` total                  | 100,000 | `total ≈ 99999` or `100000`. | Split filter; can't be larger. |
| `export_search_clip_ids`                    | 1,000,000 | Returns exactly 1,000,000 + **NEW WARNING** when total > 1M. | Split by sub-source / threshold tier; union via `merge_clip_id_lists`. |
| `vlm_judge_validate_search` (URL length)    | ~50 ids/call (when GET) | Pre-fix: returns only 1 result silently. **Now auto-chunked at 30**. | None needed — auto-chunked. |
| `expand_clip_set(max_total=N)`              | 1000 default | Docstring. | Pass explicit `max_total`. |
| `lookup_clips_batch` cliff                  | ~100 clips under load | Per-clip latency >10 s during server load. | Use `search_all_pages(..., search_clipid=...)` or split. |

## How the SDK helps

The SDK in this repo (v1.8.2+) emits `UserWarning` on the most common silent failures:

- `search(search_speed=..., data_source="MADS-1M")` → "trajectory predicates not indexed here" warning
- `search(classifier_select="Change lane to the right", data_source="MADS-1M")` → "classifier known-missing" warning
- `caption_search("...", mode='all')` with 4+ words → "FTS5 AND won't match" warning
- `export_search_clip_ids(...)` returning ≥1M when `total > 1M` → "hit cap" warning
- `vlm_judge_validate_search(...)` chunk returning ≤1 result for input ≥2 → "URL truncation" warning

Set `WHEEL_STRICT=1` to escalate the most footgun-class warnings to `WheelZeroResultError` exceptions instead. See `knowledge/anti-patterns.md`.

## How to extend this matrix

When you discover a new silent failure:

1. Run `client.get_classifier_coverage(label)` (or the equivalent probe for the relevant feature) across data sources.
2. Update the matrix above with the empirical row.
3. If it's a silent-0 case the SDK should warn on, add it to:
   - `_CLASSIFIER_KNOWN_MISSING` in `sil_wheel_agent/wheel_client.py` for classifiers, or
   - `_TRAJECTORY_UNINDEXED_SOURCES` for trajectory predicates.
4. Add a test case under `TestClassifierKnownMissingPreflight` or `TestTrajectoryUnindexedPreflight` in `tests/test_offline.py`.

The goal: **no agent should ever waste 30 minutes on a silent-0 the SDK could have warned about in 100 ms.**