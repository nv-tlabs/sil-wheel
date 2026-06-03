<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# API Timing Reference

Measured latencies for SIL Wheel operations on **production** (`your SIL Wheel server:8000`).
Use this to plan search strategies efficiently — avoid slow operations when a fast alternative exists.

## Search Operations

| Operation | Typical Time | Notes |
|-----------|:------------|-------|
| **Caption search** (FTS5) | 1–5 s | Fastest text search. Always try first. |
| **Classifier filter** | 1–3 s | Very fast. Pre-scored for all 18M clips. |
| **Annotation label filter** | 1–3 s | Fast. Binary include/exclude. |
| **World model object search** | 2–5 s | Fast. Pre-indexed counts. |
| **Country filter** | 1–2 s | Instant. Pre-indexed. |
| **Cosmos clip-to-clip** | 2–5 s | FAISS lookup. Fast. |
| **Trajectory shape similarity** | 3–10 s | FAISS on trajectory vectors. Medium. |
| **Trajectory predicate** | 5–15 s | Scans trajectory memmap. Medium. |
| **Cosmos text-to-video** | 3–5 s prod / **120 s+ dev** | Requires model inference. **Avoid on dev.** |
| **CLIP text-to-video** | 3–5 s prod / **120 s+ dev** | Same — requires model inference. **Avoid on dev.** |
| **Query rewrite** | 1–3 s | LLM call. Fast. |
| **Cluster search** | 1–3 s | Pre-computed membership. Fast. |
| **Numeric metric filter** | 1–3 s | Pre-indexed. Fast. |
| **SIL API filter** | 1–2 s | Pre-indexed. Fast. |
| **Comment search** | 1–3 s | Text search. Fast. |

## Compound Operations

| Operation | Typical Time | Notes |
|-----------|:------------|-------|
| `find_clips_for_scenario()` | 10–140 s | Caption + semantic + classifier. Bottleneck: semantic text-to-video. |
| `find_clips_for_scenario_ids()` | Same | Wrapper around above. |
| `find_similar_to_clip()` | 5–15 s | Cosmos clip-to-clip + trajectory. Both fast. |
| `expand_clip_set(N seeds)` | N × 10–15 s | Parallel across seeds (ThreadPoolExecutor, up to max_workers). Each seed runs cosmos + trajectory. |
| `search_all_pages(P pages)` | P × 2–5 s | Depends on search type. |
| `export_search_clip_ids()` | 3–10 s | Runs search + downloads CSV. |
| `score_clips()` | N × 3–5 s | N = number of search dimensions. Parallel searches (up to 6 threads). |
| `score_clips_large()` | N × 5–15 s | N dimensions, each exports full CSV. |
| `multi_search_export()` | N × 3–10 s | N searches + set merge. |
| `lookup_clips_batch()` | 1–5 s | Parallel per-clip lookups via thread pool. |

## Info / Discovery

| Operation | Typical Time | Notes |
|-----------|:------------|-------|
| `whoami()` | <1 s | |
| `get_classifiers()` | 1–3 s | |
| `get_data_stats()` | 1–3 s | |
| `get_annotations_summary()` | 2–5 s | Downloads CSV. |
| `get_leaderboard()` | 2–5 s | |
| `get_clustering_status()` | <1 s | |
| `export_classifier_weights()` | 1–3 s | |
| `scenario_inventory()` | 5–10 s | Makes 3 API calls. |

## Server Caps & Silent-Failure Behaviour

The server enforces hard caps on every search operation. Several **silently
clip** the response without raising — agents that don't read this table will
get less data than they asked for and not know it. Cross-reference
[`anti-patterns.md`](anti-patterns.md) for the user-facing warning patterns.

| Operation | Cap | Behaviour above the cap | Workaround |
|-----------|-----|------------------------|------------|
| `search(n=...)` | **default 6, max 20** results/page (server `NUM_VIDEOS_PER_PAGE=6`; `min(max(n,1),20)` cap) | `n` clamped to 20; `UserWarning` fires. | `search_all_pages(...)` for ≤500 results, `export_search_clip_ids(...)` for more. |
| `search(page=...)` | unbounded | Server re-runs full search per page (O(n) cost). | Use CSV export instead of paginating. |
| `caption_search(query, mode='all')` with 4+ words | n/a | Pre-flight `UserWarning` fires before the request (request still goes out). FTS5 AND-of-words rarely matches 4+ word natural-language queries. | Pass `mode='any'` or split into per-concept queries + `intersect_clip_id_lists`. |
| `caption_search(...)` with 3 words returning 0 | n/a | Post-call `UserWarning` recommends `caption_search_any` (still costs 1 round trip). | Same — switch to `mode='any'` for natural language. |
| `search(search_clipid='<exact-uuid>')` (UUID-shaped) | n/a | Post-call `UserWarning` recommends `lookup_clip()` (clearer intent — both routes use the same exact-key intersection). | Use `lookup_clip(clip_id) -> SearchResult \| None` directly. |
| `find_clips_for_scenario_ids(...)` | default `data_source="MADS"` (not "MADS-1M"!) | Returns clips only from MADS — agents looking for MADS-1M get 0 join hits. | Pass `data_source="MADS-1M"` explicitly, or `data_source=None` for all sources. |
| `export_search_clip_ids(...)` CSV path | **500,000** clips (server `launch_server.py:1192-1196`) | Silently returns empty list above this. | List clip IDs from S3 (`s3://processed_data/{data_source}/`), then use API for filtered subsets + set operations. |
| Composed search returning 0 (≥2 active filters incl. `search_clipid`) | n/a | Post-call `UserWarning` recommends running each filter separately + `intersect_clip_id_lists`. | Run per-filter `export_search_clip_ids` + `intersect_clip_id_lists`. |

**Why this matters**: every "silent" clip is a place where the agent's **stated
intent** ("give me 200 clips") and the **delivered result** (20 clips) diverge
without raising. The warnings exist so the agent can detect the divergence;
the table here documents the underlying server contract so the agent can
plan around it.

### Practical thresholds for autonomous workflows

- **Up to 20 clips**: `search(n=20)` — one round trip.
- **20 to 500 clips**: `search_all_pages(n=20, max_pages=25)` — pagination, server re-runs each page.
- **500 to ~50,000 clips**: `export_search_clip_ids(...)` — CSV path, single round trip.
- **50,000 to ~500,000 clips**: same `export_search_clip_ids(...)` — still CSV, still fast.
- **Above 500,000**: list IDs from S3 + API set operations (see workaround above).

**Example — MADS-1M without left-hand driving:**
```python
# S3 for the full set (bypasses 500k cap)
# boto3 paginator on s3://processed_data/mads_1M/ → 1,071,385 IDs

# CSV for the small filtered set (under 500k, instant)
lhd = client.export_search_clip_ids(data_source='MADS-1M', search_country='GB')
lhd += client.export_search_clip_ids(data_source='MADS-1M', search_country='JP')

# Subtract
rhs = WheelClient.subtract_clip_id_lists(all_ids, lhd)
```

## Strategy Guidelines

1. **For "find clips about X"**: Start with caption search (1–5s). If the user wants visual matches too, add semantic text-to-video only on production. On dev, skip it.
2. **For "find similar to clip Y"**: Use `find_similar_to_clip()` (5–15s). This avoids the slow text-to-video path entirely.
3. **For "expand from seeds"**: Budget ~12s per seed clip. For 10 seeds, expect ~2 min.
4. **For scenarios on dev server**: Skip the semantic search step. Use caption + classifier only (~5s).
5. **For bulk exports ≤500k**: Use `export_search_clip_ids()` — single CSV call, instant.
6. **For bulk exports >500k**: Use S3 listing + set operations (see § "Server Caps & Silent-Failure Behaviour" above).
7. **Avoid `search_all_pages()` with semantic search**: Each page re-triggers inference. Use export instead.
8. **Avoid pagination for large sets**: Each page re-runs the full search. Use CSV or S3 instead.

## VLM Judge Timings (v1.5.0)

VLM Judge calls the NVIDIA Inference API (Gemini 3 Flash) — same latency on prod and dev.

| Operation | Latency | Notes |
|-----------|---------|-------|
| `vlm_judge_status()` | ~1s | No auth needed, just a status check |
| `vlm_judge_caption_score()` | ~5-15s | Samples video frames + VLM call |
| `vlm_judge_score_clip()` | ~7-18s | lookup_clip + caption_score |
| `vlm_judge_validate_search()` (5 clips) | ~10-15s | Parallel, bounded by slowest |
| `vlm_judge_validate_search()` (50 clips) | ~30-60s | 20 parallel workers |
| `vlm_judge_search_and_validate()` | ~15-30s | search + validate combined |

## Dev Server Specifics

The dev server (`localhost:8018`) runs on CPU only. Everything involving model inference is ~20–30x slower:
- Caption search: ~5–15s (still usable)
- Classifier filter: ~1–5s (still fast)
- Cosmos text-to-video: **120s+** (avoid — use caption search instead)
- CLIP text-to-video: **120s+** (avoid)
- Cosmos clip-to-clip: ~1–10s (FAISS, still reasonable)
- VLM Judge: Same as production (calls external API, not local inference)