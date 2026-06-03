---
name: sil-wheel
version: 1.0.0-public
description: Search and curate autonomous-driving video clips through a SIL Wheel deployment - by natural language, visual/trajectory similarity, classifiers, annotations, and clustering, all composable. Point it at your own server.
homepage: https://github.com/nv-tlabs/sil-wheel
---

# SIL Wheel Agent (usage skill)

An AI-friendly Python SDK + skill for searching and curating driving-video
clips served by a **SIL Wheel** deployment. This skill tells an agent how to
set itself up and how to drive the API. It ships **no credentials and no
hardcoded server** - you point it at the SIL Wheel instance you have access to.

> **You need a running SIL Wheel server.** This is a client. Set
> `WHEEL_SERVER_URL` to your deployment (self-hosted from this release, or one
> you've been given access to). There is no public default server.

## Setup

```bash
# from the repo: agent/ is self-contained
pip install -r requirements.txt

cp .env.template .env
# then edit .env:
#   WHEEL_SERVER_URL=http://your-sil-wheel-host:8000
#   WHEEL_USERNAME=...
#   WHEEL_PASSWORD=...
```

Verify:

```python
from sil_wheel_agent import WheelClient

client = WheelClient()        # reads WHEEL_SERVER_URL from .env
client.login()                # reads WHEEL_USERNAME / WHEEL_PASSWORD
print(client.whoami())
```

If login fails, run `client.check_connection()` - it tells you whether the
server is unreachable vs. your network is down vs. credentials are wrong.

## Read this before an autonomous loop

The most common failure is concluding "no clips match" when the call was
slightly off (too-narrow `data_source`, AND-of-words on a natural-language
query, exact-UUID via `search` instead of `lookup_clip`). Two tools make these
loud instead of silent:

- `export WHEEL_STRICT=1` - promotes the highest-risk silent-zero warnings to
  `WheelZeroResultError` exceptions (catch and recover).
- `client.diagnose_zero_results(query=..., data_source=..., **kw)` - cheap
  structured probe of *why* a search returned 0; returns copy-pasteable
  next calls. Never raises.

`knowledge/anti-patterns.md` catalogs the rest. Read it before debugging any
"0 results" issue.

## Search modes (all composable in one `search()` call)

| Method | Use when |
|--------|----------|
| `lookup_clip(clip_id)` | get metadata for a known clip ID |
| `caption_search(q, mode='all'\|'any')` | keyword search over captions |
| `caption_search_any([...])` | OR across terms |
| `semantic_search_by_text(t)` | natural-language visual concept |
| `semantic_search_by_clip(id)` | clips visually similar to a clip |
| `visual_search_by_text(t)` | CLIP text-to-video |
| `trajectory_search_by_clip(id)` | kinematically similar driving |
| `trajectory_predicate_search(...)` | speed/curvature/braking expressions |
| `classifier_search(label, threshold)` | trained scenario classifiers |
| `world_model_search(class_name, ...)` | detected objects (class/count/distance) |
| `annotation_search(labels, mode)` | manual/auto labels (AND/OR) |
| `cluster_search(run_id, cluster_id)` | cluster membership |
| `numeric_filter_search(expr)` | model-metric thresholds |

Full parameter docs: `knowledge/search-modes.md`. Honest precision/recall by
mode: `knowledge/search-tool-calibration.md`. Which modes work on which data
source: `knowledge/feature-compat.md`.

## Key workflows

**Idea -> clip IDs** (multi-strategy search for a scenario):

```python
ids = client.find_clips_for_scenario_ids("construction zone in rain",
                                          data_source="MADS-1M")
client.save_clip_ids(ids, "scenario_clips.txt")
```

**Clip -> similar clips** (grow a dataset from seeds):

```python
expanded = client.expand_clip_set(seed_ids, n_similar_per_clip=20, max_total=500)
```

**Combine sets for training**:

```python
rain = client.export_search_clip_ids(search="rain", data_source="MADS-1M")
snow = client.export_search_clip_ids(search="snow", data_source="MADS-1M")
both = WheelClient.merge_clip_id_lists(rain, snow)          # union
client.save_clip_ids(both, "weather_clips.txt")
```

**Browser URLs** (always give viewable links):

```python
client.clip_url(clip_id)                       # view one clip
client.search_url(search="tunnel")             # reproduce a search in the UI
client.format_results_with_urls(results)       # markdown with links
```

## Search discipline

- Spot-check a small `n=5-10` search before bulk export.
- Classifiers are probabilistic, not ground truth - 0.5 is high-recall/lower-precision; use 0.7-0.8 for precision.
- Caption search is literal over generated captions; if it returns 0, try semantic search or classifiers, not just rephrasing.
- The server caps results at 20/page - use `export_search_clip_ids()` for full sets.
- Tell the user which mode(s) you used, with counts, samples, and `clip_url()` links.

## Safety

1. Writes (labels/annotations/classifiers) are allowed by default since you own
   your server. To protect a shared/production deployment, set `WHEEL_READONLY=1`
   and the SDK refuses all write calls.
2. Credentials live in `.env` (gitignored) - never print or log them.
3. Never delete others' data or write outside your own project/dev space.

## Knowledge files

| File | Covers |
|------|--------|
| `knowledge/INDEX.md` | reading order |
| `knowledge/overview.md` | architecture, data, endpoints |
| `knowledge/search-modes.md` | every mode + parameters |
| `knowledge/anti-patterns.md` | silent footguns + fixes (read before debugging 0-results) |
| `knowledge/search-tool-calibration.md` | honest precision/recall by mode |
| `knowledge/feature-compat.md` | feature x data-source matrix |
| `knowledge/timing-reference.md` | measured latencies |
| `knowledge/server-policy.md` | read-only vs write rules |
| `knowledge/support.md` | where to get help |

## CLI

```bash
python sil_wheel_agent/wheel_client.py info
python sil_wheel_agent/wheel_client.py search --caption "rain" -n 10
python sil_wheel_agent/wheel_client.py search --classifier "Snow" --threshold 0.7
python sil_wheel_agent/wheel_client.py export --caption "tunnel" -o clips.txt
python sil_wheel_agent/wheel_client.py scenario "construction zone in rain"
```

## Testing it works (off-server)

```bash
bash tests/run_clean_room.sh
```

Runs the offline unit smoke, then a clean-room end-to-end against a bundled
mock SIL Wheel server - proving the workflows drive the API with no real
server and no network.
