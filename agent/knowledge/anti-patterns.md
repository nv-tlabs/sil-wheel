<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Anti-Patterns — Recurring SDK Footguns

Distilled from real agent sessions on Agent Hub (community
[`sil_wheel`]()). Each entry follows the same
shape so you can scan quickly:

- **Trigger** — the natural call an agent reaches for first.
- **What you'll see** — the misleading observable.
- **Why it's wrong** — the real story.
- **Right call** — what to do instead.

The SDK emits `UserWarning`s for most of these so they fail loudly rather
than silently. Don't suppress them.

> Headline source: Agent Hub post `161a7f6c` (Apr 2026, flywheel-pipeline
> Cursor agent), reproduced and extended from the recurring themes across
> 30+ posts in the `sil_wheel` community.

---

## 1. Prefer `lookup_clip()` over `search(search_clipid=<exact-UUID>)`

### Trigger

```python
total, results = client.search(search_clipid="7552f3a8-fe90-..._15239..._15259...")
# total == 0
```

### What you'll see

`(0, [])` and a `UserWarning` (since the SDK now detects exact clip-key
shapes and emits a guidance warning).

### Why it's wrong

Both `search(search_clipid=...)` and `lookup_clip(clip_id)` hit the same
`/videos` endpoint and use the same exact-key intersection
(`sqlite_data_store.py:679-682` — `project_dict(current_results, [search_clipid])`).
**They give the same answer for the same clip.** The two reasons to prefer
`lookup_clip()`:

1. **Clearer intent**: `lookup_clip` says "give me this one clip"; `search`
   with a clip filter looks like a query that might return many results.
2. **Cleaner return type**: `SearchResult | None` instead of
   `(int, list[SearchResult])`.

When `search(search_clipid=...)` returns 0 hits, the most common cause is
that the clip is **not in the current filter universe** — e.g. you passed
`data_source="MADS"` but the clip lives in `MADS-1M`. Switching to
`lookup_clip` will give the same answer; what fixes the result is widening
the filter (e.g. dropping `data_source` to search across all sources).

### Right call

```python
clip = client.lookup_clip("7552f3a8-fe90-..._15239..._15259...")
# clip is SearchResult | None
```

If `clip is None`, check your `data_source` filter — the clip may live in
a different source than the one you queried. To search across all sources:

```python
total, results = client.search(search_clipid=clip_id, data_source=None)
# or just:
clip = client.lookup_clip(clip_id)  # already passes no data_source filter
```

If you really need a `(total, results)` tuple, wrap:

```python
clip = client.lookup_clip(clip_id)
total, results = (1, [clip]) if clip else (0, [])
```

---

## 2. Don't rely on the default `data_source="MADS"`

### Trigger

```python
clip_ids = client.find_clips_for_scenario_ids("FedEx truck")
```

### What you'll see

A list of UUIDs from `MADS` (NOT `MADS-1M`). When joining against a local
`MADS-1M` index they all miss — looks like a UUID-space mismatch.

### Why it's wrong

The default `data_source` for the `find_clips_for_*` family is `"MADS"`
(the smaller, human-curated set). `"MADS-1M"` (the 1M-clip set) and every
other Wheel data source require **explicit** opt-in.

### Right call

```python
clip_ids = client.find_clips_for_scenario_ids(
    "FedEx truck", data_source="MADS-1M",
)
# or, for clips from ALL sources (Waymo, AV V1+V2, OpenDV, Physical AI, ...):
clip_ids = client.find_clips_for_scenario_ids("FedEx truck", data_source=None)
```

Always inspect `client.get_data_sources()` before assuming a default.

---

## 3. Don't use `caption_search(...)` for natural-language queries

### Trigger

```python
total, results = client.caption_search("busy street with billboards")
# total == 0
```

### What you'll see

For 4+ word queries: a `UserWarning` fires **before** the request goes out
(then the request runs anyway, and most likely returns 0).
For 3-word queries: the request runs first; if it returns 0, the
`UserWarning` fires post-call. For 1-2 word queries no warning fires —
the SDK assumes you know what you're doing.

### Why it's wrong

The default mode is **FTS5 AND-of-words** — every word must appear in a
single caption for it to match. Natural-language phrases of 4+ words rarely
satisfy this. The server's caption store (`sqlite_caption_store.py`)
normalizes hyphens to spaces and otherwise passes the query straight into
SQLite FTS5 `MATCH`, where unquoted multi-token strings are AND-of-tokens.

### Right call

For natural language, use **OR semantics**:

```python
total, results = client.caption_search(
    "busy street with billboards", mode="any",
)
# Equivalent to: caption_search_any(["busy", "street", "with", "billboards"])
```

For true **AND across concepts** (e.g. "stroller appearing in the same
clip as a car"), do per-concept exports + intersect:

```python
stroller = client.export_search_clip_ids(search="stroller", data_source="MADS-1M")
car      = client.export_search_clip_ids(search="car",      data_source="MADS-1M")
both     = WheelClient.intersect_clip_id_lists(stroller, car)
```

---

## 4. Don't assume `n=200` returns 200 results

### Trigger

```python
total, results = client.caption_search_any(["rain"], n=200, data_source="MADS-1M")
# UserWarning: n=200 exceeds server page size of 20; clipped.
# len(results) == 20
```

### What you'll see

`n=` larger than 20 is silently clipped. The warning fires, but if you don't
read warnings you'll just notice your bucket is small.

### Why it's wrong

The server enforces a hard 20-results-per-page cap on `/videos`. Larger
result sets require pagination or CSV export.

### Right call

```python
# 20 to ~500 clips: paginate
results = client.search_all_pages(search="rain", data_source="MADS-1M", max_pages=10)

# 500+ clips: CSV export (single round trip, scales to ~500k)
clip_ids = client.export_search_clip_ids(search="rain", data_source="MADS-1M")
```

See [`timing-reference.md`](timing-reference.md) §"Server Caps & Silent-Failure Behaviour"
for the full table of caps and the right tool per scale.

---

## 5. Don't guess the classifier accessor name

### Trigger

```python
client.list_classifiers()
# AttributeError on older SDK builds
```

### What you'll see

Older builds: `AttributeError: 'WheelClient' object has no attribute
'list_classifiers'`. Agents searching for `list_*` (matching the
`list_data_sources()` / `list_communities()` convention used elsewhere)
don't find it.

### Why it's wrong

Historically, the classifier accessors were named inconsistently:

- `get_classifiers()` → returns the full dict (`trained`, `untrained`, counts).
- `list_classifier_names(include_untrained=False)` → returns a flat `list[str]`.

### Right call (current SDK)

`list_classifiers()` is now an alias for `list_classifier_names()` — it
follows the `list_*` convention and returns a `list[str]`:

```python
labels: list[str] = client.list_classifiers()
labels_with_pending = client.list_classifiers(include_untrained=True)

# Full dict (with annotation counts) still available:
data = client.get_classifiers()
```

---

## 6. Don't trust composed-search empty results

### Trigger

```python
total, results = client.search(
    search="rain",
    classifier_select="Construction zone",
    semantic_search_text="urban intersection",
)
# total == 0; UserWarning fires
```

### What you'll see

Composed searches with 2+ active filters often return 0 because the server
intersects all filters server-side. The SDK warns post-call.

### Why it's wrong

Server-side intersection of independent search modes is much stricter
than agents expect — almost every clip fails at least one filter. Worse,
debugging is hard because you can't see which filter excluded the clip.

### Right call

Run each filter independently + intersect client-side:

```python
caption    = client.export_search_clip_ids(search="rain")
classifier = client.export_search_clip_ids(classifier_select="Construction zone", probability_threshold=0.5)
semantic   = [r.clip_id for r in client.search(semantic_search_text="urban intersection", n=20)[1]]

intersected = WheelClient.intersect_clip_id_lists(caption, classifier, semantic)
```

This way each filter's contribution is observable and the agent can drop
the strictest filter without re-running the others.

---

## 7. Don't pass server URL params via `extra_params` without reading the atlas

### Trigger

```python
# Bypassing the SDK's translation layer
total, results = client.search(
    classifier_select="Snow",
    extra_params={"probability_threshold": 0.5},  # WRONG name
)
```

### What you'll see

`(N, results)` where N is suspiciously large — server silently dropped the
threshold. Classifier scores look unfiltered.

### Why it's wrong

The server-side parameter name has been **renamed** twice in 2026:

- `probability_threshold` (numeric) → `probability_expression` (string predicate)
- `extra_queries` → `caption_extra_queries`

The SDK translates the legacy NAMED kwargs (`probability_threshold=0.5`,
`extra_queries="a||b"`) at the URL boundary so the public Python API stays
back-compat. But params passed via `extra_params={...}` bypass that
translation and hit the server with the OLD names, which are silently
dropped.

### Right call

Use the named kwargs (preferred):
```python
client.search(classifier_select="Snow", probability_threshold=0.5)
client.search(extra_queries="a||b")
```

Or, if you really need to pass an advanced predicate, use the explicit
expression:
```python
client.search(classifier_select="Snow", probability_expression="0.3 < p < 0.7")
```

The current server URL parameter names are documented in
the SIL Wheel server source § "Silent server-side renames since
2026-03-02".

---

## 8. Don't assume historical classifier results were filtered

### Trigger

Pre-v1.6 SDK calls of:

```python
client.classifier_search("Snow", threshold=0.7, data_source="MADS")
```

### What you'll see (historically)

A list of clips. The list LOOKS like it was filtered to high-confidence Snow
matches, but **it wasn't**. The server silently dropped the threshold and
returned whatever clips matched `data_source="MADS"` from the previous
pipeline stage.

### Why it's wrong

Anatomy of the silent bug (root cause: server param rename in 2026-03):

1. Client sent `?classifier_select=Snow&probability_threshold=0.7`
2. Server's `SearchFilters.from_query` parsed `probability_expression=None`
   (it never read `probability_threshold`)
3. Server's `classifier_search.search()` checked
   `if classifier_select is not None AND probability_expression is not None: apply`
   → false because expression was None → returned `current_results` unchanged
4. `current_results` came from prior pipeline stages — typically all clips
   matching `data_source`, no classifier filtering applied
5. User saw a result list and assumed the threshold worked

### Right call

The SDK v1.6.0+ translates `probability_threshold=0.7` →
`probability_expression="p > 0.7"` at the URL boundary, so the public API
stays the same. **But any historical results from pre-v1.6 SDK calls should
be treated as untrusted** — re-run with the current SDK to get correctly
filtered results.

This is the root cause of multiple Agent Hub posts about "Fog classifier
broken on MADS" and "interesting classifier needs recalibration" — the
classifiers themselves were fine; the SDK was sending an unrecognized
parameter name.

---

## 9. Don't trust pre-v1.7 `list_classifier_names()` output (was empty)

### Trigger

Any agent or pipeline that called `client.list_classifier_names()` or
`client.list_classifiers()` between when the server schema changed
(2026-Q2) and when the SDK was patched (v1.7.0).

### What you'll see (historically)

`[]` — empty list. Agent concludes "there are no trained classifiers"
when there are actually 100+.

### Why it's wrong

Server changed the `/classifiers_status` response shape from a flat
`trained: ["Snow", "Rain", ...]` list to a typed dict
`trained_by_type: {"cosmos": [...], "caption": [...]}`. The pre-v1.7 SDK
read `data.get("trained")` which silently returned `None`.

### Right call

The current SDK handles both shapes. Re-running with v1.7+ will return
the actual trained-classifier list. Add the new optional `embed_type`
kwarg to filter to one backend:

```python
all_trained = client.list_classifier_names()  # both backends, deduped, sorted
cosmos_only = client.list_classifier_names(embed_type="cosmos")
caption_only = client.list_classifier_names(embed_type="caption")
```

For full structure with annotation counts and per-embed-type breakdown:
```python
data = client.get_classifiers()
print(data["trained_by_type"])  # {"cosmos": [...], "caption": [...]}
print(data["number_of_annotations"])  # per-label counts
```

---

## 10. Don't call `export_classifier_weights(label)` without `embed_type`

### Trigger

Any pre-v1.7 SDK call: `client.export_classifier_weights("Snow")`.

### What you'll see (historically)

`{"error": 404}` — "classifier not found". Agent concludes the
classifier doesn't exist or the export endpoint is broken.

### Why it's wrong

Server URL changed in 2026-Q2 from `/classifier/export/{label}` to
`/classifier/export/{embed_type}/{label}` (because a classifier label
can exist on either the Cosmos or Caption embedding backend, with
different weights). The pre-v1.7 SDK sent the old URL and always got 404.

### Right call

The current SDK auto-detects `embed_type` via
`get_classifier_embed_type(label)`:

```python
weights = client.export_classifier_weights("Snow")
# {'label': 'Snow', 'version': 1, 'coefficients': [...], 'intercept': [...]}
```

If you know the embed type explicitly:
```python
weights = client.export_classifier_weights("Snow", embed_type="cosmos")
```

If the auto-detection fails (label isn't in `trained_by_type`), the
return value is now `{"error": 404, "reason": "..."}` with an actionable
explanation pointing at `list_classifier_names()`.

---

## Cross-cutting principles

1. **Verify before concluding absence**. A 0-result response is a
   hypothesis, not a fact. Confirm with `lookup_clip`, `get_data_sources`,
   `list_classifiers`, or by widening the filter (e.g. drop `data_source`)
   before concluding the data isn't there. The most common cause of "0
   results" is a too-narrow filter, not missing data.

5. **Use `client.diagnose_zero_results(...)` when stuck**. It's a
   structured diagnostic that probes auth, suggests `mode='any'`,
   probes the caption embedding store, checks clip-id shape, suggests
   widening `data_source`, etc. Never raises, even in strict mode.

6. **Set `WHEEL_STRICT=1` for autonomous loops**. Strict mode promotes
   silent-bug-class warnings (composed-search-returns-0, exact-clip-key-
   footgun, FTS5-AND-on-natural-language) into `WheelZeroResultError`
   exceptions, so agents that ignore stderr stop and surface the
   failure immediately.

7. **Read the warnings**. Every `UserWarning` the SDK emits comes with a
   ready-to-paste corrective code snippet (e.g.
   `client.caption_search('rain night urban', mode='any')`). The fix is
   right there.

2. **Read warnings**. The SDK is heavily instrumented with `UserWarning`s
   that fire on the most common footguns. Suppressing them turns silent
   data loss into invisible data loss.

3. **Prefer named primitives over filter parameters**. `lookup_clip` over
   `search(search_clipid=)`. `caption_search_any` over
   `caption_search(mode='any')` if the OR semantics are part of the
   request's identity. `score_clips_large` over hand-rolled fan-out.

4. **Cite the data source explicitly**. The default of `"MADS"` for
   `find_clips_for_*` is a back-compat constant, not a recommendation.
   Pass `data_source` explicitly in every production call. To search
   across all sources, pass `data_source=None`.

---

## 11. Don't ignore `login()`'s return value (silent `total=0` trap)

### Trigger

```python
client = WheelClient()
client.login()                 # Returns False on timeout — no exception raised
total, results = client.search(search="rain", data_source="MADS-1M")
# total == 0  →  caller concludes "no rain clips" when in fact the search
# went through unauthenticated and the server quietly returned an empty page.
```

### What you'll see

`(0, [])` for every subsequent search call. **Crucially**, `WHEEL_STRICT=1`
will not save you here — the strict warnings are about silent-empty-result
**queries**, not silent-empty-result **sessions**. The diagnose_zero_results
probe will also report no obvious cause because the query itself is fine.

stderr will have a one-line warning when the failure happens:

```
Error: Login timed out after 120s connecting to http://your-sil-wheel-host:8000.
The server may be down or under heavy load. Try again later.
```

…but in tightly-piped agent contexts that line is easy to miss.

### Why it's wrong

`WheelClient.login()` returns `False` (instead of raising) on
`requests.exceptions.ReadTimeout`, `ConnectionError`, etc. This is for
back-compat with the CLI, but it lets unauthenticated requests through
silently. The wheel server returns 200 + empty payload for filters that
exclude all data the unauthenticated user can see — which is everything.

### Right call

For autonomous loops and agent code, **opt into the loud failure path**:

```python
from src.wheel_client import WheelClient, WheelAuthenticationError

client = WheelClient()
try:
    client.login(raise_on_failure=True, retries=3)
except WheelAuthenticationError as e:
    # e.reason in {"timeout", "connection_error", "bad_credentials",
    #              "missing_credentials", "vpn_down", "wheel_down"}
    if e.reason in ("timeout", "wheel_down"):
        client.wait_for_server(max_wait=1800)   # blocks until server is back
        client.login(raise_on_failure=True, retries=3)
    elif e.reason == "vpn_down":
        sys.exit("Reconnect your network and retry.")
    else:
        raise
```

Or via env (zero code change for existing scripts):

```bash
export WHEEL_LOGIN_RAISE=1
export WHEEL_LOGIN_RETRIES=3
python my_script.py
```

For non-autonomous one-shot scripts the legacy `bool` return is fine, but
**always check it**:

```python
if not client.login():
    sys.exit("login failed — check VPN / credentials / server health")
```

### Test references

- Smoke-tested in this session: `login(raise_on_failure=True, retries=0)`
  raises `WheelAuthenticationError(reason="bad_credentials")` with wrong
  creds and `(reason="missing_credentials")` when env vars are unset.
- `check_connection(probe_vpn=True)` now distinguishes `vpn_down` from
  `sil_wheel_down`, surfaced as `e.reason` after retries are exhausted.

---

## 12. Don't sort `trajectory_score` descending — it's a distance

### Trigger

```python
total, results = client.trajectory_search_by_clip(seed)
results.sort(key=lambda r: -r.trajectory_score)   # WRONG — gets LEAST similar first
top = results[:20]
```

### What you'll see

The "top" clips have `trajectory_score ≈ 200000+` (high = far away in
trajectory L2 space) and look nothing like your seed. Took the first
agent that hit this ~30 minutes to discover the inversion.

### Why it's wrong

`SearchResult.trajectory_score` is the **L2 distance** between
trajectory shapes — lower = more similar. The field name and the
generic `score` alias suggested "higher = better" by analogy with
classifier / cosine-similarity scores. The same trap exists for
`cluster_distance_score`.

### Right call (v1.8.2+ SDK)

The SDK now exposes two safer accessors:

```python
r.trajectory_distance     # alias for trajectory_score, honest name
r.trajectory_similarity   # 1 / (1 + distance), higher = better
```

And `r.best_score` now applies the similarity transform automatically
when only the distance is set, so `sort(key=lambda r: -r.best_score)`
behaves consistently across all search modes.

If you hold the raw distance directly, sort ascending:

```python
results.sort(key=lambda r: r.trajectory_score)   # most similar first
# OR
results.sort(key=lambda r: -r.trajectory_similarity)
```

Same fix applies to `cluster_distance_score` / `cluster_similarity`.

---

## 13. Don't compose multi-classifier filters in one search — intersect client-side

### Trigger

```python
client.search(
    classifier_select=["Change lane to the right",
                       "Change lane to the left"],
    probability_threshold=0.7,
)
# silently returns 0 — the server only accepts one classifier_select
```

### Why it's wrong

The server's `classifier_select` URL param is a single string. Lists are
silently coerced to a string representation that matches no classifier
name, so the filter passes nothing.

### Right call (v1.8.2+ SDK)

Use the new helper:

```python
both = client.find_clips_matching_classifiers(
    [("Change lane to the right", 0.7),
     ("Change lane to the left",  0.7)],
    op="and",
    data_source="AV V2 train",
)
# Returns sorted list of clip IDs in the intersection.
```

Behind the scenes this calls `export_search_clip_ids` per classifier and
intersects (or unions) the results client-side. The helper warns if any
individual classifier returns 0 — the most common silent failure here is
running it on a `data_source` where one of the classifiers has no
inferences. See `knowledge/feature-compat.md`.

---

## 14. Don't assume classifier inferences are uniform across data sources

### Trigger

```python
total, _ = client.classifier_search(
    "Change lane to the right",
    threshold=0.5,
    data_source="MADS-1M",
)
# total == 0 — agent concludes "no right-lane-change clips in MADS-1M"
```

### What you'll see

`(0, [])`. The search returns quickly with no warning. The agent
concludes the data is missing, switches strategies, wastes time.

### Why it's wrong

The classifier IS trained (you can see it in
`get_classifiers()['trained_by_type']['cosmos']`), but **inference was
never run on MADS-1M for this label**. The same label returns 1.4M
matches on AV V2 train.

The asymmetry is per-label: `Change lane to the LEFT` works on MADS-1M
(266K matches) but `Change lane to the RIGHT` does not. There's no
documented reason — operational state.

### Right call

Probe coverage before assuming missing data:

```python
coverage = client.get_classifier_coverage("Change lane to the right")
# {"MADS": 0, "MADS-1M": 0, "AV V1 train": 1500454, ...}
```

The v1.8.2+ SDK maintains an in-code `_CLASSIFIER_KNOWN_MISSING` map and
warns automatically when you compose `classifier_select=X
data_source=Y` for known-missing combinations. To extend the map, run
`get_classifier_coverage()` for the new label and submit a one-line
patch.

See `knowledge/feature-compat.md` for the full matrix.

---

## 15. Don't use trajectory predicates on MADS-1M / OpenDV / etc.

### Trigger

```python
total, _ = client.search(
    search_speed="max(speed_kph) > 50",
    data_source="MADS-1M",
)
# After ~80s, total == 0
```

### Why it's wrong

The trajectory store only ingests AV V1/V2/V2.2, Waymo, and celsius
sources. Calls against any other source compile, run, return 0, and
take 50–150 seconds while doing it. Compounded silent failure.

### Right call

The v1.8.2+ SDK warns when `search_speed` or `trajectory_pattern` is
combined with one of the known-unindexed sources
(MADS, MADS-1M, OpenDV-YouTube, MultiCountry-800K, Physical AI,
OGameData, NVIQ).

Move the predicate to AV V2 train (or one of the other indexed
sources):

```python
total, _ = client.trajectory_predicate_search(
    speed_expr="max(speed_kph) > 50",
    data_source="AV V2 train",
)
```

If you must work in MADS-1M, anchor with classifier or caption
filters first and fall back to client-side trajectory analysis on
result `positions` arrays.

---

## 16. Don't pass >50 clip IDs to the raw `vlm_judge_validate_search`

### Trigger (pre-v1.8.2)

```python
results = client.vlm_judge_validate_search(query, clip_ids[:1000])
# returns list with len == 1, no warning
```

### What you'll see

Only one result returned, `match: True/False` for one clip. The
remaining 999 silently skipped. Agent reports "VLM-validated 1000
clips" when actually 1.

### Why it's wrong

The endpoint is GET. 1000 UUID-formatted clip IDs joined by commas
produce a ~37 KB query string. Most servers cap at ~8 KB and silently
return only the first parsed entry.

### Right call (v1.8.2+ SDK)

The SDK now auto-chunks at `chunk_size=30` and retries 502/503/504
with exponential backoff. Just call it:

```python
results = client.vlm_judge_validate_search(
    query, clip_ids,                      # any length now OK
    chunk_size=30,                        # tune if you know server limit
    max_attempts=3,                       # retries on transient HTTP failure
    backoff_seconds=30.0,                 # base backoff
    on_chunk_complete=lambda i, r: ...,   # for incremental save
    progress_fn=lambda done, total: ...,  # for UI
)
```

A defensive warning fires inside the SDK if a chunk of 2+ clips
returns 0 results — the symptom of URL truncation that bit the
pre-v1.8.2 caller.

If a chunk fails after all retries, the SDK returns
`[{"clip_id": cid, "error": "..."}]` per id (preserving alignment) so
you can re-run or drop the chunk.

---

## How to add a new entry

When a new pattern lands on Agent Hub:

1. Reproduce it locally — confirm the bug AND that it isn't already fixed
   in `main`.
2. If it's a real footgun, add a section here using the same five-part
   template (Trigger / What you'll see / Why it's wrong / Right call /
   any test references).
3. If the SDK can detect the footgun and warn, add the warning in
   `wheel_client.py` and an offline test in `tests/test_offline.py` —
   put the new test class near the existing classes that audit the same
   method (e.g. `TestSearchClipidExactWarning`,
   `TestCaptionSearchModeKwarg`, `TestExactClipIdRegex`,
   `TestListClassifiersAlias`).
4. Reply on the original Hub post with the SHA of the fix.