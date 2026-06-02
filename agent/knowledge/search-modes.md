# SIL Wheel Search Modes

All search modes are composable — combine any subset in a single API call.
When multiple modes are active, results are **intersected**: only clips satisfying
every active filter are returned. Scored searches produce a ranking signal;
binary filters include/exclude without contributing a score.

## 1. Caption/Text Search (FTS5)

- **Parameter**: `search=<text>`
- **Backend**: SQLite FTS5 on Qwen2.5-7B generated captions
- **Speed**: Fast (~1-5s production, ~5-15s dev)
- **Best for**: Keyword and phrase search ("construction zone", "heavy rain", "tunnel entrance")
- **Query rewrite**: Set `query_rewrite=true` to enable LLM-based expansion that
  generates semantically related variants to broaden recall. Server needs `NV_INFERENCE_API_KEY`.
- **Multiple queries**: The server supports OR-combined queries via `extra_queries` parameter.
- **Result cap**: FTS5 returns at most ~5,000 matching clip IDs. For larger sets, combine with classifier search or use multiple narrower queries.

## 1b. Caption Embedding Search (Qwen3)

- **Parameter**: `caption_embed_search_text=<query>` (or via caption embedding UI)
- **Backend**: Qwen3-Embedding-8B vectors in FAISS IVF index, best-caption-per-clip scoring
- **Speed**: Fast (~2-5s)
- **Query rewrite**: Also supports LLM rewrite for expanded recall
- **Ranked mode**: Produces a ranking signal (not just filter)
- **Best for**: Semantic caption similarity — finds clips whose captions MEAN similar things even with different words. Complements FTS5 keyword search.

## 2. Cosmos Embedding Similarity — Clip-to-Clip

- **Parameter**: `semantic_search_clipid=<UUID>`
- **Backend**: FAISS IVF-PQ index on 768-dim Cosmos-Embed1 vectors
- **Speed**: Fast (~1s production, ~1-10s dev)
- **Best for**: Finding visually similar clips to a known clip

## 3. Cosmos Embedding Similarity — Text-to-Video

- **Parameter**: `semantic_search_text=<query>`
- **Backend**: Encodes text with Cosmos model, then FAISS search
- **Speed**: Fast on production GPU (~3-5s), **SLOW on dev CPU (~120s+)**
- **Best for**: Natural language visual concept search
- **Tip**: Use caption search as fast alternative on dev server

## 4. CLIP Visual Similarity

- **Text parameter**: `visual_search_text=<query>`
- **Image parameter**: Upload an image via the UI for image-to-video retrieval
- **Backend**: CLIP ViT-B/32 — 20 frames × multi-crop (140 patches/clip), text or image query
- **Speed**: Similar to Cosmos text-to-video
- **Best for**: Image-text matching (complementary to Cosmos, better for static visual concepts)
- **Ranked priority**: When combined with text, image ranking takes precedence

## 5. Trajectory Shape Similarity

- **Parameter**: `trajectory_shape_clipid=<UUID>`
- **Optional**: `trajectory_shape_start_t`, `trajectory_shape_end_t` for sub-trajectory
- **Backend**: FAISS L2 on flattened trajectory vectors
- **Speed**: Medium (~3-10s production, ~5-30s dev)
- **Index selection**: Auto-selects based on time range:
  - Full trajectory (20s): `OPQ121,IVF4096,PQ121x8`
  - 10s windows: `OPQ40,IVF4096,PQ40x8` (3 windows per clip: 0-10, 5-15, 10-20)
  - 5s windows: `OPQ20,IVF4096,PQ20x8` (4 windows per clip: 0-5, 5-10, 10-15, 15-20)
- **Best for**: Finding clips with similar driving patterns (turns, lane changes, etc.)

## 6. Trajectory Predicate Search

- **Parameter**: `trajectory_pattern=<name>` or `search_speed=<expression>`
- **Speed**: Medium
- **Named patterns**:
  - `high_curvature` — `sum(curvature > 0.15) > 10`
  - `stop_go` — vehicle stops then accelerates to >3 m/s
  - `hard_braking` — `sum(acceleration < -3.0) > 10`
  - `prolonged_stop` — `sum(speed < 0.5) > 150`
  - `idle_to_cruise` — stop then accelerate to >10 m/s
  - `high_speed_swerve` — high curvature + >50 kph
  - `moving_ego` — `sum(speed_kph > 5) > 10`
- **Custom expressions**: Python expressions over per-frame arrays
  - **Variables**: `speed`, `acceleration`, `jerk`, `curvature`, `speed_kph`
  - **Functions**: `mean`, `min`, `max`, `sum`, `all`, `any`, `len`
  - **Examples**:
    - `max(speed_kph) > 120` — high speed
    - `max(abs(curvature)) > 0.05` — sharp turns
    - `any(acceleration < -3)` — hard braking
    - `min(speed_kph) < 5 and max(speed_kph) > 30` — stop and go

## 7. Classifier Score Filtering

- **Parameters**: `classifier_select=<label>`, `probability_threshold=<float>`
- **Speed**: Fast (~1s)
- **Production classifiers (93 trained)**: Snow, Construction zone, night, tunnel, interesting, HighDynamicMotion, little_scene_dynamics, High curvature road, and many more
- **Case sensitivity**: Classifier search is **case-sensitive** on the server. Use `resolve_classifier_name("rain")` for fuzzy lookup — finds `"Heavy rain"` via substring, prefix, and difflib matching. Or use `list_classifier_names()` to browse all labels.
- **How classifiers work**: Logistic regression on 768-dim Cosmos embeddings. Train from ~50 positive labels, auto-scores all indexed clips.
- **Classifier weight export**: `GET /classifier/export/<label>` returns JSON with coefficients + intercept
- **Use `get_classifiers()` to list available labels**

## 8. Cluster Search (with TF-IDF Topic Modeling)

- **Parameters**: `cluster_run_id=<id>`, `cluster_id=<id>`
- **Speed**: Fast (precomputed)
- **Backend**: K-means clustering over Cosmos embeddings (spherical or Euclidean).
  Per-cluster TF-IDF topic keywords are extracted at clustering time from
  50 sampled clip captions per cluster (caption model auto-selected by
  coverage). Optional one-phrase LLM theme description per cluster
  (server MR !1, gracefully absent without an LLM key).
- **Results ranked by**: Distance to cluster centroid (closer = more representative)
- **Workflow**: Run clustering from UI → get `run_id` → browse/filter by cluster
- **Best for**: Exploring coherent groups within a search result set;
  data-coverage checks ("what kinds of scenes are in my 50k urban clips?")
- **Topic-aware client methods** (v1.8.0):
  - `get_clustering_results(run_id)` — full payload with `topics` + `topics_meta`
  - `get_cluster_topics(run_id)` — TF-IDF keywords + LLM descriptions per cluster
  - `get_cluster_members(run_id, cluster_id)` — full clip_id list (no paging)
  - `find_clusters_by_keyword(run_id, kw)` — substring scan over keywords/descriptions
  - `summarize_clustering_run(run_id, top_k)` — agent-friendly themes summary
  - `cluster_search(run_id, cluster_id)` — paged SearchResult view (existing)
- **Topic gotcha**: clustering runs created BEFORE the topic feature shipped
  (Apr 2026) return `topics: {}`. Re-run `run_clustering()` on the dev
  server to regenerate, or wait for the proposed `extract_topics::<run_id>`
  endpoint.

## 9. Perception Search (formerly World Model Object Search)

- **Parameters**: `wm_class_name=<class>`, `wm_min_count=<int>`, `wm_max_count=<int>`
- **Optional**: `wm_max_dist=<float>` (meters), `wm_min_time=<float>` (seconds), `wm_angle_range=<sectors>`
- **Speed**: Fast
- **Object classes**: `VEHICLE_CAR`, `VEHICLE_TRUCK`, `VEHICLE_BUS`, `BIKE_WITH_RIDER`, `BIKE_TRICYCLE`, `PEDESTRIAN_UNKNOWN`
- **Angle sectors (6)**: `FRONT`, `FRONT_LEFT`, `FRONT_RIGHT`, `BACK`, `BACK_LEFT`, `BACK_RIGHT`
- **Filter-only mode**: Narrows results but does not contribute a ranking signal
- **Best for**: Finding clips with specific object configurations (e.g., "3+ pedestrians within 10m in front")

## 10. Annotation Label Filtering

- **Parameters**: `filter=<label>` (include), `labels_to_exclude=<label>` (exclude)
- **Optional**: `label_types=manual|autolabel`, `project_source=<project>`, `filter_mode=any|all`
- **Speed**: Fast
- **687+ labels** available (manual + autolabel)
- **Key labels**: Keep lane (7M), Accelerate (3.7M), Turn right (1.4M), Highway Boring (9.9K)
- **Multiple labels**: Use `||` separator: `filter=Snow||rain`
- **AND vs OR**: `filter_mode=any` (default, OR) or `filter_mode=all` (AND — clips must have ALL listed labels)

## 11. Numeric Metric Filtering

- **Parameters**: `numeric_filter=<metric>,<min>,<max>,<ordering>`, `with_metrics=true`
- **Speed**: Fast
- **Models on leaderboard**: Policy (4 models), Human Driving (ground_truth), VLM (5 models)
- **Use case**: Filter clips by model performance metrics (PSNR, loss, comfort score, etc.)
- **Ordering**: `asc` (small to large) or `desc` (large to small)

## 12. Country/Driving-Side Filtering

- **Parameters**: `search_country=<name>`, `left_hand_driving=true`
- **Speed**: Fast
- **Use case**: Control geographic distribution of data

## 13. SIL API Applicability

- **Parameter**: `sil_apis=<api_name>`
- **Speed**: Fast
- **Use case**: Restrict results to clips tagged as applicable to specific SIL APIs

## 14. Comment Search

Search clips by user comments. Comments are free-text notes users have left on clips in the SIL Wheel UI.

**Parameters:**
- `search_comments` — text to search in comments

**Convenience method:** `client.comment_search(query)`

**Speed:** Fast (~1-3s)

**Example:**
```python
results = client.comment_search("interesting edge case")
# Or via search():
total, results = client.search(search_comments="interesting")
```

## Ranked Mode Priority

When multiple ranked modes are active, the server uses this priority order for the final sort:

`Numeric metric → Classifier → Caption embedding → Semantic text → Semantic clip → Visual (text) → Visual (image) → Trajectory shape → Cluster distance`

Filter-only modes (caption FTS5 when combined with a ranked mode, trajectory predicates, perception, annotation labels, country, SIL APIs, comments) narrow the pool but don't affect sort order.

## Composability

All modes chain together in the server's search pipeline:

```python
results = captionstore.search(filters, results)
results = trajectorystore.search(filters, results)
results = embeddingsstore.search(filters, results)
results = clipembeddingsstore.search(filters, results)
results = classifiersearch.search(filters, results)
results = clustersearch.search(filters, results)
results = wm_store.search(filters, results)
results = datastore.search(filters, results)
results = metricstore.search(filters, results)
results = bev_fetcher.search(filters, results)
```

Each store narrows the result set from the previous one (intersection). Order matters for performance — the server applies them in this fixed order.

## Choosing the Right Mode

| User Intent | Recommended Mode(s) |
|-------------|---------------------|
| Find clips by keyword | Caption search |
| Find clips matching a visual concept | Cosmos text-to-video (or caption as fast fallback) |
| Find clips similar to a known clip | Cosmos clip-to-clip + trajectory similarity |
| Find specific driving behaviors | Trajectory predicates |
| Find clips by scenario category | Classifier search |
| Find clips with specific objects | Perception search (world model) |
| Find clips by caption meaning | Caption embedding search |
| Filter by quality/performance | Numeric metric filtering |
| Explore groups within results | Cluster search |
| Filter by geography | Country/driving-side |
| Complex multi-dimensional query | Compose multiple modes |

## "Idea to Clip IDs" Workflow

When a team has a training idea (e.g., "we need more construction zone in rain clips"):

1. **Start with caption search** (`search="construction zone rain"`) — fast, broad recall
2. **Refine with classifier** (`classifier_select="Construction zone"`) — trained on labeled data
3. **Expand with semantic search** (`semantic_search_text="construction zone in rainy weather"`) — visual similarity
4. **Use query rewrite** to auto-expand captions into related terms
5. **Compose filters** to intersect multiple signals for precision
6. **Export** the final clip ID list for training pipeline integration

The `find_clips_for_scenario()` method automates steps 1-3. If all strategies return 0 (common for composite descriptions on MADS), it automatically decomposes the description into sub-queries, searches each separately, and intersects the results.

## "Clip ID to Similar Clips" Workflow

Starting from known good clip(s):

1. **Cosmos similarity** (`semantic_search_clipid=<id>`) — find visually similar clips
2. **Trajectory similarity** (`trajectory_shape_clipid=<id>`) — find kinematically similar driving
3. **Combine both** for clips that are similar in both visual content and driving behavior
4. **Iterate**: take top results, find _their_ similar clips to grow the set
5. **Export** the expanded set

The `expand_clip_set()` method automates this iterative expansion.

## VLM Judge (added v1.5.0)

VLM Judge is a server-side feature using a vision-language model (Gemini 3 Flash via
NVIDIA Inference API) to evaluate clips. Requires `NV_INFERENCE_API_KEY` on the server.

### Caption Scoring
- **Endpoint**: `GET /api/vlm_judge/caption_score?clip_id=X&caption=Y&uid=Z`
- **Client**: `client.vlm_judge_caption_score(clip_id, caption, uid=None)`
- **Returns**: Five 1-10 scores (scene, action, road_entities, temporal, overall) + reasoning
- **Use case**: Evaluate caption quality, find clips with poor/inaccurate captions

### Search Validation
- **Endpoint**: `GET /api/vlm_judge/validate_search?query=Q&clip_ids=id1,id2,...`
- **Client**: `client.vlm_judge_validate_search(query, clip_ids)`
- **Returns**: Per-clip binary match + reasoning + analysis
- **Limit**: Up to 1000 clips per request (parallel processing, ~20 workers)
- **Use case**: Measure search precision, filter false positives

### Status Check
- **Endpoint**: `GET /api/vlm_judge/status`
- **Client**: `client.vlm_judge_status()`
- **Returns**: `{enabled, healthy, in_process}`

### Combined Workflow
```python
results, validations = client.vlm_judge_search_and_validate("construction zone")
precision = sum(v["match"] for v in validations) / len(validations)
print(f"Search precision: {precision:.0%}")
```
