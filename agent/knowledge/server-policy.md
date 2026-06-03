<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Server Policy

## Production Server (`your SIL Wheel server:8000`)

**READ-ONLY for all users except the Wheel team.**

### Web UI Deep Links

The frontend uses hash-based routing. Deep links for viewing clips and searches:

- **Single clip**: `http://your-sil-wheel-host:8000/#&page=0&search_clipid={clip_id}&project_source=Alpamayo`
- **Caption search**: `http://your-sil-wheel-host:8000/#&page=0&search={query}`
- **Classifier filter**: `http://your-sil-wheel-host:8000/#&page=0&classifier_select={label}&probability_threshold=0.5`
- **Cosmos text search**: `http://your-sil-wheel-host:8000/#&page=0&semantic_search_text={text}`
- **Similar clips**: `http://your-sil-wheel-host:8000/#&page=0&semantic_search_clipid={clip_id}`
- **Data source filter**: `http://your-sil-wheel-host:8000/#&page=0&data_source={source}`
- **Combined**: Chain any parameters with `&` for composed searches

Use `client.clip_url(clip_id)` and `client.search_url(**params)` to generate these programmatically.

Always generate browser links when presenting search results to the user.

### Allowed operations
- Search (all 14 modes)
- Export clip IDs (`/current_search.csv`)
- View leaderboard metrics (`/metrics`, `/per_clip_metrics`, `/full_metrics`)
- View classifiers (`/classifiers_status`)
- Export classifier weights (`/classifier/export/<label>`)
- View data sources and stats (`/data_stats_list`)
- View clustering results (`/clustering_status`, `/clustering_results`)
- View predictions (`/predictions`)
- Query rewrite (`/rewrite`)
- Stream videos and autolabel previews

Prohibited operations:
- Upload labels or annotations
- Train classifiers
- Run clustering jobs
- Modify any data
- Spam bulk operations that could cause load

**Other teams depend on production data integrity. Never write to production.**

## Dev Server (`localhost:8018`)

**Sandbox for experimentation.** All write operations are allowed here:

- Upload labels — `client.upload_labels()` (server action: `mass_label`)
- Upload annotations — `client.upload_annotations()` (server action: `upload_annotations`)
- Train classifiers — `client.train_classifier()` (server action: `train_classifier`)
- Auto-label search results — `client.auto_label_search()` (server action: `auto_label`)
- Rename labels — `client.rename_label()` (server action: `rename_label`)
- Delete labels — `client.delete_label()` (server action: `delete_label`)
- Merge labels — `client.merge_label()` (server action: `merge_label`)
- Single-clip annotation CRUD — `client.add_annotation()`, `remove_annotation()`, `update_annotation_times()`, `verify_annotation()`
- Run clustering — `client.run_clustering()` (server action: `run_clustering`)
- Delete clustering — `client.delete_clustering()` (server action: `delete_clustering`)
- Upload captions — server action only (`upload_captions`), no client method
- Upload model metrics — server action only, no client method

The dev server runs on a physical machine managed by your SIL Wheel administrator. It uses copied/isolated databases so experiments don't affect production.

**Access**: Requires VPN or NVIDIA network. SSH access is separate from web access.

**Deep links**: Same format but with dev server URL: `http://localhost:8018/#&page=0&...`

## S3 Data Policy

- **NEVER modify `s3://processed_data/`** — this is the primary copy with no backup
- **OK to read**: All S3 data is read-only for us
- Backup of Cosmos embeddings: `s3://backup/cosmos_embeddings/`
- Videos are stored at `s3://processed_data/<dataset_name>/<clip_id>.mp4`

## Authentication

Both servers use session-cookie authentication:

1. POST `user_login::{username}::{password}` to the root URL
2. Server returns a `session_id` cookie
3. Include the cookie in all subsequent requests

The `WheelClient` handles this automatically. Credentials come from `.env`:
- `WHEEL_USERNAME` / `WHEEL_PASSWORD` / `WHEEL_SERVER_URL` for production
- `WHEEL_DEV_URL` / `WHEEL_DEV_USER` / `WHEEL_DEV_PASSWORD` for dev server
- Use `WheelClient.dev()` to create a dev-server client automatically

## Performance Considerations

- No formal rate limits, but be considerate
- Use pagination (`page` + `n` parameters) for browsing (server caps at 20 per page)
- Use `/current_search.csv` for bulk clip ID export (client defaults to 1M via `max_clips`)
- Text-to-video searches (Cosmos/CLIP) are slow on dev server (~120s) due to CPU-only inference
- Caption search and FAISS lookups are fast on both servers
- The server uses an LRU cache (size 50) for recent searches — repeated queries are instant

## Gotchas

1. **Data source permissions**: Users need `user_datasources` entries for EACH data source they can access. Empty list = no results. Admin role alone is NOT enough.
2. **Clips need video_paths**: Clips must be in BOTH `clips` AND `video_paths` tables to appear in search.
3. **Export format varies**: Over 10K results, CSV is `clip_id,data_source` only. Under 10K, it includes full annotations.
4. **NaN in JSON**: The server may return `NaN` or `Inf` values in JSON (non-standard). The client handles this.
5. **`num_videos` vs `total`**: The /videos response uses `num_videos` for total matching clips and `total` for total pages. The client normalizes this.
6. **Video path format**: Video URLs are `/video/{clip_id}.mp4` — the server looks up the S3 path from the `video_paths` table.
7. **Autolabel videos**: Depth, boxes, pointmap, MFMRH, and VIPE visualization videos are only available for clips with corresponding autolabel data.