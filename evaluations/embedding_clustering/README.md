<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Embedding-space analysis

Unsupervised exploration of the embeddings SIL-Wheel serves, for **§4.4
("Embedding Space Analysis")** of the whitepaper. For a pool of clips covered by
all three embeddings — Cosmos-Embed1 (768-d video), Caption (Qwen3-Embedding-8B
over Qwen3.5-27B captions, 4096-d), and Visual (Florence-2/SigLIP, 768-d
frame-region) — it clusters each embedding with spherical (cosine) *k*-means and
produces the figures and tables that show how the three induce different cluster
geometry on the *same* clips, what each one's clusters emphasize, and how product
quantization (PQ) perturbs the partition at serving scale.

All paths are passed in; nothing is hardcoded. Run with an environment that has
`numpy`, `scikit-learn`, `pandas`, `matplotlib`, `umap-learn`, `boto3`, `decord`,
`Pillow`, and (for `preindex_compare`) `faiss`. Set `PYTHONPATH=$REPO` so the
vendored `sil_wheel` package (and `sil_wheel.cluster_hierarchy`) imports.

## Pipeline (run order)

The figures use two clustering tracks on the same pool:

- a **k=50, exact (pre-index)** track — the source for the overlay maps,
  cluster-topic table, distinctive-terms table, and hierarchical drill-down;
- a **k=1000, after-index** track (`run_full_cluster.sh`) — the source for the
  large-scale UMAP overview and the pre/after-index compression metrics.

```
ingest_raw_embeddings.py   raw per-encoder dumps -> one deduped <encoder>.npz
build_pool_clip_ids.py     the clip-ids covered by all three embeddings
cluster_raw.py             k=50 spherical k-means per encoder  (--center for visual)
hier_raw.py                recursive k-means taxonomy per encoder
backfill_cluster_themes.py LLM one-phrase theme per cluster (keywords -> phrase)
make_*.py                  the §4.4 figures and tables (see mapping below)
preindex_compare.py        exact vs PQ-reconstructed clustering agreement + compression
upload_runs.py             push run dirs to the wheel server
```

### 1. Ingest raw embeddings and build the pool
```bash
python ingest_raw_embeddings.py --root "$RAW_DUMPS" --out ./npz \
    --encoders cosmos caption visual --pool-name pai
python build_pool_clip_ids.py --wheel-data-dir "$WHEEL_DATA_DIR" --out ./emb_pools
```
`ingest_raw_embeddings.py` writes one `<encoder>.npz` (`clip_ids`, `embeddings`)
per encoder; `build_pool_clip_ids.py` writes the clip-id pools (`full_*`,
`large_*`, optional `pai_*`) and a full-coverage caption `clip_id -> row` map.

### 2. Cluster (k=50, exact) — overlay / distinctive / drill-down source
```bash
for e in cosmos caption; do
  python cluster_raw.py --npz ./npz/$e.npz --embed $e --k 50 --spherical \
      --captions-db "$CAPTIONS_DB" --clustering-dir ./clustering --run-id k50_$e
done
# the visual encoder is region-level and anisotropic: mean-centre before cosine
python cluster_raw.py --npz ./npz/visual.npz --embed visual --k 50 --spherical \
    --center --captions-db "$CAPTIONS_DB" --clustering-dir ./clustering --run-id k50_visual
```
Each run dir gets `centroids.npy`, `cluster_assignments.parquet` (distance-sorted
within cluster), `umap.json`, and `cluster_topics.json` (TF-IDF keywords over
≤50 sampled captions/cluster). The figure scripts expect the run-ids
`k50_cosmos`, `k50_caption`, `k50_visual`.

### 3. Hierarchical taxonomy (for the drill-down)
```bash
for e in cosmos caption visual; do
  python hier_raw.py --npz ./npz/$e.npz --captions-db "$CAPTIONS_DB" \
      --branching 10 --max-depth 2 $( [ $e = visual ] && echo --center ) \
      --out ./hier/pai_$e
done
```
Writes `hier_topics.json` + `hier_assignments.parquet` (a dotted `path` per clip,
e.g. `3.7`) under each `pai_<encoder>` dir.

### 4. Backfill LLM theme summaries
```bash
python backfill_cluster_themes.py --clustering-dir ./clustering \
    --runs k50_cosmos k50_caption k50_visual --model "$LLM_MODEL" --workers 4
```
Adds a one-phrase `description` to each cluster from its keywords only (the
system prompt forbids first-person / dashcam / ego phrasing). Retries on 5xx and
null content. Routes through the provider pinned by `LLM_PROVIDER` (`.env`);
never hardcode a provider or key.

### 5. Figures and tables
```bash
# row 1: three overlay maps (UMAP + centroid-clip thumbnails of the 10 most distinct clusters)
for e in cosmos caption visual; do
  python make_cluster_overlay_table.py --run-dir ./clustering/k50_$e \
      --select distinct --k 10 --map-only --out figures/overlay_map_$e.png
done
# row 2: colour-coded cluster-topic table
python make_overlay_topics_table.py --clustering-dir ./clustering --k 10 \
    --out tables/emb_cluster_topics.tex
# row 3: contrastive distinctive-terms table (weighted log-odds)
python make_distinctive_terms.py --clustering-dir ./clustering --topn 8 \
    --out tables/emb_distinctive_terms.tex
# large-scale UMAP overview (k=1000 after-index track; see run_full_cluster.sh)
python build_fig_runs.py --runs ./emb_pools/runs.tsv --clustering-dir ./clustering --out fig_runs.json
python make_umap_overview.py --clustering-dir ./clustering --fig-runs fig_runs.json \
    --out figures/emb_cluster_umap_overview.png
# hierarchical drill-down with arrows
python make_drilldown_arrows.py --clustering-dir ./clustering --hier-base ./hier \
    --npz-dir ./npz --out figures/hier_drilldown_arrows.png
```

### 6. Pre- vs after-index comparison
```bash
# real production codebook for an encoder that ships a Flat index:
python preindex_compare.py --wheel-data-dir "$WHEEL_DATA_DIR" --embed cosmos --out preindex_compare.json
# proxy (self-quantize raw vectors) where no Flat index exists:
python preindex_compare.py --raw-npz ./npz/caption.npz --index-spec IVF4096,PQ256x8 --embed caption --out preindex_compare.json
```
Reports **ARI/NMI** (partition agreement = quality retained), **compression**
(bytes/vec exact→PQ), timing, and intrinsic Δ/silhouette. Merges into the JSON,
one key per `--embed`.

### 7. Upload runs (optional)
```bash
python upload_runs.py --clustering-dir ./clustering --runs k50_cosmos k50_caption k50_visual \
    --url "$WHEEL_URL" --username "$WHEEL_USERNAME" --password "$WHEEL_PASSWORD"
```
Credentials come from `.env` (`WHEEL_URL` / `WHEEL_USERNAME` / `WHEEL_PASSWORD`).

## Paper artifacts

| Script | Output | Paper |
| --- | --- | --- |
| `make_umap_overview.py` | `figures/emb_cluster_umap_overview.png` | `fig:emb_cluster_umap` |
| `make_cluster_overlay_table.py --map-only` (×3) | `figures/overlay_map_{cosmos,caption,visual}.png` | `fig:emb_cluster_scenarios` (row 1) |
| `make_overlay_topics_table.py` | `tables/emb_cluster_topics.tex` | `fig:emb_cluster_scenarios` (row 2) |
| `make_distinctive_terms.py` | `tables/emb_distinctive_terms.tex` | `fig:emb_cluster_scenarios` (row 3) |
| `make_drilldown_arrows.py` | `figures/hier_drilldown_arrows.png` | `fig:emb_cluster_taxonomy` |
| `preindex_compare.py` | `preindex_compare.json` | `tab::preindex_compare` |

## Shared libraries

Two small, dependency-light modules are the single source of truth shared by the
figure generators:

- **`topic_lexicon.py`** — the category lexicon (`APPEAR` / `ACTIVITY` /
  `OBJECT` / `NEUTRAL` colours + signature word sets), `categorize(word)`,
  `latex_escape(s)`, and the weighted-log-odds distinctive-terms scoring
  (`topic_profiles`, `distinctive_terms`). Only each embedding's *signature*
  vocabulary is coloured; words common to all three stay neutral so the
  distinct focus of each column stands out. This is the canonical lexicon —
  `make_taxonomy_compare.py` keeps an older copy but is no longer imported by
  the active scripts.
- **`cluster_select.py`** — `distinct_clusters(run_dir, sizes, k, min_frac)`
  (farthest-first traversal over L2-normalised centroids, seeded with the
  largest cluster, size-floored to drop tiny outliers) and `dense_xy(xs, ys)`
  (densest-bin anchor for placing a label on a UMAP blob).

## Tests
```bash
python -m pytest evaluations/embedding_clustering/tests/ -q
```
Unit tests cover the lexicon (`categorize`, `latex_escape`, category
disjointness), the distinctive-terms profiles and log-odds ranking, and the
cluster-selection helpers (farthest-first, size floor, dense-bin anchoring).
They depend only on `numpy`/`pandas`, not the figure-rendering stack.

## Instant smoke test (synthetic — seconds, no downloads)

`make_synthetic_starter.py` writes a tiny wheel-format dataset (three embeddings
+ a toy captions DB) that exercises the clustering and figure steps. It is
synthetic data to verify the workflow, **not** a real result.

```bash
O=./synth_starter
python make_synthetic_starter.py --out $O
python build_pool_clip_ids.py --wheel-data-dir $O/wheel-data --out $O/emb_pools
WHEEL_DATA_DIR=$O/wheel-data CAPTIONS_DB=$O/captions.db POOLS_DIR=$O/emb_pools \
  CLUSTER_OUT=$O/clustering POOL=full K=20 bash run_full_cluster.sh
python build_fig_runs.py --runs $O/emb_pools/runs.tsv --clustering-dir $O/clustering --out $O/fig_runs.json
python make_umap_overview.py --clustering-dir $O/clustering --fig-runs $O/fig_runs.json --out $O/umap_overview.png
```

## Notes

- **Visual is anisotropic.** Pass `--center` to `cluster_raw.py` / `hier_raw.py`
  / `preindex_compare.py` for the visual encoder: subtracting the mean and
  renormalising recovers a usable cosine gap (Δ rises from ~0.10 to ~0.75).
- **Intrinsic scores are PQ-optimistic.** In `preindex_compare`, read partition
  *agreement* (ARI/NMI), not intrinsic Δ/silhouette, as the quality axis —
  codebook discretization inflates the intrinsic scores.
- **Topics overlap by construction.** All three embeddings are described by the
  same caption set, so the cluster *themes* overlap; the discriminative signal
  is the distinctive-terms (log-odds) table, not the raw themes.
- **`figstyle.py`** registers NVIDIA Sans (from
  `SIL_Wheel_Whitepaper/NVIDIA-Sans-Font-TTF`) and sets the matplotlib palette;
  import it before plotting.
- **Superseded scripts** (`make_hier_*`, `make_taxonomy_*`, `make_overlay_grid`,
  `make_overlay_focus_bars`, `make_scenario_grid`, `make_umap_overlay`,
  `make_umap_hier`, `topic_ngram_compare`, `make_topic_focus`) are earlier
  iterations kept for reference; they are not part of the current figure set.
