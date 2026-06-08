# Embedding-clustering evaluation

Unsupervised dataset exploration with the embeddings SIL-Wheel serves. For a
pool of clips covered by all three embedding indices (Cosmos-Embed1, caption,
visual), it clusters each embedding with spherical *k*-means and produces:

1. **UMAP overview** — how the three embeddings induce different cluster
   geometries on the *same* clips (`make_umap_overview.py`).
2. **Topic focus** — what each embedding's clusters emphasize, via caption terms
   that concentrate in one embedding but disperse across the others, summarized
   by an LLM (`make_topic_focus.py`).
3. **Cluster-topic taxonomy** — recursive *k*-means with per-level topic labels,
   rendered as a sunburst/treemap (`run_hier_cluster.py`, `make_hier_viz.py`).
4. **Pre- vs after-index comparison** — how much product quantization (PQ), used
   to serve embeddings at scale, perturbs the clustering vs. exact vectors, and
   the storage/speed it buys (`preindex_compare.py`).

All paths are passed in (no hardcoded locations). Run with an environment that
has `faiss`, `numpy`, `scikit-learn`, `matplotlib`, `pandas`, and (for the
sunburst) `plotly`; install the package so `sil_wheel` and `scripts/` import.

Inputs you provide:
- `WHEEL_DATA_DIR` — the cosmos/visual/caption FAISS indices + id maps.
- a SQLite **captions DB** (for TF-IDF topic labels).
- optionally `NV_INFERENCE_API_KEY` for the LLM one-phrase titles/descriptions.

## Quickstart

Use the `wheel` conda env (so `python` has faiss); the driver also honours
`PYTHON=/path/to/python` if the env isn't active.

### A. Instant smoke test (synthetic — seconds, no downloads)

`make_synthetic_starter.py` writes a tiny wheel-format dataset (three embeddings
+ a toy captions DB) that exercises every step. It is synthetic data to verify
the workflow, **not** a real result.

```bash
O=./synth_starter
python evaluations/embedding_clustering/make_synthetic_starter.py --out $O
python evaluations/embedding_clustering/build_pool_clip_ids.py --wheel-data-dir $O/wheel-data --out $O/emb_pools
WHEEL_DATA_DIR=$O/wheel-data CAPTIONS_DB=$O/captions.db POOLS_DIR=$O/emb_pools \
  CLUSTER_OUT=$O/clustering POOL=full K=20 bash evaluations/embedding_clustering/run_full_cluster.sh
python evaluations/embedding_clustering/build_fig_runs.py --runs $O/emb_pools/runs.tsv --clustering-dir $O/clustering --out $O/fig_runs.json
python evaluations/embedding_clustering/make_umap_overview.py --clustering-dir $O/clustering --fig-runs $O/fig_runs.json --out $O/umap_overview.png
python evaluations/embedding_clustering/make_topic_focus.py  --clustering-dir $O/clustering --fig-runs $O/fig_runs.json --out $O/topic_focus.png
python evaluations/embedding_clustering/preindex_compare.py  --raw-npz $O/cosmos.npz --index-spec IVF16,PQ96x8 --embed cosmos --out $O/preindex_compare.json
```

### B. Real data (nuScenes mini)

For real embeddings on a small public split, first build a `wheel-data` dir with
the repo's nuScenes example (10 scenes; all modalities + FAISS indexes):

```bash
pip install -r examples/getting-started-nuscenes/requirements.txt
python examples/getting-started-nuscenes/setup_nuscenes.py --workdir ./wheel-data --admin-password admin
```

Then run the steps below with `WHEEL_DATA_DIR=./wheel-data`, `CAPTIONS_DB` set to
the captions DB named in `wheel-data/config.yaml`, and a small `K` (mini has only
a few hundred clips). Pre/post needs an exact (Flat) source, which the mini
pipeline doesn't build — use the synthetic `--raw-npz` path above, or your own
exact vectors.

## 1. Build the clip-id pools
```bash
python build_pool_clip_ids.py --wheel-data-dir "$WHEEL_DATA_DIR" --out ./emb_pools
# optional curated subset:  --pai-path-files paths_a.txt paths_b.txt
```
Writes `full_clip_ids.json` (the entire common set), `large_clip_ids.json` (a
sample), an optional `pai_clip_ids.json`, and a full-coverage caption
`clip_id -> row` map under `emb_pools/caption_embeddings/`.

## 2. Cluster (k=1000 spherical, topics on)
```bash
WHEEL_DATA_DIR=... CAPTIONS_DB=... POOLS_DIR=./emb_pools CLUSTER_OUT=./clustering \
  POOL=full bash run_full_cluster.sh
```
Runs cosmos → visual → caption sequentially, logging run ids to
`emb_pools/runs.tsv`. Each run dir gets `cluster_assignments.parquet`,
`umap.json`, `cluster_topics.json`, etc.

## 3. Figures
Build `fig_runs.json` from `runs.tsv` with `build_fig_runs.py` (or copy
`fig_runs.example.json` and edit run ids by hand).
```bash
python build_fig_runs.py --runs ./emb_pools/runs.tsv --clustering-dir ./clustering --out fig_runs.json
python make_umap_overview.py --clustering-dir ./clustering --fig-runs fig_runs.json \
    --out umap_overview.png
NV_INFERENCE_API_KEY=... python make_topic_focus.py --clustering-dir ./clustering \
    --fig-runs fig_runs.json --out topic_focus.png
```
`make_topic_focus.py` ranks each embedding's caption terms by a contrastive
weighted log-odds (concentrated here, dispersed in the other embeddings of the
same pool), draws the top terms as bars, and titles each panel with an LLM
one-phrase summary (bars-only without an API key).

## 4. Cluster-topic taxonomy (optional)
```bash
PYTHONPATH=$REPO:$REPO/scripts python run_hier_cluster.py \
    --wheel-data-dir "$WHEEL_DATA_DIR" --captions-db "$CAPTIONS_DB" \
    --pools-dir ./emb_pools --pool full --embed cosmos \
    --branching 10 --max-depth 2 --out ./hier/full_cosmos
python make_hier_viz.py --hier-dir ./hier/full_cosmos --title "Full / Cosmos-Embed1" --png
```

## 5. Pre- vs after-index comparison
```bash
# real production codebook, for an encoder that ships a Flat index (e.g. cosmos):
python preindex_compare.py --wheel-data-dir "$WHEEL_DATA_DIR" --embed cosmos --n 0
# proxy (self-quantize raw vectors) where no Flat index exists:
python preindex_compare.py --raw-npz caption.npz --index-spec IVF4096,PQ256x8 --embed caption
```
Reports **ARI/NMI** (partition agreement = quality retained), **compression**
(stored bytes/vec exact→PQ), reconstruct/cluster **timing**, and intrinsic
Δ/silhouette. Results merge into `preindex_compare.json` (one key per `--embed`).

## Notes
- **Quality retained, much smaller:** PQ preserves the coarse partition (high
  NMI) while storing 32–64× fewer bytes/vec, which is what makes million-scale
  clustering tractable on one host. Fine assignments shift modestly (moderate
  ARI). Intrinsic Δ/silhouette are **PQ-optimistic** (codebook discretization
  inflates them) — read agreement, not intrinsic scores, as the quality axis.
- **Intrinsic-vs-index:** report quality-vs-labels on exact embeddings; report
  large-scale clustering on after-index embeddings (what the server serves).
- **Visual encoder** clusters on the first-frame embedding per video (via
  `embed_io`); it is region-level and anisotropic — pass `--center` to
  `preindex_compare.py` to recover a usable cosine gap.
- The contrastive stopword lists in `make_topic_focus.py` (`_DROP`,
  `_FILLER_TOKENS`) are tuned for AV captions; adjust for another corpus.
