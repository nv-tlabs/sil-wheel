<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Embedding-space analysis

An unsupervised look at how the embeddings SIL-Wheel serves organize a video corpus, and how that structure survives the product-quantized (PQ) index used to serve them at scale.

Three embeddings cluster the **same** pool of clips:

- **Cosmos-Embed1**: 768-d video embedding.
- **Caption**: Qwen3-Embedding-8B over Qwen3.5-27B captions, 4096-d.
- **Visual**: Florence-2/SigLIP frame-region, 768-d.

## The analysis

Cluster each embedding on the same clips, name every cluster from its members' captions, and compare. That surfaces which scene groups *all* embeddings recover (a property of the data), where they *differ* in emphasis, and how a coarse cluster refines into finer ones. A separate check measures how much the PQ serving index shifts the clustering versus exact vectors.

Each step writes plain files the next one reads, and all paths are passed in. Install the heavy deps with `pip install -e ".[embedding-clustering]"` (adds `faiss-cpu` + `scikit-learn`; the rest — `numpy`, `pandas`, `matplotlib`, `umap-learn`, `boto3`, `decord`, `Pillow` — ship with the base package), and set `PYTHONPATH=$REPO:$REPO/scripts` so `sil_wheel` and `embed_io` import.

```
ingest_raw_embeddings.py    wheel-data embeddings -> one deduped <encoder>.npz
prep.py pool                the clip-ids covered by all three embeddings
cluster_raw.py              cluster each embedding (flat; --hierarchical for a taxonomy)
make_figures.py themes      name each cluster with an LLM (keywords -> one phrase)
make_figures.py <fig>       the figures and tables (see "What each output shows")
preindex_compare.py         exact vs PQ clustering: agreement + compression
```

Two clustering passes feed the figures: a small **k=50 exact** pass per embedding (overlay maps, topic/distinctive tables, drill-down) and a large **k=1000** pass (`run_full_cluster.sh`) for the at-scale overview and the pre/after-index check.

### 1. Ingest and find the shared pool

First produce a Physical AI wheel-data directory with the public getting-started example, which downloads the dataset from HuggingFace and runs the extract steps (Cosmos / caption / Florence-2+SigLIP embeddings + captions DB):

```bash
python examples/getting-started-physical-ai-autonomous-vehicles/setup_physical_ai.py \
    --workdir ./wheel-data-physical-ai --chunks 0-3
```

Then ingest those embeddings into one `<encoder>.npz` (`clip_ids`, `embeddings`) per encoder:

```bash
python ingest_raw_embeddings.py --root ./wheel-data-physical-ai --out ./npz \
    --encoders cosmos caption visual --pool-name pai
```

`ingest_raw_embeddings.py` also writes `pai_clip_ids.json` — the clips covered by **all** requested encoders — which `cluster_raw.py --pool` uses to keep every embedding on the same clips. Set `CAPTIONS_DB=./wheel-data-physical-ai/captions.db` for the topic steps below.

> Internal research dumps (`<root>/<encoder>/physical_ai/{avfoundation,alpamayo}/...`, plus the qwen3_vl / pe_core encoders) ingest the same way with `--layout internal`. The large-scale, after-index passes below (`prep.py pool` + `run_full_cluster.sh`) read a served FAISS index and are internal-only; the public flow uses the exact-vector `ingest -> cluster_raw` path.

### 2. Cluster each embedding (k=50, exact)

```bash
for e in cosmos caption; do
  python cluster_raw.py --npz ./npz/$e.npz --embed $e --k 50 --spherical \
      --captions-db "$CAPTIONS_DB" --clustering-dir ./clustering --run-id k50_$e
done
# visual is region-level and anisotropic: mean-centre before cosine
python cluster_raw.py --npz ./npz/visual.npz --embed visual --k 50 --spherical --center \
    --captions-db "$CAPTIONS_DB" --clustering-dir ./clustering --run-id k50_visual
```

Each run dir gets `centroids.npy`, `cluster_assignments.parquet`, `umap.json`, and `cluster_topics.json` (TF-IDF keywords over ≤50 sampled captions/cluster). Figure scripts expect run-ids `k50_cosmos`, `k50_caption`, `k50_visual`.

### 3. Build the taxonomy (recursive clustering)

```bash
for e in cosmos caption visual; do
  python cluster_raw.py --npz ./npz/$e.npz --embed $e --hierarchical \
      --captions-db "$CAPTIONS_DB" --branching 10 --max-depth 2 \
      $( [ $e = visual ] && echo --center ) --out ./hier/pai_$e
done
```

`--hierarchical` re-clusters each cluster into sub-clusters, writing `hier_topics.json` + `hier_assignments.parquet` (dotted `path` per clip, e.g. `3.7`).

### 4. Name the clusters

```bash
python make_figures.py themes --clustering-dir ./clustering \
    --runs k50_cosmos k50_caption k50_visual --model "$LLM_MODEL" --workers 4
```

Adds a one-phrase `description` per cluster from its keywords. Routes through `LLM_PROVIDER` (`.env`); never hardcode a provider or key.

### 5. Figures and tables

```bash
# per embedding: its UMAP with the 10 most distinct clusters pinned to a clip
for e in cosmos caption visual; do
  python make_figures.py overlay-maps --run-dir ./clustering/k50_$e \
      --select distinct --k 10 --map-only --out figures/overlay_map_$e.png
done
# both comparison tables in one call
python make_figures.py tables --clustering-dir ./clustering --what both \
    --topics-out tables/emb_cluster_topics.tex --distinctive-out tables/emb_distinctive_terms.tex
# at-scale UMAP overview of the k=1000 pass
python prep.py fig-runs --runs ./emb_pools/runs.tsv --clustering-dir ./clustering --out fig_runs.json
python make_figures.py umap-overview --clustering-dir ./clustering --fig-runs fig_runs.json --out figures/emb_cluster_umap_overview.png
# hierarchical drill-down
python make_figures.py hierarchical --clustering-dir ./clustering --hier-base ./hier --npz-dir ./npz --out figures/hier_drilldown_arrows.png
```

### 6. Pre- vs after-index check

```bash
# real production codebook (encoder that ships a Flat index):
python preindex_compare.py --wheel-data-dir "$WHEEL_DATA_DIR" --embed cosmos --out preindex_compare.json
# proxy (self-quantize raw vectors) where no Flat index exists:
python preindex_compare.py --raw-npz ./npz/caption.npz --index-spec IVF4096,PQ256x8 --embed caption --out preindex_compare.json
```

Reports ARI/NMI (partition agreement), compression (bytes/vec exact→PQ), timing, and intrinsic Δ/silhouette; one key per `--embed`.

## What each output shows

| Script | Output | Shows |
| --- | --- | --- |
| `make_figures.py umap-overview` | `emb_cluster_umap_overview.png` | how each embedding lays out the same clips at scale |
| `make_figures.py overlay-maps --map-only` | `overlay_map_{cosmos,caption,visual}.png` | each embedding's UMAP with its 10 most distinct clusters, pinned to a representative clip |
| `make_figures.py tables --what topics` | `emb_cluster_topics.tex` | those clusters' topic phrases per embedding: the scene groups each recovers |
| `make_figures.py tables --what distinctive` | `emb_distinctive_terms.tex` | the terms that most concentrate in each embedding vs the others: what separates them |
| `make_figures.py hierarchical` | `hier_drilldown_arrows.png` | one branch per embedding drilled into finer sub-clusters |
| `preindex_compare.py` | `preindex_compare.json` | how much the PQ index perturbs the partition, and the compression it buys |

Exact figure layout and any paper labels live in the whitepaper source, not here.

## Shared library

`figlib.py` is one dependency-light module shared by every figure generator:

- **Topic lexicon**: category colours + signature word sets, `categorize`, `latex_escape`, and the weighted-log-odds distinctive-terms scoring (`topic_profiles`, `distinctive_terms`). Only each embedding's *signature* vocabulary is coloured; shared road/place words stay neutral.
- **Cluster selection**: `distinct_clusters` (farthest-first over L2-normalised centroids, size-floored) and `dense_xy` (densest-bin anchor for a label on a UMAP blob).

`figlib.use_nvidia_style()` registers NVIDIA Sans and sets the matplotlib palette; it imports matplotlib lazily so `figlib`'s lexicon/selection helpers (and their tests) stay matplotlib-free.

## Tests

```bash
python -m pytest evaluations/embedding_clustering/tests/ -q
```

`test_figlib.py` covers the lexicon, the log-odds ranking, and the cluster-selection helpers. Depends only on `numpy`/`pandas`, not the figure stack.

## Smoke test (synthetic, seconds, no downloads)

`prep.py synthetic` writes a tiny wheel-format dataset that exercises the clustering and figure steps. It verifies the workflow and is **not** a real result.

```bash
O=./synth_starter
python prep.py synthetic --out $O
python prep.py pool --wheel-data-dir $O/wheel-data --out $O/emb_pools
WHEEL_DATA_DIR=$O/wheel-data CAPTIONS_DB=$O/captions.db POOLS_DIR=$O/emb_pools \
  CLUSTER_OUT=$O/clustering POOL=full K=20 bash run_full_cluster.sh
python prep.py fig-runs --runs $O/emb_pools/runs.tsv --clustering-dir $O/clustering --out $O/fig_runs.json
python make_figures.py umap-overview --clustering-dir $O/clustering --fig-runs $O/fig_runs.json --out $O/umap_overview.png
```

## Notes

- **Visual is anisotropic**: pass `--center` to `cluster_raw.py` and `preindex_compare.py` for it; mean-centring recovers a usable cosine gap (Δ ~0.10 → ~0.75).
- **Intrinsic scores are PQ-optimistic**: in `preindex_compare`, read agreement (ARI/NMI), not Δ/silhouette, as the quality axis.
- **Topics overlap by construction**: all embeddings share the same captions, so the themes overlap; the distinctive-terms table is the discriminator.
