<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Embedding-space analysis

How the embeddings SIL-Wheel serves organize a video corpus, and how that structure survives the product-quantized (PQ) index used to serve them at scale. Three embeddings cluster the **same** clips:

- **Cosmos-Embed1** — 768-d video.
- **Caption** — Qwen3-Embedding-8B over Qwen3.5-27B captions, 4096-d.
- **Visual** — Florence-2/SigLIP frame-region, 768-d.

We name every cluster from its members' captions and compare: which scene groups all three recover, where they differ, and how a coarse cluster refines into finer ones. A separate check measures how much the PQ index shifts the clustering versus exact vectors.

Install with `pip install -e ".[embedding-clustering]"` (adds `faiss-cpu` + `scikit-learn`; the rest ship with the base package) and set `PYTHONPATH=$REPO:$REPO/scripts`. Each step writes plain files the next reads:

```
ingest_raw_embeddings.py    wheel-data embeddings -> one deduped <encoder>.npz
prep.py pool                clip-ids covered by all three embeddings
cluster_raw.py              cluster each embedding (flat; --hierarchical for a taxonomy)
**make_figures**.py themes      name each cluster with an LLM (keywords -> one phrase)
make_figures.py <fig>       figures and tables (see Outputs)
preindex_compare.py         exact vs PQ clustering: agreement + compression
```

Two passes feed the figures: a **k=50 exact** pass per embedding (overlay maps, tables, drill-down) and a **k=1000** pass for the at-scale overview and the pre/after-index check.

### Run everything (one command)

`run_embedding_clustering.py` sequences all stages (flat clustering + topics, hierarchical taxonomy, pre/after-index comparison, caption PC ablation, figures) and writes a single `summary.json`. It is the clustering-side counterpart to `embedding_quality/run_embedding_quality.py` and supersedes the old shell driver; a stage whose inputs are missing is skipped rather than failing.

```bash
python run_embedding_clustering.py \
  --npz-dir ./embeddings --pool ./emb_pools/full_clip_ids.json --pool-name full \
  --captions-db ./captions.db --output-dir ./out --k 1000
# subset of stages: --stages flat preindex
```

The per-stage scripts below remain runnable on their own; the runner just calls them in-process.

### 1. Build embeddings from the Physical AI dataset, then ingest

Everything downstream reads one `.npz` per encoder. Produce them from the public [Physical AI Autonomous Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) dataset with the getting-started example, then ingest. Prerequisites: a CUDA GPU (the extract steps run Cosmos-Embed1, Qwen captioning, caption embeddings, and Florence-2/SigLIP2), `ffmpeg`/`ffprobe` on PATH, and `huggingface-cli login` (the dataset is gated).

```bash
# Download a few chunks (~96 clips each) and run every extractor into a wheel-data workdir.
python "$REPO"/examples/getting-started-physical-ai-autonomous-vehicles/setup_physical_ai.py \
    --workdir ./wheel-data-physical-ai --chunks 0-3
# Fold the per-encoder shards into one deduped npz each + the shared-clip pool.
python ingest_raw_embeddings.py --root ./wheel-data-physical-ai --out ./npz \
    --encoders cosmos caption visual --pool-name pai
export CAPTIONS_DB=./wheel-data-physical-ai/captions.db
```

The loader writes `cosmos_embeddings/`, `caption_embeddings/`, `visual_embeddings/` shards plus `captions.db` under the workdir. Ingest turns those into `npz/cosmos.npz`, `npz/caption.npz`, `npz/visual.npz` (each holds `clip_ids` + `embeddings`), `npz/pai_clip_ids.json` (clips covered by every encoder, for `cluster_raw.py --pool`), and `npz/pool_summary.json`. The same npz files feed `evaluations/embedding_quality`.

The public example uses smaller query-time models than the paper (Qwen3-Embedding-0.6B captions, SigLIP2-base, Qwen3-VL-4B captions), so embedding dimensions and absolute numbers differ from the reported runs while the workflow is identical. Internal research dumps use `--layout internal` (adds the `qwen3_vl` and `pe_core` encoders); the after-index passes (`prep.py pool` + `run_embedding_clustering.py`) read a served FAISS index and are internal-only.

### 2. Cluster each embedding (k=50, exact)

```bash
for e in cosmos caption; do
  python cluster_raw.py --npz ./npz/$e.npz --embed $e --k 50 --spherical \
      --captions-db "$CAPTIONS_DB" --clustering-dir ./clustering --run-id k50_$e
done
# visual is anisotropic: mean-centre before cosine
python cluster_raw.py --npz ./npz/visual.npz --embed visual --k 50 --spherical --center \
    --captions-db "$CAPTIONS_DB" --clustering-dir ./clustering --run-id k50_visual
```

Each run dir gets `centroids.npy`, `cluster_assignments.parquet`, `umap.json`, `cluster_topics.json`. Figure scripts expect run-ids `k50_{cosmos,caption,visual}`.

### 3. Taxonomy (recursive clustering)

```bash
for e in cosmos caption visual; do
  python cluster_raw.py --npz ./npz/$e.npz --embed $e --hierarchical \
      --captions-db "$CAPTIONS_DB" --branching 10 --max-depth 2 \
      $( [ $e = visual ] && echo --center ) --out ./hier/pai_$e
done
```

Writes `hier_topics.json` + `hier_assignments.parquet` (dotted `path` per clip, e.g. `3.7`).

### 4. Name the clusters

```bash
python make_figures.py themes --clustering-dir ./clustering \
    --runs k50_cosmos k50_caption k50_visual --model "$LLM_MODEL" --workers 4
```

Adds a one-phrase `description` per cluster. Routes through `LLM_PROVIDER` (`.env`); never hardcode a provider or key.

### 5. Figures and tables

```bash
for e in cosmos caption visual; do
  python make_figures.py overlay-maps --run-dir ./clustering/k50_$e \
      --select distinct --k 10 --map-only --out figures/overlay_map_$e.png
done
python make_figures.py tables --clustering-dir ./clustering --what both \
    --topics-out tables/emb_cluster_topics.tex --distinctive-out tables/emb_distinctive_terms.tex
python prep.py fig-runs --runs ./emb_pools/runs.tsv --clustering-dir ./clustering --out fig_runs.json
python make_figures.py umap-overview --clustering-dir ./clustering --fig-runs fig_runs.json --out figures/emb_cluster_umap_overview.png
python make_figures.py hierarchical --clustering-dir ./clustering --hier-base ./hier --npz-dir ./npz --out figures/hier_drilldown_arrows.png
```

### 6. Pre- vs after-index check

```bash
# real production codebook (encoder that ships a Flat index):
python preindex_compare.py --wheel-data-dir "$WHEEL_DATA_DIR" --embed cosmos --out preindex_compare.json
# proxy (self-quantize raw vectors) where no Flat index exists:
python preindex_compare.py --raw-npz ./npz/caption.npz --index-spec IVF4096,PQ256x8 --embed caption --out preindex_compare.json
```

Reports ARI/NMI (agreement), compression, timing, and intrinsic Δ/silhouette; one key per `--embed`.

## Outputs

| Command | Output | Shows |
| --- | --- | --- |
| `umap-overview` | `emb_cluster_umap_overview.png` | how each embedding lays out the same clips at scale |
| `overlay-maps --map-only` | `overlay_map_{cosmos,caption,visual}.png` | each embedding's UMAP, 10 most distinct clusters pinned to a clip |
| `tables --what topics` | `emb_cluster_topics.tex` | per-embedding topic phrases: the scene groups each recovers |
| `tables --what distinctive` | `emb_distinctive_terms.tex` | terms that most concentrate in each embedding vs the others |
| `hierarchical` | `hier_drilldown_arrows.png` | one branch per embedding drilled into finer sub-clusters |
| `preindex_compare.py` | `preindex_compare.json` | how much PQ perturbs the partition, and the compression it buys |

## Shared code

`figlib.py` backs every figure generator: the topic lexicon (`categorize`, `topic_profiles`, `distinctive_terms`) and cluster selection (`distinct_clusters`, `dense_xy`). `use_nvidia_style()` registers NVIDIA Sans from `NVIDIA_SANS_DIR`, falling back to DejaVu when absent.

## Tests

```bash
python -m pytest evaluations/embedding_clustering/tests/ -q
```

`test_figlib.py` covers the lexicon, log-odds ranking, and cluster selection; `test_ingest.py` covers the npz ingest.

## Smoke test (synthetic, no downloads)

```bash
O=./synth_starter
python prep.py synthetic --out $O
python prep.py pool --wheel-data-dir $O/wheel-data --out $O/emb_pools
python run_embedding_clustering.py --npz-dir $O/wheel-data \
  --pool $O/emb_pools/full_clip_ids.json --pool-name full \
  --captions-db $O/captions.db --output-dir $O/clustering --k 20 --stages flat
python prep.py fig-runs --runs $O/emb_pools/runs.tsv --clustering-dir $O/clustering --out $O/fig_runs.json
python make_figures.py umap-overview --clustering-dir $O/clustering --fig-runs $O/fig_runs.json --out $O/umap_overview.png
```

## Caption PC ablation

Caption clusters look diffuse because a few leading principal components carry
broad, shared scene structure (lighting / road type / weather) that is largely
orthogonal to the target scenarios. `pc_topics.py` shows what those components
encode; `caption_pc_ablation.py` projects them out (fit on the full corpus) and
writes a `caption_pc<r>.npz` encoder that can be clustered / scored like any
other. Removing them sharpens label-relevant structure only slightly.

```bash
# what the leading components encode
python evaluations/embedding_clustering/pc_topics.py \
  --caption-npz ./embeddings/caption.npz --captions captions.parquet \
  --out ./out/pc_topics.json

# build the ablated encoder (then score it via embedding_quality)
python evaluations/embedding_clustering/caption_pc_ablation.py \
  --caption-npz ./embeddings/caption.npz --out-dir ./embeddings --remove-pcs 5
```

## Notes

- **Visual is anisotropic**: pass `--center`; mean-centring recovers the cosine gap (Δ ~0.10 → ~0.75).
- **Intrinsic scores are PQ-optimistic**: read ARI/NMI, not Δ/silhouette, as the quality axis.
- **Topics overlap by construction**: all embeddings share captions; the distinctive-terms table is the discriminator.
