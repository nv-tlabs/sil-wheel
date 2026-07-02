<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Embedding quality

Supervised label probes for the embeddings SIL-Wheel serves. The workflow
measures whether local neighbourhoods and k-means clusters are consistent with
human labels, then renders the paper's per-label embedding-quality table.

## Producing the embeddings from the Physical AI dataset

The encoders come from the public [Physical AI Autonomous Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) dataset. The getting-started example downloads a slice and runs every extractor; `ingest_raw_embeddings.py` (in `evaluations/embedding_clustering`) folds the per-encoder shards into the `<encoder>.npz` files this workflow reads. Needs a CUDA GPU, `ffmpeg` on PATH, and `huggingface-cli login` (the dataset is gated).

```bash
python examples/getting-started-physical-ai-autonomous-vehicles/setup_physical_ai.py \
    --workdir ./wheel-data-physical-ai --chunks 0-3
python evaluations/embedding_clustering/ingest_raw_embeddings.py \
    --root ./wheel-data-physical-ai --out ./embeddings \
    --encoders cosmos caption visual --pool-name pai
```

This writes `embeddings/{cosmos,caption,visual}.npz`, each with the `clip_ids` + `embeddings` arrays described below, so they drop straight into the runner with `--embeddings-dir ./embeddings --embeddings cosmos caption visual`. The remaining paper rows (`qwen3_vl_8b`, `pe_core_g14`, `trajectory`, `random`) come from internal dumps and are not part of the public slice.

Labels are not shipped with the dataset: the public run writes no manual annotations, so supply `labels.csv` (`clip_id,label`) and `negatives.csv` (`clip_id`) yourself, or use the synthetic smoke test below to exercise the workflow end-to-end. The public example also uses smaller query-time models than the paper (Qwen3-Embedding-0.6B captions, SigLIP2-base), so dimensions and absolute numbers differ from the reported table.

## Inputs

Most encoders are one `.npz` file per encoder:

```bash
embeddings/
  cosmos.npz
  qwen3_vl_8b.npz
  pe_core_g14.npz
  caption.npz
  trajectory.npz
  random.npz
```

Every file must contain row-aligned arrays:

```python
clip_ids      # shape (N,), string-like
embeddings    # shape (N, D), numeric
```

Labels are CSVs. Multi-label mode uses `clip_id,label` rows. Negatives use a
single `clip_id` column. The runner scores every encoder on the intersection
of clips found in all loaded embeddings.

### Region encoders (Florence-2 / SigCLIP detections)

Two encoders are aggregated on the fly from the *per-detection* Florence-2 /
SigCLIP archive (one `__full_frame__` row per clip plus one SigLIP2 embedding
per grounded-detection crop), not from a per-clip `.npz`:

- `florence2_sigclip_grounding_balanced_bof` — **Region BoF**, a TF-IDF
  bag-of-features histogram over a visual vocabulary fit on the detections
  (a per-clip vector).
- `florence2_sigclip_grounding_balanced_chamfer_kmedoids` — **Region Chamfer**,
  which treats each clip as the *set* of its detections and clusters natively
  on the symmetric set-to-set Chamfer similarity (K-medoids); no per-clip
  vector is ever formed.

Both read the archive given by `--region-archive` (a glob over the
grounded-balanced `.pkl` shards produced by
`scripts/extract_florence2_sigclip_embeddings.py` with grounding-phrase
detection + label-balanced selection). They are scored on the same clip
intersection as every other encoder, so all rows are directly comparable.

## Environment

Install the optional dependencies before running the paper-parity path:

```bash
pip install -e ".[embedding-quality]"
```

The default runner uses FAISS spherical k-means, matching the paper experiment.
If `faiss-cpu` is unavailable, pass `--no-spherical-kmeans` to use sklearn
KMeans for smoke tests and development checks.

## Run

```bash
python evaluations/embedding_quality/run_embedding_quality.py \
  --labels-csv labels.csv \
  --negative-csv negatives.csv \
  --embeddings-dir embeddings \
  --embeddings cosmos qwen3_vl_8b pe_core_g14 caption \
    florence2_sigclip_grounding_balanced_bof \
    florence2_sigclip_grounding_balanced_chamfer_kmedoids \
    trajectory random \
  --region-archive "/path/to/visual_embeddings_grounding_balanced_benchmark/**/*.pkl" \
  --ks 1 10 \
  --cluster-ks 32 64 \
  --few-shot-n 5 20 \
  --few-shot-trials 20 \
  --output-dir results/embedding_quality
```

Outputs:

```bash
results/embedding_quality/summary.json
results/embedding_quality/<embedding>/numbers.json
```

Render the paper table:

```bash
python evaluations/embedding_quality/build_table.py \
  --summary results/embedding_quality/summary.json \
  --output-stem results/embedding_quality/table
```

This writes `table.csv` and `table_paper.tex`. The paper table keeps the eight
public rows in this order: Cosmos-Embed1, Qwen3-VL-8B, PE-Core-G14, Caption
embedding, Region BoF, Region Chamfer, Trajectory shape, Random Gaussian.

## Smoke test

The synthetic generator creates planted-signal vectors and matching labels. It
is only a workflow check.

```bash
O=./synth_embedding_quality
python evaluations/embedding_quality/make_synthetic.py --out "$O" --n-per-label 12 --n-neg 48
python evaluations/embedding_quality/run_embedding_quality.py \
  --labels-csv "$O/labels.csv" \
  --negative-csv "$O/negatives.csv" \
  --embeddings-dir "$O/embeddings" \
  --ks 1 10 \
  --cluster-ks 4 \
  --few-shot-n 5 \
  --few-shot-trials 5 \
  --no-spherical-kmeans \
  --output-dir "$O/results"
python evaluations/embedding_quality/build_table.py \
  --summary "$O/results/summary.json" \
  --output-stem "$O/results/table" \
  --purity-ks 4 \
  --intrinsic-k 4
```

Drop `--no-spherical-kmeans` when `faiss-cpu` is installed and exact paper
parity matters.

## Caption faceting

Whether the caption embedding can be *reshaped* to cluster better without
re-captioning. The dense caption is split into four facets aligned with the
caption-quality axes (scene / road entities / action / temporal), each embedded
separately and scored on the standard vector path as ordinary `<name>.npz`
encoders. (The related common-component-removal ablation and PC-topics
diagnostic live in `evaluations/embedding_clustering`.)

**1. Decompose captions into the 12-field schema** (text-only, no video/labels).
gpt-oss-20b via the NVIDIA inference API; set `NV_INFERENCE_API_KEYS`
(comma-separated pool) or `NV_INFERENCE_API_KEY`:

```bash
python evaluations/embedding_quality/cap_facet_emb.py segment \
  --captions captions.parquet \      # clip_id + summary/caption column
  --out ./facets/facets.jsonl --workers-per-key 12
```

**2. Build the facet encoders** (embeds facet texts with Qwen3-Embedding-8B;
needs a CUDA GPU):

```bash
python evaluations/embedding_quality/cap_facet_emb.py embed \
  --facets ./facets/facets.jsonl --out-dir ./embeddings
# writes caption_facet_{scene,road_entities,action,temporal}.npz
```

**3. Score them** alongside the monolithic caption (same runner, same labels;
add `caption_pc5` too once built, see embedding_clustering):

```bash
python evaluations/embedding_quality/run_embedding_quality.py \
  --embeddings-dir ./embeddings --output-dir ./out \
  --embeddings caption caption_facet_scene caption_facet_road_entities \
      caption_facet_action caption_facet_temporal
```

## Tests

```bash
python -m pytest evaluations/embedding_quality/tests/ -q
```
