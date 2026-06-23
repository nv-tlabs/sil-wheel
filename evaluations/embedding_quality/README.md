<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Embedding quality

Supervised label probes for the embeddings SIL-Wheel serves. The workflow
measures whether local neighbourhoods and k-means clusters are consistent with
human labels, then renders the paper's per-label embedding-quality table.

## Inputs

Each encoder is one `.npz` file:

```bash
embeddings/
  cosmos.npz
  qwen3_vl_8b.npz
  pe_core_g14.npz
  caption.npz
  florence2_sigclip.npz
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
  --embeddings cosmos qwen3_vl_8b pe_core_g14 caption florence2_sigclip trajectory random \
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

This writes `table.csv` and `table_paper.tex`. The paper table keeps the seven
public rows in this order: Cosmos-Embed1, Qwen3-VL-8B, PE-Core-G14, Caption
embedding, Region embeddings, Trajectory shape, Random Gaussian.

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

## Tests

```bash
python -m pytest evaluations/embedding_quality/tests/ -q
```
