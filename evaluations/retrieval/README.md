<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Video retrieval benchmark

A 1:1 paired text↔video retrieval driver that plugs in different
datasets through a small module interface. Reports T2V and V2T
Recall@{1, 5, 10} and MedR for every standalone modality, and for
every cross-family RRF and z-score fusion of every pair and triplet
of those modalities. Writes a Markdown leaderboard, a JSON dump of
the same numbers, and optionally a per-modality (and per-fusion)
T2V-failure dump for qualitative inspection.

Supported datasets:

- `msrvtt`: MSR-VTT 1K-A (Yu et al. 2018), the split that CLIP4Clip,
  X-CLIP, and InternVideo2 report on. 1000 (video, caption) pairs.
- `pvdbench`: the 15000-video held-out slice of `facebook/PE-Video`
  (PE-Core paper, Section 2.3 / Appendix B.1.2). 1:1
  (video, human caption) pairs.

The single `MODALITIES` list in `run_benchmark.py` is shared across
both datasets. Adding a third dataset means writing a
`load_<name>_split` function and adding an entry in the `DATASETS`
dict.

## Modalities

| Modality | What `compute_sim` reads |
| --- | --- |
| 8 visual encoders (3× Cosmos-Embed1, 3× PE-Core, 2× Qwen3-VL-Embedding, Florence-2/SigLIP) | one precomputed `<model>_group_0_1.parquet` per encoder under `--embeddings-dir` (Florence/SigLIP is per-crop pickles in a `florence_sigclip/` subdir) |
| `caption_embedding` | precomputed Qwen3-Embedding-8B vectors over Qwen3-VL captions in `caption_embeddings_group_0_1.parquet` |

Fusion uses **RRF** (Cormack et al. 2009, `1 / (k + rank)`, k=60) and
per-row **z-score sum** of the cosine similarities. Combinations are
restricted to cross-family triples / pairs (e.g., two Cosmos-Embed1
resolutions are not fused with each other).

## Running

### MSR-VTT

```bash
python evaluations/retrieval/run_benchmark.py \
    --dataset msrvtt \
    --embeddings-dir /path/to/msrvtt_retrieval_benchmark/msrvtt_embeddings \
    --cache-dir      /path/to/msrvtt_retrieval_benchmark/text_cache \
    --results-json   /path/to/msrvtt_retrieval_benchmark/results_full.json \
    --results-md     /path/to/msrvtt_retrieval_benchmark/results_full.md
```

Required data layout under `--embeddings-dir`:

```
msrvtt_embeddings/
  cosmos_embed1_{224,336,448}p_group_0_1.parquet
  qwen3_vl_embed_{2b,8b}_group_0_1.parquet
  pe_core_{b16_224,l14_336,g14_448}p_group_0_1.parquet
  caption_embeddings_group_0_1.parquet
  florence_sigclip/florence2_sigclip_group_*.pkl
```

The JSFusion 1K-A split CSV is expected at
`<cache-dir>/MSRVTT_JSFUSION_test.csv` (download from
`huggingface.co/datasets/friedrichor/MSR-VTT`,
`raw_data/MSRVTT_JSFUSION_test.csv`). Encoded test-caption matrices
are cached under `--cache-dir` too, keyed by encoder name and the
hash of the input sentence list, so reruns are matmul-only.

### PVD-Bench

```bash
python evaluations/retrieval/run_benchmark.py \
    --dataset pvdbench \
    --embeddings-dir   /path/to/pvd_retrieval_benchmark/pvd_benchmark_embeddings \
    --metadata-parquet /path/to/pvd_retrieval_benchmark/test.parquet \
    --cache-dir        /path/to/pvd_retrieval_benchmark/text_cache \
    --results-json     /path/to/pvd_retrieval_benchmark/results_full.json \
    --results-md       /path/to/pvd_retrieval_benchmark/results_full.md
```

`--metadata-parquet` is the output of
`evaluations/retrieval/extract_pe_video_metadata.py`, which scans the
locally-cached `facebook/PE-Video` test tars and emits one row per
clip with `human_caption`, `category`, FPS, dimensions, etc. The
PVD-Bench split is every row of that parquet whose `human_caption`
is non-null (15000 clips).

### Dumping failures for qualitative inspection

Add `--failures-json <path>` to write per-modality T2V failures
(queries whose GT rank exceeds `--failures-rank-threshold`, default
10). Each entry has `clip_id`, `gt_caption`, `rank`, `top1_clip_id`,
`top1_caption`.

To also dump failures for a specific fusion combo, add
`--failures-fusion-combo <mod1+mod2+...:RRF|zscore>`. The combo
string is used verbatim as the JSON key, e.g.

```bash
... --failures-json failures_full.json \