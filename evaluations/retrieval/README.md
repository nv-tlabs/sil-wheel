# Video Retrieval Benchmark

Measures paired text-to-video and video-to-text retrieval for every embedding
modality in SIL-Wheel, both standalone and fused. The script reports
Recall@{1, 5, 10} and MedR to an output file, and can optionally write a log of
per-query T2V failures for inspection. Fusion combines embeddings with RRF or
z-score. Adding a dataset takes one function.

Supported datasets:

- `msrvtt`: MSR-VTT 1K-A (Yu et al. 2018), the split that CLIP4Clip,
  X-CLIP, and InternVideo2 report on. 1000 (video, caption) pairs.
- `pvdbench`: the 15000-video held-out slice of `facebook/PE-Video`
  (PE-Core paper, Section 2.3 / Appendix B.1.2).
- `opendv`: OpenDV driving clips with synthetic captions in three
  lengths (`short`, `medium`, `long`).


## Adding a dataset

Create a loader that reads the dataset's ground-truth file and returns its
`(video, caption)` pairs. Once this is done, you can register the loaders
in the `supported_datasets` map  and you are ready to go.

## Modalities

The script scores ten modalities: it loads precomputed video embeddings from
`--embeddings-dir` and encodes the query text with each model's own text encoder.

| Modality | Query text encoder | Precomputed embeddings |
| --- | --- | --- |
| `cosmos_embed1_{224,336,448}p` | Cosmos-Embed1 (per resolution) | `<modality>_group_0_1.parquet` |
| `qwen3_vl_embed_{2b,8b}` | Qwen3-VL-Embedding | `<modality>_group_0_1.parquet` |
| `pe_core_{b16_224,l14_336,g14_448}p` | PE-Core (per resolution) | `<modality>_group_0_1.parquet` |
| `florence_sigclip2` | SigLIP2 (`google/siglip2-base-patch16-224`) | per-crop pickles in `florence_sigclip2/`, max-pooled per clip |
| `caption_embedding` | Qwen3-Embedding-8B | `caption_embeddings_group_0_1.parquet` (over Qwen3-VL captions) |

Fusion uses RRF (`1 / (k + rank)`, k=60) and per-row z-score sum of the cosine
similarities, restricted to cross-family combinations i.e. two Cosmos-Embed1
resolutions are not fused with each other.

## Expected embeddings format

Every dataset uses the same `--embeddings-dir` layout, so embeddings produced
from any data source can be benchmarked as long as they match this contract:

```
<embeddings-dir>/
  cosmos_embed1_{224,336,448}p_group_0_1.parquet
  qwen3_vl_embed_{2b,8b}_group_0_1.parquet
  pe_core_{b16_224,l14_336,g14_448}p_group_0_1.parquet
  caption_embeddings_group_0_1.parquet
  florence_sigclip2/
    florence2_sigclip2_group_*.pkl
```

- **`<modality>_group_0_1.parquet`** (Cosmos-Embed1, Qwen3-VL, PE-Core; one file
  per modality): columns `clip_id` (str) and `embeddings` (1-D float vector),
  L2-normalised at load, so raw or normalised both work.
- **`caption_embeddings_group_0_1.parquet`**: columns `clip_id` (str) and
  `embedding` (1-D float vector); multiple rows per clip are max-pooled.
- **`florence_sigclip2/*.pkl`**: pickled dicts with `embeddings` (768-d SigLIP2
  vectors, one per crop) and `items` (each carrying at least `clip_id`); a
  clip's score is the max over its crops.

A modality whose file is entirely absent is skipped; a clip missing from a
modality that is present raises at load time.

## Running

### MSR-VTT

```bash
python evaluations/retrieval/run_benchmark.py \
    --dataset msrvtt \
    --embeddings-dir /path/to/msrvtt_retrieval_benchmark/msrvtt_embeddings \
    --gt-path        /path/to/msrvtt_retrieval_benchmark/MSRVTT_JSFUSION_test.csv \
    --cache-dir      /path/to/msrvtt_retrieval_benchmark/text_cache \
    --results-md     /path/to/msrvtt_retrieval_benchmark/results_full.md
```

`--gt-path` is the JSFusion 1K-A split CSV from `huggingface.co/datasets/friedrichor/MSR-VTT`.

### PVD-Bench

```bash
python evaluations/retrieval/run_benchmark.py \
    --dataset pvdbench \
    --embeddings-dir /path/to/pvd_retrieval_benchmark/pvd_benchmark_embeddings \
    --gt-path        /path/to/pvd_retrieval_benchmark/test.parquet \
    --cache-dir      /path/to/pvd_retrieval_benchmark/text_cache \
    --results-md     /path/to/pvd_retrieval_benchmark/results_full.md
```

`--gt-path` is the output of `extract_pe_video_metadata.py` (one row per
`facebook/PE-Video` test clip).

### OpenDV

`--gt-path` points at the captions JSONL (one object per clip with `short`,
`medium`, `long`), and `--caption-length` selects the query variant:

```bash
python evaluations/retrieval/run_benchmark.py \
    --dataset opendv \
    --embeddings-dir /path/to/opendv_embeddings/uniform_1pm \
    --gt-path        /path/to/opendv_embeddings/uniform_1pm_captions_all_lengths.jsonl \
    --caption-length long \
    --cache-dir      /path/to/opendv_embeddings/text_cache \
    --results-md     /path/to/opendv_embeddings/results/uniform_1pm_long.md
```

The two variants (`uniform_1pm`, `diverse_annot`) times three caption lengths
give six runs: switch `--embeddings-dir`/`--gt-path` per variant and
`--caption-length` per length.

### Saving failures

Add `--failures-json <path>` to save per-modality T2V failures, i.e. queries
whose ground-truth video is not ranked within the top `--failures-rank-threshold`
results (default 10). Each failure records the `clip_id`, `gt_caption`, `rank`,
`top1_clip_id`, and `top1_caption`.

Only the standalone modalities are saved by default. Pass
`--failures-fusion-combo <mod1+mod2+...:RRF|zscore>` to also save the failures of
one fused combination, e.g. `cosmos_embed1_448p+caption_embedding:RRF`.
