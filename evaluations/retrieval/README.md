# Video retrieval benchmark

Measures 1:1 paired text-to-video and video-to-text retrieval for every
embedding modality in SIL-Wheel, both standalone and fused. It reports
Recall@{1, 5, 10} and MedR, writes a Markdown leaderboard and a matching
JSON dump, and can optionally dump per-query T2V failures for inspection.
Fusion covers every cross-family pair and triplet of modalities under
both RRF and z-score. Adding a dataset takes one function.

Supported datasets:

- `msrvtt`: MSR-VTT 1K-A (Yu et al. 2018), the split that CLIP4Clip,
  X-CLIP, and InternVideo2 report on. 1000 (video, caption) pairs.
- `pvdbench`: the 15000-video held-out slice of `facebook/PE-Video`
  (PE-Core paper, Section 2.3 / Appendix B.1.2). 1:1
  (video, human caption) pairs.
- `opendv`: OpenDV driving clips with synthetic captions in three
  lengths (`short`, `medium`, `long`). Two prepared variants ship
  alongside the embeddings, `uniform_1pm` and `diverse_annot`.

The single `BASELINES` registry in `run_benchmark.py` is shared across
every dataset, and so is the `--embeddings-dir` layout (see
[Expected embeddings format](#expected-embeddings-format)). The only
dataset-specific piece is the split loader.

## Adding a dataset

Write a `load_<name>_split(args) -> Split(video_ids, sentences)`
function and register it in the `DATASETS` dict at the top of
`run_benchmark.py`. The scoring, fusion, and embedding-loading code is
shared, so that loader is the only per-dataset logic: read the ground
truth at `--gt-path` with `read_csv`, `read_jsonl`, or `read_parquet`
(or add a reader for a new format) and map its rows to `video_ids` and
`sentences`. Every dataset reads its ground truth from `--gt-path`;
only an extra option (like `opendv`'s `--caption-length`) needs a new
flag.

`Split.video_ids[i]` is the video paired with `Split.sentences[i]`;
Recall@K treats the identity permutation as ground truth, so
`video_ids` must be unique and each id must match a `clip_id` present
in every embeddings file. If your raw ground-truth ids differ from the
embedding ids, normalise them inside the loader. For example, `opendv`
ground truth uses `vid__vid_2460-2480` but the embeddings use
`vid_2460-2480`, so `load_opendv_split` strips the prefix at the first
`__`.

## Modalities

`compute_sim` scores ten modalities: it loads precomputed video
embeddings from `--embeddings-dir` and encodes the query text with each
model's own text tower, cached under `--cache-dir`:

| Modality | Query text encoder | Precomputed embeddings |
| --- | --- | --- |
| `cosmos_embed1_{224,336,448}p` | Cosmos-Embed1 (per resolution) | `<modality>_group_0_1.parquet` |
| `qwen3_vl_embed_{2b,8b}` | Qwen3-VL-Embedding | `<modality>_group_0_1.parquet` |
| `pe_core_{b16_224,l14_336,g14_448}p` | PE-Core (per resolution) | `<modality>_group_0_1.parquet` |
| `florence_sigclip2` | SigLIP2 (`google/siglip2-base-patch16-224`) | per-crop pickles in `florence_sigclip2/`, max-pooled per clip |
| `caption_embedding` | Qwen3-Embedding-8B | `caption_embeddings_group_0_1.parquet` (over Qwen3-VL captions) |

For `florence_sigclip2` the query tower is pinned to SigLIP2
(`google/siglip2-base-patch16-224`) to match the precomputed video
embeddings. SigLIP and SigLIP2 base both emit 768-d vectors, so a
dimension check cannot catch a SigLIP/SigLIP2 mismatch; the checkpoint
must be fixed by name.

Fusion uses **RRF** (Cormack et al. 2009, `1 / (k + rank)`, k=60) and
per-row **z-score sum** of the cosine similarities. Combinations are
restricted to cross-family triples / pairs (e.g., two Cosmos-Embed1
resolutions are not fused with each other).

## Expected embeddings format

Every dataset uses the same `--embeddings-dir` layout, so a directory
of embeddings produced anywhere (i.e. from any data source) can be
benchmarked as long as it matches this contract:

```
<embeddings-dir>/
  cosmos_embed1_{224,336,448}p_group_0_1.parquet
  qwen3_vl_embed_{2b,8b}_group_0_1.parquet
  pe_core_{b16_224,l14_336,g14_448}p_group_0_1.parquet
  caption_embeddings_group_0_1.parquet
  florence_sigclip2/
    florence2_sigclip2_group_*.pkl
```

- **Per-video encoder parquets** (the eight Cosmos-Embed1 / Qwen3-VL /
  PE-Core files): one row per clip, with columns `clip_id` (str) and
  `embeddings` (a 1-D float vector, model-specific dimension). Vectors
  are L2-normalised at load time, so raw or normalised both work.
- **`caption_embeddings_group_0_1.parquet`**: columns `clip_id` (str)
  and `embedding` (a 1-D float vector). One row per clip is the common
  case; if a clip has several rows they are max-pooled into a single
  per-clip score.
- **`florence_sigclip2/florence2_sigclip2_group_*.pkl`**: one or more
  shards under the `florence_sigclip2/` subdirectory. Each shard is a
  pickled dict with `embeddings` (768-d SigLIP2 vectors, one per crop)
  and `items` (dicts carrying at least `clip_id` for the crop's parent
  clip). A clip's score is the max over its crops.

Every embeddings file must cover the split's `video_ids`: a clip
missing from any modality (parquet, caption, or Florence/SigLIP2)
raises at load time rather than scoring that clip silently wrong. A
modality whose file is entirely absent is skipped instead, so a
dataset can omit one (e.g. ship no caption embeddings).

## Running

### MSR-VTT

```bash
python evaluations/retrieval/run_benchmark.py \
    --dataset msrvtt \
    --embeddings-dir /path/to/msrvtt_retrieval_benchmark/msrvtt_embeddings \
    --gt-path        /path/to/msrvtt_retrieval_benchmark/MSRVTT_JSFUSION_test.csv \
    --cache-dir      /path/to/msrvtt_retrieval_benchmark/text_cache \
    --results-json   /path/to/msrvtt_retrieval_benchmark/results_full.json \
    --results-md     /path/to/msrvtt_retrieval_benchmark/results_full.md
```

`--gt-path` is the JSFusion 1K-A split CSV (download from
`huggingface.co/datasets/friedrichor/MSR-VTT`,
`raw_data/MSRVTT_JSFUSION_test.csv`). Encoded test-caption matrices
are cached under `--cache-dir`, keyed by encoder name and the hash of
the input sentence list, so reruns are matmul-only.

### PVD-Bench

```bash
python evaluations/retrieval/run_benchmark.py \
    --dataset pvdbench \
    --embeddings-dir /path/to/pvd_retrieval_benchmark/pvd_benchmark_embeddings \
    --gt-path        /path/to/pvd_retrieval_benchmark/test.parquet \
    --cache-dir      /path/to/pvd_retrieval_benchmark/text_cache \
    --results-json   /path/to/pvd_retrieval_benchmark/results_full.json \
    --results-md     /path/to/pvd_retrieval_benchmark/results_full.md
```

`--gt-path` is the output of
`evaluations/retrieval/extract_pe_video_metadata.py`, which scans the
locally-cached `facebook/PE-Video` test tars and emits one row per
clip with `human_caption`, `category`, FPS, dimensions, etc. The
PVD-Bench split is every row of that parquet whose `human_caption`
is non-null (15000 clips).

### OpenDV

`--gt-path` points at the captions JSONL (one object per clip with
`short`, `medium`, `long` variants), and `--caption-length` selects
which variant becomes the query text:

```bash
python evaluations/retrieval/run_benchmark.py \
    --dataset opendv \
    --embeddings-dir /path/to/opendv_embeddings/uniform_1pm \
    --gt-path        /path/to/opendv_embeddings/uniform_1pm_captions_all_lengths.jsonl \
    --caption-length long \
    --cache-dir      /path/to/opendv_embeddings/text_cache \
    --results-json   /path/to/opendv_embeddings/results/uniform_1pm_long.json \
    --results-md     /path/to/opendv_embeddings/results/uniform_1pm_long.md
```

The two data variants (`uniform_1pm`, `diverse_annot`) crossed with
the three caption lengths give six runs: switch `--embeddings-dir` and
`--gt-path` together per variant, and `--caption-length` per length.

### Dumping failures for qualitative inspection

Add `--failures-json <path>` to write per-modality T2V failures
(queries whose GT rank exceeds `--failures-rank-threshold`, default
10). Each entry has `clip_id`, `gt_caption`, `rank`, `top1_clip_id`,
`top1_caption`.

To also dump failures for a specific fusion combo, add
`--failures-fusion-combo <mod1+mod2+...:RRF|zscore>`. The combo
string is used verbatim as the JSON key, e.g.

```bash
python evaluations/retrieval/run_benchmark.py \
    --dataset opendv \
    --embeddings-dir /path/to/opendv_embeddings/uniform_1pm \
    --gt-path        /path/to/opendv_embeddings/uniform_1pm_captions_all_lengths.jsonl \
    --caption-length long \
    --cache-dir      /path/to/opendv_embeddings/text_cache \
    --failures-json  /path/to/opendv_embeddings/results/failures_uniform_1pm_long.json \
    --failures-fusion-combo cosmos_embed1_448p+caption_embedding:RRF
```
