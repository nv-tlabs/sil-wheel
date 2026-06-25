# Getting Started: SIL-Wheel + Physical AI Autonomous Vehicles

A single script that takes you from a fresh SIL-Wheel checkout to a
fully-loaded SIL-Wheel UI on a slice of NVIDIA's
[Physical AI Autonomous Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
dataset, with every retrieval modality populated.

It runs the same end-to-end pipeline as the
[nuScenes example](../getting-started-nuscenes/) and reuses the same `scripts/`.
The Physical AI dataset is served from HuggingFace as `.zip` shards; the full
dataset is ~300k clips across 3146 chunks per camera (each camera chunk zip is
~2 GiB), so the script processes **one camera and a handful of chunks** at a time.

## How this differs from the nuScenes example

The Physical AI dataset is not a local download you extract once; it streams
from HuggingFace as per-chunk `.zip` archives. Two small, reusable additions to
the shared code make the existing scripts work with it:

* **`sil_wheel/datasets/base_dataset.py` → `HuggingFaceZipDataset`**: a
  zip-shard analogue of the existing tar-based `HuggingFaceTarDataset`.
  `scripts/prepare_data.py` picks it automatically when a `--hf-repo-id`
  ships `.zip` shards, so video download + compression needs **no** change to
  `prepare_data.py` itself.
* **`scripts/extract_trajectory_stats.py` → `physical_ai` source**: a new
  trajectory source branch (auto-detected from the `.egomotion.offline.parquet`
  filename) that reads the `timestamp` and `x/y/z` columns of the dataset's
  `egomotion.offline` parquet, resamples them onto the camera frame timestamps,
  and reuses the existing speed/acceleration/jerk/curvature math.

## Prerequisites

* The SIL-Wheel conda env from the main README is already created and active
  (`conda activate wheel`). It already includes everything this example needs
  (huggingface_hub, vllm, qwen-vl-utils, …); no extra install step is required.
* Linux, Python 3.12.
* `ffmpeg` and `ffprobe` on `PATH` (`sudo apt install ffmpeg`).
* A CUDA GPU. The pipeline was developed on a single RTX 4090. Skip GPU stages
  individually with `--skip-cosmos / --skip-captions /
  --skip-caption-embeddings / --skip-visual-embeddings` if your GPU is smaller
  or absent (the server still boots; affected modalities just return empty
  results).
* **HuggingFace access to the gated dataset.** The dataset is distributed
  under the NVIDIA Autonomous Vehicle Dataset License:
  1. Visit
     <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles>
     and accept the license terms.
  2. Log in locally so `huggingface_hub` can download on your behalf:
     `huggingface-cli login` (or set `HF_TOKEN`).
* Disk: **each camera chunk zip is ~2 GiB** and lands in your HuggingFace cache
  (`--hf-cache-dir` or `$HF_HOME`), plus a few hundred MB of processed videos,
  embeddings, and indexes per chunk. Budget ~5 GiB of free space per chunk.
* Internet access for the dataset and HuggingFace model downloads on first use
  (Cosmos, Qwen2.5-VL, Qwen3-Embedding, Florence2, SigCLIP2).

## Setup

Run from the SIL-Wheel repository root with the `wheel` conda env active:

```bash
# 1. Run the pipeline on the first chunk of the front-wide camera.
#    The chunk zip (~2 GiB) downloads once; --max-clips keeps the GPU stages
#    quick for a first run. Drop --max-clips to process the whole chunk.
python examples/getting-started-physical-ai-autonomous-vehicles/setup_physical_ai.py \
    --workdir ./wheel-data-physical-ai \
    --camera camera_front_wide_120fov \
    --chunks 0 \
    --max-clips 20 \
    --admin-password admin

# 2. Launch the SIL-Wheel server.
python scripts/launch_server.py wheel-data-physical-ai/config.yaml
```

Open <http://127.0.0.1:8012/> and log in.

### Login credentials

The script creates a single admin user. By default:

| Username | Password |
| -------- | -------- |
| `admin`  | `admin`  |

Override with `--admin-user / --admin-password / --admin-email`. The script
prints the chosen credentials at the end of its run. Re-running the script does
**not** rotate an existing user's password. Delete `users.db` first to change
it.

### Choosing how much data to process

```
--camera CAM        One of the seven cameras (default camera_front_wide_120fov).
                    One camera per run; the forward-wide camera is the closest
                    analogue to nuScenes' CAM_FRONT.
--chunks SPEC       Which chunks to process. Comma-separated indices and/or
                    ranges: "0", "0,1,2", "0-3,7". Each camera chunk zip is
                    ~2 GiB and holds ~100 clips. Default: 0.
--max-clips N       Cap the clips fed to the GPU stages / DB. The full chunk
                    zip is still downloaded (zip members can't be fetched
                    individually), but only N clips are captioned / embedded /
                    indexed. Handy for a quick smoke test.
--hf-cache-dir DIR  Where the raw chunk zips are cached (defaults to $HF_HOME).
```

To scale up, increase `--chunks` (e.g. `--chunks 0-9`) and drop `--max-clips`.
Chunk indices line up across features, so chunk *N* of the camera and chunk *N*
of `egomotion.offline` describe the same clips.

### Useful flags

```
python examples/getting-started-physical-ai-autonomous-vehicles/setup_physical_ai.py --help
  --workdir DIR                    where to write artifacts
  --host HOST --port PORT          bind address baked into config.yaml
  --admin-user / --admin-password / --admin-email
  --cosmos-index-spec SPEC         FAISS index spec for cosmos; default FLAT
  --qwen-model-size {3,7,32,72}    Qwen2.5-VL size for captioning
  --gpu-memory-utilization F       vLLM GPU memory fraction
  --no-enforce-eager               let vLLM capture cudagraphs (faster, more VRAM)
  --max-model-len N                vLLM context window
  --caption-embed-model MODEL      SentenceTransformer for caption embeddings
  --skip-prepare / --skip-cosmos / --skip-captions
  --skip-caption-embeddings / --skip-visual-embeddings / --skip-trajectory
```

Every stage is independently re-runnable; use the `--skip-*` flags to resume
from a partial run.

## Pipeline at a glance

```
setup_physical_ai.py
├── run_prepare_data_hf            scripts/prepare_data.py --hf-repo-id ...
│                                  (HuggingFaceZipDataset → processed_videos/)
├── run_extract_cosmos            cosmos_embed1_448p → cosmos_embeddings/*.parquet
│   └── materialize_cosmos_index   FAISS index files (Flat)
├── run_extract_qwen_captions      Qwen2.5-VL-3B → captions/*.parquet
│   └── load_captions_into_db      FTSCaptionStore.insert_from_dataframe
├── run_extract_caption_embeddings Qwen3-Embedding-0.6B → caption_embeddings/*.parquet
│   └── materialize_caption_embeddings_index   FAISS index files (Flat)
├── run_extract_visual_embeddings  Florence2-base + SigCLIP2 → visual_embeddings/*.pkl
│   └── materialize_visual_embeddings_index    FAISS index files (Flat)
├── download_egomotion             egomotion.offline + camera frame timestamps → egomotion/
│   └── run_extract_trajectories   scripts/extract_trajectory_stats.py (resamples ego→frame times)
│       └── build_trajectory_memmap_and_index  memmap + FAISS (full / 10s / 5s, Flat)
├── init_annotations_db            clips, video_paths, datasets
├── init_users_db                  single admin user
├── write_required_stubs           wm_stats.parquet, clips_to_apis.json, predictions/
└── write_config                   config.yaml the SIL-Wheel server reads
```

## What you get after setup

| Modality | Source | SIL-Wheel store |
| --- | --- | --- |
| Caption full-text search | Qwen2.5-VL captions | `FTSCaptionStore` |
| Cosmos text→video / clip→clip | `cosmos_embed1_448p` | `CosmosEmbeddingsStore` |
| Caption-embedding semantic search | `Qwen3-Embedding-0.6B` | `CaptionEmbeddingsStore` |
| Visual text→region search | Florence2 + SigCLIP2 | `Florence2SigCLIPEmbeddingStore` |
| Trajectory pattern (`hard_braking`, `stop_go`, …) | `egomotion.offline` | `TrajectoryStore` |
| Trajectory shape (clip→clip) | `egomotion.offline` | `TrajectoryStore` |
| HTTP-range video streaming | local files | `LocalFileFetcher` |

The script also creates a single admin user so you can log into the UI.

### On-disk layout

```
wheel-data-physical-ai/
├── processed_videos/<clip_id>.mp4       scripts/prepare_data.py output
├── clip_manifest.json                   per-clip metadata snapshot
├── video_paths_*.txt                    input lists for scripts/extract_*.py
├── cosmos_embeddings/                   parquet shards + FAISS index
├── captions/                            qwen captions parquet
├── caption_embeddings/                  parquet shards + FAISS index
├── visual_embeddings/                   pkl shards + FAISS index
├── egomotion/<clip_id>.egomotion.offline.parquet   ego x/y/z @ ~10 Hz
│   └── <clip_id>.timestamps.parquet                camera frame times @ ~30 Hz
├── trajectory_data/
│   ├── shards/trajectory_data_downsampled_d5_0.safetensors
│   ├── trajectory_data.dat              memmap (all rows concatenated)
│   ├── clip_to_idx.json                 clip → row range
│   └── trajectory_*_p1.index            FAISS index (full / 10s / 5s)
├── annotations.db                       SQLite: clips, video_paths, datasets
├── captions.db                          SQLite: FTS5 caption index
├── users.db                             SQLite: admin user
├── predictions/<origin-stub>.json       opened by ModelsWithMetricsDataStore
├── wm_stats.parquet                     opened by WMStore (empty df)
├── clips_to_apis.json                   parsed by AutolabelsDataStore (empty)
└── config.yaml                          read by scripts/launch_server.py
```

The raw ~2 GiB chunk zips live in your HuggingFace cache (`--hf-cache-dir` /
`$HF_HOME`), not under the workdir.

## What is intentionally not built

* Perception-based search (object class / count / proximity / direction): only
  an empty `wm_stats.parquet` stub is written.
* BEV viewer / metrics filter (no `predictions/` data populated).
* Arena evaluation mode.
* Classifier and cluster search (`classifiers/` and `clustering/` remain empty;
  the search is lazy and just returns nothing).
* Country / driving-side filters: the dataset's `clip_index` carries no
  per-clip geography, so `country` is left blank rather than guessed.

## Troubleshooting

* **`401 / GatedRepoError` on download.** Accept the dataset license at
  <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles> and
  run `huggingface-cli login` (or export `HF_TOKEN`).
* **`ffmpeg` / `ffprobe` not found.** `apt install ffmpeg` (Ubuntu) or
  `brew install ffmpeg` (macOS).
* **`No CUDA GPU visible to PyTorch`.** Re-run with the relevant `--skip-*`
  flags or run on a GPU host.
* **Out of disk.** Each chunk zip is ~2 GiB in the HF cache. Point
  `--hf-cache-dir` at a roomy disk and process fewer `--chunks` at a time.
* **vLLM OOM on a 24 GiB GPU.** Lower `--gpu-memory-utilization` (default 0.7)
  or `--max-model-len` (default 32768).
* **"Address already in use".** Port 8012 is busy; pass `--port 18012`.
* **Re-running.** Every step is idempotent. Mix `--skip-prepare`,
  `--skip-cosmos`, `--skip-captions`, `--skip-caption-embeddings`,
  `--skip-visual-embeddings`, `--skip-trajectory` to resume from any stage.
