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

The dataset streams from HuggingFace as per-chunk `.zip` archives rather than
being a local download you extract once. Four additions to the shared code make
the existing scripts work with it:

* **`HuggingFaceZipDataset`** (`sil_wheel/datasets/base_dataset.py`): a zip-shard
  analogue of `HuggingFaceTarDataset`, picked automatically by
  `scripts/prepare_data.py` when a `--hf-repo-id` ships `.zip` shards.
* **`physical_ai` trajectory source** (`scripts/extract_trajectory_stats.py`):
  reads the `egomotion.offline` parquet and resamples it onto the camera frame
  timestamps, reusing the existing speed/acceleration/jerk/curvature math.
* **`scripts/build_bev_data.py`**: turns the obstacle and egomotion labels into
  the per-clip msgpack the BEV viewer reads.
* **`BEVFetcher` prefix** (`scripts/launch_server.py`): an absolute `bev_store`
  prefix is served from local disk, a relative one from S3.

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
  (Cosmos, Qwen3-VL, Qwen3-Embedding, Florence2, SigCLIP2).

## Setup

Run from the SIL-Wheel repository root with the `wheel` conda env active:

```bash
# 1. Run the pipeline on 500 clips from the front-wide camera.
python examples/getting-started-physical-ai-autonomous-vehicles/setup_physical_ai.py \
    --workdir ./wheel-data-physical-ai \
    --camera camera_front_wide_120fov \
    --max-clips 500 \
    --admin-password admin

# 2. Launch the SIL-Wheel server.
python scripts/launch_server.py wheel-data-physical-ai/config.yaml
```

Open <http://127.0.0.1:8012/> and log in.

> [!NOTE]
> `127.0.0.1` only accepts connections from the machine running the server. On a
> remote host pass `--host <its-address>` above, or launch with
> `--override server.bindto=<its-address>:8012`.

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

`--max-clips` and `--chunks` are mutually exclusive; passing both is an error.

```
--max-clips N       Process exactly N clips, downloading chunks from 0 until N
                    are on disk.
--chunks SPEC       Process these chunks in full: "0", "0,1,2", "0-3,7".
                    Default when neither flag is given: chunk 0.
--camera CAM        One of the seven cameras (default camera_front_wide_120fov).
--hf-cache-dir DIR  Where the raw chunk zips are cached (defaults to $HF_HOME).
```

Downloads are whole chunks, ~2 GiB for ~100 clips, so `--max-clips 500` pulls
about five.

### Useful flags

```
python examples/getting-started-physical-ai-autonomous-vehicles/setup_physical_ai.py --help
  --workdir DIR                    where to write artifacts
  --host HOST --port PORT          bind address baked into config.yaml
  --admin-user / --admin-password / --admin-email
  --cosmos-index-spec SPEC         FAISS index spec for cosmos; default FLAT
  --gpu-memory-utilization F       vLLM GPU memory fraction
  --no-enforce-eager               let vLLM capture cudagraphs (faster, more VRAM)
  --max-model-len N                vLLM context window
  --caption-embed-model MODEL      SentenceTransformer for caption embeddings
  --skip-prepare / --skip-cosmos / --skip-captions
  --skip-caption-embeddings / --skip-visual-embeddings / --skip-trajectory
  --skip-bev                       skip building the BEV viewer data
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
├── run_extract_qwen_captions      Qwen3-VL-4B → captions/*.parquet
│   └── load_captions_into_db      FTSCaptionStore.insert_from_dataframe
├── run_extract_caption_embeddings Qwen3-Embedding-0.6B → caption_embeddings/*.parquet
│   └── materialize_caption_embeddings_index   FAISS index files (Flat)
├── run_extract_visual_embeddings  Florence2-base + SigCLIP2 → visual_embeddings/*.pkl
│   └── materialize_visual_embeddings_index    FAISS index files (Flat)
├── download_egomotion             egomotion.offline + camera frame timestamps → egomotion/
│   └── run_extract_trajectories   scripts/extract_trajectory_stats.py (resamples ego→frame times)
│       └── build_trajectory_memmap_and_index  memmap + FAISS (full / 10s / 5s, Flat)
├── download_obstacles             obstacle.offline + vehicle_dimensions → obstacles/
│   └── run_build_bev              scripts/build_bev_data.py → bev_data/, bev_index/
├── fetch_clip_countries           metadata/data_collection.parquet → ISO alpha-2 codes
├── init_annotations_db            clips (incl. country), video_paths, datasets
├── init_users_db                  single admin user
├── write_required_stubs           wm_stats.parquet, clips_to_apis.json, predictions/
└── write_config                   config.yaml the SIL-Wheel server reads
```

## What you get after setup

| Modality | Source | SIL-Wheel store |
| --- | --- | --- |
| Caption full-text search | Qwen3-VL-4B captions | `FTSCaptionStore` |
| Cosmos text→video / clip→clip | `cosmos_embed1_448p` | `CosmosEmbeddingsStore` |
| Caption-embedding semantic search | `Qwen3-Embedding-0.6B` | `CaptionEmbeddingsStore` |
| Visual text→region search | Florence2 + SigCLIP2 | `Florence2SigCLIPEmbeddingStore` |
| Trajectory pattern (`hard_braking`, `stop_go`, …) | `egomotion.offline` | `TrajectoryStore` |
| Trajectory shape (clip→clip) | `egomotion.offline` | `TrajectoryStore` |
| Country / driving-side filter | `metadata/data_collection.parquet` | `SQLiteDataStore` (`clips.country`) |
| BEV viewer (ego + tracked objects) | `labels/obstacle.offline` + `egomotion.offline` | `BEVFetcher` (`bev_data/`) |
| HTTP-range video streaming | local files | `LocalFileFetcher` |

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
├── obstacles/<clip_id>.obstacle.offline.parquet    3D boxes, rig frame
├── bev_data/<clip_id>.msgpack                      BEV frames served by BEVFetcher
├── bev_index/clips_with_bev_set.pkl                backs the "With BEV" filter
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

## What is intentionally not built

* Perception-based search (object class / count / proximity / direction): only
  an empty `wm_stats.parquet` stub is written.
* Model metrics and the leaderboard (no `predictions/` data populated).
* Lane markings and road boundaries in the BEV viewer: the official dataset
  release ships no map geometry, so the BEV shows the ego vehicle and tracked
  objects on a blank ground plane.
* Arena evaluation mode.

## Troubleshooting

* **`401 / GatedRepoError` on download.** Accept the dataset license at
  <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles> and
  run `huggingface-cli login` (or export `HF_TOKEN`).
* **`ffmpeg` / `ffprobe` not found.** `apt install ffmpeg` (Ubuntu) or
  `brew install ffmpeg` (macOS).
* **`No CUDA GPU visible to PyTorch`.** Re-run with the relevant `--skip-*`
  flags or run on a GPU host.
* **Out of disk.** Each chunk zip is ~2 GiB in the HF cache. Point
  `--hf-cache-dir` at a roomy disk and lower `--max-clips` (or process fewer
  `--chunks`) at a time.
* **vLLM OOM on a 24 GiB GPU.** Lower `--gpu-memory-utilization` (default 0.7)
  or `--max-model-len` (default 32768).
* **"Address already in use".** Port 8012 is busy; pass `--port 18012`.
