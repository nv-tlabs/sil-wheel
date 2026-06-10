<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Getting Started: SIL-Wheel + nuScenes

A single script that takes you from a fresh SIL-Wheel checkout to a
fully-loaded SIL-Wheel UI on a public nuScenes split, with every retrieval
modality populated.

It performs the same end-to-end pipeline that SIL-Wheel's
[docs/data-preparation.md](../../docs/data-preparation.md) describes, wired
up for the public nuScenes mini split (10 scenes, no AWS account required)
and serving videos straight off the local filesystem.

## Prerequisites

* The SIL-Wheel conda env from the main README is already created and
  active (`conda activate wheel`).
* Linux, Python 3.12.
* `ffmpeg` and `ffprobe` on `PATH` (`sudo apt install ffmpeg`).
* A CUDA GPU. The pipeline was developed on a single RTX 4090. Skip GPU
  stages individually with `--skip-cosmos / --skip-captions /
  --skip-caption-embeddings / --skip-visual-embeddings` if your GPU is
  smaller, or absent (the server still boots; affected modalities just
  return empty results).
* ~15 GB free disk for v1.0-mini (4 GB nuScenes archive, ~5 GB extracted +
  encoded videos, embeddings + FAISS indexes).
* Internet access for the nuScenes download and HuggingFace model downloads
  on first use (Cosmos, Qwen2.5-VL, Qwen3-Embedding, Florence2, SigCLIP2).

## Setup

Run from the SIL-Wheel repository root with the `wheel` conda env active:

```bash
# 1. Install this example's extras.
conda install -n wheel -c conda-forge ffmpeg -y
pip install --no-deps nuscenes-devkit

# 2. Run the full pipeline. The 10-scene mini split takes a few minutes on a
#    4090 (plus a one-time model download on first run); larger splits take
#    proportionally longer.
python examples/getting-started-nuscenes/setup_nuscenes.py \
    --workdir ./wheel-data \
    --admin-password admin

# 3. Launch the SIL-Wheel server.
python scripts/launch_server.py wheel-data/config.yaml
```

Open <http://127.0.0.1:8012/> and log in.

### Login credentials

The script creates a single admin user. By default:

| Username | Password |
| -------- | -------- |
| `admin`  | `admin`  |

Override with `--admin-user / --admin-password / --admin-email`. The script
prints the chosen credentials at the end of its run. Re-running the script
does **not** rotate an existing user's password. Delete `users.db` first if
you need to change it.

### Useful flags

```
python examples/getting-started-nuscenes/setup_nuscenes.py --help
  --workdir DIR             where to download nuScenes / write artifacts
                            (default: ./wheel-data)
  --version V               v1.0-mini (default, auto-downloads ~4 GiB)
                            v1.0-trainval (~300 GiB, manual download)
                            v1.0-test     (~60 GiB, manual download)
  --host HOST --port PORT   bind address baked into config.yaml
  --admin-user / --admin-password / --admin-email
  --n-encode-workers N      parallel ffmpeg jobs for raw video encoding
  --cosmos-index-spec SPEC  FAISS index spec for cosmos; default FLAT
  --qwen-model-size {3,7,32,72}    Qwen2.5-VL size for captioning
  --gpu-memory-utilization F       vLLM GPU memory fraction
  --no-enforce-eager               let vLLM capture cudagraphs (faster, more VRAM)
  --max-model-len N                vLLM context window
  --caption-embed-model MODEL      SentenceTransformer for caption embeddings
  --skip-download / --skip-encode / --skip-prepare
  --skip-cosmos / --skip-captions / --skip-caption-embeddings
  --skip-visual-embeddings / --skip-trajectory
```

Every stage is independently re-runnable; use the `--skip-*` flags to resume
from a partial run.

### Choosing a nuScenes split

Default `--version v1.0-mini` downloads automatically from a public CDN URL.
For the larger splits the script does **not** download for you (those URLs
require an account + Terms-of-Use acceptance). Workflow:

1. Register at <https://www.nuscenes.org/nuscenes#download>, accept the ToU.
2. Download the metadata + blob `tgz` files for `v1.0-trainval` (10 blob
   archives + `_meta`) or `v1.0-test`.
3. Extract them under `<workdir>/nuscenes/` so `samples/`, `sweeps/`, and
   `v1.0-trainval/` (or `v1.0-test/`) sit at the top level.
4. Re-run the script with `--version v1.0-trainval --skip-download`.

The full trainval split has 850 scenes; expect proportionally longer
encoding / extraction times.

## Pipeline at a glance

```
setup_nuscenes.py
├── ensure_nuscenes_split            v1.0-mini.tgz (~4 GB) auto, others manual
├── encode_raw_videos                ffmpeg concat → raw_videos/<scene>.mp4
├── run_prepare_data                 scripts/prepare_data.py → processed_videos/
├── run_extract_cosmos               cosmos_embed1_448p → cosmos_embeddings/*.parquet
│   └── materialize_cosmos_index     FAISS index files (Flat)
├── run_extract_qwen_captions        Qwen2.5-VL-3B → captions/*.parquet
│   └── load_captions_into_db        FTSCaptionStore.insert_from_dataframe
├── run_extract_caption_embeddings   Qwen3-Embedding-0.6B → caption_embeddings/*.parquet
│   └── materialize_caption_embeddings_index   FAISS index files (Flat)
├── run_extract_visual_embeddings    Florence2-base + SigCLIP2 → visual_embeddings/*.pkl
│   └── materialize_visual_embeddings_index   FAISS index files (Flat)
├── extract_nuscenes_trajectories    nuscenes-devkit ego_pose → safetensors
│   └── build_trajectory_memmap_and_index     memmap + FAISS (full / 10s / 5s, Flat)
├── init_annotations_db              clips, video_paths, datasets
├── init_users_db                    single admin user
├── write_required_stubs             wm_stats.parquet, clips_to_apis.json, predictions/
└── write_config                     config.yaml the SIL-Wheel server reads
```

## What you get after setup

Once the script finishes you have a SIL-Wheel instance with every retrieval
modality populated, alongside all the on-disk artifacts the server reads.

### Search modalities in the UI

| Modality | Source | SIL-Wheel store |
| --- | --- | --- |
| Caption full-text search | Qwen2.5-VL captions | `FTSCaptionStore` |
| Cosmos text→video / clip→clip | `cosmos_embed1_448p` | `CosmosEmbeddingsStore` |
| Caption-embedding semantic search | `Qwen3-Embedding-0.6B` | `CaptionEmbeddingsStore` |
| Visual text→region search | Florence2 + SigCLIP2 | `Florence2SigCLIPEmbeddingStore` |
| Trajectory pattern (`hard_braking`, `stop_go`, …) | nuScenes ego_pose | `TrajectoryStore` |
| Trajectory shape (clip→clip) | nuScenes ego_pose | `TrajectoryStore` |
| Country / data-source filters | metadata in `annotations.db` | `SQLiteDataStore` |
| HTTP-range video streaming | local files | `LocalFileFetcher` |

The script also creates a single admin user so you can log into the UI.

### On-disk layout

```
wheel-data/
├── nuscenes/                            raw nuScenes (samples/, sweeps/, …)
├── raw_videos/<scene>.mp4               ffmpeg-stitched CAM_FRONT @ 12 Hz
├── processed_videos/<scene>.mp4         scripts/prepare_data.py output
├── video_paths_*.txt                    input lists for scripts/extract_*.py
├── clip_manifest.json                   per-scene metadata snapshot
├── cosmos_embeddings/                   parquet shards + FAISS index
├── captions/                            qwen captions parquet
├── caption_embeddings/                  parquet shards + FAISS index
├── visual_embeddings/                   pkl shards + FAISS index
├── trajectory_data/
│   ├── shards/*.safetensors             per-scene (T, 7) tensors
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

`classifiers/`, `clustering/`, and `bev_index/` are referenced from
`config.yaml` but not populated; their stores are lazy and don't require the
directories to exist.

## What is intentionally not built

* Perception-based search (object class / count / proximity / direction) —
  only an empty `wm_stats.parquet` stub is written, so it returns nothing.
* BEV viewer / metrics filter (no `predictions/` data populated).
* Arena evaluation mode.
* Classifier and cluster search (`classifiers/` and `clustering/` directories
  remain empty; the search is lazy and just returns nothing).

## Troubleshooting

* **`ffmpeg` / `ffprobe` not found.** `apt install ffmpeg` (Ubuntu) or
  `brew install ffmpeg` (macOS).
* **`No CUDA GPU visible to PyTorch`.** Re-run with the relevant
  `--skip-*` flags or run on a GPU host.
* **Download fails partway.** The script writes to a `.part` file and
  resumes from where it left off on the next run.
* **vLLM OOM on a 24 GiB GPU.** Lower `--gpu-memory-utilization` (default
  0.7) or `--max-model-len` (default 32768). The default is already tuned
  to not OOM a 4090 alongside ~2 GB held by another process.
* **HuggingFace download stalls.** First launch pulls Cosmos / Qwen2.5-VL /
  Qwen3-Embedding / Florence2 / SigCLIP2 weights. Ensure `huggingface_hub`
  can reach the hub (proxy / token).
* **"Address already in use".** Port 8012 is busy; pass `--port 18012` so
  the generated config binds elsewhere.
* **Re-running.** Every step is idempotent. Mix `--skip-download`,
  `--skip-encode`, `--skip-prepare`, `--skip-cosmos`, `--skip-captions`,
  `--skip-caption-embeddings`, `--skip-visual-embeddings`,
  `--skip-trajectory` to resume from any stage.