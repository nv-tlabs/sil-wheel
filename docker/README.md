<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Docker images

If you'd rather not set up the conda environment, you can run SIL-Wheel from
Docker instead. Most people need two of these images: `server` to browse and
search a dataset you already prepared, and `pipeline` to build that dataset from
raw video. Both are thin layers on a shared `base`, so you build `base` once and
then whichever you need.

| Image | Build with | What it does |
| :--- | :--- | :--- |
| `silwheel:base` | `docker/base.Dockerfile` | The shared runtime (the package plus FAISS). Build it first; you don't run it directly. |
| `silwheel:server` | `docker/server.Dockerfile` | Starts the SIL-Wheel web UI and search over a prepared dataset. |
| `silwheel:pipeline` | `docker/pipeline.Dockerfile` | Runs the preparation pipeline that turns raw video into a dataset you can serve. |

```bash
docker build -f docker/base.Dockerfile     -t silwheel:base .
docker build -f docker/server.Dockerfile   -t silwheel:server .
docker build -f docker/pipeline.Dockerfile -t silwheel:pipeline .
```

Add `--gpus all` when you run an image so FAISS and the model encoders can use
your GPUs. Everything still works on CPU without it, though the pipeline is only
practical on a GPU. On Kubernetes, request `nvidia.com/gpu: 1` and the device
plugin takes care of the rest.

## Running the server

The server serves a prepared `wheel-data` directory: a `config.yaml` together
with the embedding, caption, and trajectory stores it points to. The example
walkthroughs under [`examples/`](../examples) produce exactly this layout. Mount
the directory, pass its config, and the server listens on the address set in the
config's `bindto`.

```bash
docker run --gpus all --rm -p 8012:8012 \
  -v /path/to/wheel-data-physical-ai:/data/wheel-data:ro \
  silwheel:server /data/wheel-data/config.yaml
```

Open the printed bind address in a browser. The text and image search encoders
use the GPU when one is available and fall back to CPU otherwise. If a dataset
has no embeddings yet the server still starts and the UI comes up; those searches
just return nothing.

## Running the pipeline

The pipeline image runs the getting-started walkthrough end to end. It downloads
a slice of NVIDIA's [Physical AI Autonomous Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
dataset and prepares a `wheel-data` directory (caption and video embeddings,
captions, visual embeddings, trajectories, and a `config.yaml`) that the server
image can then serve.

```bash
docker run --gpus all --rm \
  -v /path/to/out:/data/out \
  -v $HF_HOME:/data/hf -e HF_HOME=/data/hf -e HF_TOKEN=hf_xxx \
  silwheel:pipeline \
  examples/getting-started-physical-ai-autonomous-vehicles/setup_physical_ai.py \
  --workdir /data/out/wheel-data-physical-ai --chunks 0-3
```

The dataset is gated, so accept its license and provide a Hugging Face token with
`-e HF_TOKEN=...`, or mount a cache you have already logged into. The entrypoint
is `python`, so you can run any other extractor under `scripts/` the same way.
When it finishes, point the server image at
`/data/out/wheel-data-physical-ai/config.yaml`.

## How the images fit together

The base image installs the full SIL-Wheel runtime, so the server needs nothing
extra and the pipeline only adds vLLM and ffmpeg for captioning and video
decoding. FAISS comes from the base as `faiss-gpu`. If you'd rather install
SIL-Wheel with pip outside Docker, `pip install -e ".[server]"` (or `.[pipeline]`)
pulls a portable `faiss-cpu` build, and vLLM for the pipeline.
