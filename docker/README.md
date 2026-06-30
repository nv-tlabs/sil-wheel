<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Docker images

If you'd rather not set up the conda environment, you can run SIL-Wheel from
Docker instead. Most people need two of these images: `pipeline` to build a
dataset from raw video, and `server` to browse and search it. Both are thin
layers on a shared `base`, so you build `base` once and then whichever you need.

| Image | Build with | What it does |
| :--- | :--- | :--- |
| `silwheel:base` | `docker/base.Dockerfile` | The shared runtime (the package plus FAISS). Build it first; you don't run it directly. |
| `silwheel:pipeline` | `docker/pipeline.Dockerfile` | Runs the preparation pipeline that turns raw video into a dataset. |
| `silwheel:server` | `docker/server.Dockerfile` | Starts the SIL-Wheel web UI and search over a prepared dataset. |

```bash
docker build -f docker/base.Dockerfile     -t silwheel:base .
docker build -f docker/pipeline.Dockerfile -t silwheel:pipeline .
docker build -f docker/server.Dockerfile   -t silwheel:server .
```

Add `--gpus all` when you run an image so FAISS and the model encoders can use
your GPUs. Everything still works on CPU without it, though the pipeline is only
practical on a GPU. On Kubernetes, request `nvidia.com/gpu: 1` and the device
plugin takes care of the rest.

## How it fits together

The two images run one after the other, not at once. The pipeline is a one-shot
job that prepares a `wheel-data` directory and exits; the server is a long-lived
process that mounts that directory and serves it. The directory on disk is the
handoff between them.

```
  Hugging Face dataset
          │
          ▼   silwheel:pipeline     one-shot job, exits when done
   ┌───────────────┐
   │  wheel-data/  │   embeddings · captions · trajectories · config.yaml
   └───────────────┘
          │
          ▼   silwheel:server       long-lived service
   browser / API   ──▶   search the prepared corpus
```

On Kubernetes this maps to a `Job` for the pipeline and a `Deployment` plus
`Service` for the server, sharing one `PersistentVolumeClaim` (the pipeline
mounts it read-write, the server read-only). On a single host it is the same
idea with a host directory in place of the volume.

One thing to get right: `setup_physical_ai.py` writes **absolute** paths into
`config.yaml`, so the server has to see the data at the same path the pipeline
wrote it to. The simplest way to guarantee that is to mount the same host
directory at the same container path in both steps, as below.

## 1. Build a dataset (pipeline)

The pipeline runs the getting-started walkthrough end to end. It downloads a
slice of NVIDIA's [Physical AI Autonomous Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
dataset and prepares a `wheel-data` directory (caption and video embeddings,
captions, visual embeddings, trajectories, and a `config.yaml`).

```bash
docker run --gpus all --rm \
  -v /srv/wheel:/srv/wheel \
  -v $HF_HOME:/data/hf -e HF_HOME=/data/hf -e HF_TOKEN=hf_xxx \
  silwheel:pipeline \
  examples/getting-started-physical-ai-autonomous-vehicles/setup_physical_ai.py \
  --workdir /srv/wheel/physical-ai --chunks 0-3
```

The dataset is gated, so accept its license and provide a Hugging Face token with
`-e HF_TOKEN=...`, or mount a cache you have already logged into. The entrypoint
is `python`, so you can run any other extractor under `scripts/` the same way.

## 2. Serve it (server)

Point the server at the `config.yaml` the pipeline wrote, mounting the same host
directory at the same path so those absolute paths resolve.

```bash
docker run --gpus all --rm -p 8012:8012 \
  -v /srv/wheel:/srv/wheel:ro \
  silwheel:server /srv/wheel/physical-ai/config.yaml
```

Open the bind address printed at startup in a browser. The text and image search
encoders use the GPU when one is available and fall back to CPU otherwise. If a
dataset has no embeddings yet the server still starts and the UI comes up; those
searches just return nothing.

If you already have a prepared `wheel-data` directory, you can skip step 1 and go
straight here, mounting it at whatever path its `config.yaml` expects.

## Dependencies

The base image installs the full SIL-Wheel runtime, so the server needs nothing
extra and the pipeline only adds vLLM and ffmpeg for captioning and video
decoding. FAISS comes from the base as `faiss-gpu`. If you'd rather install
SIL-Wheel with pip outside Docker, `pip install -e ".[server]"` (or `.[pipeline]`)
pulls a portable `faiss-cpu` build, and vLLM for the pipeline.

The PE-Core encoder is optional and off by default. It needs `perception_models`,
which pip installs from git (a corporate TLS proxy can block that), and the
getting-started flow does not use it. To include it, build the base with the
encoder turned on:

```bash
docker build -f docker/base.Dockerfile --build-arg INSTALL_PECORE=true -t silwheel:base .
```
