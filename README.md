<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

<h1 align="center">
  <img src="sil_wheel/app/static/images/car.png" width="40" alt="SIL-Wheel logo" />
  SIL-Wheel
</h1>
<p align="center"><b>A Multi-Modal Search and Curation Platform for Physical AI</b></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg" alt="Code License" /></a>
  <a href="https://www.python.org/downloads/release/python-3120/"><img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python 3.12" /></a>
  <a href="https://arxiv.org/abs/"><img src="https://img.shields.io/badge/arXiv-TBD-brightgreen.svg" alt="Technical Report" /></a>
</p>

<p align="center">
  <a href="#installation">How to Install</a>
  &nbsp;✦&nbsp;
  <a href="https://research.nvidia.com/labs/sil/projects/sil-wheel-docs/index.html">Detailed Documentation</a>
  &nbsp;✦&nbsp;
  <a href="https://research.nvidia.com/labs/sil/projects/sil-wheel-docs/tutorials.html">Tutorials &amp; Recipes</a>
  &nbsp;✦&nbsp;
  <a href="#citation">Technical Report</a>
</p>

<p align="center">
  <img src="docs/assets/wheel_ui_demo.gif" width="800" alt="SIL-Wheel UI walkthrough" />
</p>

SIL-Wheel is a framework for building searchable, curated video datasets for
Physical AI. With SIL-Wheel, we can discover clips that match specific
criterion, validate retrieval results, construct training and evaluation slices
and analyze model behavior on those slices.

The core idea is to keep search, curation, and evaluation in the same loop.
Users can retrieve clips using captions, embeddings, visual content, ego-motion
patterns, object filters, metadata, and classifier predictions. These signals
can be composed in a single query, so users can express targeted scenarios such
as hard braking at intersections, dense pedestrian scenes, or rare trajectory
patterns. Retrieved clips can then be reviewed, annotated, refined with
classifiers or clustering, exported as curated datasets, or used directly for
slice-based model evaluation.

## Key Features

- **Composable multi-modal search:** Search large video corpora using caption
full-text search, caption embeddings, video embeddings, visual embeddings,
ego-trajectory shape matching, motion-pattern filters, perception-derived
object filters, classifier scores, and metadata constraints. Multiple signals
can be combined in the same query to retrieve clips that satisfy all active
conditions.

- **Interactive validation and annotation:** Review retrieved clips in the web
interface, inspect captions and trajectories, add manual labels, edit temporal
spans, and convert validated results into reusable annotations or curated
slices.

- **Dataset and benchmark slice construction:** Build targeted slices of data for
training, evaluation, or failure analysis purposes. Candidate clips can be expanded
with learned classifiers, filtered based on various criterion,
and exported for downstream applications.

- **Targeted model evaluation:** Evaluate models on curated slices rather than
only on broad aggregate datasets. SIL-Wheel supports slice-level metrics,
leaderboard-style comparisons, per-clip inspection, and human-powered pairwise
preference evaluation workflows.

- **Flexible raw video ingestion:** Process raw video datasets from the local
filesystem, S3 object storage, or the Hugging Face Hub using the same
preparation pipeline. Local and S3 inputs can be individual `.mp4` files or
`.tar` archives, while Hugging Face datasets can be provided as `.tar` or
`.zip` shards.

- **End-to-end preprocessing pipeline:** Scripts under `scripts/` process raw
videos and metadata into the artifacts needed by SIL-Wheel, including
captions, embeddings, metadata tables, search indices, and launch
configurations.See [`docs/data-preparation.md`](docs/data-preparation.md) for the full
data preparation pipeline.

- **Web UI and Python clients:** Use the web interface for browsing, searching,
annotating, curating, and evaluating models, or query SIL-Wheel
programmatically through `WheelClient` and `WheelHTTPClient`. Both interfaces
share the same search composition, ranking, and caching logic.

## Installation

SIL-Wheel runs on Python 3.12. After cloning the repository, the simplest way to
make sure that all dependencies are properly installed is to create the
provided conda environment:


```bash
git clone https://github.com/nv-tlabs/sil-wheel.git
cd sil-wheel
conda env create -f environment.yml
conda activate wheel

python setup.py build_ext --inplace
pip install -e .
```

> [!NOTE]
> The preprocessing stages that extract embeddings depend on the `core` module from [perception_models](https://github.com/facebookresearch/perception_models).
> After you have created the `wheel` conda environment please install it as well as follows:
> 
> ```bash
> pip install --no-deps git+https://github.com/facebookresearch/perception_models.git
> ```
>
> Note that we use `--no-deps` so that the package does not replace the
> versions of dependencies already installed dependencies.

### Optional Setup Steps

#### 1. S3 Access
If you plan to read videos from S3 paths, install and configure the AWS CLI:
```bash
pip install awscli
```

#### 2. Hugging Face Access
Models are pulled from Hugging Face Hub, so make sure that you have it properly authenticated
```bash
pip install --upgrade huggingface_hub
hf_transfer_login # or: huggingface-cli login
```

#### 3. FlashAttention
`flash-attn` is optional. SIL-Wheel works completely fine without it by using the standard PyTorch attention fallback.

> [!WARNING]
> It may not build cleanly with some CUDA and PyTorch combinations, including CUDA 13.0 and PyTorch 2.10.

**Prefer containers?** You can build and run SIL-Wheel from Docker instead of the
conda environment. See [`docker/`](docker/README.md) for the data preparation and
server images.

## Quickstart

The quickest way to try SIL-Wheel is to run one of the example walkthroughs.
Each walkthrough downloads a dataset, runs the preparation pipeline,
builds the search indices, and writes a ready-to-run `config.yaml` with the
available search modalities populated. These examples use the same scripts as
the general data preparation pipelines (see `scripts/`), so they can also serve as references for
preparing your own datasets.

**nuScenes** is the simplest starting point. The public [nuScenes](https://www.nuscenes.org/nuscenes) mini split
contains 10 scenes, requires no account, and can be processed in a few minutes
on a single RTX 4090. Prepare the data, then launch the server from the
`config.yaml` the setup writes:

```bash
python examples/getting-started-nuscenes/setup_nuscenes.py
python scripts/launch_server.py wheel-data/config.yaml
```

**Physical AI Autonomous Vehicles** runs the same pipeline on a slice of
NVIDIA's [Physical AI Autonomous Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
dataset, streamed from the Hugging Face Hub as `.zip` shards. Prepare it, then
launch the server from the `config.yaml` the setup writes:

```bash
python examples/getting-started-physical-ai-autonomous-vehicles/setup_physical_ai.py --max-clips 20
python scripts/launch_server.py wheel-data-physical-ai/config.yaml
```

See the [nuScenes](examples/getting-started-nuscenes/README.md) and
[Physical AI](examples/getting-started-physical-ai-autonomous-vehicles/README.md)
READMEs for prerequisites and the full set of options.

For your own data, prepare it with the same `scripts/`, point a copy of
`config/wheel_launch_dev_server_config.yaml` at your artifacts, and launch:

```bash
python scripts/launch_server.py config/wheel_launch_dev_server_config.yaml
```

Open the bind address printed at startup. This is the `server.bindto` value in
the YAML configuration.

## Python API

SIL-Wheel can also be driven directly from Python through two clients:

- `WheelClient` for local access, when your process can read the datasets,
  indices, metadata, and artifacts referenced in the launch YAML.
- `WheelHTTPClient` for remote access, when you connect to a running SIL-Wheel
  server over HTTP.

### Local Client

Use `WheelClient` when running in the same environment as the indexed
artifacts.

```python
from sil_wheel.client import WheelClient

client = WheelClient.from_config("config/wheel_launch_dev_server_config.yaml")

# Single-modality search via a convenience helper.
result = client.search_caption("hard braking at intersection")
print(len(result), "clips matched")
# first 10 clip_ids
print(result.head(10))

# Compose multiple modalities and filters in one query, fused with RRF.
result = client.search(
    # caption FTS
    search="hard braking at intersection",
    # text->video embedding
    semantic_search_text="hard braking at intersection",
    data_source=["nuscenes"],
    rank_mode="rrf",
)
# per-clip per-modality scores
df = result.as_dataframe()
print(df.head(10))
```

### HTTP Client

Use `WheelHTTPClient` when connecting to an already running SIL-Wheel server. Its
surface is identical to `WheelClient`, except that `result.scores` is empty for
remote results (the server returns clip IDs only), so use `result.clip_ids`
directly.

```python
from sil_wheel.http_client import WheelHTTPClient

client = WheelHTTPClient(
    server_url="http://wheel-host:8012",
    username="alice",
    password="...",
)

result = client.search_caption("hard braking at intersection")
print(result.clip_ids[:10])

# Or copy a URL straight out of the UI's address bar:
result = client.search_from_url(
    "http://wheel-host:8012/?search=hard+braking&data_source=nuscenes"
)
```

Full client surface, search composition, and ranking modes are in the
[online documentation](https://research.nvidia.com/labs/sil/projects/sil-wheel-docs/index.html).

## Documentation

Full user and developer documentation, including guides, tutorials, and the
complete API reference, can be found at
[SIL-Wheel documentation site](https://research.nvidia.com/labs/sil/projects/sil-wheel-docs/index.html).

For quick in-repo references:

| Topic | Reference |
| :--- | :--- |
| Data preparation: video processing, embeddings, metadata, and S3 upload | [`docs/data-preparation.md`](docs/data-preparation.md) |
| Arena: blind model evaluation, manifest format, and Glicko-2 ratings | [`docs/arena.md`](docs/arena.md) |
| Bug reporting: the Google Sheets backed form | [`docs/bug-reporting.md`](docs/bug-reporting.md) |

## Citation

If you use SIL-Wheel in your research, please cite:

```bibtex
@misc{sil-wheel,
  title  = {SIL-Wheel: A Multi-Modal Search and Curation Platform for Physical AI},
  author = {NVIDIA},
  year   = {2026},
  url    = {https://github.com/nv-tlabs/sil-wheel}
}
```

## Contributors

Maintained by the [NVIDIA SIL team](https://research.nvidia.com/labs/sil/).
This project is currently not accepting external contributions.
