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

SIL-Wheel is a framework for searching, curating, and evaluating
large-scale video datasets. It combines multiple search modalities,
including caption full-text search, caption and video embedding
retrieval, visual region search, ego-trajectory shape and pattern
matching, world-model filters, metadata filters, and classifier scores.

Searches in SIL-Wheel are composable: users can combine any number of
modalities and filters in a single query, and SIL-Wheel returns only the
clips that satisfy all active constraints. Retrieved clips can then be
validated, annotated, curated into datasets or benchmark slices, and
used for model evaluation within the same system.

## Key Features

- **Composable multi-modal search.** Combine multiple retrieval signals in a
single query, including caption full-text search, caption embeddings, video
embeddings, visual embeddings, trajectory shape and motion-pattern matching,
classifier scores, perception-based object filters, and metadata filters.

- **Closed-loop data curation.** Move from **search → validation →
curation → evaluation** in a single framework. Retrieve candidate clips,
validate them through manual annotation or VLM Judge, expand and refine slices
using classifiers and clustering, and evaluate models on the resulting curated
subsets.

- **Flexible raw video ingestion.** Run the same preparation pipeline on raw
video from the **local filesystem**, an **S3** object store, or the **Hugging
Face Hub**. Local and S3 inputs are raw `.mp4` files or `.tar` archives; Hugging
Face datasets are `.tar` or `.zip` archives.

- **End-to-end preprocessing pipeline.** Scripts under `scripts/` take
  raw video and metadata all the way to populated search indices, with a
  choice of captioning and embedding backends. See
  [`docs/data-preparation.md`](docs/data-preparation.md) for the full
  data preparation pipeline.

- **Agentic workflows.** A set of skills that any agent (Cursor, Claude
Code, or your own) can load to drive SIL-Wheel's search, data
curation, and evaluation in natural language. See [`agent/`](agent/README.md)
for setup and usage.

- **Web UI and Python clients.** Use the full-featured **Web UI** for
  browsing, annotating, curating, and running model arenas, or query
  SIL-Wheel programmatically through `WheelClient` and `WheelHTTPClient`.
  Both interfaces use the same search composition, ranking, and caching
  logic.

## Installation

SIL-Wheel runs on Python 3.12. Clone the repository and create the conda
environment, which carries the heavy dependencies:

```bash
git clone https://github.com/nv-tlabs/sil-wheel.git
cd sil-wheel
conda env create -f environment.yml
conda activate wheel

python setup.py build_ext --inplace
pip install -e .
```

The embedding and extraction stages import the `core` module from
`perception_models`, so install it too. The `--no-deps` flag keeps it from
downgrading `transformers` or `numpy`:

```bash
pip install --no-deps git+https://github.com/facebookresearch/perception_models.git
```

Two optional extras:

- To stream videos directly from S3, install the AWS CLI with
  `pip install awscli`.
- Models are pulled from the Hugging Face Hub on first use, so authenticate to
  reach gated checkpoints such as Cosmos-Embed1:

  ```bash
  pip install --upgrade huggingface_hub
  hf auth login
  ```

`flash-attn` does not build cleanly against CUDA 13.0 and PyTorch 2.10. We
recommend skipping it; the PyTorch fallback is slightly slower but works fine.

**Prefer containers?** You can build and run SIL-Wheel from Docker instead of the
conda environment. See [`docker/`](docker/README.md) for the data preparation and
server images.

## Quickstart

The quickest way to have a working version of SIL-Wheel is to run one of the example
walkthroughs. Each one downloads a dataset, runs the full preparation pipeline,
builds every index, and starts a server with every search modality populated.
They share the same `scripts/`, so they also serve as worked references for your
own data.

**nuScenes** is the simplest starting point. The public nuScenes mini split
(10 scenes) needs no account and finishes in a few minutes on a single 4090,
plus a one-time model download on the first run.

```bash
python examples/getting-started-nuscenes/setup_nuscenes.py
```

**Physical AI Autonomous Vehicles** runs the same pipeline on a slice of
NVIDIA's [Physical AI Autonomous Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
dataset, streamed from the Hugging Face Hub as `.zip` shards. The dataset is
gated, so accept its license and run `hf auth login` first.

```bash
python examples/getting-started-physical-ai-autonomous-vehicles/setup_physical_ai.py --max-clips 20
```

See the [nuScenes](examples/getting-started-nuscenes/README.md) and
[Physical AI](examples/getting-started-physical-ai-autonomous-vehicles/README.md)
READMEs for prerequisites and the full set of options.

If your artifacts are already prepared, launch a server directly from its
config:

```bash
python scripts/launch_server.py config/wheel_launch_dev_server_config.yaml
```

Open the bind address printed at startup (the `server.bindto` value in the
YAML) in a browser.

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

📚 Full user and developer documentation, including guides, tutorials, and the
complete API reference, lives on the
**[SIL-Wheel documentation site](https://research.nvidia.com/labs/sil/projects/sil-wheel-docs/index.html)**.

For quick in-repo references:

| Topic | Reference |
| :--- | :--- |
| Data preparation: video processing, embeddings, metadata, and S3 upload | [`docs/data-preparation.md`](docs/data-preparation.md) |
| Arena: blind model evaluation, manifest format, and ELO ratings | [`docs/arena.md`](docs/arena.md) |
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
