# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Offline extraction / ingest: turns raw video into a wheel-data dir (Cosmos /
# Qwen captions / caption + visual embeddings / trajectories) that the server
# image then serves. Heaviest tier; pulls vLLM and a CUDA GPU is mandatory.
#
#   docker build -f docker/base.Dockerfile     -t silwheel:base .
#   docker build -f docker/pipeline.Dockerfile -t silwheel:pipeline .
#   docker run --gpus all --rm \
#       -v /path/to/out:/data/out -v $HF_HOME:/data/hf -e HF_HOME=/data/hf \
#       -e HF_TOKEN=hf_xxx \
#       silwheel:pipeline \
#       examples/getting-started-physical-ai-autonomous-vehicles/setup_physical_ai.py \
#       --workdir /data/out/wheel-data-physical-ai --chunks 0-3
#
# The dataset is gated: pass `-e HF_TOKEN=...` or mount a logged-in HF cache.
# ffmpeg/libgl1 back the decord/av/opencv video decode used by the extractors.

FROM silwheel:base

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# The getting-started driver lives under examples/ and shells out to scripts/.
COPY examples ./examples

# vLLM (Qwen captioning) is the one extraction dep not in install_requires.
RUN pip install vllm

ENTRYPOINT ["python"]
CMD ["-c", "print('Run an extractor, e.g. examples/getting-started-physical-ai-autonomous-vehicles/setup_physical_ai.py --help')"]
