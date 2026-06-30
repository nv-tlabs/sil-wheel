# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared SIL-Wheel runtime. Build this first; the server and pipeline images
# build FROM it:
#
#   docker build -f docker/base.Dockerfile -t silwheel:base .
#
# It installs the package (full app runtime) plus the FAISS GPU build. faiss-gpu
# runs CPU-only when no GPU is exposed; pass `--gpus all` at run time to use one.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/sil-wheel:/opt/sil-wheel/scripts \
    PIP_NO_CACHE_DIR=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    CUPY_CACHE_DIR=/tmp/cupy_cache \
    OMP_NUM_THREADS=8

WORKDIR /opt/sil-wheel

# build-essential for native wheels; libgomp1 for the FAISS OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY setup.py README.md LICENSE NOTICE ./
COPY sil_wheel ./sil_wheel
COPY scripts ./scripts

# FAISS (shared by serving + index building) then the package runtime.
RUN pip install "faiss-gpu==1.14.3" \
    && pip install -e .

# sil_wheel.embeddings imports the `core` module from perception_models at
# import time, so the server and extractors need it. --no-deps keeps it from
# downgrading transformers/numpy (matches the conda install in the root README).
# flash-attn is intentionally skipped; the PyTorch attention fallback works.
RUN pip install --no-deps \
        git+https://github.com/facebookresearch/perception_models.git
