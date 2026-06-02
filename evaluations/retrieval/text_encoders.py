# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Encode test captions with each model's text tower, cached to .npy."""
import gc
import hashlib
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from embeddings_utils import l2_normalize
from sil_wheel.embeddings.cosmos_embed1 import CosmosEmbed1
from sil_wheel.embeddings.pe_core import PECore
from sil_wheel.embeddings.qwen3_vl_embed import Qwen3VLEmbed
from sil_wheel.embeddings.sentence_transformer_loader import (
    load_sentence_transformer,
)


def _cache_path(cache_dir, name, texts):
    h = hashlib.blake2b(digest_size=12)
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{name}__{h.hexdigest()}.npy"


def _release(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _to_numpy(x):
    return x.detach().cpu().float().numpy() if hasattr(x, "detach") else x


def _encode_per_text(model, texts):
    rows = [_to_numpy(model.get_text_embeddings(t)) for t in texts]
    return np.concatenate(rows, axis=0).astype(np.float32)


def _encode_cosmos(name, texts):
    m = CosmosEmbed1(model_type=name)
    try:
        return l2_normalize(_encode_per_text(m, texts))
    finally:
        _release(m)


def _encode_qwen3_vl(name, texts):
    m = Qwen3VLEmbed(model_type=name)
    try:
        return _encode_per_text(m, texts)  # model normalizes internally
    finally:
        _release(m)


def _encode_pe_core(name, texts):
    m = PECore(model_type=name)
    try:
        return _encode_per_text(m, texts)  # model normalizes internally
    finally:
        _release(m)


def _encode_qwen3_embedding(texts, model_name="Qwen/Qwen3-Embedding-8B"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = load_sentence_transformer(
        model_name, device=device,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    try:
        return m.encode(
            list(texts), batch_size=16, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=True,
        ).astype(np.float32)
    finally:
        _release(m)


def _encode_siglip(texts, model_name="google/siglip2-base-patch16-224"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    m = AutoModel.from_pretrained(model_name).to(device).eval()
    try:
        rows = []
        with torch.no_grad():
            for start in range(0, len(texts), 64):
                inputs = processor(
                    text=list(texts[start:start + 64]),
                    return_tensors="pt", padding="max_length",
                    truncation=True,
                ).to(device)
                rows.append(
                    m.get_text_features(**inputs).cpu().float().numpy()
                )
        return l2_normalize(np.concatenate(rows, axis=0))
    finally:
        _release(m)


def encode_text(encoder, texts, cache_dir):
    """Encode ``texts`` with ``encoder``'s text tower; cached on disk by hash."""
    cache = _cache_path(cache_dir, encoder, texts)
    if cache.exists():
        return np.load(cache)
    if encoder.startswith("cosmos_embed1_"):
        out = _encode_cosmos(encoder, texts)
    elif encoder.startswith("qwen3_vl_embed_"):
        out = _encode_qwen3_vl(encoder, texts)
    elif encoder.startswith("pe_core_"):
        out = _encode_pe_core(encoder, texts)
    elif encoder == "florence_sigclip":
        out = _encode_siglip(texts)
    elif encoder == "caption_embedding":
        out = _encode_qwen3_embedding(texts)
    else:
        raise ValueError(f"unknown encoder: {encoder}")
    np.save(cache, out)
    return out
