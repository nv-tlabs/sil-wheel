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

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import orjson
import pandas as pd


@dataclass
class Split:
    """Thin wrapper holding the gt-annotations per video_id."""
    video_ids: list
    sentences: list


def l2_normalize(matrix, eps=1e-12):
    denom = np.clip(
        np.linalg.norm(matrix, axis=-1, keepdims=True), eps, None
    )
    return matrix / denom


def read_csv(path):
    """Rows of a CSV file as dicts."""
    with open(path) as f:
        return list(csv.DictReader(f))


def read_jsonl(path):
    """Rows of a JSONL file as dicts."""
    return [
        orjson.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def read_parquet(path):
    """Rows of a parquet file as dicts."""
    return pd.read_parquet(path).to_dict("records")


def load_video_embeddings(parquet_path, video_ids):
    """Per-video embeddings (1 row per video), aligned to ``video_ids``."""
    df = pd.read_parquet(parquet_path)
    by_id = dict(zip(df["clip_id"], df["embeddings"]))
    missing = [v for v in video_ids if v not in by_id]
    if missing:
        raise KeyError(
            f"{len(missing)} videos missing from {parquet_path}: "
            f"e.g. {missing[:3]}"
        )
    matrix = np.asarray([by_id[v] for v in video_ids], dtype=np.float32)
    return l2_normalize(matrix)


def load_florence_sigclip_embeddings(pickle_dir, video_ids):
    shards = sorted(Path(pickle_dir).glob("florence2_sigclip2_group_*.pkl"))
    if not shards:
        raise FileNotFoundError(
            f"No florence2_sigclip2_group_*.pkl under {pickle_dir}"
        )
    kept = set(video_ids)
    rows, owners = [], []
    for path in shards:
        with open(path, "rb") as f:
            shard = pickle.load(f)
        for emb, item in zip(shard["embeddings"], shard["items"]):
            if item["clip_id"] in kept:
                rows.append(emb)
                owners.append(item["clip_id"])
    missing = kept - set(owners)
    if missing:
        raise KeyError(
            f"{len(missing)} videos have no Florence/SigLIP2 crops in "
            f"{pickle_dir}: e.g. {sorted(missing)[:3]}"
        )
    return l2_normalize(np.stack(rows).astype(np.float32)), owners


def load_subclip_caption_embeddings(parquet_path, video_ids):
    """Per-sub-clip caption embeddings + the parent video_id of each row."""
    df = pd.read_parquet(parquet_path)
    keep = df[df["clip_id"].isin(set(video_ids))].reset_index(drop=True)
    missing = set(video_ids) - set(keep["clip_id"])
    if missing:
        raise KeyError(
            f"{len(missing)} videos missing from {parquet_path}: "
            f"e.g. {sorted(missing)[:3]}"
        )
    matrix = np.stack(
        [np.asarray(v, dtype=np.float32) for v in keep["embedding"]]
    )
    return l2_normalize(matrix), keep["clip_id"].tolist()


def score_per_video(text_emb, row_emb, row_owners=None, video_ids=None):
    """Return a ``(n_text, n_video)`` cosine-similarity matrix.

    With ``row_owners=None`` there is one row per video and the result is
    plain ``text_emb @ row_emb.T``. Otherwise a video may span several
    rows (e.g. crops): ``row_owners`` gives the video each row belongs to,
    ``video_ids`` fixes the column order, and each video's score is the
    max over its rows.
    """
    assert text_emb.ndim == 2, f"text_emb must be 2D, got {text_emb.shape}"
    assert row_emb.ndim == 2, f"row_emb must be 2D, got {row_emb.shape}"
    assert text_emb.shape[1] == row_emb.shape[1], (
        f"feature dim mismatch: text {text_emb.shape[1]} vs "
        f"row {row_emb.shape[1]}"
    )
    sim = text_emb.astype(np.float32) @ row_emb.T.astype(np.float32)
    if row_owners is None:
        assert sim.shape == (text_emb.shape[0], row_emb.shape[0])
        return sim
    assert len(row_owners) == row_emb.shape[0], (
        f"owners length {len(row_owners)} != row_emb rows {row_emb.shape[0]}"
    )
    assert video_ids is not None, "video_ids required when row_owners is set"
    video_id_to_col = {v: i for i, v in enumerate(video_ids)}
    owner_cols = np.fromiter(
        (video_id_to_col[v] for v in row_owners),
        dtype=np.int64, count=len(row_owners),
    )
    out = np.full(
        (sim.shape[0], len(video_ids)), -np.inf, dtype=np.float32
    )
    for t in range(sim.shape[0]):
        np.maximum.at(out[t], owner_cols, sim[t])
    return out
