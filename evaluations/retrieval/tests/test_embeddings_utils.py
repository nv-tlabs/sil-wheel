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

import pickle

import numpy as np
import pandas as pd
import pytest

from embeddings_utils import (
    load_florence_sigclip_embeddings,
    load_subclip_caption_embeddings,
    score_per_video,
)


def test_per_video_is_plain_matmul():
    text = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    video = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32
    )
    sim = score_per_video(text, video)
    np.testing.assert_allclose(
        sim, np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    )


def test_multirow_max_pools_per_video():
    text = np.array([[1.0, 0.0]], dtype=np.float32)
    rows = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32
    )
    owners = ["v1", "v1", "v2"]
    video_ids = ["v1", "v2"]
    sim = score_per_video(text, rows, owners, video_ids)
    np.testing.assert_allclose(sim, [[1.0, 0.5]])


def test_multirow_picks_best_subclip_per_query():
    text = np.eye(2, dtype=np.float32)
    rows = np.array(
        [[0.1, 0.9], [0.7, 0.3], [0.4, 0.4], [0.6, 0.2]],
        dtype=np.float32,
    )
    owners = ["vA", "vA", "vB", "vB"]
    video_ids = ["vA", "vB"]
    sim = score_per_video(text, rows, owners, video_ids)
    np.testing.assert_allclose(sim, [[0.7, 0.6], [0.9, 0.4]])


def test_caption_loader_raises_on_missing_coverage(tmp_path):
    p = tmp_path / "caption_embeddings_group_0_1.parquet"
    pd.DataFrame(
        {"clip_id": ["v1", "v2"], "embedding": [[1.0, 0.0], [0.0, 1.0]]}
    ).to_parquet(p)
    with pytest.raises(KeyError):
        load_subclip_caption_embeddings(p, ["v1", "v2", "v3"])
    matrix, owners = load_subclip_caption_embeddings(p, ["v1", "v2"])
    assert owners == ["v1", "v2"]
    assert matrix.shape == (2, 2)


def _write_florence_shard(path, clip_ids):
    with open(path, "wb") as f:
        pickle.dump(
            {
                "embeddings": [np.ones(4, dtype=np.float32) for _ in clip_ids],
                "items": [{"clip_id": c} for c in clip_ids],
            },
            f,
        )


def test_florence_loader_raises_on_missing_coverage(tmp_path):
    _write_florence_shard(
        tmp_path / "florence2_sigclip2_group_0_1.pkl", ["v1", "v2"]
    )
    with pytest.raises(KeyError):
        load_florence_sigclip_embeddings(tmp_path, ["v1", "v2", "v3"])
    _, owners = load_florence_sigclip_embeddings(tmp_path, ["v1", "v2"])
    assert set(owners) == {"v1", "v2"}


def test_florence_loader_raises_filenotfound_when_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_florence_sigclip_embeddings(tmp_path, ["v1"])
