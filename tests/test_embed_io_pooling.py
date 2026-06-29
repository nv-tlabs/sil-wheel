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

"""Tests for all-rows-per-clip loading and mean pooling (scripts/embed_io.py)."""
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from embed_io import load_clip_to_rows, pool_clip_features  # noqa: E402


def test_load_clip_to_rows_groups_all_rows(tmp_path):
    # rows:      0   1   2   3   4   5
    # clip pos:  a   a   b   c   c   c
    np.save(tmp_path / "visual_clip_ids_t.npy", np.array(["a", "b", "c"], dtype=object))
    np.save(tmp_path / "visual_position_of_row_t.npy", np.array([0, 0, 1, 2, 2, 2]))
    mapping = load_clip_to_rows(str(tmp_path), "visual", "t")
    assert mapping == {"a": [0, 1], "b": [2], "c": [3, 4, 5]}


def test_load_clip_to_rows_wanted_restricts(tmp_path):
    np.save(tmp_path / "visual_clip_ids_t.npy", np.array(["a", "b", "c"], dtype=object))
    np.save(tmp_path / "visual_position_of_row_t.npy", np.array([0, 0, 1, 2, 2, 2]))
    # streamed wanted-path must match the full grouping for the requested clips
    mapping = load_clip_to_rows(str(tmp_path), "visual", "t", wanted=["a", "c"], chunk=4)
    assert mapping == {"a": [0, 1], "c": [3, 4, 5]}


def test_pool_clip_features_means_over_rows():
    d = 4
    vecs = np.array([
        [1, 0, 0, 0],
        [3, 0, 0, 0],   # clip a = mean of rows 0,1 -> [2,0,0,0]
        [0, 4, 0, 0],   # clip b = row 2
    ], dtype=np.float32)
    index = faiss.IndexFlatL2(d)
    index.add(vecs)
    clip_to_rows = {"a": [0, 1], "b": [2]}
    feats = pool_clip_features(index, clip_to_rows, ["a", "b"])
    assert feats.shape == (2, d)
    np.testing.assert_allclose(feats[0], [2, 0, 0, 0], atol=1e-5)
    np.testing.assert_allclose(feats[1], [0, 4, 0, 0], atol=1e-5)


def test_pool_clip_features_aligns_to_clip_id_order():
    d = 2
    index = faiss.IndexFlatL2(d)
    index.add(np.array([[1, 1], [2, 2], [10, 10]], dtype=np.float32))
    feats = pool_clip_features(index, {"x": [2], "y": [0, 1]}, ["y", "x"])
    np.testing.assert_allclose(feats[0], [1.5, 1.5], atol=1e-5)   # y first
    np.testing.assert_allclose(feats[1], [10, 10], atol=1e-5)     # x second
