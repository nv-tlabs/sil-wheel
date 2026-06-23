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

from __future__ import annotations

import math

import numpy as np
import pytest

from embeddings_io import load_embeddings
from metrics import cluster_metrics, few_shot_binary_knn, knn_purity


def _two_blob_dataset(
    *, n_pos: int = 80, n_neg: int = 80, dim: int = 16, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    center_pos = np.zeros(dim)
    center_neg = np.zeros(dim)
    center_pos[0] = 3.0
    center_neg[1] = 3.0
    pos = rng.normal(loc=center_pos, scale=0.2, size=(n_pos, dim))
    neg = rng.normal(loc=center_neg, scale=0.2, size=(n_neg, dim))
    X = np.vstack([pos, neg]).astype(np.float32)
    y = np.concatenate(
        [np.ones(n_pos, dtype=np.int8), np.zeros(n_neg, dtype=np.int8)]
    )
    return X, y


def test_knn_purity_high_on_separable_blobs():
    X, y = _two_blob_dataset()
    out = knn_purity(X, y, k_values=[1, 10])
    assert out["nn_purity_k1"] > 0.95
    assert out["nn_purity_k10"] > 0.95
    assert out["n_pos"] == 80
    assert out["n_neg"] == 80


def test_cluster_metrics_sklearn_path():
    X, y = _two_blob_dataset(n_pos=60, n_neg=60)
    out = cluster_metrics(X, y, k_values=[2, 4], spherical=False, seed=0)
    assert out["cluster_purity_k2"] > 0.95
    assert out["nmi_k2"] > 0.8
    assert "intra_sim_k4" in out
    assert "inter_sim_k4" in out


def test_cluster_metrics_rejects_invalid_k():
    X, y = _two_blob_dataset(n_pos=3, n_neg=3)
    with pytest.raises(ValueError):
        cluster_metrics(X, y, k_values=[6], spherical=False)


def test_few_shot_knn_skip_and_determinism():
    X, y = _two_blob_dataset(n_pos=12, n_neg=50)
    out_a = few_shot_binary_knn(X, y, n_values=[5, 20], S=4, seed=7)
    out_b = few_shot_binary_knn(X, y, n_values=[5, 20], S=4, seed=7)
    assert out_a["fewshot_skipped_ns"] == [20]
    assert not math.isnan(out_a["fewshot_acc_n5_mean"])
    assert math.isnan(out_a["fewshot_acc_n20_mean"])
    assert out_a["fewshot_acc_n5_mean"] == out_b["fewshot_acc_n5_mean"]


def test_load_embeddings_filters_and_coerces(tmp_path):
    clip_ids = np.array(["a", "b", "c", "d"], dtype=object)
    X = np.arange(12, dtype=np.float64).reshape(4, 3)
    np.savez(tmp_path / "demo.npz", clip_ids=clip_ids, embeddings=X)

    ids, loaded = load_embeddings(tmp_path, "demo", {"c", "a", "missing"})

    assert ids == ["a", "c"]
    assert loaded.dtype == np.float32
    assert loaded.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(loaded, X[[0, 2]].astype(np.float32))
