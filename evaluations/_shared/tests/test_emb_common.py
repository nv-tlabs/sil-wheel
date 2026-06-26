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

import numpy as np

import emb_common


def test_load_npz_roundtrip(tmp_path):
    ids = np.array(["a", "b", "c"], dtype=object)
    X = np.arange(6, dtype=np.float64).reshape(3, 2)
    np.savez(tmp_path / "e.npz", clip_ids=ids, embeddings=X)
    clip_ids, emb = emb_common.load_npz(tmp_path / "e.npz")
    assert clip_ids == ["a", "b", "c"]
    assert emb.dtype == np.float32 and emb.shape == (3, 2)
    np.testing.assert_allclose(emb, X.astype(np.float32))


def test_filter_to_ids_order_and_empty():
    ids = ["a", "b", "c", "d"]
    X = np.arange(8, dtype=np.float32).reshape(4, 2)
    kept, Xf = emb_common.filter_to_ids(ids, X, {"c", "a", "missing"})
    assert kept == ["a", "c"]
    np.testing.assert_allclose(Xf, X[[0, 2]])
    empty, Xe = emb_common.filter_to_ids(ids, X, set())
    assert empty == [] and Xe.shape == (0, 2)


def test_l2_normalize_unit_rows_zero_safe():
    X = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    Xn = emb_common.l2_normalize(X)
    np.testing.assert_allclose(np.linalg.norm(Xn[0]), 1.0, rtol=1e-6)
    assert np.all(np.isfinite(Xn))


def test_mean_center_unit_rows_float32():
    rng = np.random.default_rng(0)
    X = (rng.normal(size=(20, 5)) + 5.0).astype(np.float32)
    Xc = emb_common.mean_center(X)
    np.testing.assert_allclose(np.linalg.norm(Xc, axis=1), 1.0, rtol=1e-5)
    assert Xc.dtype == np.float32 and np.all(np.isfinite(Xc))
