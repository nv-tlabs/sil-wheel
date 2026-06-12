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
import pandas as pd

import cluster_select as cs


def test_farthest_first_picks_opposite_then_orthogonal():
    # Four unit directions; seed at +x. The farthest (cosine) is -x, then either
    # +y or -y (both orthogonal). A near-duplicate of +x must come last.
    X = np.array([[1, 0], [0.999, 0.001], [-1, 0], [0, 1]], float)
    chosen = cs.farthest_first(X, seed=0, k=3)
    assert chosen[0] == 0
    assert chosen[1] == 2          # -x is the single farthest point
    assert 1 not in chosen[:3]     # the +x near-duplicate is never preferred


def test_farthest_first_caps_at_n_points():
    X = np.eye(3)
    assert len(cs.farthest_first(X, seed=0, k=10)) == 3


def test_farthest_first_no_repeats():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 8))
    chosen = cs.farthest_first(X, seed=3, k=6)
    assert len(chosen) == len(set(chosen)) == 6


def test_distinct_clusters_uses_centroids_and_size_floor(tmp_path):
    # Cluster 3 is tiny (below the size floor) and should be excluded even though
    # its centroid is the most distinct direction.
    cents = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], float)
    np.save(tmp_path / "centroids.npy", cents)
    sizes = pd.Series({0: 100, 1: 90, 2: 80, 3: 1})  # cluster 3 << mean
    out = cs.distinct_clusters(tmp_path, sizes, k=3, min_frac=0.5)
    assert 3 not in out
    assert out[0] == 0             # seeded with the largest cluster
    assert set(out) <= {0, 1, 2}
    assert len(out) == 3


def test_distinct_clusters_falls_back_when_all_filtered(tmp_path):
    # If the floor would empty the candidate set, fall back to all clusters.
    cents = np.array([[1, 0], [-1, 0]], float)
    np.save(tmp_path / "centroids.npy", cents)
    sizes = pd.Series({0: 1, 1: 1})
    out = cs.distinct_clusters(tmp_path, sizes, k=2, min_frac=10.0)
    assert sorted(out) == [0, 1]


def test_dense_xy_finds_the_cluster_not_the_outlier():
    # 100 points tightly around (0, 0) plus one outlier far away. The label anchor
    # must sit on the dense blob, not be dragged to the outlier.
    xs = [0.0] * 100 + [50.0]
    ys = [0.0] * 100 + [50.0]
    x, y = cs.dense_xy(xs, ys)
    assert abs(x) < 1.0 and abs(y) < 1.0


def test_dense_xy_small_cloud_uses_median():
    x, y = cs.dense_xy([1, 2, 3], [10, 20, 30])
    assert x == 2.0 and y == 20.0
