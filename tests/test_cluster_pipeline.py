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

"""Tests for FaissKMeans clustering quality properties."""
import numpy as np
import pytest

from cluster_clips_and_select import FaissKMeans


@pytest.fixture(scope="module")
def clustered_data():
    """200 points in 8D arranged in 4 well-separated clusters."""
    rng = np.random.default_rng(42)
    # Centers are orthogonal and far apart (distance 10) so cluster structure
    # is unambiguous regardless of K or random seed.
    centers = np.eye(4, 8, dtype=np.float32) * 10
    return np.vstack(
        [center + rng.standard_normal((50, 8)).astype(np.float32) * 0.5
         for center in centers]
    )


def _sse(kmeans: FaissKMeans, X: np.ndarray) -> float:
    _, distances = kmeans.predict(X)
    return float(distances.sum())


def _fit(n_clusters: int, X: np.ndarray, **kwargs) -> FaissKMeans:
    km = FaissKMeans(
        feature_dim=X.shape[1],
        n_clusters=n_clusters,
        niter=25,
        nredo=1,
        verbose=False,
        use_gpu=False,
        seed=42,
        **kwargs,
    )
    km.fit(X)
    return km


class TestFaissKMeans:
    def test_more_clusters_lower_sse(self, clustered_data):
        """Increasing K on well-separated data strictly reduces total SSE."""
        X = clustered_data
        prev_sse = None
        for k in [2, 4, 10, 40]:
            sse = _sse(_fit(k, X), X)
            if prev_sse is not None:
                assert sse < prev_sse, (
                    f"SSE did not decrease when K increased to {k}: "
                    f"{sse:.4f} >= {prev_sse:.4f}"
                )
            prev_sse = sse

    def test_n_clusters_equals_n_points_sse_zero(self):
        """When K == N every point becomes its own centroid → SSE ≈ 0."""
        n = 30
        X = np.random.default_rng(0).standard_normal((n, 4)).astype(np.float32)
        km = _fit(n, X)
        assert _sse(km, X) < 1e-2, (
            f"Expected SSE ≈ 0 for K=N={n}, got {_sse(km, X):.6f}"
        )

    def test_centroids_shape(self, clustered_data):
        """Centroids array has shape (n_clusters, feature_dim)."""
        X = clustered_data
        k = 10
        km = _fit(k, X)
        assert km.centroids.shape == (k, X.shape[1])

    def test_assignments_in_range(self, clustered_data):
        """Every cluster assignment is a valid cluster index in [0, n_clusters)."""
        X = clustered_data
        k = 10
        km = _fit(k, X)
        assignments, _ = km.predict(X)
        assert assignments.min() >= 0
        assert assignments.max() < k
        assert len(assignments) == len(X)

    def test_reproducibility(self, clustered_data):
        """Two runs with the same seed produce identical assignments."""
        X = clustered_data
        k = 8
        a1, _ = _fit(k, X).predict(X)
        a2, _ = _fit(k, X).predict(X)
        np.testing.assert_array_equal(a1, a2)
