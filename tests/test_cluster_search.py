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

"""Tests for ClusterSearch parquet-based cluster lookup."""
import time

import numpy as np
import pandas as pd
import pytest

from sil_wheel.stores.cluster_search import ClusterSearch
from sil_wheel.stores.search_utils import SearchFilters, SearchResults

RUN_ID = "run-001"

# Three clusters, each with 5 clips and distances
_CLUSTER_DATA = {
    "0": {
        "clip_ids": [f"clip-c0-{i:02d}" for i in range(5)],
        "distances": [float(i) * 0.1 for i in range(5)],
    },
    "1": {
        "clip_ids": [f"clip-c1-{i:02d}" for i in range(5)],
        "distances": [float(i) * 0.2 for i in range(5)],
    },
    "2": {
        "clip_ids": [f"clip-c2-{i:02d}" for i in range(5)],
        "distances": [float(i) * 0.3 for i in range(5)],
    },
}


def _all_clip_ids():
    ids = {}
    for cluster in _CLUSTER_DATA.values():
        for cid in cluster["clip_ids"]:
            ids[cid] = SearchResults.default
    return ids


def _make_filters(run_id=None, cluster_id=None, cluster_ids=None):
    f = SearchFilters.from_query({})
    object.__setattr__(f, "cluster_run_id", run_id)
    if cluster_ids is None and cluster_id is not None:
        cluster_ids = [cluster_id]
    object.__setattr__(f, "cluster_ids", cluster_ids or [])
    return f


@pytest.fixture()
def cluster_store(tmp_path):
    run_dir = tmp_path / RUN_ID
    run_dir.mkdir(parents=True)

    rows = [
        {
            "clip_id": clip_id,
            "cluster_id": np.int32(cluster_id),
            "distance": np.float32(distance),
        }
        for cluster_id_str, data in _CLUSTER_DATA.items()
        for cluster_id in [int(cluster_id_str)]
        for clip_id, distance in zip(data["clip_ids"], data["distances"])
    ]
    pd.DataFrame(rows).to_parquet(
        run_dir / "cluster_assignments.parquet", index=False
    )

    return ClusterSearch(str(tmp_path))


class TestClusterSearch:
    def test_search_returns_cluster_members(self, cluster_store):
        """Valid run_id + cluster_id → exactly the 5 clips in that cluster."""
        universe = dict(_all_clip_ids())
        filters = _make_filters(run_id=RUN_ID, cluster_id="0")
        result = cluster_store.search(filters, universe)
        expected = set(_CLUSTER_DATA["0"]["clip_ids"])
        assert set(result.keys()) == expected
        assert len(result) == 5

    def test_search_scores_attached(self, cluster_store):
        """Results have cluster_distance_score set to non-None."""
        universe = dict(_all_clip_ids())
        filters = _make_filters(run_id=RUN_ID, cluster_id="1")
        result = cluster_store.search(filters, universe)
        for clip_id, search_result in result.items():
            assert search_result.cluster_distance_score is not None, (
                f"{clip_id} missing cluster_distance_score"
            )

    def test_search_unknown_cluster(self, cluster_store):
        """Cluster ID not in parquet → results unchanged (universe size)."""
        universe = dict(_all_clip_ids())
        original_size = len(universe)
        filters = _make_filters(run_id=RUN_ID, cluster_id="9999")
        result = cluster_store.search(filters, universe)
        assert len(result) == original_size

    def test_search_no_cluster_filter(self, cluster_store):
        """cluster_run_id=None → results unchanged."""
        universe = dict(_all_clip_ids())
        filters = _make_filters(run_id=None, cluster_id=None)
        result = cluster_store.search(filters, universe)
        assert len(result) == len(universe)

    def test_search_missing_run_dir(self, cluster_store):
        """Non-existent run_id → results unchanged (run directory not found)."""
        universe = dict(_all_clip_ids())
        filters = _make_filters(run_id="__no_such_run__", cluster_id="0")
        result = cluster_store.search(filters, universe)
        assert len(result) == len(universe)

    def test_invalidate_drops_caches(self, cluster_store, tmp_path):
        """invalidate(run_id) makes the next read see updated parquet."""
        cluster_store._load_run(RUN_ID)
        cluster_store._centroid_cache[RUN_ID] = np.zeros((3, 4), dtype=np.float32)
        assert RUN_ID in cluster_store._cache
        assert RUN_ID in cluster_store._centroid_cache

        replacement = pd.DataFrame([
            {"clip_id": "new-clip", "cluster_id": np.int32(0),
             "distance": np.float32(0.0)},
        ])
        replacement.to_parquet(
            tmp_path / RUN_ID / "cluster_assignments.parquet", index=False
        )

        cluster_store.invalidate(RUN_ID)
        assert RUN_ID not in cluster_store._cache
        assert RUN_ID not in cluster_store._centroid_cache

        df = cluster_store._load_run(RUN_ID)
        assert list(df["clip_id"]) == ["new-clip"]

    def test_invalidate_unknown_run_is_noop(self, cluster_store):
        """invalidate on a run that was never cached doesn't raise."""
        cluster_store.invalidate("never-loaded")

    def test_universe_subset(self, cluster_store):
        """Clips in the cluster but absent from the universe are not returned."""
        subset = {
            cid: SearchResults.default
            for cid in _CLUSTER_DATA["0"]["clip_ids"][:3]
        }
        filters = _make_filters(run_id=RUN_ID, cluster_id="0")
        result = cluster_store.search(filters, subset)
        assert set(result.keys()) == set(_CLUSTER_DATA["0"]["clip_ids"][:3])

    def test_distances_preserved(self, cluster_store):
        """cluster_distance_score matches the distances written to parquet."""
        universe = dict(_all_clip_ids())
        filters = _make_filters(run_id=RUN_ID, cluster_id="2")
        result = cluster_store.search(filters, universe)
        expected = dict(
            zip(_CLUSTER_DATA["2"]["clip_ids"], _CLUSTER_DATA["2"]["distances"])
        )
        for clip_id, search_result in result.items():
            assert search_result.cluster_distance_score == pytest.approx(
                expected[clip_id], rel=1e-5
            ), f"Distance mismatch for {clip_id}"

    def test_multi_cluster_returns_union(self, cluster_store):
        """cluster_ids=[0, 1] returns the union of both clusters' clips."""
        universe = dict(_all_clip_ids())
        filters = _make_filters(run_id=RUN_ID, cluster_ids=["0", "1"])
        result = cluster_store.search(filters, universe)
        expected = set(_CLUSTER_DATA["0"]["clip_ids"]) | set(_CLUSTER_DATA["1"]["clip_ids"])
        assert set(result.keys()) == expected

    def test_range_ignored_when_multi_cluster(self, cluster_store):
        """Per-cluster percentile range is a no-op when multiple clusters
        are selected — picking 'closest 30%' has no clean meaning across
        clusters."""
        universe = dict(_all_clip_ids())
        filters = _make_filters(run_id=RUN_ID, cluster_ids=["0", "1"])
        object.__setattr__(filters, "cluster_distance_max", 30.0)
        result = cluster_store.search(filters, universe)
        expected = set(_CLUSTER_DATA["0"]["clip_ids"]) | set(_CLUSTER_DATA["1"]["clip_ids"])
        assert set(result.keys()) == expected

    def test_invert_yields_disjoint_clip_set(self, cluster_store):
        """Picking [0, 38] vs [62, 100] (the JS Invert button's output) yields
        the same number of clips but a *disjoint* clip set — the closest 38%
        and the farthest 38% never overlap."""
        universe = dict(_all_clip_ids())

        f_close = _make_filters(run_id=RUN_ID, cluster_id="2")
        object.__setattr__(f_close, "cluster_distance_max", 38.0)
        close_ids = set(cluster_store.search(f_close, dict(universe)).keys())

        f_far = _make_filters(run_id=RUN_ID, cluster_id="2")
        object.__setattr__(f_far, "cluster_distance_min", 62.0)
        far_ids = set(cluster_store.search(f_far, dict(universe)).keys())

        # Both ranges are 38% of the cluster — same count.
        assert len(close_ids) == len(far_ids)
        # …but clip ids must not overlap.
        assert close_ids.isdisjoint(far_ids)
        # And both should be subsets of the cluster.
        cluster_ids = set(_CLUSTER_DATA["2"]["clip_ids"])
        assert close_ids.issubset(cluster_ids)
        assert far_ids.issubset(cluster_ids)

    def test_distance_range_filter(self, cluster_store):
        """[lo, hi] percentile range slices the cluster by index on sorted distance."""
        universe = dict(_all_clip_ids())
        filters = _make_filters(run_id=RUN_ID, cluster_id="0")
        # Cluster 0 has 5 clips. lo=20, hi=80 → indices [1, 4) → 3 middle clips.
        object.__setattr__(filters, "cluster_distance_min", 20.0)
        object.__setattr__(filters, "cluster_distance_max", 80.0)
        result = cluster_store.search(filters, universe)
        expected = set(_CLUSTER_DATA["0"]["clip_ids"][1:4])
        assert set(result.keys()) == expected

    def test_distance_range_lo_only(self, cluster_store):
        """lo>0 with hi at default keeps the upper portion (outliers)."""
        universe = dict(_all_clip_ids())
        filters = _make_filters(run_id=RUN_ID, cluster_id="0")
        object.__setattr__(filters, "cluster_distance_min", 60.0)
        result = cluster_store.search(filters, universe)
        # n=5, lo=60 → start_idx=3 → keeps indices 3, 4.
        expected = set(_CLUSTER_DATA["0"]["clip_ids"][3:])
        assert set(result.keys()) == expected

    def test_distance_range_hi_only(self, cluster_store):
        """hi<100 with lo at default keeps the lower portion (closest)."""
        universe = dict(_all_clip_ids())
        filters = _make_filters(run_id=RUN_ID, cluster_id="0")
        object.__setattr__(filters, "cluster_distance_max", 40.0)
        result = cluster_store.search(filters, universe)
        # n=5, hi=40 → end_idx=2 → keeps indices 0, 1.
        expected = set(_CLUSTER_DATA["0"]["clip_ids"][:2])
        assert set(result.keys()) == expected

    def test_distance_range_full_range_no_filter(self, cluster_store):
        """[0, 100] is identity — same as no range set."""
        universe = dict(_all_clip_ids())
        filters = _make_filters(run_id=RUN_ID, cluster_id="0")
        object.__setattr__(filters, "cluster_distance_min", 0.0)
        object.__setattr__(filters, "cluster_distance_max", 100.0)
        result = cluster_store.search(filters, universe)
        assert set(result.keys()) == set(_CLUSTER_DATA["0"]["clip_ids"])

    def test_members_returns_full_cluster_sorted(self, cluster_store):
        """members() returns all clips for a cluster, sorted by distance asc."""
        clip_ids, distances = cluster_store.members(RUN_ID, 1)
        expected_ids = _CLUSTER_DATA["1"]["clip_ids"]
        expected_dists = _CLUSTER_DATA["1"]["distances"]
        assert clip_ids == expected_ids
        assert distances == pytest.approx(expected_dists, rel=1e-5)

    def test_members_unknown_cluster(self, cluster_store):
        """members() for a missing cluster_id returns empty lists."""
        clip_ids, distances = cluster_store.members(RUN_ID, 9999)
        assert clip_ids == []
        assert distances == []

    def test_latency(self, cluster_store):
        """Any valid cluster search on the small fixture completes in ≤ 0.2 s."""
        universe = dict(_all_clip_ids())
        filters = _make_filters(run_id=RUN_ID, cluster_id="2")
        t0 = time.perf_counter()
        cluster_store.search(filters, universe)
        elapsed = time.perf_counter() - t0
        assert elapsed <= 0.2, f"Cluster search took {elapsed:.4f}s"

    def test_cluster_for_clips_batch(self, cluster_store):
        """Returns (cluster_id, distance) for every queried clip in the run."""
        clip_ids = (
            _CLUSTER_DATA["0"]["clip_ids"][:2]
            + _CLUSTER_DATA["2"]["clip_ids"][:1]
        )
        result = cluster_store.cluster_for_clips(clip_ids, RUN_ID)
        assert set(result.keys()) == set(clip_ids)
        # Cluster 0 clips
        assert result[_CLUSTER_DATA["0"]["clip_ids"][0]] == (0, pytest.approx(0.0))
        assert result[_CLUSTER_DATA["0"]["clip_ids"][1]] == (0, pytest.approx(0.1))
        # Cluster 2 clip
        assert result[_CLUSTER_DATA["2"]["clip_ids"][0]] == (2, pytest.approx(0.0))

    def test_cluster_for_clips_unknown_clips_omitted(self, cluster_store):
        """Clips not present in the run are silently dropped from the result."""
        result = cluster_store.cluster_for_clips(
            ["unknown-clip-1", _CLUSTER_DATA["1"]["clip_ids"][0]], RUN_ID
        )
        assert "unknown-clip-1" not in result
        assert _CLUSTER_DATA["1"]["clip_ids"][0] in result

    def test_cluster_for_clips_empty_input(self, cluster_store):
        result = cluster_store.cluster_for_clips([], RUN_ID)
        assert result == {}

    def test_cluster_for_clips_missing_run(self, cluster_store):
        """Non-existent run_id → empty dict (no exception)."""
        result = cluster_store.cluster_for_clips(
            _CLUSTER_DATA["0"]["clip_ids"], "__no_such_run__"
        )
        assert result == {}

    def test_centroids_missing_returns_none(self, cluster_store):
        """Runs predating centroid persistence → centroids() returns None."""
        assert cluster_store.centroids(RUN_ID) is None

    def test_closest_clusters_ranks_by_l2_distance(self, tmp_path):
        run_dir = tmp_path / "run-cc"
        run_dir.mkdir()
        pd.DataFrame({
            "clip_id": ["a"], "cluster_id": np.int32([0]), "distance": np.float32([0.0]),
        }).to_parquet(run_dir / "cluster_assignments.parquet", index=False)
        # 4 centroids in 3-D, hand-picked so distances to query are unambiguous.
        centroids = np.array([
            [1.0, 0.0, 0.0],   # cluster 0  → d=1.41 from q
            [0.0, 1.0, 0.0],   # cluster 1  → d=1.41
            [10.0, 10.0, 10.0],# cluster 2  → d=15.7
            [0.5, 0.5, 0.5],   # cluster 3  → d=0.87
        ], dtype=np.float32)
        np.save(run_dir / "centroids.npy", centroids)

        store = ClusterSearch(str(tmp_path))
        q = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        ranked = store.closest_clusters(q, "run-cc", k=4)
        # Cluster 3 is closest; cluster 2 farthest.
        assert [cid for cid, _ in ranked] == [3, 0, 1, 2]
        assert ranked[0][1] == pytest.approx(0.5 * 3 ** 0.5, rel=1e-5)
        assert ranked[-1][1] == pytest.approx(10.0 * 3 ** 0.5, rel=1e-5)

    def test_closest_clusters_k_caps_results(self, tmp_path):
        run_dir = tmp_path / "run-k"
        run_dir.mkdir()
        pd.DataFrame({
            "clip_id": ["a"], "cluster_id": np.int32([0]), "distance": np.float32([0.0]),
        }).to_parquet(run_dir / "cluster_assignments.parquet", index=False)
        np.save(run_dir / "centroids.npy", np.eye(8, dtype=np.float32))

        store = ClusterSearch(str(tmp_path))
        ranked = store.closest_clusters(np.zeros(8, dtype=np.float32), "run-k", k=3)
        assert len(ranked) == 3

    def test_closest_clusters_no_centroids_returns_empty(self, cluster_store):
        """Old runs without centroids.npy → empty list, no exception."""
        ranked = cluster_store.closest_clusters(np.zeros(4), RUN_ID, k=5)
        assert ranked == []

    def test_closest_clusters_dim_mismatch_raises(self, tmp_path):
        run_dir = tmp_path / "run-dim"
        run_dir.mkdir()
        pd.DataFrame({
            "clip_id": ["a"], "cluster_id": np.int32([0]), "distance": np.float32([0.0]),
        }).to_parquet(run_dir / "cluster_assignments.parquet", index=False)
        np.save(run_dir / "centroids.npy", np.eye(4, dtype=np.float32))

        store = ClusterSearch(str(tmp_path))
        with pytest.raises(ValueError, match="does not match centroid dim"):
            store.closest_clusters(np.zeros(8), "run-dim", k=2)

    def test_centroids_loaded_when_present(self, tmp_path):
        """centroids.npy on disk is loaded with correct shape and dtype."""
        run_dir = tmp_path / "run-with-centroids"
        run_dir.mkdir()
        # Minimal parquet so _load_run() doesn't break unrelated calls.
        pd.DataFrame({
            "clip_id": ["a"], "cluster_id": np.int32([0]), "distance": np.float32([0.0]),
        }).to_parquet(run_dir / "cluster_assignments.parquet", index=False)
        centroids = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
        np.save(run_dir / "centroids.npy", centroids)

        store = ClusterSearch(str(tmp_path))
        loaded = store.centroids("run-with-centroids")
        assert loaded is not None
        assert loaded.shape == (4, 8)
        assert loaded.dtype == np.float32
        np.testing.assert_array_equal(loaded, centroids)

        # Cache hit: second call returns the same object.
        assert store.centroids("run-with-centroids") is loaded
