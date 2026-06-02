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

from pathlib import Path

import numpy as np
import pandas as pd

from sil_wheel.stores.search_utils import project_starmap
from sil_wheel.stores.time_utils import Timer


class ClusterSearch:
    def __init__(self, clustering_dir):
        self.clustering_dir = Path(clustering_dir)
        self.timers = Timer()
        # Cache: run_id -> DataFrame with columns [clip_id, cluster_id, distance]
        self._cache = {}
        # Cache: run_id -> np.ndarray of shape (n_clusters, feature_dim)
        self._centroid_cache = {}

    def _load_run(self, run_id):
        """Load and cache clip assignments for a run. Returns a DataFrame."""
        if run_id in self._cache:
            return self._cache[run_id]

        parquet_path = self.clustering_dir / run_id / "cluster_assignments.parquet"
        df = pd.read_parquet(parquet_path)

        self._cache[run_id] = df
        return df

    def invalidate(self, run_id):
        """Drop cached state for ``run_id`` so the next call re-reads from disk.

        Called after a run's files are replaced on disk (e.g. by the
        ``/upload_clustering`` endpoint) so stale assignments/centroids
        don't survive the swap.
        """
        self._cache.pop(run_id, None)
        self._centroid_cache.pop(run_id, None)

    def centroids(self, run_id):
        """Return the K-means centroid matrix for a run, or None if missing.

        Older runs predate centroid persistence and will return None; new
        runs save ``centroids.npy`` alongside ``cluster_assignments.parquet``.
        """
        if run_id in self._centroid_cache:
            return self._centroid_cache[run_id]

        path = self.clustering_dir / run_id / "centroids.npy"
        if not path.exists():
            return None
        arr = np.load(path)
        self._centroid_cache[run_id] = arr
        return arr

    def members(self, run_id, cluster_id):
        """Return ([clip_ids], [distances]) for one cluster, sorted ascending."""
        df = self._load_run(run_id)
        cluster_df = df[df["cluster_id"] == int(cluster_id)]
        cluster_df = cluster_df.sort_values("distance")
        return (
            cluster_df["clip_id"].astype(str).tolist(),
            cluster_df["distance"].astype(float).tolist(),
        )

    def closest_clusters(self, query_embedding, run_id, k=10):
        """Rank clusters by L2 distance from `query_embedding` to centroid.

        Returns list of ``(cluster_id, distance)`` sorted by distance ascending.
        Returns ``[]`` if centroids weren't persisted for this run (older
        runs predating centroid persistence).
        """
        centroids = self.centroids(run_id)
        if centroids is None:
            return []
        q = np.asarray(query_embedding, dtype=np.float32).flatten()
        if q.shape[0] != centroids.shape[1]:
            raise ValueError(
                f"query dim {q.shape[0]} does not match centroid dim "
                f"{centroids.shape[1]} for run {run_id}"
            )
        distances = np.linalg.norm(centroids - q, axis=1)
        n = min(k, len(distances))
        top = np.argpartition(distances, n - 1)[:n]
        top = top[np.argsort(distances[top])]
        return [(int(i), float(distances[i])) for i in top]

    def cluster_for_clips(self, clip_ids, run_id):
        """Return ``{clip_id: (cluster_id, distance)}`` for the given clips.

        Clips not present in the run are silently omitted from the result.
        Used by `/videos` to enrich per-clip cards with their cluster
        membership when a clustering run is active.
        """
        if not clip_ids:
            return {}
        run_dir = self.clustering_dir / run_id
        if not run_dir.exists():
            return {}
        df = self._load_run(run_id)
        sub = df[df["clip_id"].isin(clip_ids)]
        return {
            row["clip_id"]: (int(row["cluster_id"]), float(row["distance"]))
            for _, row in sub.iterrows()
        }

    def representatives(self, run_id):
        """Return {cluster_id: clip_id_closest_to_centroid} for this run.

        Relies on the clustering builder's invariant that clips within
        each cluster are sorted by distance-to-centroid ascending, so the
        first row per cluster_id is the representative.
        """
        df = self._load_run(run_id)
        firsts = df.drop_duplicates(subset=["cluster_id"], keep="first")
        return dict(zip(
            firsts["cluster_id"].astype(int),
            firsts["clip_id"].astype(str),
        ))

    def search(self, filters, current_results):
        if filters.cluster_run_id is None or not filters.cluster_ids:
            return current_results

        run_dir = self.clustering_dir / filters.cluster_run_id
        if not run_dir.exists():
            return current_results

        df = self._load_run(filters.cluster_run_id)
        ids = [int(c) for c in filters.cluster_ids]
        cluster_df = df[df["cluster_id"].isin(ids)]

        # Percentile range filter is per-cluster and only meaningful
        # when exactly one cluster is selected — for multi-cluster
        # selection ("closest 38%" across multiple clusters has no
        # clean semantics) we leave the union as-is.
        if len(ids) == 1:
            lo = filters.cluster_distance_min if filters.cluster_distance_min is not None else 0
            hi = filters.cluster_distance_max if filters.cluster_distance_max is not None else 100
            if (lo > 0 or hi < 100) and len(cluster_df) > 0:
                sorted_df = cluster_df.sort_values("distance")
                n = len(sorted_df)
                lo_idx = int(round(n * lo / 100.0))
                hi_idx = int(round(n * hi / 100.0))
                cluster_df = sorted_df.iloc[lo_idx:hi_idx]

        clip_ids = cluster_df["clip_id"].tolist()
        distances = cluster_df["distance"].tolist()

        if not clip_ids:
            return current_results

        self.timers.tic()
        current_results = project_starmap(
            lambda r, s: r.with_cluster_distance_score(s),
            current_results,
            zip(clip_ids, distances),
        )
        print(f"The cluster search took {self.timers.toc()}")
        return current_results
