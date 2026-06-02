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

"""Tests for the offline clustering builder: sil_wheel.cluster_build."""
import json
import re

import numpy as np
import pandas as pd

from sil_wheel.cluster_build import build_clustering_run, generate_run_id
from sil_wheel.stores.cluster_search import ClusterSearch


def _toy_embeddings(n_per_cluster=40, dim=8, n_clusters=5, seed=0):
    """N=200 well-separated clusters in 8D with stable clip_ids."""
    rng = np.random.default_rng(seed)
    centers = np.eye(n_clusters, dim, dtype=np.float32) * 10.0
    rows = [c + rng.standard_normal((n_per_cluster, dim)).astype(np.float32) * 0.3
            for c in centers]
    embs = np.vstack(rows)
    clip_ids = [f"clip-{i:04d}" for i in range(len(embs))]
    return embs, clip_ids


def test_build_clustering_run_writes_required_files(tmp_path):
    embs, clip_ids = _toy_embeddings()
    run_dir = build_clustering_run(
        tmp_path, embs, clip_ids, n_clusters=5, run_id="run01",
    )
    assert run_dir == (tmp_path / "run01").resolve()

    expected = {
        "cluster_assignments.parquet",
        "representative_by_cluster.json",
        "centroids.npy",
        "umap.json",
        "metadata.json",
    }
    actual = {p.name for p in run_dir.iterdir()}
    assert expected <= actual

    df = pd.read_parquet(run_dir / "cluster_assignments.parquet")
    assert list(df.columns) == ["clip_id", "cluster_id", "distance"]
    assert pd.api.types.is_string_dtype(df["clip_id"])
    assert np.issubdtype(df["cluster_id"].dtype, np.integer)
    assert np.issubdtype(df["distance"].dtype, np.floating)
    assert len(df) == len(embs)
    assert set(df["clip_id"]) == set(clip_ids)

    with open(run_dir / "representative_by_cluster.json") as f:
        reps = json.load(f)
    assert set(reps.keys()) == {str(i) for i in range(5)}
    assert all("cluster_size" in v for v in reps.values())
    assert sum(v["cluster_size"] for v in reps.values()) == len(embs)

    with open(run_dir / "umap.json") as f:
        umap = json.load(f)
    assert set(umap.keys()) == {"centroids", "clips", "clip_ids", "distances"}

    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)
    assert meta["status"] == "done"
    assert meta["n_clusters"] == 5
    assert meta["n_input_clips"] == len(embs)
    assert meta["embed_type"] == "cosmos"
    assert meta["run_id"] == "run01"

    cs = ClusterSearch(tmp_path)
    centroids = cs.centroids("run01")
    assert centroids.shape == (5, embs.shape[1])
    for cid in range(5):
        members, dists = cs.members("run01", cid)
        assert len(members) == reps[str(cid)]["cluster_size"]
        assert dists == sorted(dists)


def test_build_run_writes_centroids_npy(tmp_path):
    embs, clip_ids = _toy_embeddings(n_per_cluster=20, dim=4, n_clusters=3)
    run_dir = build_clustering_run(
        tmp_path, embs, clip_ids, n_clusters=3, run_id="r2",
    )
    centroids = np.load(run_dir / "centroids.npy")
    assert centroids.shape == (3, 4)
    assert centroids.dtype == np.float32


def test_build_run_skips_topics_without_db(tmp_path):
    embs, clip_ids = _toy_embeddings(n_per_cluster=10, dim=4, n_clusters=2)
    run_dir = build_clustering_run(
        tmp_path, embs, clip_ids, n_clusters=2, run_id="r3",
        captions_db_path=None,
    )
    assert not (run_dir / "cluster_topics.json").exists()


def test_run_id_auto_generated(tmp_path):
    embs, clip_ids = _toy_embeddings(n_per_cluster=10, dim=4, n_clusters=2)
    run_dir = build_clustering_run(tmp_path, embs, clip_ids, n_clusters=2)
    assert re.fullmatch(r"[a-z0-9]{10}", run_dir.name)
    assert (run_dir / "metadata.json").exists()


def test_generate_run_id_format():
    rid = generate_run_id()
    assert re.fullmatch(r"[a-z0-9]{10}", rid)
    assert generate_run_id(length=16) != generate_run_id(length=16)
