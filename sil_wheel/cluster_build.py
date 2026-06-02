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
"""Build a clustering run from embeddings."""
import json
import secrets
import string
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.decomposition import PCA

from sil_wheel.stores.cluster_topics import extract_topics_for_run


class FaissKMeans:
    def __init__(
        self,
        feature_dim: int,
        n_clusters: int = 1000,
        niter: int = 25,
        nredo: int = 1,
        verbose: bool = True,
        seed: int = 1234,
        spherical_kmeans: bool = False,
        use_gpu: Optional[bool] = None,
        max_points_per_centroid: int = 256,
    ) -> None:
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        self.use_gpu = bool(use_gpu)

        device = "cuda" if self.use_gpu else "cpu"
        print(f"Running clustering on {device} ....")
        print(
            f"FAISS training cap: {n_clusters} clusters × "
            f"{max_points_per_centroid} max_points_per_centroid = "
            f"{n_clusters * max_points_per_centroid:,} points"
        )

        self.kmeans = faiss.Kmeans(
            d=feature_dim,
            k=n_clusters,
            niter=niter,
            nredo=nredo,
            verbose=verbose,
            seed=seed,
            spherical=spherical_kmeans,
            gpu=self.use_gpu,
            max_points_per_centroid=max_points_per_centroid,
        )

        self.feature_dim = feature_dim
        self.n_clusters = n_clusters
        self.max_points_per_centroid = max_points_per_centroid
        self.seed = seed
        self._centroids = None

    @property
    def centroids(self) -> np.ndarray:
        if self._centroids is None:
            self._centroids = self.kmeans.centroids
        return self._centroids

    def fit(self, X: np.ndarray) -> "FaissKMeans":
        """Compute K-Means clustering."""
        t0 = time.perf_counter()
        n_train = self.n_clusters * self.max_points_per_centroid
        if n_train < len(X):
            rng = np.random.default_rng(self.seed)
            X = X[rng.choice(len(X), n_train, replace=False)]
        X = np.ascontiguousarray(X)
        print(
            f"Running Kmeans clustering using faiss on dataset of shape {X.shape} ...."
        )
        self.kmeans.train(X)
        self._centroids = np.ascontiguousarray(self.kmeans.centroids)
        elapsed = time.perf_counter() - t0
        print(f"Time for clustering (sec): {elapsed:.5f}")
        return self

    def predict(self, X: np.ndarray):
        """Predict nearest centroid and its distance for each row in X."""
        t0 = time.perf_counter()
        n_samples = X.shape[0]

        all_cluster_assignments = np.zeros(n_samples, dtype=np.int64)
        all_distances = np.zeros(n_samples, dtype=np.float32)

        batch_size = 500_000
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = np.ascontiguousarray(X[start_idx:end_idx])
            distances, cluster_assignments = self.kmeans.index.search(X_batch, 1)
            all_cluster_assignments[start_idx:end_idx] = cluster_assignments.squeeze(1)
            all_distances[start_idx:end_idx] = distances.squeeze(1)

        elapsed = time.perf_counter() - t0
        print(f"Time for assigning points (sec): {elapsed:.5f}")
        return all_cluster_assignments, all_distances


def generate_run_id(length: int = 10) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def validate_run_dir(run_dir) -> None:
    """Verify ``run_dir`` is a valid clustering run.

    Raises ``ValueError`` with a specific message on the first failed check.
    Used by ``/upload_clustering`` to reject malformed uploads before they
    are renamed into the live clustering directory.
    """
    run_dir = Path(run_dir)

    parquet_path = run_dir / "cluster_assignments.parquet"
    if not parquet_path.exists():
        raise ValueError("missing cluster_assignments.parquet")
    df = pd.read_parquet(parquet_path)
    if set(df.columns) != {"clip_id", "cluster_id", "distance"}:
        raise ValueError(
            f"cluster_assignments.parquet columns {set(df.columns)} != "
            "{clip_id, cluster_id, distance}"
        )
    if not np.issubdtype(df["cluster_id"].dtype, np.integer):
        raise ValueError("cluster_id column is not integer-typed")
    if not np.issubdtype(df["distance"].dtype, np.floating):
        raise ValueError("distance column is not float-typed")

    reps_path = run_dir / "representative_by_cluster.json"
    if not reps_path.exists():
        raise ValueError("missing representative_by_cluster.json")
    with open(reps_path) as f:
        reps = json.load(f)
    if not isinstance(reps, dict):
        raise ValueError("representative_by_cluster.json is not a JSON object")
    for cid, payload in reps.items():
        if not isinstance(payload, dict) or "cluster_size" not in payload:
            raise ValueError(
                f"representative_by_cluster.json[{cid!r}] missing cluster_size"
            )

    umap_path = run_dir / "umap.json"
    if not umap_path.exists():
        raise ValueError("missing umap.json")
    with open(umap_path) as f:
        umap_data = json.load(f)
    if set(umap_data.keys()) != {"centroids", "clips", "clip_ids", "distances"}:
        raise ValueError(
            f"umap.json keys {set(umap_data.keys())} != "
            "{centroids, clips, clip_ids, distances}"
        )

    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError("missing metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    for key in ("embed_type", "n_clusters", "n_input_clips"):
        if key not in metadata:
            raise ValueError(f"metadata.json missing key: {key}")


def write_cluster_assignments(run_dir, cluster_assignments, distances, clip_ids, n_clusters):
    """Write representative_by_cluster.json + cluster_assignments.parquet.

    The parquet rows are sorted by distance ascending within each cluster,
    so ``ClusterSearch.representatives()`` (first-row-per-cluster) Just Works.
    """
    run_dir = Path(run_dir)
    clusters = {}
    all_clip_ids = []
    all_cluster_ids = []
    all_distances = []
    for cid in range(n_clusters):
        idx = np.where(cluster_assignments == cid)[0]
        cluster_size = int(idx.size)
        clusters[str(cid)] = {"cluster_size": cluster_size}
        if cluster_size == 0:
            continue
        idx_sorted = idx[np.argsort(distances[idx], kind="mergesort")]
        all_clip_ids.extend(clip_ids[i] for i in idx_sorted.tolist())
        all_cluster_ids.append(np.full(len(idx_sorted), cid, dtype=np.int32))
        all_distances.append(distances[idx_sorted])

    with open(run_dir / "representative_by_cluster.json", "w") as f:
        json.dump(clusters, f)

    pd.DataFrame({
        "clip_id": all_clip_ids,
        "cluster_id": np.concatenate(all_cluster_ids) if all_cluster_ids else np.array([], dtype=np.int32),
        "distance": np.concatenate(all_distances) if all_distances else np.array([], dtype=np.float32),
    }).to_parquet(run_dir / "cluster_assignments.parquet", index=False)


def write_centroids(run_dir, centroids):
    np.save(Path(run_dir) / "centroids.npy", centroids)


def fit_and_write_umap(
    run_dir,
    n_clusters,
    centroids,
    sub_embeddings,
    sub_cluster_assignments,
    sub_clip_ids,
    sub_distances,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    pca_dim: int = 128,
):
    """Fit PCA→UMAP on np.vstack([centroids, sub_embeddings]) and write umap.json.

    Skipped (writes an empty-shaped JSON) when n_clusters < 2; fitting UMAP on
    a single point doesn't help.
    """
    run_dir = Path(run_dir)
    K = n_clusters
    umap_data = {
        "centroids": {},
        "clips":     {str(cid): [] for cid in range(K)},
        "clip_ids":  {str(cid): [] for cid in range(K)},
        "distances": {str(cid): [] for cid in range(K)},
    }
    if K >= 2 and len(sub_embeddings) > 0:
        t0 = time.perf_counter()
        X_all = np.vstack([centroids, sub_embeddings])
        print(
            f"Fitting PCA→UMAP on {len(X_all):,} points "
            "(silent; ~30s-2min for ~10k points)..."
        )
        X_pca = PCA(n_components=min(pca_dim, X_all.shape[1])).fit_transform(X_all)
        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            n_jobs=8,
            init="pca",
            verbose=True,
        )
        umap_result = umap_model.fit_transform(X_pca)
        for cid in range(K):
            umap_data["centroids"][str(cid)] = umap_result[cid].tolist()
        for i, cid in enumerate(np.asarray(sub_cluster_assignments).tolist()):
            umap_data["clips"][str(cid)].append(umap_result[K + i].tolist())
            umap_data["clip_ids"][str(cid)].append(str(sub_clip_ids[i]))
            umap_data["distances"][str(cid)].append(float(sub_distances[i]))
        print(
            f"Time for UMAP ({len(X_all)} points): "
            f"{time.perf_counter() - t0:.2f}s"
        )

    with open(run_dir / "umap.json", "w") as f:
        json.dump(umap_data, f)


def write_metadata(
    run_dir,
    run_id,
    n_clusters,
    n_input_clips,
    embed_type,
    spherical_kmeans,
    max_points_per_centroid,
    search_params=None,
):
    """Write/merge metadata.json with status='done'.

    Don't overwrite fields the server may have pre-written when launching
    this run from the UI (embed_type, search_params, etc.) — only
    n_clusters/n_input_clips/status are always-update.
    """
    run_dir = Path(run_dir)
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}
    metadata.setdefault("started_at", time.time())
    metadata.setdefault("run_id", run_id)
    metadata.setdefault("embed_type", embed_type)
    metadata.setdefault("spherical_kmeans", bool(spherical_kmeans))
    metadata.setdefault("max_points_per_centroid", max_points_per_centroid)
    metadata.setdefault("search_params", search_params)
    metadata["n_clusters"] = n_clusters
    metadata["n_input_clips"] = n_input_clips
    metadata["status"] = "done"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)


def build_clustering_run(
    output_dir,
    embeddings,
    clip_ids,
    n_clusters,
    embed_type: str = "cosmos",
    spherical: bool = False,
    max_points_per_centroid: int = 256,
    seed: int = 1234,
    n_iter: int = 25,
    n_redo: int = 1,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.1,
    umap_max_clips: int = 50_000,
    captions_db_path=None,
    caption_model=None,
    run_id: Optional[str] = None,
    search_params: Optional[str] = None,
    verbose: bool = False,
) -> Path:
    """Run K-means + UMAP on `embeddings`, write the 6-file run directory.

    Parameters
    ----------
    output_dir
        Either a parent dir (a ``run_id`` subdir is created) or a path
        whose ``.name`` already equals ``run_id`` (treated as the run dir
        directly — useful for tests).
    embeddings
        ``(n_clips, d)`` float32 array. Will be cast to contiguous float32.
    clip_ids
        Length ``n_clips``, aligned with ``embeddings`` rows.
    n_clusters
        Number of K-means clusters. Not capped — caller's choice.
    captions_db_path
        Optional. When given, runs ``extract_topics_for_run`` after the
        parquet is written so the run also gets ``cluster_topics.json``.
    run_id
        Auto-generated 10-char alphanumeric id when None.

    Returns
    -------
    Path
        Absolute path to the run directory.
    """
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if len(embeddings) != len(clip_ids):
        raise ValueError(
            f"embeddings rows ({len(embeddings)}) != clip_ids ({len(clip_ids)})"
        )
    n_total = len(embeddings)
    n_clusters = int(n_clusters)

    if run_id is None:
        run_id = generate_run_id()

    output_dir = Path(output_dir)
    run_dir = output_dir if output_dir.name == run_id else output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    kmeans = FaissKMeans(
        feature_dim=embeddings.shape[1],
        n_clusters=n_clusters,
        niter=n_iter,
        nredo=n_redo,
        verbose=verbose,
        seed=seed,
        spherical_kmeans=spherical,
        max_points_per_centroid=max_points_per_centroid,
    )
    kmeans.fit(embeddings)
    cluster_assignments, distances = kmeans.predict(embeddings)

    write_cluster_assignments(
        run_dir, cluster_assignments, distances, clip_ids, n_clusters,
    )
    write_centroids(run_dir, kmeans.centroids)

    n_sub = min(umap_max_clips, n_total)
    sub_idx = np.random.default_rng(seed).choice(n_total, n_sub, replace=False)
    sub_embeddings = embeddings[sub_idx]
    sub_cluster_assignments = cluster_assignments[sub_idx]
    sub_distances = distances[sub_idx]
    sub_clip_ids = [clip_ids[i] for i in sub_idx.tolist()]

    # Topics first so the server's done-detection (which keys on
    # umap.json existing) flips after topics are also persisted.
    if captions_db_path:
        try:
            extract_topics_for_run(
                run_dir, str(captions_db_path), model_name=caption_model,
            )
        except Exception as e:
            print(f"[topics] extraction failed: {e}")

    fit_and_write_umap(
        run_dir,
        n_clusters=n_clusters,
        centroids=kmeans.centroids,
        sub_embeddings=sub_embeddings,
        sub_cluster_assignments=sub_cluster_assignments,
        sub_clip_ids=sub_clip_ids,
        sub_distances=sub_distances,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
    )
    write_metadata(
        run_dir,
        run_id=run_id,
        n_clusters=n_clusters,
        n_input_clips=n_total,
        embed_type=embed_type,
        spherical_kmeans=spherical,
        max_points_per_centroid=max_points_per_centroid,
        search_params=search_params,
    )
    return run_dir.resolve()
