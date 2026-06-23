#!/usr/bin/env python
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

"""Embedding-quality metrics for binary one-vs-rest label probes.

The module keeps the math used for the SIL-Wheel paper's supervised
embedding-quality table:

* kNN consistency: for each clip, fraction of cosine nearest neighbours
  that share the query clip's binary label.
* Cluster purity and NMI: k-means cluster assignments compared against
  one-vs-rest labels.
* Few-shot binary kNN: sample n positive and n negative seeds, classify
  held-out clips by their nearest seed, and average over trials.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

try:
    import faiss
except ImportError:
    faiss = None


def knn_purity(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    k_values: list[int],
) -> dict[str, float | int]:
    """Compute cosine kNN consistency against binary labels."""
    assert X.ndim == 2, f"X must be 2D, got shape {X.shape}"
    assert labels.ndim == 1, f"labels must be 1D, got shape {labels.shape}"
    assert X.shape[0] == labels.shape[0], (
        f"X and labels must have same length: "
        f"{X.shape[0]} rows vs {labels.shape[0]} labels"
    )
    label_vals = set(np.unique(labels).tolist())
    assert label_vals.issubset({0, 1}), (
        f"labels must be binary {{0, 1}}; got unique values {sorted(label_vals)}"
    )

    n_clips = X.shape[0]
    k_max = max(k_values)
    if k_max >= n_clips:
        raise ValueError(f"k_max={k_max} must be < N={n_clips}")

    nn = NearestNeighbors(n_neighbors=k_max + 1, metric="cosine").fit(X)
    _, idx = nn.kneighbors(X)

    rows = np.arange(n_clips)
    self_mask = idx == rows[:, None]
    self_in_row = self_mask.any(axis=1)
    drop_col = np.where(self_in_row, np.argmax(self_mask, axis=1), k_max)
    keep_mask = np.ones_like(idx, dtype=bool)
    keep_mask[rows, drop_col] = False
    neighbors = idx[keep_mask].reshape(n_clips, k_max)

    neigh_labels = labels[neighbors]
    q_labels = labels[:, None]
    pos_mask = labels == 1
    neg_mask = labels == 0

    out: dict[str, float | int] = {
        "n_clips": int(n_clips),
        "n_pos": int(pos_mask.sum()),
        "n_neg": int(neg_mask.sum()),
    }
    for k in k_values:
        frac_same = (neigh_labels[:, :k] == q_labels).mean(axis=1)
        pos_count = neigh_labels[:, :k].sum(axis=1)
        pred_correct = ((2 * pos_count > k).astype(labels.dtype) == labels)

        out[f"nn_purity_k{k}"] = float(frac_same.mean())
        out[f"pos_purity_k{k}"] = (
            float(frac_same[pos_mask].mean())
            if pos_mask.any() else float("nan")
        )
        out[f"neg_purity_k{k}"] = (
            float(frac_same[neg_mask].mean())
            if neg_mask.any() else float("nan")
        )
        out[f"nn_acc_k{k}"] = float(pred_correct.mean())
    return out


def _faiss_spherical_kmeans(
    Xn: np.ndarray,
    k: int,
    seed: int,
    niter: int = 25,
    nredo: int = 10,
) -> np.ndarray:
    """Spherical k-means via faiss.Kmeans(spherical=True)."""
    if faiss is None:
        raise ImportError(
            "faiss is required for spherical k-means. Install faiss-cpu or "
            "pass --no-spherical-kmeans to use sklearn KMeans."
        )
    X32 = np.ascontiguousarray(Xn, dtype=np.float32)
    km = faiss.Kmeans(
        d=X32.shape[1],
        k=k,
        niter=niter,
        nredo=nredo,
        seed=seed,
        spherical=True,
        verbose=False,
    )
    km.train(X32)
    _, labels = km.index.search(X32, 1)
    return labels.squeeze(1)


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def _purity(
    cluster_labels: np.ndarray,
    gt_labels: np.ndarray,
    k: int,
) -> tuple[float, int, int]:
    """Per-cluster majority-vote purity vs. binary labels."""
    n_clips = cluster_labels.shape[0]
    majority_total = 0
    n_pos_clusters = 0
    n_empty = 0
    for cid in range(k):
        mask = cluster_labels == cid
        if not mask.any():
            n_empty += 1
            continue
        n_c_pos = int((gt_labels[mask] == 1).sum())
        n_c_neg = int((gt_labels[mask] == 0).sum())
        if n_c_pos > n_c_neg:
            majority_total += n_c_pos
            n_pos_clusters += 1
        else:
            majority_total += n_c_neg
    return majority_total / n_clips, n_pos_clusters, n_empty


def _intra_sim(Xn: np.ndarray, cluster_labels: np.ndarray, k: int) -> float:
    """Mean per-cluster pairwise cosine across non-singleton clusters."""
    sims: list[float] = []
    for cid in range(k):
        mask = cluster_labels == cid
        n_c = int(mask.sum())
        if n_c < 2:
            continue
        sub = Xn[mask]
        gram_sum = float((sub @ sub.T).sum())
        sims.append((gram_sum - n_c) / (n_c * (n_c - 1)))
    if not sims:
        return float("nan")
    return float(sum(sims) / len(sims))


def _inter_sim(Xn: np.ndarray, cluster_labels: np.ndarray, k: int) -> float:
    """Mean cosine between L2-normalized centroids of non-empty clusters."""
    centroids: list[np.ndarray] = []
    for cid in range(k):
        mask = cluster_labels == cid
        if not mask.any():
            continue
        cm = Xn[mask].mean(axis=0)
        cm_norm = float(np.linalg.norm(cm))
        centroids.append(cm / max(cm_norm, 1e-12))
    n_valid = len(centroids)
    if n_valid < 2:
        return float("nan")
    cs = np.stack(centroids)
    g = cs @ cs.T
    return float((g.sum() - n_valid) / (n_valid * (n_valid - 1)))


def cluster_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    k_values: list[int],
    seed: int = 0,
    spherical: bool = True,
) -> dict[str, float | int]:
    """Compute k-means purity, NMI, and intrinsic cluster geometry."""
    assert X.ndim == 2, f"X must be 2D, got shape {X.shape}"
    assert labels.ndim == 1, f"labels must be 1D, got shape {labels.shape}"
    assert X.shape[0] == labels.shape[0], (
        f"X and labels length mismatch: {X.shape[0]} vs {labels.shape[0]}"
    )
    label_vals = set(np.unique(labels).tolist())
    assert label_vals.issubset({0, 1}), (
        f"labels must be binary {{0, 1}}; got {sorted(label_vals)}"
    )

    n_clips = X.shape[0]
    Xn = _l2_normalize(X)

    out: dict[str, float | int] = {
        "n_clips": int(n_clips),
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
    }
    for k in k_values:
        if not (1 < k < n_clips):
            raise ValueError(f"k={k} must satisfy 1 < k < N={n_clips}")
        if spherical:
            cluster_labels = _faiss_spherical_kmeans(Xn, k, seed=seed)
        else:
            cluster_labels = KMeans(
                n_clusters=k, random_state=seed, n_init=10
            ).fit(Xn).labels_

        purity, n_pos_clusters, n_empty = _purity(cluster_labels, labels, k)
        nmi = float(normalized_mutual_info_score(
            labels, cluster_labels, average_method="arithmetic"
        ))
        intra = _intra_sim(Xn, cluster_labels, k)
        inter = _inter_sim(Xn, cluster_labels, k)
        separation = (
            intra - inter
            if not (np.isnan(intra) or np.isnan(inter)) else float("nan")
        )

        out[f"cluster_purity_k{k}"] = purity
        out[f"nmi_k{k}"] = nmi
        out[f"n_pos_clusters_k{k}"] = n_pos_clusters
        out[f"n_empty_clusters_k{k}"] = n_empty
        out[f"intra_sim_k{k}"] = intra
        out[f"inter_sim_k{k}"] = inter
        out[f"separation_k{k}"] = separation
    return out


def few_shot_binary_knn(
    X: np.ndarray,
    binary_labels: np.ndarray,
    *,
    n_values: list[int],
    S: int,
    seed: int,
) -> dict:
    """Evaluate a seed-subsampled binary nearest-neighbour classifier."""
    assert X.ndim == 2, f"X must be 2D, got shape {X.shape}"
    assert binary_labels.ndim == 1, (
        f"binary_labels must be 1D, got shape {binary_labels.shape}"
    )
    assert X.shape[0] == binary_labels.shape[0], (
        f"X ({X.shape[0]}) and binary_labels ({binary_labels.shape[0]}) "
        "must agree on N"
    )
    label_vals = set(np.unique(binary_labels).tolist())
    assert label_vals.issubset({0, 1}), (
        f"binary_labels must be {{0, 1}}; got {sorted(label_vals)}"
    )
    assert S >= 1, f"S must be >= 1, got {S}"

    n_clips = X.shape[0]
    pos_idx = np.flatnonzero(binary_labels == 1)
    neg_idx = np.flatnonzero(binary_labels == 0)

    out: dict = {
        "fewshot_n_trials": int(S),
        "fewshot_skipped_ns": [],
    }
    for n in n_values:
        if len(pos_idx) < n + 1 or len(neg_idx) < n + 1:
            out["fewshot_skipped_ns"].append(int(n))
            out[f"fewshot_acc_n{n}_mean"] = float("nan")
            out[f"fewshot_acc_n{n}_std"] = float("nan")
            out[f"fewshot_pos_recall_n{n}"] = float("nan")
            out[f"fewshot_neg_recall_n{n}"] = float("nan")
            continue

        accs: list[float] = []
        pos_recs: list[float] = []
        neg_recs: list[float] = []
        for s in range(S):
            rng = np.random.default_rng(seed * 10_000 + s)
            pos_seeds = rng.choice(pos_idx, size=n, replace=False)
            neg_seeds = rng.choice(neg_idx, size=n, replace=False)
            seed_idx = np.concatenate([pos_seeds, neg_seeds])
            seed_mask = np.zeros(n_clips, dtype=bool)
            seed_mask[seed_idx] = True
            held_out = np.flatnonzero(~seed_mask)

            seed_labels = binary_labels[seed_idx]
            nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(X[seed_idx])
            _, nbr = nn.kneighbors(X[held_out])
            pred = seed_labels[nbr[:, 0]]
            truth = binary_labels[held_out]

            accs.append(float((pred == truth).mean()))
            pos_q = truth == 1
            neg_q = truth == 0
            assert pos_q.any() and neg_q.any()
            pos_recs.append(float((pred[pos_q] == 1).mean()))
            neg_recs.append(float((pred[neg_q] == 0).mean()))

        out[f"fewshot_acc_n{n}_mean"] = float(np.mean(accs))
        out[f"fewshot_acc_n{n}_std"] = (
            float(np.std(accs, ddof=1)) if S >= 2 else float("nan")
        )
        out[f"fewshot_pos_recall_n{n}"] = float(np.mean(pos_recs))
        out[f"fewshot_neg_recall_n{n}"] = float(np.mean(neg_recs))
    return out
