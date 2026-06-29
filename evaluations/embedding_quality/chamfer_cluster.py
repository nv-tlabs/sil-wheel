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

"""Chamfer-native clustering + metrics for set-of-embeddings clips.

A clip is a **set** of detection embeddings, not a single vector. The only
geometry between clips is the N×N symmetric Chamfer similarity

    S[i,j] = (1/2) [ mean_{a∈A_i} max_{b∈A_j} cos(a,b)
                   + mean_{b∈A_j} max_{a∈A_i} cos(a,b) ]   ∈ [0,1]

with dissimilarity D = 1 − S (since on unit-normalized embeddings
min_b ‖a−b‖-style distance ⇔ 1 − max_b cos(a,b)).

Chamfer is a **semimetric**: non-negative, symmetric, identity-of-
indiscernibles, but the triangle inequality FAILS. Consequences:

  - Fine for any algorithm that consumes a precomputed dissimilarity and
    only compares distances (K-medoids, agglomerative, spectral).
  - Breaks exact Euclidean embedding (classical MDS / KernelPCA produce
    negative eigenvalues — the matrix isn't PSD). So we never embed clips
    into a vector space; we cluster directly on D and define every metric
    natively on S.

This is the deliberate contrast with single-vector encoders, whose
Euclidean ``(ids, X)`` contract is the wrong space for set geometry.

Benchmark-only: S is dense N×N, so this is feasible at the ~3K-clip label
scale, not the 273K-clip corpus.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score

from metrics import _purity

ALGOS = ("kmedoids", "agglomerative", "spectral")


def build_chamfer_similarity(D: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Symmetric Chamfer similarity over clips from a stacked detection matrix.

    Args:
        D: (N_total, dim) L2-normalized detection embeddings, clip-contiguous.
        offsets: (n_clips+1,) int; clip i owns rows D[offsets[i]:offsets[i+1]].

    Returns:
        S: (n_clips, n_clips) float32, symmetric, entries in [0,1], diag≈1.

    Cost: bottleneck is A_i @ D.T per clip = N_total² × dim flops total. At
    3K clips / ~180K detections / 768-d ≈ 24T flops ≈ 3–6 min on CPU.
    """
    n_clips = len(offsets) - 1
    seg_starts = offsets[:-1]
    clip_sizes = np.diff(offsets).astype(np.float32)
    S = np.zeros((n_clips, n_clips), dtype=np.float32)
    for i in range(n_clips):
        A_i = D[offsets[i]:offsets[i + 1]]              # (|A_i|, dim)
        S_i = A_i @ D.T                                  # (|A_i|, N_total)
        # i→j: each d_i's best match in clip j, then mean over A_i.
        seg_max = np.maximum.reduceat(S_i, seg_starts, axis=1)  # (|A_i|, n_clips)
        sim_itoJ = seg_max.mean(axis=0)                  # (n_clips,)
        # j→i: each pool detection's best match in A_i, then mean per clip j.
        best_from_Ai = S_i.max(axis=0)                   # (N_total,)
        sim_jtoI = np.add.reduceat(best_from_Ai, seg_starts) / clip_sizes
        S[i] = (sim_itoJ + sim_jtoI) * 0.5
    np.clip(S, 0.0, 1.0, out=S)
    return S


# --------------------------------------------------------------------------
# Clustering primitives on a precomputed dissimilarity D = 1 − S
# --------------------------------------------------------------------------
def _kpp_init(D: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++-style medoid seeding on a precomputed distance matrix."""
    N = D.shape[0]
    medoids = [int(rng.integers(N))]
    closest = D[medoids[0]].copy()
    for _ in range(1, k):
        probs = closest ** 2
        total = probs.sum()
        nxt = (int(rng.integers(N)) if total == 0
               else int(rng.choice(N, p=probs / total)))
        medoids.append(nxt)
        closest = np.minimum(closest, D[nxt])
    return np.array(medoids, dtype=np.int64)


def fast_pam(
    D: np.ndarray, k: int, *, seed: int = 0, n_init: int = 10, max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """K-medoids (Voronoi-iteration PAM) on a precomputed distance matrix.

    Centers are real data points (medoids); assignment minimizes distance to
    medoid; medoid update picks the member minimizing within-cluster distance
    sum. No averaging — works on any dissimilarity, metric or not.

    Returns (labels (N,), medoid_indices (k,)).
    """
    rng = np.random.default_rng(seed)
    N = D.shape[0]
    best_labels: np.ndarray | None = None
    best_medoids: np.ndarray | None = None
    best_cost = np.inf
    for _ in range(n_init):
        medoids = _kpp_init(D, k, rng)
        for _ in range(max_iter):
            labels = D[:, medoids].argmin(axis=1)
            new_medoids = medoids.copy()
            for c in range(k):
                members = np.where(labels == c)[0]
                if len(members) == 0:
                    continue
                sub = D[np.ix_(members, members)]
                new_medoids[c] = members[sub.sum(axis=1).argmin()]
            if np.array_equal(np.sort(new_medoids), np.sort(medoids)):
                medoids = new_medoids
                break
            medoids = new_medoids
        labels = D[:, medoids].argmin(axis=1)
        cost = float(D[np.arange(N), medoids[labels]].sum())
        if cost < best_cost:
            best_cost, best_labels, best_medoids = cost, labels, medoids
    return best_labels, best_medoids


def _cluster(
    algo: str, S: np.ndarray, D: np.ndarray, k: int, seed: int,
) -> np.ndarray:
    """Return cluster labels (N,) for one algorithm at cluster count k."""
    if algo == "kmedoids":
        labels, _ = fast_pam(D, k, seed=seed)
        return labels
    if algo == "agglomerative":
        return AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average",
        ).fit_predict(D)
    if algo == "spectral":
        return SpectralClustering(
            n_clusters=k, affinity="precomputed", random_state=seed,
            assign_labels="kmeans", n_init=10,
        ).fit_predict(S)
    raise ValueError(f"unknown algo: {algo}")


def _medoids_for(D: np.ndarray, labels: np.ndarray, k: int) -> list[int]:
    """Representative clip per cluster = member minimizing within-cluster
    distance sum. Defined uniformly for every algorithm so inter_sim has a
    consistent medoid-to-medoid meaning."""
    medoids: list[int] = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        sub = D[np.ix_(members, members)]
        medoids.append(int(members[sub.sum(axis=1).argmin()]))
    return medoids


def _intra_sim(S: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Mean within-cluster pairwise Chamfer similarity, averaged across
    non-singleton clusters (mirrors metrics._intra_sim, but on S)."""
    sims: list[float] = []
    for c in range(k):
        members = np.where(labels == c)[0]
        n_c = len(members)
        if n_c < 2:
            continue
        sub = S[np.ix_(members, members)]
        # Off-diagonal mean: subtract n_c self-pairs (diag≈1) from gram sum.
        sims.append((float(sub.sum()) - float(np.diag(sub).sum()))
                    / (n_c * (n_c - 1)))
    return float(np.mean(sims)) if sims else float("nan")


def _inter_sim(S: np.ndarray, medoids: list[int]) -> float:
    """Mean Chamfer similarity between distinct cluster medoids."""
    if len(medoids) < 2:
        return float("nan")
    m = np.array(medoids)
    sub = S[np.ix_(m, m)]
    n = len(medoids)
    return float((sub.sum() - np.diag(sub).sum()) / (n * (n - 1)))


def knn_purity_precomputed(
    S: np.ndarray, labels: np.ndarray, *, k_values: list[int],
) -> dict[str, float | int]:
    """kNN-purity from the Chamfer similarity matrix directly.

    Neighbors = largest S[i,·] (excluding self). Mirrors metrics.knn_purity's
    output keys so the table renderer is unchanged.
    """
    N = S.shape[0]
    k_max = max(k_values)
    if k_max >= N:
        raise ValueError(f"k_max={k_max} must be < N={N}")
    # Descending similarity; column 0 is self (S[i,i]≈1 is the max).
    order = np.argsort(-S, axis=1)
    arange = np.arange(N)
    self_mask = order == arange[:, None]
    self_col = np.argmax(self_mask, axis=1)
    keep = np.ones_like(order, dtype=bool)
    keep[arange, self_col] = False
    neighbors = order[keep].reshape(N, N - 1)[:, :k_max]   # (N, k_max)
    neigh_labels = labels[neighbors]
    q_labels = labels[:, None]

    pos_mask = labels == 1
    neg_mask = labels == 0
    out: dict[str, float | int] = {
        "n_clips": int(N),
        "n_pos": int(pos_mask.sum()),
        "n_neg": int(neg_mask.sum()),
    }
    for k in k_values:
        frac_same = (neigh_labels[:, :k] == q_labels).mean(axis=1)
        pos_count = neigh_labels[:, :k].sum(axis=1)
        pred_correct = ((2 * pos_count > k).astype(labels.dtype) == labels)
        out[f"nn_purity_k{k}"] = float(frac_same.mean())
        out[f"pos_purity_k{k}"] = (float(frac_same[pos_mask].mean())
                                   if pos_mask.any() else float("nan"))
        out[f"neg_purity_k{k}"] = (float(frac_same[neg_mask].mean())
                                   if neg_mask.any() else float("nan"))
        out[f"nn_acc_k{k}"] = float(pred_correct.mean())
    return out


def cluster_all(
    S: np.ndarray, *, algo: str, k_values: list[int], seed: int = 0,
) -> dict[int, np.ndarray]:
    """Cluster S with `algo` at each k. Label-INDEPENDENT (no GT used), so
    the result is shared across all one-vs-rest targets in multi-label mode.

    Returns {k: cluster_labels (N,)}.
    """
    N = S.shape[0]
    D = 1.0 - S
    by_k: dict[int, np.ndarray] = {}
    for k in k_values:
        if not (1 < k < N):
            raise ValueError(f"k={k} must satisfy 1 < k < N={N}")
        by_k[k] = _cluster(algo, S, D, k, seed)
    return by_k


def metrics_from_clusters(
    S: np.ndarray,
    gt_labels: np.ndarray,
    clusters_by_k: dict[int, np.ndarray],
) -> dict[str, float | int]:
    """Per-k metrics from precomputed cluster assignments, native to Chamfer.

    Output keys mirror metrics.cluster_metrics (cluster_purity_k, nmi_k,
    n_pos_clusters_k, n_empty_clusters_k, intra_sim_k, inter_sim_k,
    separation_k) so downstream table code is unchanged — except intra/inter
    are Chamfer *set* similarities, not cosines between vectors. intra/inter/
    separation/n_empty are label-independent (clustering uses no GT).
    """
    D = 1.0 - S
    out: dict[str, float | int] = {
        "n_clips": int(S.shape[0]),
        "n_pos": int((gt_labels == 1).sum()),
        "n_neg": int((gt_labels == 0).sum()),
    }
    for k, cluster_labels in clusters_by_k.items():
        purity, n_pos_clusters, n_empty = _purity(cluster_labels, gt_labels, k)
        nmi = float(normalized_mutual_info_score(
            gt_labels, cluster_labels, average_method="arithmetic"))
        intra = _intra_sim(S, cluster_labels, k)
        inter = _inter_sim(S, _medoids_for(D, cluster_labels, k))
        separation = (intra - inter
                      if not (np.isnan(intra) or np.isnan(inter))
                      else float("nan"))
        out[f"cluster_purity_k{k}"] = purity
        out[f"nmi_k{k}"] = nmi
        out[f"n_pos_clusters_k{k}"] = n_pos_clusters
        out[f"n_empty_clusters_k{k}"] = n_empty
        out[f"intra_sim_k{k}"] = intra
        out[f"inter_sim_k{k}"] = inter
        out[f"separation_k{k}"] = separation
    return out


def few_shot_binary_knn_precomputed(
    sim: np.ndarray,
    binary_labels: np.ndarray,
    *,
    n_values: list[int],
    n_trials: int,
    seed: int,
) -> dict:
    """Few-shot binary 1-NN classifier on a precomputed similarity matrix.

    Mirrors metrics.few_shot_binary_knn exactly (same sampling, same output
    keys, same `seed * 10_000 + s` per-trial RNG so trials are reproducible
    and comparable to the vector encoders) — but the nearest seed is the one
    with the largest Chamfer similarity rather than the smallest cosine
    distance. No vector ever formed.

    Args:
        sim: (N, N) Chamfer similarity matrix (rows = queries, cols = seeds).
        binary_labels: (N,) int {0, 1}.
        n_values: seeds-per-class values (e.g. [5, 20]).
        n_trials: number of independent trials (== few_shot_binary_knn's `S`).
        seed: base RNG seed.
    """
    assert sim.shape[0] == sim.shape[1] == binary_labels.shape[0]
    N = sim.shape[0]
    pos_idx = np.flatnonzero(binary_labels == 1)
    neg_idx = np.flatnonzero(binary_labels == 0)

    out: dict = {"fewshot_n_trials": int(n_trials), "fewshot_skipped_ns": []}
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
        for s in range(n_trials):
            rng = np.random.default_rng(seed * 10_000 + s)
            seed_idx = np.concatenate([rng.choice(pos_idx, size=n, replace=False),
                                       rng.choice(neg_idx, size=n, replace=False)])
            seed_mask = np.zeros(N, dtype=bool)
            seed_mask[seed_idx] = True
            held_out = np.flatnonzero(~seed_mask)
            seed_labels = binary_labels[seed_idx]
            # Nearest seed = largest similarity from each held-out clip.
            nearest = sim[np.ix_(held_out, seed_idx)].argmax(axis=1)
            pred = seed_labels[nearest]
            truth = binary_labels[held_out]
            accs.append(float((pred == truth).mean()))
            pos_q = truth == 1
            neg_q = truth == 0
            assert pos_q.any() and neg_q.any()
            pos_recs.append(float((pred[pos_q] == 1).mean()))
            neg_recs.append(float((pred[neg_q] == 0).mean()))
        out[f"fewshot_acc_n{n}_mean"] = float(np.mean(accs))
        out[f"fewshot_acc_n{n}_std"] = (
            float(np.std(accs, ddof=1)) if n_trials >= 2 else float("nan"))
        out[f"fewshot_pos_recall_n{n}"] = float(np.mean(pos_recs))
        out[f"fewshot_neg_recall_n{n}"] = float(np.mean(neg_recs))
    return out
