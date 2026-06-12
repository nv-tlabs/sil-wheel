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

"""Cluster selection and label-placement helpers shared by the §4.4 figures.

``distinct_clusters`` chooses the k most mutually-distinct clusters of a run for
the overlay map and topic table; ``dense_xy`` finds a good (x, y) anchor for a
text label inside a 2D point cloud (the densest histogram bin) so labels sit on
the body of a UMAP blob rather than on a stray outlier.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def farthest_first(X, seed: int, k: int) -> list:
    """Farthest-first traversal by cosine distance.

    Returns the row indices of ``k`` mutually-distant vectors of ``X``, seeded
    with row ``seed`` and greedily adding the row whose minimum cosine distance to
    the already-chosen set is largest. ``X`` is L2-normalised internally, so the
    caller need not normalise."""
    X = np.asarray(X, dtype=float)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    chosen = [int(seed)]
    while len(chosen) < min(k, len(X)):
        mind = (1.0 - X @ X[chosen].T).min(axis=1)   # min cosine-dist to chosen set
        for i in chosen:
            mind[i] = -1.0
        chosen.append(int(np.argmax(mind)))
    return chosen


def distinct_clusters(run_dir, sizes, k: int, min_frac: float) -> list:
    """The ``k`` most mutually-distinct clusters of a run.

    Farthest-first traversal (cosine) over the run's L2-normalised centroids,
    restricted to clusters whose size is at least ``min_frac`` of the mean size so
    tiny outliers don't dominate, seeded with the largest cluster. ``sizes`` is a
    pandas Series indexed by cluster id; returns cluster ids."""
    cents = np.load(Path(run_dir) / "centroids.npy")
    ids = sorted(int(c) for c in sizes.index)
    mean_sz = float(sizes.mean())
    cand = [c for c in ids if sizes[c] >= min_frac * mean_sz] or ids
    seed = max(range(len(cand)), key=lambda i: sizes[cand[i]])
    chosen = farthest_first(cents[cand].astype(float), seed, k)
    return [cand[i] for i in chosen]


def dense_xy(xs, ys, bins: int = 20):
    """Anchor (x, y) for a label over a 2D point cloud: the mean of the points in
    the densest 2D-histogram bin (falls back to the median for tiny clouds)."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if len(xs) < 5:
        return float(np.median(xs)), float(np.median(ys))
    H, xe, ye = np.histogram2d(xs, ys, bins=bins)
    bx, by = np.unravel_index(int(np.argmax(H)), H.shape)
    m = (xs >= xe[bx]) & (xs <= xe[bx + 1]) & (ys >= ye[by]) & (ys <= ye[by + 1])
    return (float(xs[m].mean()), float(ys[m].mean())) if m.any() \
        else (float(np.median(xs)), float(np.median(ys)))
