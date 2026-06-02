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

"""Per-modality fusion terms. Sum them across modalities to fuse rankings.

Mirrors the ordinal-rank RRF used in ``sil_wheel.stores.search_utils.rrf_rank``;
ties get arbitrary order (stable argsort), matching the core convention.
"""
import numpy as np


def rrf_term(sim, k=60):
    """Reciprocal Rank Fusion contribution ``1 / (k + rank)`` per query.

    ``sim`` is a ``(n_query, n_candidate)`` similarity matrix; returns the
    same shape.
    """
    assert sim.ndim == 2, f"expected 2D sim, got shape {sim.shape}"
    order = np.argsort(-sim, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks, order,
        np.arange(1, sim.shape[1] + 1)[None, :].repeat(sim.shape[0], axis=0),
        axis=1,
    )
    return 1.0 / (k + ranks)


def zscore_term(sim):
    """Per-row z-score contribution; ``(n_query, n_candidate)`` in and out.

    Non-finite entries (e.g. BM25 ``-inf`` misses) are excluded from the
    mean/std and mapped to ``-10`` in the output so they still rank below
    every real hit.
    """
    assert sim.ndim == 2, f"expected 2D sim, got shape {sim.shape}"
    finite = np.isfinite(sim)
    masked = np.where(finite, sim, np.nan).astype(np.float64)
    mean = np.nanmean(masked, axis=1, keepdims=True)
    std = np.nanstd(masked, axis=1, keepdims=True) + 1e-8
    z = (masked - mean) / std
    return np.where(finite, z, -10.0)
