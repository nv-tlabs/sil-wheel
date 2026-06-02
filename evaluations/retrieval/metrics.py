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

"""Recall@K for 1:1 paired retrieval (InternVideo2 protocol)."""
from dataclasses import dataclass

import numpy as np


@dataclass
class RetrievalMetrics:
    r1: float
    r5: float
    r10: float
    median_rank: float

    def as_dict(self):
        return {
            "R@1": self.r1,
            "R@5": self.r5,
            "R@10": self.r10,
            "MedR": self.median_rank,
        }


def ranks_for_paired(sim_matrix, gt_indices):
    assert sim_matrix.ndim == 2, (
        f"expected 2D sim, got shape {sim_matrix.shape}"
    )
    assert sim_matrix.shape[0] == len(gt_indices), (
        f"row/gt length mismatch: {sim_matrix.shape[0]} vs {len(gt_indices)}"
    )
    gt = np.asarray(gt_indices)
    gt_scores = sim_matrix[np.arange(sim_matrix.shape[0]), gt]
    # Average-rank-on-tie (scipy.stats.rankdata method="average"):
    # rank = 1 + #strict_greater + #equal/2 (the -1 cancels GT itself).
    # When phrase-FTS returns ``-inf`` for every gallery item, GT ties
    # with every row and lands at the expected rank N/2, not at fake 1.
    strict_greater = (sim_matrix > gt_scores[:, None]).sum(axis=1)
    equal = (sim_matrix == gt_scores[:, None]).sum(axis=1)
    return strict_greater + 1 + (equal - 1) / 2.0


def recall_at_k(sim_matrix, gt_indices):
    ranks = ranks_for_paired(sim_matrix, gt_indices)
    return RetrievalMetrics(
        r1=float((ranks <= 1).mean()),
        r5=float((ranks <= 5).mean()),
        r10=float((ranks <= 10).mean()),
        median_rank=float(np.median(ranks)),
    )


def t2v_and_v2t(sim_matrix):
    """Return (T2V, V2T) metrics for a square 1:1 paired sim matrix."""
    n = sim_matrix.shape[0]
    assert sim_matrix.shape == (n, n), (
        f"expected a square 1:1 paired matrix; got shape {sim_matrix.shape}"
    )
    gt = np.arange(n)
    return (
        recall_at_k(sim_matrix, gt),
        recall_at_k(sim_matrix.T, gt),
    )
