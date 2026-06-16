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

import numpy as np
import pytest

from metrics import ranks_for_paired, recall_at_k, t2v_and_v2t


def test_recall_perfect_identity():
    # Identity sim matrix: each query's GT is its own row.
    sim = np.eye(5, dtype=np.float32)
    m = recall_at_k(sim, gt_indices=np.arange(5))
    assert m.r1 == 1.0
    assert m.r5 == 1.0
    assert m.r10 == 1.0
    assert m.median_rank == 1.0


def test_recall_worst_case():
    # Reverse-order ground truth: the correct video always has the
    # *lowest* similarity, so the rank of the GT is N (the number of videos).
    n = 10
    sim = np.arange(n * n, dtype=np.float32).reshape(n, n)  # row r ranks col n-1 highest
    gt = np.zeros(n, dtype=np.int64)  # GT is col 0 → strictly worst
    m = recall_at_k(sim, gt)
    assert m.r1 == 0.0
    assert m.r5 == 0.0
    assert m.r10 == 1.0  # k=10 captures the worst rank of 10
    assert m.median_rank == n


def test_ranks_match_recall():
    # Mix of correct and incorrect; verify ranks line up.
    sim = np.array(
        [
            [0.9, 0.1, 0.0],  # rank of col 0 = 1
            [0.4, 0.5, 0.6],  # rank of col 1 = 2
            [0.1, 0.2, 0.3],  # rank of col 0 = 3
        ],
        dtype=np.float32,
    )
    gt = np.array([0, 1, 0])
    ranks = ranks_for_paired(sim, gt)
    np.testing.assert_array_equal(ranks, [1, 2, 3])
    m = recall_at_k(sim, gt)
    # 1 query at rank 1, 1 at rank 2, 1 at rank 3 → R@1 = 1/3.
    assert m.r1 == pytest.approx(1 / 3)
    assert m.r5 == pytest.approx(1.0)


def test_t2v_v2t_symmetric_on_identity():
    sim = np.eye(20, dtype=np.float32)
    t2v, v2t = t2v_and_v2t(sim)
    assert t2v.as_dict() == v2t.as_dict()


def test_t2v_v2t_transpose():
    rng = np.random.default_rng(0)
    sim = rng.normal(size=(8, 8)).astype(np.float32)
    t2v, v2t = t2v_and_v2t(sim)
    # T2V on `sim` must equal V2T on `sim.T` (definition of V2T).
    t2v_of_transpose, _ = t2v_and_v2t(sim.T)
    assert t2v_of_transpose.as_dict() == v2t.as_dict()


def test_t2v_v2t_rejects_non_square():
    with pytest.raises(AssertionError):
        t2v_and_v2t(np.zeros((5, 6), dtype=np.float32))


def test_all_tied_scores_give_expected_rank():
    # When every candidate ties, GT lands at the average rank (N+1)/2.
    n = 10
    sim = np.zeros((n, n), dtype=np.float32)
    ranks = ranks_for_paired(sim, np.arange(n))
    np.testing.assert_allclose(ranks, np.full(n, (n + 1) / 2.0))
