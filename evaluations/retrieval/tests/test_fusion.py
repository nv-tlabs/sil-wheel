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

from fusion import rrf_term, zscore_term
from metrics import recall_at_k


def _rrf(matrices):
    return sum(rrf_term(s) for s in matrices)


def _zscore(matrices):
    return sum(zscore_term(s) for s in matrices)


def test_rrf_single_input_preserves_ranking():
    sim = np.array([[0.1, 0.5, 0.4], [0.8, 0.2, 0.6]], dtype=np.float32)
    fused = _rrf([sim])
    np.testing.assert_array_equal(fused.argmax(axis=1), sim.argmax(axis=1))


def test_rrf_combination_can_outperform_components():
    visual = np.array(
        [
            [0.4, 0.9, 0.1],
            [0.0, 0.7, 0.2],
            [0.0, 0.1, 0.7],
        ],
        dtype=np.float32,
    )
    caption = np.array(
        [
            [0.9, 0.1, 0.2],
            [0.4, 0.3, 0.6],
            [0.1, 0.0, 0.8],
        ],
        dtype=np.float32,
    )
    gt = np.array([0, 1, 2])
    fused = _rrf([visual, caption])
    visual_r1 = recall_at_k(visual, gt).r1
    caption_r1 = recall_at_k(caption, gt).r1
    fused_r1 = recall_at_k(fused, gt).r1
    assert fused_r1 >= max(visual_r1, caption_r1)


def test_zscore_handles_neg_inf_misses():
    visual = np.array([[0.4, 0.7, 0.3]], dtype=np.float32)
    bm25 = np.array([[-np.inf, 5.0, -np.inf]], dtype=np.float32)
    fused = _zscore([visual, bm25])
    # The -inf entries must not produce NaN and must rank below the hit.
    assert np.isfinite(fused).all()
    assert fused.argmax(axis=1)[0] == 1
