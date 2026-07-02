# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for common-component removal (no GPU / no LLM)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from caption_pc_ablation import remove_top_pcs


def test_remove_top_pcs_returns_unit_rows_and_strips_shared_direction():
    rng = np.random.default_rng(0)
    shared = rng.normal(size=(1, 16))
    # every row = big shared component + small idiosyncratic part
    X = 5.0 * shared + 0.1 * rng.normal(size=(200, 16))
    out = remove_top_pcs(X, r=1, fit_on=X)
    assert out.shape == X.shape
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    # projecting the dominant shared direction out shrinks its alignment
    before = np.abs(X @ shared.ravel()).mean()
    after = np.abs(out @ shared.ravel()).mean()
    assert after < before


def test_remove_top_pcs_r0_is_centering():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 8)).astype(np.float32)
    out = remove_top_pcs(X, r=0, fit_on=X)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
