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

"""Public embedding input helpers.

Every encoder is represented by one ``<name>.npz`` file with:

* ``clip_ids``: row-aligned clip identifiers.
* ``embeddings``: a 2D numeric array, one row per clip.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def load_embeddings(
    embeddings_dir: Path,
    name: str,
    wanted: set[str],
) -> tuple[list[str], np.ndarray]:
    """Load and filter ``<embeddings_dir>/<name>.npz`` to wanted clip IDs."""
    npz_path = embeddings_dir / f"{name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    t0 = time.time()
    data = np.load(npz_path, allow_pickle=True)
    all_ids = [str(x) for x in data["clip_ids"]]
    all_embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    assert all_embeddings.ndim == 2, (
        f"{npz_path}: embeddings must be 2D, got {all_embeddings.shape}"
    )
    assert len(all_ids) == all_embeddings.shape[0], (
        f"{npz_path}: {len(all_ids)} clip_ids for "
        f"{all_embeddings.shape[0]} embedding rows"
    )
    assert len(set(all_ids)) == len(all_ids), f"{npz_path}: duplicate clip_ids"

    rows = [i for i, clip_id in enumerate(all_ids) if clip_id in wanted]
    if not rows:
        elapsed = time.time() - t0
        print(
            f"[embeddings] {name}: 0 / {len(wanted)} clips matched "
            f"({elapsed:.1f}s)",
            flush=True,
        )
        return [], np.zeros((0, all_embeddings.shape[1]), dtype=np.float32)

    row_idx = np.asarray(rows, dtype=np.int64)
    ids = [all_ids[i] for i in rows]
    X = np.ascontiguousarray(all_embeddings[row_idx], dtype=np.float32)
    elapsed = time.time() - t0
    print(
        f"[embeddings] {name}: {len(ids)} / {len(wanted)} clips matched, "
        f"{X.shape} in {elapsed:.1f}s",
        flush=True,
    )
    return ids, X
