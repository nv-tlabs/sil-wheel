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

"""Public embedding input helpers. Each encoder is one ``<name>.npz`` with
row-aligned ``clip_ids`` + ``embeddings``."""

import time

import numpy as np


def load_embeddings(embeddings_dir, name, wanted):
    """Load ``<embeddings_dir>/<name>.npz`` and filter it to the wanted clip ids."""
    npz_path = embeddings_dir / f"{name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    t0 = time.time()
    data = np.load(npz_path, allow_pickle=True)
    all_ids = [str(x) for x in data["clip_ids"]]
    all_emb = np.ascontiguousarray(data["embeddings"], dtype=np.float32)
    assert all_emb.ndim == 2, f"{npz_path}: embeddings must be 2D, got {all_emb.shape}"
    assert len(all_ids) == all_emb.shape[0], f"{npz_path}: clip_id/row count mismatch"
    assert len(set(all_ids)) == len(all_ids), f"{npz_path}: duplicate clip_ids"

    rows = [i for i, clip_id in enumerate(all_ids) if clip_id in wanted]
    ids = [all_ids[i] for i in rows]
    X = (
        np.ascontiguousarray(all_emb[np.asarray(rows, dtype=np.int64)])
        if rows
        else np.zeros((0, all_emb.shape[1]), dtype=np.float32)
    )
    print(
        f"[embeddings] {name}: {len(ids)} / {len(wanted)} clips matched, "
        f"{X.shape} in {time.time() - t0:.1f}s",
        flush=True,
    )
    return ids, X
