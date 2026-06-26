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

"""npz loader + geometry helpers shared by the embedding-clustering and
embedding-quality evals (each encoder = one ``<name>.npz`` of ``clip_ids`` +
``embeddings``). Consumers add this dir to ``sys.path``, then import."""

import numpy as np


def load_npz(path):
    """Return ``(clip_ids, embeddings)`` from an npz: str ids + 2D float32 array."""
    data = np.load(path, allow_pickle=True)
    clip_ids = [str(x) for x in data["clip_ids"]]
    X = np.ascontiguousarray(data["embeddings"], dtype=np.float32)
    assert X.ndim == 2, f"{path}: embeddings must be 2D, got {X.shape}"
    assert len(clip_ids) == X.shape[0], (
        f"{path}: {len(clip_ids)} clip_ids for {X.shape[0]} embedding rows"
    )
    return clip_ids, X


def filter_to_ids(clip_ids, X, keep):
    """Restrict rows to clip ids in ``keep``, preserving the original order."""
    keep = {str(x) for x in keep}
    rows = [i for i, c in enumerate(clip_ids) if c in keep]
    ids = [clip_ids[i] for i in rows]
    Xf = np.ascontiguousarray(X[np.asarray(rows, dtype=np.int64)]) if rows else (
        np.zeros((0, X.shape[1]), dtype=np.float32)
    )
    return ids, Xf


def l2_normalize(X, eps=1e-12):
    """Row-wise L2 normalize."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def mean_center(X):
    """Mean-center then row-renormalize (anisotropy fix, e.g. Florence-2/SigLIP)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(Xc / norms, dtype=np.float32)
