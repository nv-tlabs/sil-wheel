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

Each encoder is one ``<name>.npz`` with row-aligned ``clip_ids`` + ``embeddings``.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from emb_common import filter_to_ids, load_npz  # noqa: E402


def load_embeddings(embeddings_dir, name, wanted):
    """Load ``<embeddings_dir>/<name>.npz`` and filter it to the wanted clip ids."""
    npz_path = embeddings_dir / f"{name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    t0 = time.time()
    all_ids, all_embeddings = load_npz(npz_path)
    assert len(set(all_ids)) == len(all_ids), f"{npz_path}: duplicate clip_ids"
    ids, X = filter_to_ids(all_ids, all_embeddings, wanted)
    elapsed = time.time() - t0
    if not ids:
        print(
            f"[embeddings] {name}: 0 / {len(wanted)} clips matched ({elapsed:.1f}s)",
            flush=True,
        )
    else:
        print(
            f"[embeddings] {name}: {len(ids)} / {len(wanted)} clips matched, "
            f"{X.shape} in {elapsed:.1f}s",
            flush=True,
        )
    return ids, X
