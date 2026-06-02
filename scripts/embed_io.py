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

"""Shared helpers for the offline cluster/classifier scripts."""
import pickle
from pathlib import Path

import numpy as np


def load_clip_to_index(path_to_embeddings, embed_type, index_tag):
    """Return {clip_id: faiss_row} for the given (embed_type, index_tag).

    Tries (in order of cost):
      1. Compact .npy sidecars (visual_clip_ids_*.npy + visual_position_of_row_*.npy
         and analogues for caption); built once, fast to load.
      2. Cached <embed_type>_clip_to_index_<tag>.pkl. The native form
         for cosmos; cached form for caption/visual.
      3. Legacy <embed_type>_index_to_clip_<tag>.pkl. Slow path:
         load + invert + cache as (2).
    """
    root = Path(path_to_embeddings)
    clip_ids_npy = root / f"{embed_type}_clip_ids_{index_tag}.npy"
    position_npy = root / f"{embed_type}_position_of_row_{index_tag}.npy"
    cti_pkl = root / f"{embed_type}_clip_to_index_{index_tag}.pkl"
    itc_pkl = root / f"{embed_type}_index_to_clip_{index_tag}.pkl"

    if clip_ids_npy.exists() and position_npy.exists():
        clip_ids = np.load(clip_ids_npy, allow_pickle=True)
        position_of_row = np.asarray(np.load(position_npy))
        # First-row-per-position via np.unique. Returns sorted unique
        # positions (0..n_unique-1) with the row index of each first
        # occurrence.
        _, first_rows = np.unique(position_of_row, return_index=True)
        return dict(zip(
            [str(c) for c in clip_ids.tolist()],
            first_rows.tolist(),
        ))

    if cti_pkl.exists():
        with open(cti_pkl, "rb") as f:
            return pickle.load(f)

    if itc_pkl.exists():
        with open(itc_pkl, "rb") as f:
            index_to_clips = pickle.load(f)
        clip_to_index = {}
        for idx, cid in index_to_clips.items():
            if cid not in clip_to_index:
                clip_to_index[cid] = idx
        with open(cti_pkl, "wb") as f:
            pickle.dump(clip_to_index, f)
        return clip_to_index

    raise FileNotFoundError(
        f"No clip mapping found for {embed_type}/{index_tag} under {root}"
    )
