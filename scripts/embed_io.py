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


def load_clip_to_rows(path_to_embeddings, embed_type, index_tag, wanted=None,
                      chunk=50_000_000):
    """Return {clip_id: [faiss_row, ...]} with *all* rows for each clip.

    Multi-row encoders (e.g. region embeddings: one row per detection crop)
    keep several rows per clip. ``load_clip_to_index`` collapses these to the
    first row; this returns the full set so a clip can be pooled over all of
    its rows. Requires the compact .npy sidecars
    (``<embed_type>_clip_ids_<tag>.npy`` + ``<embed_type>_position_of_row_<tag>.npy``).

    ``wanted`` restricts the result to that set of clip_ids and is strongly
    recommended: a full visual index can have ~1e9 rows, so building the whole
    mapping is expensive. With ``wanted`` set, ``position_of_row`` is streamed in
    chunks and only the requested clips' rows are gathered.
    """
    root = Path(path_to_embeddings)
    clip_ids_npy = root / f"{embed_type}_clip_ids_{index_tag}.npy"
    position_npy = root / f"{embed_type}_position_of_row_{index_tag}.npy"
    if not (clip_ids_npy.exists() and position_npy.exists()):
        raise FileNotFoundError(
            f"Mean pooling needs the .npy sidecars for {embed_type}/{index_tag} "
            f"under {root} (clip_ids + position_of_row)."
        )
    clip_ids = np.load(clip_ids_npy, allow_pickle=True)
    pos = np.load(position_npy, mmap_mode="r")  # position_of_row[r] -> clip position

    if wanted is None:  # full mapping (heavy; only sane for small indices)
        p = np.asarray(pos)
        order = np.argsort(p, kind="stable")
        starts = np.searchsorted(p[order], np.arange(len(clip_ids) + 1))
        return {str(clip_ids[i]): order[starts[i]:starts[i + 1]].tolist()
                for i in range(len(clip_ids))}

    wanted = {str(c) for c in wanted}
    keep_pos = np.fromiter((str(c) in wanted for c in clip_ids),
                           dtype=bool, count=len(clip_ids))
    sel_pos, sel_row = [], []
    for s in range(0, len(pos), chunk):
        block = np.asarray(pos[s:s + chunk])
        m = keep_pos[block]
        idx = np.nonzero(m)[0]
        if len(idx):
            sel_pos.append(block[idx])
            sel_row.append(idx + s)
    if not sel_pos:
        return {}
    sp = np.concatenate(sel_pos)
    sr = np.concatenate(sel_row)
    order = np.argsort(sp, kind="stable")
    sp, sr = sp[order], sr[order]
    uniq, starts = np.unique(sp, return_index=True)
    bounds = list(starts) + [len(sp)]
    return {str(clip_ids[p]): sr[bounds[i]:bounds[i + 1]].tolist()
            for i, p in enumerate(uniq)}


def pool_clip_features(features_index, clip_to_rows, clip_ids, batch_size=1_000_000):
    """Mean-pool each clip's rows into a single vector, reconstructing in batches.

    Returns an ``(len(clip_ids), d)`` float32 matrix aligned with ``clip_ids``.
    """
    d = features_index.d
    sums = np.zeros((len(clip_ids), d), dtype=np.float32)
    flat_rows, flat_clip = [], []
    for i, c in enumerate(clip_ids):
        rows = clip_to_rows[c]
        flat_rows.extend(rows)
        flat_clip.extend([i] * len(rows))
    flat_rows = np.asarray(flat_rows, dtype="int64")
    flat_clip = np.asarray(flat_clip, dtype="int64")
    counts = np.bincount(flat_clip, minlength=len(clip_ids)).astype(np.float32)
    for s in range(0, len(flat_rows), batch_size):
        recon = features_index.reconstruct_batch(flat_rows[s:s + batch_size])
        np.add.at(sums, flat_clip[s:s + batch_size], recon)
    return sums / np.maximum(counts, 1.0)[:, None]
