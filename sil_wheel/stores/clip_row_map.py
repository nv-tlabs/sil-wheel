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

"""Compact bidirectional map between FAISS row ids and clip_ids.

A FAISS index holds many rows (often 10s–100s of millions) and many rows
share the same clip_id: one row per caption for text indices, one row
per frame/region for visual indices. The naive representation is a
Python ``dict[int, str]`` plus an inverse ``dict[str, list[int]]``.
That is correct but wastes memory.

The core idea here is to store each clip_id exactly once in a small
numpy array ``clip_ids`` and have every FAISS row hold only its
*position* in that array (a 4-byte int32 in ``position_of_row``)

Two lookup directions are both needed at query time:

  - row -> clip_id, used to translate FAISS search results. One array
    index chain: ``clip_ids[position_of_row[row]]``.

  - allowed clip_ids -> rows, used to build a FAISS ``IDSelector`` that
    restricts search to a pre-filtered pool of clips. Implemented on
    top of a CSR-style grouping (``rows_by_position`` /
    ``row_offsets``) precomputed once at load time, so each lookup is
    a slice rather than an O(ntotal) scan.

  - row:      a FAISS row id, 0..ntotal-1.
  - position: an index into ``clip_ids``, 0..len(clip_ids)-1.
"""

import sys

import numpy as np


class ClipRowMap:
    def __init__(self, clip_ids, position_of_row):
        # Intern each unique clip_id so later dict lookups and equality
        # checks can short-circuit on identity. str() coerces numpy.str_
        # (a str subclass) which sys.intern rejects.
        self.clip_ids = np.array(
            [sys.intern(str(c)) for c in clip_ids], dtype=object,
        )
        self.position_of_row = np.asarray(position_of_row, dtype=np.int32)
        self.position_of_clip_id = {
            cid: i for i, cid in enumerate(self.clip_ids)
        }

        # Group rows by the position they point to so that the
        # clip_id -> rows lookup is a constant number of slices per
        # requested clip instead of a full scan over position_of_row.
        # For position p, its rows live in:
        #   rows_by_position[row_offsets[p] : row_offsets[p + 1]]
        order = np.argsort(self.position_of_row, kind="stable")
        self.rows_by_position = order.astype(np.int64)
        counts = np.bincount(
            self.position_of_row, minlength=len(self.clip_ids)
        )
        self.row_offsets = np.concatenate(([0], np.cumsum(counts)))

    def clip_id_for_row(self, row):
        return self.clip_ids[self.position_of_row[row]]

    def rows_for_clips(self, allowed_clip_ids):
        # Silently drop unknown clip_ids: the caller's pool may contain
        # clips that exist in other stores but not in this FAISS index.
        positions = np.fromiter(
            (
                self.position_of_clip_id[c]
                for c in allowed_clip_ids
                if c in self.position_of_clip_id
            ),
            dtype=np.int32,
        )
        if len(positions) == 0:
            return np.empty(0, dtype=np.int64)
        return np.concatenate([
            self.rows_by_position[
                self.row_offsets[p] : self.row_offsets[p + 1]
            ]
            for p in positions
        ])

    @classmethod
    def build(cls, clip_ids_per_row):
        # Given one clip_id per FAISS row (row order must match the
        # FAISS index insertion order), factor out the unique clip_ids
        # and produce the compact arrays.
        arr = np.asarray(clip_ids_per_row, dtype=object)
        clip_ids, position_of_row = np.unique(arr, return_inverse=True)
        return cls(clip_ids, position_of_row.astype(np.int32))
