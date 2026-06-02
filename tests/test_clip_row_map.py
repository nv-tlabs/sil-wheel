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

from sil_wheel.stores.clip_row_map import ClipRowMap


@pytest.fixture
def sample_map():
    # 3 unique clips, 6 FAISS rows. Row order is the FAISS insertion order.
    rows = ["clip-A", "clip-B", "clip-A", "clip-C", "clip-B", "clip-A"]
    return rows, ClipRowMap.build(rows)


def test_build_deduplicates_clip_ids(sample_map):
    _, m = sample_map
    assert sorted(m.clip_ids.tolist()) == ["clip-A", "clip-B", "clip-C"]
    assert len(m.position_of_row) == 6


def test_clip_id_for_row_matches_insertion_order(sample_map):
    rows, m = sample_map
    for row_idx, expected in enumerate(rows):
        assert m.clip_id_for_row(row_idx) == expected


def test_rows_for_clips_returns_every_occurrence(sample_map):
    _, m = sample_map
    rows_for_A = sorted(m.rows_for_clips(["clip-A"]).tolist())
    assert rows_for_A == [0, 2, 5]

    rows_for_B = sorted(m.rows_for_clips(["clip-B"]).tolist())
    assert rows_for_B == [1, 4]


def test_rows_for_clips_handles_multiple_and_unknown(sample_map):
    _, m = sample_map
    rows = sorted(m.rows_for_clips(["clip-A", "clip-C", "unknown"]).tolist())
    assert rows == [0, 2, 3, 5]


def test_rows_for_clips_empty_inputs(sample_map):
    _, m = sample_map
    assert m.rows_for_clips([]).tolist() == []
    assert m.rows_for_clips(["nope"]).tolist() == []


def test_position_of_row_uses_int32(sample_map):
    _, m = sample_map
    # Compact storage is the whole point; verify we aren't using int64.
    assert m.position_of_row.dtype == np.int32


def test_rows_for_clips_union_matches_manual_scan():
    rng = np.random.default_rng(0)
    clip_ids = [f"clip-{i % 7}" for i in rng.integers(0, 1000, size=500)]
    m = ClipRowMap.build(clip_ids)

    allowed = {"clip-1", "clip-4", "clip-6"}
    expected = sorted(i for i, c in enumerate(clip_ids) if c in allowed)
    got = sorted(m.rows_for_clips(list(allowed)).tolist())
    assert got == expected
