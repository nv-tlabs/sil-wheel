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

"""Tests for benchmark_semantic_search_indexes helpers.

Covers:
  - fmt_score / fmt_recall — NaN handling and float formatting
  - recall — Recall against a FLAT baseline
  - load_annotation_slices — data-source slices, annotation-key slices, and
    their pairwise intersections, using an in-memory SQLite database
"""
import math
import sqlite3

import pytest

from benchmark_semantic_search_indexes import (
    DataSlice,
    fmt_recall,
    fmt_score,
    recall,
    load_annotation_slices,
)


# ---------------------------------------------------------------------------
# fmt_score
# ---------------------------------------------------------------------------


class TestFmtScore:
    def test_nan_returns_nan_string(self):
        assert fmt_score(float("nan")) == "nan"

    def test_normal_float_formatted_to_4dp(self):
        assert fmt_score(0.12345) == "0.1235"

    def test_zero(self):
        assert fmt_score(0.0) == "0.0000"

    def test_one(self):
        assert fmt_score(1.0) == "1.0000"

    def test_negative(self):
        assert fmt_score(-0.5) == "-0.5000"


# ---------------------------------------------------------------------------
# fmt_recall
# ---------------------------------------------------------------------------


class TestFmtRecall:
    def test_nan_returns_na(self):
        assert fmt_recall(float("nan")) == "n/a"

    def test_perfectrecall(self):
        assert fmt_recall(1.0) == "1.000"

    def test_partialrecall(self):
        assert fmt_recall(0.5) == "0.500"

    def test_zerorecall(self):
        assert fmt_recall(0.0) == "0.000"


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------


class TestRecall:
    def test_perfectrecall(self):
        assert recall(["a", "b", "c"], ["a", "b", "c", "d"]) == pytest.approx(1.0)

    def test_zerorecall(self):
        assert recall(["a", "b", "c"], ["d", "e"]) == pytest.approx(0.0)

    def test_partialrecall(self):
        # 2 of 4 baseline items recovered; x/y don't appear in baseline
        assert recall(["a", "b", "c", "d"], ["a", "b", "x", "y"]) == pytest.approx(0.5)

    def test_extra_items_in_tested_do_not_inflaterecall(self):
        assert recall(["a", "b"], ["z1", "z2", "a", "b"]) == pytest.approx(1.0)

    def test_empty_baseline_returns_nan(self):
        assert math.isnan(recall([], ["a", "b"]))

    def test_empty_tested_returns_nan(self):
        assert math.isnan(recall(["a", "b"], []))


# ---------------------------------------------------------------------------
# load_annotation_slices — in-memory SQLite fixture
# ---------------------------------------------------------------------------


def _make_db(tmp_path) -> str:
    """Create a minimal annotations.db under tmp_path and return its path."""
    db_path = str(tmp_path / "annotations.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE clips (
            clip_id TEXT PRIMARY KEY,
            data_source TEXT,
            country TEXT
        );
        CREATE TABLE annotations (
            uid TEXT PRIMARY KEY,
            project TEXT,
            clip_id TEXT,
            key TEXT,
            value REAL,
            start_time REAL,
            end_time REAL,
            label_type TEXT
        );
        """
    )
    # clips: 3 from MADS, 2 from Nexar, 1 from Other
    clips = [
        ("mads-1", "MADS", "US"),
        ("mads-2", "MADS", "US"),
        ("mads-3", "MADS", "DE"),
        ("nexar-1", "Nexar", "GB"),
        ("nexar-2", "Nexar", "FR"),
        ("other-1", "Other", "AU"),
    ]
    conn.executemany("INSERT INTO clips VALUES (?,?,?)", clips)

    # annotations: night on mads-1, mads-2; sign on nexar-1; both on mads-3
    annotations = [
        ("a1", "proj", "mads-1", "night_scene", None, 0, 5, "manual"),
        ("a2", "proj", "mads-2", "night_driving", None, 0, 5, "manual"),
        ("a3", "proj", "mads-3", "night_scene", None, 0, 5, "manual"),
        ("a4", "proj", "mads-3", "sign_visible", None, 0, 5, "manual"),
        ("a5", "proj", "nexar-1", "person_with_sign", None, 0, 5, "manual"),
    ]
    conn.executemany(
        "INSERT INTO annotations VALUES (?,?,?,?,?,?,?,?)", annotations
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
def db_path(tmp_path):
    return _make_db(tmp_path)


ALL_IDS = {
    "mads-1", "mads-2", "mads-3", "nexar-1", "nexar-2", "other-1"
}


class TestLoadAnnotationSlices:
    def _slice_by_name(self, slices, name) -> DataSlice:
        for s in slices:
            if s.name == name:
                return s
        raise KeyError(f"No slice named {name!r}")

    def test_data_source_mads(self, db_path):
        slices = load_annotation_slices(db_path, ALL_IDS, ["MADS"], [])
        sl = self._slice_by_name(slices, "mads")
        assert sl.clip_ids == {"mads-1", "mads-2", "mads-3"}

    def test_data_source_nexar(self, db_path):
        slices = load_annotation_slices(db_path, ALL_IDS, ["Nexar"], [])
        sl = self._slice_by_name(slices, "nexar")
        assert sl.clip_ids == {"nexar-1", "nexar-2"}

    def test_data_source_intersected_with_index(self, db_path):
        """Clips not in all_clip_ids are excluded even if they match the DS."""
        subset = {"mads-1", "mads-2"}  # mads-3 is absent from the index
        slices = load_annotation_slices(db_path, subset, ["MADS"], [])
        sl = self._slice_by_name(slices, "mads")
        assert sl.clip_ids == {"mads-1", "mads-2"}

    def test_annotation_slice_night(self, db_path):
        slices = load_annotation_slices(
            db_path, ALL_IDS, [], [("night", "%night%")]
        )
        sl = self._slice_by_name(slices, "night")
        assert sl.clip_ids == {"mads-1", "mads-2", "mads-3"}

    def test_annotation_slice_sign(self, db_path):
        slices = load_annotation_slices(
            db_path, ALL_IDS, [], [("sign", "%sign%")]
        )
        sl = self._slice_by_name(slices, "sign")
        assert sl.clip_ids == {"mads-3", "nexar-1"}

    def test_pairwise_intersection_mads_night(self, db_path):
        slices = load_annotation_slices(
            db_path, ALL_IDS, ["MADS"], [("night", "%night%")]
        )
        sl = self._slice_by_name(slices, "mads_night")
        # MADS clips with a night annotation
        assert sl.clip_ids == {"mads-1", "mads-2", "mads-3"}

    def test_pairwise_intersection_nexar_sign(self, db_path):
        slices = load_annotation_slices(
            db_path, ALL_IDS, ["Nexar"], [("sign", "%sign%")]
        )
        sl = self._slice_by_name(slices, "nexar_sign")
        assert sl.clip_ids == {"nexar-1"}

    def test_pairwise_intersection_empty(self, db_path):
        """MADS × sign should only contain mads-3 (has both DS and sign ann)."""
        slices = load_annotation_slices(
            db_path, ALL_IDS, ["MADS"], [("sign", "%sign%")]
        )
        sl = self._slice_by_name(slices, "mads_sign")
        assert sl.clip_ids == {"mads-3"}

    def test_number_of_slices(self, db_path):
        """With 2 data-sources and 2 annotation specs → 2 + 2 + 4 = 8 slices."""
        slices = load_annotation_slices(
            db_path,
            ALL_IDS,
            ["MADS", "Nexar"],
            [("night", "%night%"), ("sign", "%sign%")],
        )
        assert len(slices) == 8  # 2 DS + 2 ann + 4 intersections

    def test_empty_db_ids_filtered(self, db_path):
        """Passing an empty index set yields empty slices."""
        slices = load_annotation_slices(
            db_path, set(), ["MADS"], [("night", "%night%")]
        )
        for sl in slices:
            assert len(sl.clip_ids) == 0
