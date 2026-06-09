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

import sqlite3

import pytest

from caption_quality import load_pairs


def _make_captions_db(path, with_question=False):
    con = sqlite3.connect(path)
    q_col = ", question TEXT" if with_question else ""
    con.execute(
        f"CREATE TABLE captions(uid INTEGER PRIMARY KEY, clip_id TEXT, model_name TEXT, "
        f"caption TEXT, data_source TEXT{q_col}, start_time REAL, end_time REAL)"
    )
    rows = []
    for i in range(3):
        cid = f"clip{i}"
        if with_question:
            rows.append((cid, "ref_model", f"reference {i}", "src", f"question {i}"))
            rows.append((cid, "pred_model", f"prediction {i}", "src", f"question {i}"))
        else:
            rows.append((cid, "ref_model", f"reference {i}", "src"))
            rows.append((cid, "pred_model", f"prediction {i}", "src"))
    cols = "clip_id, model_name, caption, data_source" + (", question" if with_question else "")
    ph = ",".join("?" * (5 if with_question else 4))
    con.executemany(f"INSERT INTO captions({cols}) VALUES ({ph})", rows)
    con.commit()
    con.close()


def test_caption_vs_caption_pairs(tmp_path):
    db = str(tmp_path / "captions.db")
    _make_captions_db(db)
    pairs = load_pairs(db, reference_model="ref_model", prediction_model="pred_model")
    assert len(pairs) == 3
    p = next(p for p in pairs if p["clip_id"] == "clip0")
    assert p["reference"] == "reference 0"
    assert p["prediction"] == "prediction 0"
    assert p["data_source"] == "src"
    # No question column -> key absent (not None).
    assert "question" not in p


def test_question_column_passthrough(tmp_path):
    db = str(tmp_path / "captions.db")
    _make_captions_db(db, with_question=True)
    pairs = load_pairs(db, reference_model="ref_model", prediction_model="pred_model")
    assert len(pairs) == 3
    assert all(p["question"] == f"question {p['clip_id'][-1]}" for p in pairs)


def test_num_samples_caps_pairs_deterministically(tmp_path):
    db = str(tmp_path / "captions.db")
    _make_captions_db(db)
    a = load_pairs(db, "ref_model", "pred_model", num_samples=2, seed=7)
    b = load_pairs(db, "ref_model", "pred_model", num_samples=2, seed=7)
    assert len(a) == 2
    assert [p["clip_id"] for p in a] == [p["clip_id"] for p in b]


def test_human_mode_requires_annotations_db(tmp_path):
    db = str(tmp_path / "captions.db")
    _make_captions_db(db)
    with pytest.raises(ValueError, match="annotations DB"):
        load_pairs(db, reference_model="human", prediction_model="pred_model")
