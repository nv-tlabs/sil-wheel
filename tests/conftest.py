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

"""Shared pytest fixtures for server search API tests."""
import sqlite3

import pytest

from sil_wheel.stores.sqlite_caption_store import FTSCaptionStore
from sil_wheel.stores.sqlite_data_store import SQLiteDataStore
from sil_wheel.stores.users_data_store import UsersDataStore


@pytest.fixture()
def users_db(tmp_path):
    """Empty UsersDataStore on a temp SQLite file."""
    return UsersDataStore(str(tmp_path / "users.db"))


# ---------------------------------------------------------------------------
# Caption store fixture
# ---------------------------------------------------------------------------

_CAPTIONS = [
    # (clip_id, caption, data_source)
    ("clip-A-01", "A vehicle makes a sharp left turn at the intersection.", "src-A"),
    ("clip-A-02", "Heavy rain reduces visibility on the highway.", "src-A"),
    ("clip-A-03", "A pedestrian crosses the street at a red traffic light.", "src-A"),
    ("clip-A-04", "The car brakes hard to avoid a cyclist.", "src-A"),
    ("clip-A-05", "Stop and go traffic in an urban environment.", "src-A"),
    ("clip-A-06", "The vehicle accelerates on the highway after the merge.", "src-A"),
    ("clip-A-07", "A truck signals and turns right at the intersection.", "src-A"),
    ("clip-A-08", "Night driving with wet road surface after rain.", "src-A"),
    ("clip-A-09", "Emergency vehicle passes on the left lane.", "src-A"),
    ("clip-A-10", "Red light stop followed by smooth acceleration.", "src-A"),
    ("clip-B-01", "Merging onto a busy freeway with fast moving traffic.", "src-B"),
    ("clip-B-02", "Roundabout navigation with multiple yielding events.", "src-B"),
    ("clip-B-03", "School zone with pedestrians and slow speed driving.", "src-B"),
    ("clip-B-04", "Construction zone with lane shifts and speed restrictions.", "src-B"),
    ("clip-B-05", "Off-ramp turn followed by city street driving.", "src-B"),
    ("clip-B-06", "The ego vehicle stops at a pedestrian crossing.", "src-B"),
    ("clip-B-07", "High speed lane change on the highway.", "src-B"),
    ("clip-B-08", "U-turn performed in a residential area.", "src-B"),
    ("clip-B-09", "Traffic cone avoidance with gentle steering.", "src-B"),
    ("clip-B-10", "Parallel parking maneuver completed successfully.", "src-B"),
]


@pytest.fixture()
def caption_db(tmp_path):
    """FTSCaptionStore populated with ~20 synthetic captions."""
    db_path = str(tmp_path / "captions.db")
    store = FTSCaptionStore(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = [
        (clip_id, "test_model", caption, data_source, -1.0, -1.0)
        for clip_id, caption, data_source in _CAPTIONS
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO captions
            (clip_id, model_name, caption, data_source, start_time, end_time)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()

    # Backfill clip_fts and clip_fts_index via the store's own method so the
    # fixture uses the same aggregation logic as production.
    clip_ids = [clip_id for clip_id, _, _ in _CAPTIONS]
    with store.lock, store.conn:
        store._update_clip_fts(clip_ids)

    return store


# ---------------------------------------------------------------------------
# Data store fixture
# ---------------------------------------------------------------------------

_CLIPS = [
    # (clip_id, data_source, country)
    ("ds-A-clip-01", "src-A", "DE"),
    ("ds-A-clip-02", "src-A", "DE"),
    ("ds-A-clip-03", "src-A", "US"),
    ("ds-A-clip-04", "src-A", "US"),
    ("ds-A-clip-05", "src-A", "DE"),
    ("ds-B-clip-01", "src-B", "FR"),
    ("ds-B-clip-02", "src-B", "FR"),
    ("ds-B-clip-03", "src-B", "DE"),
    ("ds-B-clip-04", "src-B", "US"),
    ("ds-B-clip-05", "src-B", "GB"),
]

_ANNOTATIONS = [
    # (uid, project, clip_id, key, start_time, end_time, label_type)
    ("ann-01", "proj-A", "ds-A-clip-01", "turn_left",  -1.0, -1.0, "manual"),
    ("ann-02", "proj-A", "ds-A-clip-01", "brake",       -1.0, -1.0, "manual"),
    ("ann-03", "proj-A", "ds-A-clip-02", "turn_left",  -1.0, -1.0, "manual"),
    ("ann-04", "proj-A", "ds-A-clip-03", "turn_right", -1.0, -1.0, "manual"),
    ("ann-05", "proj-A", "ds-A-clip-04", "brake",       -1.0, -1.0, "manual"),
    ("ann-06", "proj-A", "ds-B-clip-01", "turn_left",  -1.0, -1.0, "manual"),
    ("ann-07", "proj-A", "ds-B-clip-02", "brake",       -1.0, -1.0, "manual"),
    ("ann-08", "proj-A", "ds-B-clip-03", "turn_right", -1.0, -1.0, "manual"),
    # Clips 05 and 04 of each source are unannotated
]


@pytest.fixture()
def data_store(tmp_path):
    """SQLiteDataStore with ~10 clips and ~8 annotations across 2 data sources."""
    db_path = str(tmp_path / "annotations.db")

    # Pre-populate via direct SQLite so we control the schema exactly
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS clips (
            clip_id TEXT PRIMARY KEY,
            data_source TEXT,
            country TEXT,
            has_time INTEGER DEFAULT 0,
            has_manual_annotations INTEGER DEFAULT 0,
            has_autolabels INTEGER DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS annotations (
            uid TEXT PRIMARY KEY,
            project TEXT,
            clip_id TEXT,
            key TEXT,
            value REAL,
            start_time REAL,
            end_time REAL,
            label_type TEXT,
            FOREIGN KEY (clip_id) REFERENCES clips (clip_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS video_paths (
            clip_id TEXT PRIMARY KEY,
            path TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO clips (clip_id, data_source, country) VALUES (?, ?, ?)",
        _CLIPS,
    )
    conn.executemany(
        "INSERT INTO video_paths (clip_id, path) VALUES (?, ?)",
        [(c[0], f"/fake/{c[0]}.mp4") for c in _CLIPS],
    )
    conn.executemany(
        """
        INSERT INTO annotations
            (uid, project, clip_id, key, start_time, end_time, label_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        _ANNOTATIONS,
    )
    conn.commit()
    conn.close()

    return SQLiteDataStore(db_path)
