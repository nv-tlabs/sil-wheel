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

"""Tests for the per-cluster topic-extraction pipeline.

Builds a synthetic clustering run (parquet + SQLite captions DB) on disk in
a tmp_path and exercises ``extract_topics_for_run`` end-to-end. The test
mirrors what the clustering subprocess does at runtime — read the parquet,
fetch captions from SQLite, run TF-IDF, write ``cluster_topics.json`` —
without touching FAISS or running an actual KMeans job.
"""
import json
import sqlite3

import pandas as pd
import pytest

from sil_wheel.stores.cluster_topics import (
    DEFAULT_CAPTION_MODEL,
    _canonical_key,
    _summarize_topics,
    extract_topics_for_run,
    pick_highest_coverage_captions,
    read_topics,
)


# ---------------------------------------------------------------------------
# Fixture: synthetic clustering run on disk
# ---------------------------------------------------------------------------

# Three clusters with distinct themes. Each cluster shares a fixed
# environment prefix (so env terms are bursty within a cluster) plus a pool
# of varied event sentences (so per-clip TF-IDF + cluster-mean aggregation
# can surface action / agent terms on top of the env baseline).
_CLUSTER_THEMES = {
    0: {
        "env": "It is night and the ego vehicle is at an urban intersection.",
        "events": [
            "A pedestrian crosses the crosswalk in front of the ego vehicle.",
            "A cyclist passes through the intersection from the right.",
            "The ego vehicle yields to a delivery truck turning left.",
            "A person on a wheelchair waits at the corner.",
            "The traffic light turns green and the ego vehicle accelerates.",
            "Two pedestrians wait at the crosswalk while the light is red.",
            "A police car with flashing lights drives past.",
        ],
    },
    1: {
        "env": "It is a sunny day on a multi-lane highway.",
        "events": [
            "The ego vehicle merges into the right lane behind a trailer.",
            "A motorcycle weaves between the lanes ahead.",
            "The ego vehicle overtakes a slow truck on the left.",
            "A bus changes lane abruptly causing the ego vehicle to brake.",
            "The ego vehicle accelerates to pass a sedan.",
            "A delivery van enters from the on-ramp.",
            "A police vehicle with sirens passes on the shoulder.",
        ],
    },
    2: {
        "env": "Daytime in a quiet residential neighborhood.",
        "events": [
            "A child runs into the street chasing a ball.",
            "A dog crosses the road off-leash.",
            "An elderly person with a walker steps off the curb.",
            "The ego vehicle slows for a stroller crossing.",
            "A garbage truck blocks half the lane.",
            "A skateboarder rides down the middle of the road.",
            "A jogger emerges from between two parked cars.",
        ],
    },
}

_CLIPS_PER_CLUSTER = 80


def _build_run(tmp_path, clips_per_cluster=_CLIPS_PER_CLUSTER, seed=0):
    """Materialize a synthetic clustering run under tmp_path.

    Writes ``representative_by_cluster.json``, ``cluster_assignments.parquet``,
    and a ``captions.db`` SQLite file populated with one caption per clip.
    Returns ``(run_dir, captions_db_path)``.
    """
    import random
    rng = random.Random(seed)

    clip_ids, cluster_ids, distances = [], [], []
    captions = []  # parallel: (clip_id, caption)
    for cid, spec in _CLUSTER_THEMES.items():
        for i in range(clips_per_cluster):
            clip_id = f"c{cid}_{i:04d}"
            clip_ids.append(clip_id)
            cluster_ids.append(cid)
            # Distance is just a placeholder here — sampling is random, not
            # by-distance, so the value doesn't influence the test.
            distances.append(rng.random())
            event = rng.choice(spec["events"])
            captions.append((clip_id, f"{spec['env']} {event}"))

    df = (
        pd.DataFrame({
            "clip_id": clip_ids,
            "cluster_id": cluster_ids,
            "distance": distances,
        })
        .sort_values(["cluster_id", "distance"])
    )
    df.to_parquet(tmp_path / "cluster_assignments.parquet", index=False)

    clusters = {
        str(cid): {"cluster_size": clips_per_cluster}
        for cid in _CLUSTER_THEMES
    }
    (tmp_path / "representative_by_cluster.json").write_text(json.dumps(clusters))

    db_path = tmp_path / "captions.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE captions (
            uid INTEGER PRIMARY KEY,
            clip_id TEXT,
            model_name TEXT,
            caption TEXT,
            data_source TEXT,
            start_time REAL,
            end_time REAL
        )
        """
    )
    conn.executemany(
        "INSERT INTO captions(clip_id, model_name, caption, data_source, "
        "start_time, end_time) VALUES (?, ?, ?, ?, ?, ?)",
        [
            (clip_id, DEFAULT_CAPTION_MODEL, caption, "test", 0.0, 1.0)
            for clip_id, caption in captions
        ],
    )
    conn.commit()
    conn.close()

    return tmp_path, str(db_path)


@pytest.fixture()
def synthetic_run(tmp_path):
    """Three-cluster synthetic run with environment + event captions."""
    return _build_run(tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractTopicsForRun:
    def test_writes_topics_json(self, synthetic_run):
        """The function persists cluster_topics.json under run_dir, and the
        ``topics`` field of the on-disk wrapper matches the in-memory
        return value."""
        run_dir, db = synthetic_run
        result = extract_topics_for_run(run_dir, db, n_threads=2)
        out = run_dir / "cluster_topics.json"
        assert out.exists()
        on_disk = json.loads(out.read_text())
        assert on_disk["topics"] == result
        # Meta fields the UI relies on are present.
        assert "caption_model" in on_disk
        assert "captions_found" in on_disk
        assert "captions_total" in on_disk

    def test_one_entry_per_cluster(self, synthetic_run):
        """Every cluster id from the parquet appears in the result."""
        run_dir, db = synthetic_run
        result = extract_topics_for_run(run_dir, db, n_threads=2)
        assert set(result.keys()) == {"0", "1", "2"}

    def test_keywords_only_schema(self, synthetic_run):
        """Each entry exposes a non-empty `keywords` list and nothing else
        (the structured-bucket fields were intentionally removed)."""
        run_dir, db = synthetic_run
        result = extract_topics_for_run(run_dir, db, n_threads=2)
        for cid, entry in result.items():
            assert set(entry.keys()) == {"keywords"}, (
                f"cluster {cid} has unexpected keys: {set(entry.keys())}"
            )
            assert isinstance(entry["keywords"], list)
            assert len(entry["keywords"]) > 0
            assert all(isinstance(t, str) and t for t in entry["keywords"])

    def test_keywords_are_deduplicated(self, synthetic_run):
        """No two surviving keywords in a cluster should share a canonical
        form — guards against "pedestrian"/"pedestrians" or
        "lane changing"/"changing lane" both surviving the top-K cut."""
        run_dir, db = synthetic_run
        result = extract_topics_for_run(run_dir, db, n_threads=2)
        for cid, entry in result.items():
            canons = [_canonical_key(t) for t in entry["keywords"]]
            assert len(canons) == len(set(canons)), (
                f"cluster {cid} has near-duplicate keywords: "
                f"{entry['keywords']!r}"
            )

    def test_themes_distinguishable(self, synthetic_run):
        """Each cluster's keywords contain at least one term unambiguously
        tied to its theme — confirms TF-IDF + cluster-mean is actually
        separating clusters rather than collapsing onto shared boilerplate
        like 'ego vehicle'.
        """
        run_dir, db = synthetic_run
        result = extract_topics_for_run(run_dir, db, n_threads=2)

        markers = {
            "0": {"intersection", "crosswalk", "cyclist", "pedestrian", "night", "urban"},
            "1": {"highway", "lane", "motorcycle", "merges", "overtakes", "sunny"},
            "2": {"residential", "neighborhood", "child", "dog", "stroller", "skateboarder"},
        }
        for cid, expected in markers.items():
            kw_words = {w for term in result[cid]["keywords"] for w in term.split()}
            hit = expected & kw_words
            assert hit, (
                f"cluster {cid} keywords {result[cid]['keywords']!r} "
                f"contain none of {expected}"
            )

    def test_deterministic_across_runs(self, synthetic_run):
        """Same fixture + same sample_seed → identical output."""
        run_dir, db = synthetic_run
        a = extract_topics_for_run(run_dir, db, n_threads=2, sample_seed=7)
        b = extract_topics_for_run(run_dir, db, n_threads=2, sample_seed=7)
        assert a == b

    def test_seed_changes_sampling(self, tmp_path):
        """Different sample_seed picks a different subset → at least one
        cluster's keyword list shifts. Uses small samples_per_cluster so
        the variance is observable on a small fixture."""
        run_dir, db = _build_run(tmp_path, clips_per_cluster=40)
        a = extract_topics_for_run(
            run_dir, db, n_threads=2,
            samples_per_cluster=10, sample_seed=1,
        )
        b = extract_topics_for_run(
            run_dir, db, n_threads=2,
            samples_per_cluster=10, sample_seed=2,
        )
        assert a != b

    def test_samples_per_cluster_zero_uses_all_clips(self, synthetic_run):
        """``samples_per_cluster=0`` disables sampling and uses every clip.
        Sanity-checked by asserting the result is identical regardless of
        sample_seed (no sampling = no randomness)."""
        run_dir, db = synthetic_run
        a = extract_topics_for_run(
            run_dir, db, n_threads=2,
            samples_per_cluster=0, sample_seed=1,
        )
        b = extract_topics_for_run(
            run_dir, db, n_threads=2,
            samples_per_cluster=0, sample_seed=999,
        )
        assert a == b

    def test_missing_parquet_returns_empty(self, tmp_path):
        """If cluster_assignments.parquet is missing, the function returns
        an empty dict and writes no file."""
        # Only create the clusters json — no parquet.
        (tmp_path / "representative_by_cluster.json").write_text("{}")
        result = extract_topics_for_run(tmp_path, "/nonexistent.db")
        assert result == {}
        assert not (tmp_path / "cluster_topics.json").exists()

    def test_no_captions_writes_empty_marker(self, tmp_path):
        """When the captions DB has nothing for any clip in the run, an
        empty cluster_topics.json (with meta) is still written so the
        server stops polling and the done-detection gate flips."""
        run_dir, _ = _build_run(tmp_path)
        # Point at a fresh DB with the right schema but zero rows.
        empty_db = tmp_path / "empty.db"
        conn = sqlite3.connect(empty_db)
        conn.execute(
            "CREATE TABLE captions (uid INTEGER PRIMARY KEY, clip_id TEXT, "
            "model_name TEXT, caption TEXT, data_source TEXT, "
            "start_time REAL, end_time REAL)"
        )
        conn.commit()
        conn.close()

        result = extract_topics_for_run(run_dir, str(empty_db), n_threads=2)
        out = run_dir / "cluster_topics.json"
        assert out.exists()
        on_disk = json.loads(out.read_text())
        assert on_disk["topics"] == {}
        assert on_disk["caption_model"] is None
        assert on_disk["captions_found"] == 0
        assert on_disk["captions_total"] > 0  # we sampled some clip_ids
        assert result == {}

    def test_auto_select_picks_highest_coverage_model(self, tmp_path):
        """When ``model_name=None``, the function should pick whichever
        caption model_name covers the most clips in the run.

        Fixture: per cluster, half the clips get a "MajorityModel" caption
        and a smaller subset get a "MinorityModel" caption. Auto-select
        should land on MajorityModel."""
        # Hand-rolled DB so we control which model covers which clips.
        run_dir = tmp_path
        clips_per_cluster = 30
        clip_ids, cluster_ids, distances = [], [], []
        for cid in range(2):
            for i in range(clips_per_cluster):
                clip_ids.append(f"c{cid}_{i:03d}")
                cluster_ids.append(cid)
                distances.append(float(i))
        pd.DataFrame(
            {"clip_id": clip_ids, "cluster_id": cluster_ids, "distance": distances}
        ).to_parquet(run_dir / "cluster_assignments.parquet", index=False)
        (run_dir / "representative_by_cluster.json").write_text(
            json.dumps({"0": {"cluster_size": clips_per_cluster},
                        "1": {"cluster_size": clips_per_cluster}})
        )

        db_path = run_dir / "captions.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE captions (uid INTEGER PRIMARY KEY, clip_id TEXT, "
            "model_name TEXT, caption TEXT, data_source TEXT, "
            "start_time REAL, end_time REAL)"
        )
        # Distinct theme per cluster so TF-IDF can find at least one
        # discriminating term after pruning.
        themes = {
            0: "highway sedan merges lane multi traffic",
            1: "intersection pedestrian crosses crosswalk light",
        }
        rows = []
        # MajorityModel: every clip in both clusters.
        for cid in range(2):
            for i in range(clips_per_cluster):
                rows.append((
                    f"c{cid}_{i:03d}", "MajorityModel",
                    themes[cid], "test", 0.0, 1.0,
                ))
        # MinorityModel: only first 5 clips per cluster.
        for cid in range(2):
            for i in range(5):
                rows.append((
                    f"c{cid}_{i:03d}", "MinorityModel",
                    themes[cid], "test", 0.0, 1.0,
                ))
        conn.executemany(
            "INSERT INTO captions(clip_id, model_name, caption, data_source, "
            "start_time, end_time) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        conn.close()

        extract_topics_for_run(run_dir, str(db_path), n_threads=2)
        on_disk = json.loads((run_dir / "cluster_topics.json").read_text())
        assert on_disk["caption_model"] == "MajorityModel", (
            f"expected MajorityModel, got {on_disk['caption_model']!r}"
        )
        # MajorityModel covered every clip we sampled.
        assert on_disk["captions_found"] == on_disk["captions_total"]

    def test_explicit_model_name_overrides_auto_select(self, synthetic_run):
        """Passing ``model_name=...`` skips auto-select and uses the
        provided string verbatim, even if a different model has more
        coverage."""
        run_dir, db = synthetic_run
        # The fixture writes captions under DEFAULT_CAPTION_MODEL; pinning a
        # nonexistent model should produce zero coverage.
        extract_topics_for_run(
            run_dir, db, n_threads=2, model_name="NoSuchModel"
        )
        on_disk = json.loads((run_dir / "cluster_topics.json").read_text())
        assert on_disk["caption_model"] == "NoSuchModel"
        assert on_disk["captions_found"] == 0
        assert on_disk["topics"] == {}


class TestReadTopics:
    def test_round_trip(self, synthetic_run):
        """``read_topics`` returns the full cluster_topics.json payload."""
        run_dir, db = synthetic_run
        topics = extract_topics_for_run(run_dir, db, n_threads=2)
        payload = read_topics(run_dir)
        assert payload["topics"] == topics
        assert payload["caption_model"] == DEFAULT_CAPTION_MODEL
        assert payload["captions_found"] > 0
        assert payload["captions_total"] >= payload["captions_found"]

    def test_missing_file_returns_empty(self, tmp_path):
        assert read_topics(tmp_path) == {}

    def test_corrupt_file_returns_empty(self, tmp_path):
        (tmp_path / "cluster_topics.json").write_text("not valid json {{{")
        assert read_topics(tmp_path) == {}


class TestPickHighestCoverageCaptions:
    def test_picks_higher_coverage_model(self, tmp_path):
        db = tmp_path / "captions.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE captions (uid INTEGER PRIMARY KEY, clip_id TEXT, "
            "model_name TEXT, caption TEXT, data_source TEXT, "
            "start_time REAL, end_time REAL)"
        )
        # ModelA covers 10 clips, ModelB covers 4. Both have one row per clip.
        rows = []
        for i in range(10):
            rows.append((f"clip-{i:02d}", "ModelA", "x", "t", 0.0, 1.0))
        for i in range(4):
            rows.append((f"clip-{i:02d}", "ModelB", "x", "t", 0.0, 1.0))
        conn.executemany(
            "INSERT INTO captions(clip_id, model_name, caption, data_source, "
            "start_time, end_time) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        conn.close()

        clip_ids = [f"clip-{i:02d}" for i in range(10)]
        model, count = pick_highest_coverage_captions(str(db), clip_ids)
        assert model == "ModelA"
        assert count == 10

    def test_returns_none_when_no_captions(self, tmp_path):
        db = tmp_path / "empty.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE captions (uid INTEGER PRIMARY KEY, clip_id TEXT, "
            "model_name TEXT, caption TEXT, data_source TEXT, "
            "start_time REAL, end_time REAL)"
        )
        conn.commit()
        conn.close()
        model, count = pick_highest_coverage_captions(str(db), ["clip-A", "clip-B"])
        assert model is None
        assert count == 0

    def test_empty_clip_ids_returns_none(self, tmp_path):
        # No DB query should run when the input list is empty.
        model, count = pick_highest_coverage_captions("/nonexistent.db", [])
        assert model is None
        assert count == 0


class TestSummarizeTopics:
    def test_attaches_descriptions_and_isolates_failures(self, monkeypatch):
        """Happy path attaches a sanitized description; per-cluster failures
        (None content, raised exception) are isolated so other clusters in
        the same batch still get descriptions."""
        class Client:
            config = type("C", (), {"api_key": "fake"})()
            def generate(self, prompt, system_prompt, **kw):
                if "raise_me" in prompt:
                    raise RuntimeError("provider exploded")
                if "none_me" in prompt:
                    return None  # filtered / empty completion
                # Wrap in quotes + period to verify sanitization.
                return f'"{prompt.replace("Keywords: ", "")[:20]}."'

        import sys, types
        mod = types.ModuleType("sil_wheel.llm.llm_client")
        mod.LLMClient = lambda *a, **k: Client()
        monkeypatch.setitem(sys.modules, "sil_wheel.llm.llm_client", mod)

        topics = {
            "0": {"keywords": ["lane", "highway"]},   # happy path
            "1": {"keywords": ["raise_me"]},          # raises
            "2": {"keywords": ["none_me"]},           # None content
        }
        _summarize_topics(topics, n_threads=2)
        assert topics["0"]["description"]
        assert not topics["0"]["description"].endswith(".")
        assert "description" not in topics["1"]
        assert "description" not in topics["2"]


class TestCanonicalKey:
    def test_canonicalization(self):
        # Plural/inflection variants collapse.
        assert _canonical_key("pedestrian") == _canonical_key("pedestrians")
        assert _canonical_key("merging") == _canonical_key("merged")
        assert _canonical_key("city") == _canonical_key("cities")
        # Order-independent for n-grams.
        assert _canonical_key("lane changing") == _canonical_key("changing lane")
        # Distinct concepts stay distinct.
        assert _canonical_key("highway") != _canonical_key("highway lane")
        assert _canonical_key("bus") != _canonical_key("car")
        # Short words aren't over-stripped ("bus" -> "bus", not "b").
        assert _canonical_key("bus") == ("bus",)
