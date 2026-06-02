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

"""Tests for FTSCaptionStore search API."""
import sqlite3
import time

import pandas as pd
import pytest

from sil_wheel.stores.search_utils import SearchFilters, SearchResults
from sil_wheel.stores.sqlite_caption_store import FTSCaptionStore


class TestSanitizeQuery:
    def test_single_token(self):
        assert FTSCaptionStore._sanitize_fts5_query("pedestrian") == '"pedestrian"'

    def test_multi_word_kept_as_phrase(self):
        assert FTSCaptionStore._sanitize_fts5_query("homeless on the road") == '"homeless on the road"'

    def test_quoted_phrase_preserved(self):
        assert FTSCaptionStore._sanitize_fts5_query('"red light"') == '"red light"'

    def test_mixed_quoted_and_unquoted(self):
        assert FTSCaptionStore._sanitize_fts5_query('"red light" on the road') == '"red light" "on the road"'

    def test_hyphen_normalized(self):
        assert FTSCaptionStore._sanitize_fts5_query("left-turn signal") == '"left turn signal"'


class TestCaptionSearch:
    def test_search_returns_matching_clips(self, caption_db):
        """A known keyword returns at least one result."""
        results = caption_db._inner_search("pedestrian")
        assert len(results) >= 1

    def test_search_no_match_returns_empty(self, caption_db):
        """A query for a non-existent term returns empty."""
        results = caption_db._inner_search("xyzzy_nonexistent_term_42")
        assert results == []

    def test_search_with_data_source_filter(self, caption_db):
        """Filtering by one data source returns a subset of unfiltered results."""
        all_results = caption_db._inner_search("highway")
        filtered = caption_db._inner_search("highway", data_sources=["src-A"])

        # Both should be non-empty given our fixture data
        assert len(all_results) >= 1
        # Filtered results must be a subset
        assert set(filtered).issubset(set(all_results))

    def test_search_phrase_match(self, caption_db):
        """FTS5 phrase query returns only clips containing the exact phrase."""
        # "red light" appears in some captions; phrase search narrows results
        results_phrase = caption_db._inner_search('"red light"')
        results_word = caption_db._inner_search("red")

        # Phrase results must be subset of single-word results
        assert set(results_phrase).issubset(set(results_word))

    def test_search_limit_respected(self, caption_db):
        """search() with limit=2 returns at most 2 clips."""
        # "the" appears in many captions; limit must cap the result
        results = caption_db._inner_search("the", limit=2)
        assert len(results) <= 2

    def test_search_cache_hit_fast(self, caption_db):
        """The second identical call uses the LRU cache and is fast (≤ 0.01 s)."""
        query = "pedestrian"
        # Prime the cache
        caption_db._inner_search(query)

        t0 = time.perf_counter()
        caption_db._inner_search(query)
        elapsed = time.perf_counter() - t0

        assert elapsed <= 0.01, f"Cache hit took {elapsed:.4f}s, expected ≤ 0.01s"

    def test_search_latency(self, caption_db):
        """Any query on ~20 rows completes in ≤ 0.5 s."""
        t0 = time.perf_counter()
        caption_db._inner_search("highway")
        elapsed = time.perf_counter() - t0

        assert elapsed <= 0.5, f"Search took {elapsed:.4f}s, expected ≤ 0.5s"

    def test_search_via_store_method(self, caption_db):
        """The public search() method on SearchFilters delegates correctly."""
        filters = SearchFilters.from_query({"search": ["pedestrian"]})
        universe = {
            clip_id: SearchResults.default
            for clip_id, _, _ in [
                ("clip-A-03", None, None),
                ("clip-B-03", None, None),
                ("clip-B-06", None, None),
                ("clip-A-01", None, None),
            ]
        }
        result = caption_db.search(filters, universe)
        # At least one pedestrian clip must survive the intersection
        assert len(result) >= 1

    def test_full_query_matches_only_complete_clips(self, tmp_path):
        """Of N clips, only the M containing the exact phrase are returned.

        'man on scooter' → FTS5 '"man on scooter"' (phrase).
        Clips that contain only some words must not appear in the results.
        """
        store = FTSCaptionStore(str(tmp_path / "precision.db"))
        M, N = 4, 10
        rows = (
            [
                {
                    "clip_id": f"full-{i:02d}",
                    "summary": "man on scooter weaving through traffic",
                    "start_time": 0.0,
                    "end_time": 5.0,
                }
                for i in range(M)
            ]
            + [
                {
                    "clip_id": f"partial-{i:02d}",
                    # has "man" and "scooter" but not the phrase "man on scooter"
                    "summary": "man riding scooter",
                    "start_time": 0.0,
                    "end_time": 5.0,
                }
                for i in range(N - M)
            ]
        )
        store.insert_from_dataframe(
            pd.DataFrame(rows), model_name="test", data_source="test-src"
        )

        results = store._inner_search("man on scooter")
        assert len(results) == M
        assert all(r.startswith("full-") for r in results)

    def test_search_data_source_filter_excludes_other(self, caption_db):
        """Filtering by src-A excludes clips from src-B."""
        results = caption_db._inner_search("driving", data_sources=["src-A"])
        # All returned clip_ids must start with "clip-A-"
        for clip_id in results:
            assert clip_id.startswith("clip-A-"), (
                f"Expected src-A clip but got {clip_id!r}"
            )


class TestCaptionGet:
    def test_get_all_models(self, caption_db):
        """get() with no model filter returns a dict keyed by model_name."""
        result = caption_db.get("clip-A-01")
        assert "test_model" in result
        assert isinstance(result["test_model"], list)
        assert len(result["test_model"]) >= 1

    def test_get_single_model(self, caption_db):
        """get() scoped to model_name returns only that model's entries."""
        result = caption_db.get("clip-A-01", model_name="test_model")
        assert list(result.keys()) == ["test_model"]
        assert len(result["test_model"]) >= 1

    def test_get_unknown_model_returns_empty_list(self, caption_db):
        """get() for a known clip but unknown model returns an empty list."""
        result = caption_db.get("clip-A-01", model_name="no_such_model")
        assert result == {"no_such_model": []}

    def test_get_unknown_clip_returns_empty_dict(self, caption_db):
        """get() for a clip not in the DB returns an empty dict."""
        result = caption_db.get("nonexistent-clip")
        assert result == {}

    def test_get_caption_fields(self, caption_db):
        """Each caption entry exposes caption, start_time, and end_time."""
        result = caption_db.get("clip-A-01")
        entry = result["test_model"][0]
        assert "caption" in entry
        assert "start_time" in entry
        assert "end_time" in entry


class TestFTSSchema:
    def test_data_source_sanitization(self, tmp_path):
        """data_source tokens with special chars are sanitized on both insert
        and query — e.g. 'AV-V2.1_train' still matches via sources:avv21train."""
        store = FTSCaptionStore(str(tmp_path / "san.db"))
        conn = sqlite3.connect(str(tmp_path / "san.db"))
        conn.execute(
            "INSERT INTO captions"
            " (clip_id, model_name, caption, data_source, start_time, end_time)"
            " VALUES ('clip-x', 'model', 'highway driving in rain',"
            " 'AV-V2.1_train', -1, -1)"
        )
        conn.commit()
        conn.close()

        with store.lock, store.conn:
            store._update_clip_fts(["clip-x"])

        results = store._inner_search("highway", data_sources=["AV-V2.1_train"])
        assert "clip-x" in results

    def test_multi_source_or_filter(self, caption_db):
        """data_sources list applies OR logic: combined results equal the union
        of each individual filter."""
        a = set(caption_db._inner_search("driving", data_sources=["src-A"]))
        b = set(caption_db._inner_search("driving", data_sources=["src-B"]))
        both = set(
            caption_db._inner_search("driving", data_sources=["src-A", "src-B"])
        )
        assert both == a | b
        assert len(both) > len(a)  # src-B contributes additional clips

    def test_cache_key_order_independent(self, caption_db):
        """data_sources in different orders must hit the same cache entry."""
        caption_db.searches.clear()
        r1 = caption_db._inner_search("highway", data_sources=["src-B", "src-A"])
        assert len(caption_db.searches) == 1  # one entry written

        r2 = caption_db._inner_search("highway", data_sources=["src-A", "src-B"])
        assert len(caption_db.searches) == 1  # same entry hit, not a second one
        assert set(r1) == set(r2)

    def test_incremental_fts_update(self, caption_db):
        """Adding a new caption to an existing clip updates its clip_fts row."""
        assert "clip-A-01" not in caption_db._inner_search("braking")

        with caption_db.lock, caption_db.conn:
            caption_db.conn.execute(
                "INSERT INTO captions"
                " (clip_id, model_name, caption, data_source, start_time, end_time)"
                " VALUES ('clip-A-01', 'test_model', 'Sudden braking event.',"
                " 'src-A', 1.0, 2.0)"
            )
            caption_db._update_clip_fts(["clip-A-01"])
        caption_db.searches.clear()

        assert "clip-A-01" in caption_db._inner_search("braking")


class TestInsertFromDataframe:
    def test_basic_insert_and_search(self, tmp_path):
        """insert_from_dataframe populates captions and makes them searchable."""
        store = FTSCaptionStore(str(tmp_path / "df.db"))
        df = pd.DataFrame([
            {
                "clip_id": "c1",
                "summary": "A cyclist crosses the road.",
                "start_time": 0.0,
                "end_time": 2.0,
            },
            {
                "clip_id": "c2",
                "summary": "Snowy highway conditions.",
                "start_time": 0.0,
                "end_time": 2.0,
            },
        ])
        store.insert_from_dataframe(df, model_name="m", data_source="src-test")

        assert "c1" in store._inner_search("cyclist")
        assert "c2" in store._inner_search("snowy")

    def test_insert_missing_columns_raises(self, tmp_path):
        """insert_from_dataframe raises ValueError on missing required columns."""
        store = FTSCaptionStore(str(tmp_path / "df2.db"))
        df = pd.DataFrame([{"clip_id": "x", "text": "wrong column name"}])
        with pytest.raises(ValueError, match="Missing required columns"):
            store.insert_from_dataframe(df, model_name="m", data_source="s")

    def test_insert_searchable_by_data_source(self, tmp_path):
        """After insert_from_dataframe, the data_source filter returns the
        inserted clips."""
        store = FTSCaptionStore(str(tmp_path / "df3.db"))
        df = pd.DataFrame([{
            "clip_id": "c1",
            "summary": "Highway exit ramp.",
            "start_time": 0.0,
            "end_time": 2.0,
        }])
        store.insert_from_dataframe(df, model_name="m", data_source="src-df")

        results = store._inner_search("highway", data_sources=["src-df"])
        assert "c1" in results


# ---------------------------------------------------------------------------
# Multi-query OR search (_inner_search with list + search() with caption_extra_queries)
# ---------------------------------------------------------------------------

# Fixture data (from conftest) for reference:
#   pedestrian → clip-A-03, clip-B-03, clip-B-06
#   highway    → clip-A-02, clip-A-06, clip-B-07
#   turn       → clip-A-01, clip-A-07, clip-B-05, clip-B-08
#   intersection → clip-A-01, clip-A-07

_ALL_CLIP_IDS = [
    "clip-A-01", "clip-A-02", "clip-A-03", "clip-A-04", "clip-A-05",
    "clip-A-06", "clip-A-07", "clip-A-08", "clip-A-09", "clip-A-10",
    "clip-B-01", "clip-B-02", "clip-B-03", "clip-B-04", "clip-B-05",
    "clip-B-06", "clip-B-07", "clip-B-08", "clip-B-09", "clip-B-10",
]


class TestTwoPathSearch:
    def test_targeted_search_post_filters_correctly(self, caption_db):
        """With a prior filter active, only universe clips matching the caption are returned."""
        universe = {cid: SearchResults.default for cid in _ALL_CLIP_IDS}
        filters = SearchFilters.from_query(
            {"search": ["pedestrian"], "trajectory_pattern": ["left_turn"]}
        )
        result = caption_db.search(filters, universe)
        expected = set(caption_db._inner_search("pedestrian")) & set(universe)
        assert set(result.keys()) == expected

    def test_targeted_search_excludes_non_universe_clips(self, caption_db):
        """targeted_search must not return clips outside current_results."""
        # Pass only src-A clips as the universe
        universe = {
            cid: SearchResults.default
            for cid in _ALL_CLIP_IDS
            if cid.startswith("clip-A-")
        }
        filters = SearchFilters.from_query(
            {"search": ["pedestrian"], "semantic_search_text": ["crossing"]}
        )
        result = caption_db.search(filters, universe)
        for clip_id in result:
            assert clip_id in universe

    def test_primary_path_used_when_no_prior_filter(self, caption_db):
        """Without prior filters, the full FTS scan path is taken."""
        universe = {cid: SearchResults.default for cid in _ALL_CLIP_IDS}
        filters = SearchFilters.from_query({"search": ["pedestrian"]})
        result = caption_db.search(filters, universe)
        expected = set(caption_db._inner_search("pedestrian")) & set(universe)
        assert set(result.keys()) == expected


class TestMultiQuerySearch:
    def test_union_covers_both_queries(self, caption_db):
        """OR query covers all clips that match either query."""
        r_ped = set(caption_db._inner_search("pedestrian"))
        r_hwy = set(caption_db._inner_search("highway"))
        assert r_ped and r_hwy  # fixture sanity

        union = set(caption_db._inner_search(["pedestrian", "highway"]))
        assert union == r_ped | r_hwy

    def test_no_duplicates_when_clip_matches_both(self, caption_db):
        """A clip matching both terms appears exactly once in OR results."""
        combined = caption_db._inner_search(["turn", "intersection"])
        assert len(combined) == len(set(combined))

    def test_single_element_matches_string_search(self, caption_db):
        """Single-element list returns the same clips as a plain string search."""
        assert set(caption_db._inner_search(["pedestrian"])) == set(
            caption_db._inner_search("pedestrian")
        )

    def test_repeated_query_deduplicates(self, caption_db):
        """Duplicate entries in the query list produce the same result as one."""
        once = set(caption_db._inner_search(["pedestrian"]))
        twice = set(caption_db._inner_search(["pedestrian", "pedestrian"]))
        assert once == twice

    def test_more_queries_return_more_results(self, caption_db):
        """Adding OR terms never shrinks results and grows them when new clips match."""
        one = caption_db._inner_search(["pedestrian"])
        two = caption_db._inner_search(["pedestrian", "highway"])
        three = caption_db._inner_search(["pedestrian", "highway", "turn"])

        assert len(two) > len(one), (
            f"2-query OR ({len(two)}) should exceed 1-query ({len(one)})"
        )
        assert len(three) > len(two), (
            f"3-query OR ({len(three)}) should exceed 2-query ({len(two)})"
        )
        assert set(one).issubset(set(two))
        assert set(two).issubset(set(three))

    def test_with_data_source_filter(self, caption_db):
        """Data source filter applied to OR query keeps results within the source."""
        result = caption_db._inner_search(
            ["pedestrian", "highway"], data_sources=["src-A"]
        )
        assert result
        for clip_id in result:
            assert clip_id.startswith("clip-A-"), clip_id

    def test_search_with_caption_extra_queries(self, caption_db):
        """search() with caption_extra_queries returns the union of all queries."""
        universe = {cid: SearchResults.default for cid in _ALL_CLIP_IDS}

        filters = SearchFilters.from_query(
            {"search": ["pedestrian"], "caption_extra_queries": ["highway||turn"]}
        )
        result = caption_db.search(filters, universe)

        expected = (
            set(caption_db._inner_search("pedestrian"))
            | set(caption_db._inner_search("highway"))
            | set(caption_db._inner_search("turn"))
        ) & set(universe)
        assert set(result.keys()) == expected

    def test_search_deduplicates_original_in_extra(self, caption_db):
        """If the original query appears in caption_extra_queries it runs only once."""
        universe = {cid: SearchResults.default for cid in _ALL_CLIP_IDS}

        filters = SearchFilters.from_query(
            {"search": ["pedestrian"], "caption_extra_queries": ["pedestrian"]}
        )
        result = caption_db.search(filters, universe)

        expected = set(caption_db._inner_search("pedestrian")) & set(universe)
        assert set(result.keys()) == expected

    def test_limit_sweep_timing(self, tmp_path):
        """Latency across increasing limits on a 1 000-row DB.

        Asserts that result counts grow monotonically with limit and that every
        query — even at the largest limit — completes within 1 s on a cold cache.
        Prints a timing table so the numbers are visible in test output.
        """
        store = FTSCaptionStore(str(tmp_path / "sweep.db"))
        captions = [
            "vehicle drives through intersection at night",
            "pedestrian crosses the road near the school zone",
            "emergency vehicle approaches with siren on highway",
            "sharp left turn at a busy urban intersection",
            "cyclist weaving through traffic in the rain",
        ]
        rows = [
            {
                "clip_id": f"clip-{i:04d}",
                "summary": captions[i % len(captions)],
                "start_time": 0.0,
                "end_time": 5.0,
            }
            for i in range(1000)
        ]
        store.insert_from_dataframe(
            pd.DataFrame(rows), model_name="m", data_source="sweep-src"
        )

        limits = [5_000, 10_000, 30_000, 50_000, 100_000, 200_000, 500_000]
        query = "vehicle"

        print(f"\n{'limit':>6}  {'seconds':>8}  {'count':>6}")
        print("-" * 26)
        prev_count = -1
        for lim in limits:
            store.searches.clear()
            t0 = time.perf_counter()
            results = store._inner_search(query, limit=lim)
            elapsed = time.perf_counter() - t0
            count = len(results)
            print(f"{lim:>6}  {elapsed:>8.4f}  {count:>6}")
            assert count >= prev_count, (
                f"count regressed: limit={lim} gave {count}, "
                f"previous was {prev_count}"
            )
            assert elapsed < 1.0, f"limit={lim} took {elapsed:.3f}s (> 1s)"
            prev_count = count

    @pytest.mark.skip(
        reason="Timing-sensitive perf test; flaky on busy machines. "
        "Run manually when validating FTS5 OR-query performance."
    )
    def test_or_query_faster_than_sequential(self, tmp_path):
        """Single OR query is faster than the same N queries run independently."""
        store = FTSCaptionStore(str(tmp_path / "perf.db"))
        captions = [
            "man on scooter weaving through traffic near crosswalk",
            "vehicle turning left at busy intersection with pedestrians",
            "pedestrian crossing road against red light in urban area",
            "highway at night with heavy rain reducing visibility",
            "cyclist near intersection signaling before lane change",
            "bus stopping abruptly causing vehicles to brake suddenly",
            "construction zone with workers on road and slow traffic",
            "emergency vehicle passing through red light with sirens",
            "parking lot with multiple vehicles reversing simultaneously",
            "scooter rider on sidewalk narrowly avoiding pedestrian",
        ]
        # 50 000 rows needed so FTS5 has enough work that the single OR query's
        # savings (one conn.execute + one BM25 scan) beat 5× sequential queries.
        rows = [
            {
                "clip_id": f"clip-{i:05d}",
                "summary": captions[i % len(captions)],
                "start_time": 0.0,
                "end_time": 5.0,
            }
            for i in range(50000)
        ]
        store.insert_from_dataframe(
            pd.DataFrame(rows), model_name="perf", data_source="perf-src"
        )

        queries = [
            "man on scooter",
            "vehicle turning left",
            "pedestrian crossing road",
            "highway at night",
            "cyclist near intersection",
        ]
        N = 20

        times_seq, times_or = [], []
        for _ in range(N):
            store.searches.clear()
            t0 = time.perf_counter()
            for q in queries:
                store._inner_search(q)
            times_seq.append(time.perf_counter() - t0)

            store.searches.clear()
            t0 = time.perf_counter()
            store._inner_search(queries)
            times_or.append(time.perf_counter() - t0)

        min_seq = min(times_seq)
        min_or = min(times_or)
        assert min_or < min_seq, (
            f"OR query min={min_or:.4f}s should be faster than "
            f"{len(queries)} sequential queries min={min_seq:.4f}s"
        )
