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

"""Tests for SQLiteDataStore search and retrieval methods."""
import time

from sil_wheel.stores.search_utils import SearchFilters

PROJECT = "proj-A"


def _filters(query_dict):
    """Helper: build SearchFilters with a fixed project_source."""
    query_dict.setdefault("project_source", [PROJECT])
    return SearchFilters.from_query(query_dict)


def _default(store):
    return store.default_results


class TestGetFiltered:
    def test_get_filtered_no_filter(self, data_store):
        """No filter → all clips with video paths are returned."""
        results = _default(data_store)
        # All 10 clips have video paths inserted in the fixture
        assert len(results) == 10

    def test_get_filtered_by_data_source(self, data_store):
        """data_sources=['src-A'] → only src-A clips."""
        filters = _filters({"data_source": ["src-A"]})
        results = data_store.search(filters, _default(data_store))
        assert len(results) == 5
        for clip_id in results:
            assert clip_id.startswith("ds-A-")

    def test_get_filtered_by_annotation(self, data_store):
        """annotation_filter=['turn_left'] → only clips labelled turn_left."""
        filters = _filters({"filter": ["turn_left"]})
        results = data_store.search(filters, _default(data_store))
        # Clips labelled turn_left in fixture: ds-A-clip-01, ds-A-clip-02, ds-B-clip-01
        assert len(results) == 3
        for clip_id in results:
            assert clip_id in ("ds-A-clip-01", "ds-A-clip-02", "ds-B-clip-01")

    def test_get_filtered_exclude_label(self, data_store):
        """labels_to_exclude=['turn_left'] → annotated turn_left clips excluded."""
        filters = _filters({"labels_to_exclude": ["turn_left"]})
        results = data_store.search(filters, _default(data_store))
        for clip_id in results:
            assert clip_id not in (
                "ds-A-clip-01",
                "ds-A-clip-02",
                "ds-B-clip-01",
            )

    def test_get_filtered_without_annotations(self, data_store):
        """without_ann=True → only unannotated clips."""
        filters = _filters({"without_ann": ["true"]})
        results = data_store.search(filters, _default(data_store))
        # Unannotated clips: ds-A-clip-05, ds-B-clip-04, ds-B-clip-05
        assert len(results) >= 1
        # None of the annotated clips should be present
        annotated = {
            "ds-A-clip-01",
            "ds-A-clip-02",
            "ds-A-clip-03",
            "ds-A-clip-04",
            "ds-B-clip-01",
            "ds-B-clip-02",
            "ds-B-clip-03",
        }
        assert not set(results).intersection(annotated)

    def test_get_filtered_combined(self, data_store):
        """data_source + annotation_filter → intersection."""
        filters = _filters(
            {"data_source": ["src-A"], "filter": ["turn_left"]}
        )
        results = data_store.search(filters, _default(data_store))
        # Only src-A clips with turn_left: ds-A-clip-01, ds-A-clip-02
        assert len(results) == 2
        for clip_id in results:
            assert clip_id in ("ds-A-clip-01", "ds-A-clip-02")


class TestGetClipsDict:
    def test_get_clips_dict(self, data_store):
        """Hydrating 5 clips returns the correct clip_ids."""
        clip_ids = ["ds-A-clip-01", "ds-A-clip-02", "ds-B-clip-01"]
        result = data_store.get_clips_dict(clip_ids, [PROJECT])
        assert set(result.keys()) == set(clip_ids)

    def test_get_clips_dict_annotations_present(self, data_store):
        """Hydrated clips include the expected annotations."""
        clip_ids = ["ds-A-clip-01"]
        result = data_store.get_clips_dict(clip_ids, [PROJECT])
        clip = result["ds-A-clip-01"]
        keys = {a.key for a in clip.annotations}
        assert "turn_left" in keys
        assert "brake" in keys

    def test_get_single_clip(self, data_store):
        """get() returns a Clip dataclass for the requested clip_id."""
        clip = data_store.get("ds-A-clip-01", [PROJECT])
        assert clip.clip_id == "ds-A-clip-01"
        assert clip.data_source == "src-A"


class TestDataSourceIndex:
    def test_get_clip_ids_for_data_sources(self, data_store):
        """Reverse-index lookup returns correct clips for src-A."""
        ids = data_store.get_clip_ids_for_data_sources(("src-A",))
        assert len(ids) == 5
        for cid in ids:
            assert cid.startswith("ds-A-")

    def test_get_clip_ids_for_data_sources_src_b(self, data_store):
        """Reverse-index lookup returns correct clips for src-B."""
        ids = data_store.get_clip_ids_for_data_sources(("src-B",))
        assert len(ids) == 5
        for cid in ids:
            assert cid.startswith("ds-B-")


class TestCountryFilter:
    def test_search_filters_country(self, data_store):
        """Clips with country='DE' are returned when filtering by DE."""
        filters = _filters({"search_country": ["DE"]})
        results = data_store.search(filters, _default(data_store))
        # DE clips: ds-A-clip-01, ds-A-clip-02, ds-A-clip-05, ds-B-clip-03
        assert len(results) >= 1
        de_clips = {
            "ds-A-clip-01",
            "ds-A-clip-02",
            "ds-A-clip-05",
            "ds-B-clip-03",
        }
        assert set(results).issubset(de_clips)


class TestLatency:
    def test_latency_default_results(self, data_store):
        """Accessing default_results completes in ≤ 0.1 s."""
        t0 = time.perf_counter()
        _ = data_store.default_results
        elapsed = time.perf_counter() - t0
        assert elapsed <= 0.1, f"default_results took {elapsed:.4f}s"

    def test_latency_filtered(self, data_store):
        """search() with a single data-source filter completes in ≤ 0.5 s."""
        filters = _filters({"data_source": ["src-A"]})
        t0 = time.perf_counter()
        data_store.search(filters, _default(data_store))
        elapsed = time.perf_counter() - t0
        assert elapsed <= 0.5, f"search(data_source) took {elapsed:.4f}s"


class TestFilterMode:
    def test_or_mode_two_labels(self, data_store):
        """OR (default): clips with turn_left OR brake → union of both sets."""
        filters = _filters({"filter": ["turn_left||brake"]})
        results = data_store.search(filters, _default(data_store))
        # turn_left: ds-A-clip-01, ds-A-clip-02, ds-B-clip-01
        # brake:     ds-A-clip-01, ds-A-clip-04, ds-B-clip-02
        # union:     5 distinct clips
        assert len(results) == 5
        assert set(results) == {
            "ds-A-clip-01", "ds-A-clip-02", "ds-A-clip-04",
            "ds-B-clip-01", "ds-B-clip-02",
        }

    def test_and_mode_two_labels(self, data_store):
        """AND: clips with turn_left AND brake → only ds-A-clip-01."""
        filters = _filters({"filter": ["turn_left||brake"], "filter_mode": ["all"]})
        results = data_store.search(filters, _default(data_store))
        assert len(results) == 1
        assert set(results) == {"ds-A-clip-01"}

    def test_and_mode_returns_fewer_than_or(self, data_store):
        """AND must return a strict subset of OR results."""
        or_filters = _filters({"filter": ["turn_left||brake"]})
        and_filters = _filters({"filter": ["turn_left||brake"], "filter_mode": ["all"]})
        or_results = set(data_store.search(or_filters, _default(data_store)))
        and_results = set(data_store.search(and_filters, _default(data_store)))
        assert and_results < or_results

    def test_and_mode_no_overlap(self, data_store):
        """AND with labels no single clip holds both → empty result."""
        filters = _filters({"filter": ["turn_left||turn_right"], "filter_mode": ["all"]})
        results = data_store.search(filters, _default(data_store))
        assert len(results) == 0
