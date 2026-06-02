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

"""Smoke tests for `WheelClient`.

`WheelClient.from_config` is not tested here (needs real FAISS indices,
parquet shards, etc.). Instead we build the client directly with the
conftest fixtures + MagicMocks for the heavy stores, mirroring
`test_server_launch.py`.
"""
from unittest.mock import MagicMock

import pandas as pd
import pytest

from sil_wheel.client import WheelClient, WheelSearchResult
from sil_wheel.search.search_pipeline import SearchPipeline


def _passthrough_mock():
    """A store mock whose `.search(filters, results)` just returns `results`."""
    m = MagicMock()
    m.search = MagicMock(side_effect=lambda filters, results: results)
    return m


@pytest.fixture()
def client(data_store, caption_db):
    pipeline = SearchPipeline(
        datastore=data_store,
        captionstore=caption_db,
        captionembeddingsstore=_passthrough_mock(),
        embeddingsstore=_passthrough_mock(),
        clipembeddingsstore=_passthrough_mock(),
        classifiersearch=_passthrough_mock(),
        clustersearch=_passthrough_mock(),
        cliplistsearch=_passthrough_mock(),
        trajectorystore=_passthrough_mock(),
        metricstore=_passthrough_mock(),
        bev_fetcher=None,
        wm_store=_passthrough_mock(),
    )
    return WheelClient(pipeline=pipeline)


def test_search_returns_search_result(client):
    """Generic search returns a WheelSearchResult."""
    result = client.search()
    assert isinstance(result, WheelSearchResult)
    assert isinstance(result.clip_ids, list)
    assert isinstance(result.scores, dict)


def test_search_caption_propagates_to_filters(client):
    """search_caption sets `filters.search` on the SearchFilters."""
    result = client.search_caption("turn left")
    assert result.filters.search == "turn left"


def test_search_caption_kwargs_compose(client):
    """Combining caption + country filters yields a SearchFilters with both set."""
    result = client.search_caption("traffic", search_country="DE")
    assert result.filters.search == "traffic"
    assert result.filters.search_country == "DE"


def test_search_helpers_set_correct_filter_field(client):
    """Each convenience helper sets exactly the filter field its name implies."""
    cases = [
        ("search_caption", "x", "search"),
        ("search_semantic_text", "x", "semantic_search_text"),
        ("search_visual_text", "x", "visual_search_text"),
        ("search_clip", "abc", "semantic_search_clipid"),
        ("search_country", "US", "search_country"),
        ("search_trajectory_pattern", "hard_brake", "trajectory_pattern"),
    ]
    for method_name, value, expected_field in cases:
        result = getattr(client, method_name)(value)
        assert getattr(result.filters, expected_field) == value, method_name


def test_search_classifier_routes_run_and_expression(client):
    result = client.search_classifier("run-abc", "p > 0.9")
    assert result.filters.classifier_run_id == "run-abc"
    assert result.filters.probability_expression == "p > 0.9"


def test_search_cluster_routes_run_and_id(client):
    result = client.search_cluster("run-A", 42)
    assert result.filters.cluster_run_id == "run-A"
    assert result.filters.cluster_ids == ["42"]


def test_search_cluster_accepts_list(client):
    result = client.search_cluster("run-A", [1, 2, 3])
    assert result.filters.cluster_ids == ["1", "2", "3"]


def test_as_dataframe_returns_pandas(client):
    """Empty search (no filters) ranks every default clip — DataFrame is non-empty."""
    result = client.search()
    df = result.as_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(result.clip_ids)
    if len(df) > 0:
        assert "clip_id" in df.columns
        assert "rank" in df.columns


def test_head_truncates(client):
    result = client.search()
    assert len(result.head(3)) <= 3
