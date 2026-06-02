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

"""Tests for SearchFilters utilities."""
from sil_wheel.stores.search_utils import SearchFilters


class TestHasPriorFilters:
    def test_no_filters_returns_false(self):
        filters = SearchFilters.from_query({"search": ["pedestrian"]})
        assert not filters.has_prior_filters("search")

    def test_trajectory_pattern_returns_true(self):
        filters = SearchFilters.from_query(
            {"search": ["pedestrian"], "trajectory_pattern": ["left_turn"]}
        )
        assert filters.has_prior_filters("search")

    def test_semantic_search_text_returns_true(self):
        filters = SearchFilters.from_query(
            {"search": ["pedestrian"], "semantic_search_text": ["night driving"]}
        )
        assert filters.has_prior_filters("search")

    def test_caption_embed_returns_true(self):
        filters = SearchFilters.from_query(
            {"search": ["pedestrian"], "caption_embed_search": ["rainy highway"]}
        )
        assert filters.has_prior_filters("search")

    def test_excluded_filter_not_counted(self):
        # search itself must not count as a prior filter for caption search
        filters = SearchFilters.from_query({"search": ["pedestrian"]})
        assert not filters.has_prior_filters("search")

    def test_other_filter_excluded(self):
        # trajectory_pattern alone must not count as a prior filter for itself
        filters = SearchFilters.from_query({"trajectory_pattern": ["left_turn"]})
        assert not filters.has_prior_filters("trajectory_pattern")
