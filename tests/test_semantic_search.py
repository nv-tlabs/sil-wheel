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

"""Tests for CosmosEmbeddingsStore text→video and video→video search.

The CosmosEmbeddingsStore constructor is bypassed via object.__new__ to avoid
loading FAISS indices or the Cosmos embedding model from disk. A flat
IndexFlatIP and a lightweight mock text model are injected instead.
"""
from threading import Lock

import faiss
import numpy as np
import pytest
import torch

from sil_wheel.stores.cosmos_embeddings_store import CosmosEmbeddingsStore
from sil_wheel.stores.search_utils import SearchFilters, SearchResults
from sil_wheel.stores.time_utils import Timer
from sil_wheel.stores.utils import LRUDict

DIM = 768
N_CLIPS = 20


# ---------------------------------------------------------------------------
# Mock text model
# ---------------------------------------------------------------------------


class _MockTextModel:
    """Returns a predetermined embedding for each query string."""

    def __init__(self, embeddings: dict):
        # query_text -> np.ndarray of shape (1, DIM)
        self._embeddings = embeddings

    def get_text_embeddings(self, query_text: str):
        vec = self._embeddings.get(
            query_text, np.zeros((1, DIM), dtype=np.float32)
        )
        return torch.from_numpy(vec)


# ---------------------------------------------------------------------------
# Store factory
# ---------------------------------------------------------------------------


def _make_store(
    clip_embeddings: dict, text_model=None
) -> CosmosEmbeddingsStore:
    """Build a CosmosEmbeddingsStore with injected data (no file I/O)."""
    store = object.__new__(CosmosEmbeddingsStore)
    store.lock = Lock()
    store.searches = LRUDict(size=24)
    store.timers = Timer()
    store.path_to_embeddings = None
    store._index_tag = None
    store.text_to_video_model = text_model

    index = faiss.IndexFlatIP(DIM)
    clips_to_index = {}
    index_to_clips = {}

    clip_ids = list(clip_embeddings.keys())
    feats = np.vstack([clip_embeddings[c] for c in clip_ids]).astype(np.float32)
    faiss.normalize_L2(feats)
    index.add(feats)

    for i, cid in enumerate(clip_ids):
        clips_to_index[cid] = i
        index_to_clips[i] = cid

    store.features_index = index
    store.clips_to_index = clips_to_index
    store.index_to_clips = index_to_clips
    store.clips = np.array(clip_ids)

    # Mirror the production store: a compact array reverse-mapping for O(1)
    # FAISS-row -> clip_id translation in search.
    store.row_to_clip = np.empty(index.ntotal, dtype=object)
    for cid, row in clips_to_index.items():
        store.row_to_clip[row] = cid

    return store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def clip_embeddings():
    """20 random normalized embeddings keyed clip-00 … clip-19."""
    rng = np.random.default_rng(42)
    embeddings = {}
    for i in range(N_CLIPS):
        v = rng.standard_normal((1, DIM)).astype(np.float32)
        faiss.normalize_L2(v)
        embeddings[f"clip-{i:02d}"] = v
    return embeddings


@pytest.fixture()
def semantic_store(clip_embeddings):
    """Store whose mock text model maps 'query-A' to clip-00's embedding."""
    text_model = _MockTextModel({"query-A": clip_embeddings["clip-00"]})
    return _make_store(clip_embeddings, text_model=text_model)


# ---------------------------------------------------------------------------
# Text → video search
# ---------------------------------------------------------------------------


class TestTextSearch:
    def _current(self, clip_embeddings):
        return {cid: SearchResults.default for cid in clip_embeddings}

    def test_returns_results(self, semantic_store, clip_embeddings):
        filters = SearchFilters.from_query({"semantic_search_text": ["query-A"]})
        result = semantic_store.search(filters, self._current(clip_embeddings))
        assert len(result) > 0

    def test_top_result_matches_query_embedding(self, semantic_store, clip_embeddings):
        """Mock model returns clip-00's embedding for 'query-A' → clip-00 ranks first."""
        filters = SearchFilters.from_query({"semantic_search_text": ["query-A"]})
        result = semantic_store.search(filters, self._current(clip_embeddings))
        top = max(result, key=lambda c: result[c].semantic_search_text_score)
        assert top == "clip-00"

    def test_scores_in_valid_range(self, semantic_store, clip_embeddings):
        """Cosine similarity of normalized vectors must be in (-1, 1] (±float32 eps)."""
        filters = SearchFilters.from_query({"semantic_search_text": ["query-A"]})
        result = semantic_store.search(filters, self._current(clip_embeddings))
        for r in result.values():
            assert -1.0 - 1e-5 <= r.semantic_search_text_score <= 1.0 + 1e-5

    def test_count_bounded_by_k(self, semantic_store, clip_embeddings):
        filters = SearchFilters.from_query({"semantic_search_text": ["query-A"]})
        result = semantic_store.search(
            filters, self._current(clip_embeddings), k=5
        )
        assert len(result) <= 5

    def test_result_subset_of_current_results(self, semantic_store, clip_embeddings):
        """search() only returns clips that were in current_results."""
        subset = {f"clip-{i:02d}": SearchResults.default for i in range(5)}
        filters = SearchFilters.from_query({"semantic_search_text": ["query-A"]})
        result = semantic_store.search(filters, subset)
        assert set(result.keys()).issubset(set(subset.keys()))

    def test_cache_populated_after_search(self, semantic_store):
        """search_with_text caches the query result when no selector params are passed."""
        # The store-level cache is only populated when params=None; the public
        # search() entrypoint passes a SearchParametersIVF selector to restrict
        # to the active result pool, so it intentionally bypasses the cache.
        # This test exercises the cache path directly via search_with_text.
        semantic_store.search_with_text("query-A", k=5)
        assert "query-A" in semantic_store.searches


# ---------------------------------------------------------------------------
# Video → video search
# ---------------------------------------------------------------------------


class TestVideoSearch:
    def _current(self, clip_embeddings):
        return {cid: SearchResults.default for cid in clip_embeddings}

    def test_returns_results(self, semantic_store, clip_embeddings):
        filters = SearchFilters.from_query({"semantic_search_clipid": ["clip-01"]})
        result = semantic_store.search(filters, self._current(clip_embeddings))
        assert len(result) > 0

    def test_self_is_top_match(self, semantic_store, clip_embeddings):
        """A clip searched against itself should be the top-ranked result."""
        filters = SearchFilters.from_query({"semantic_search_clipid": ["clip-01"]})
        result = semantic_store.search(filters, self._current(clip_embeddings))
        top = max(result, key=lambda c: result[c].semantic_search_clip_score)
        assert top == "clip-01"

    def test_scores_in_valid_range(self, semantic_store, clip_embeddings):
        filters = SearchFilters.from_query({"semantic_search_clipid": ["clip-01"]})
        result = semantic_store.search(filters, self._current(clip_embeddings))
        for r in result.values():
            assert -1.0 <= r.semantic_search_clip_score <= 1.0

    def test_unknown_clip_returns_empty(self, semantic_store, clip_embeddings):
        """A clip_id not in the index produces an empty result."""
        filters = SearchFilters.from_query(
            {"semantic_search_clipid": ["nonexistent-clip"]}
        )
        result = semantic_store.search(filters, self._current(clip_embeddings))
        assert len(result) == 0

    def test_result_subset_of_current_results(self, semantic_store, clip_embeddings):
        subset = {f"clip-{i:02d}": SearchResults.default for i in range(5)}
        filters = SearchFilters.from_query({"semantic_search_clipid": ["clip-01"]})
        result = semantic_store.search(filters, subset)
        assert set(result.keys()).issubset(set(subset.keys()))
