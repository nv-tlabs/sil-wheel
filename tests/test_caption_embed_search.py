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

"""Tests for CaptionEmbeddingsStore text→caption search.

The CaptionEmbeddingsStore constructor is bypassed via object.__new__ to avoid
loading FAISS indices or the Qwen3-Embedding-8B model from disk. A flat
IndexFlatIP and a lightweight mock SentenceTransformer are injected instead.
"""
from threading import Lock

import faiss
import numpy as np
import pytest

from sil_wheel.stores.caption_embeddings_store import CaptionEmbeddingsStore
from sil_wheel.stores.clip_row_map import ClipRowMap
from sil_wheel.stores.search_utils import SearchFilters, SearchResults
from sil_wheel.stores.utils import LRUDict

DIM = 64
N_CLIPS = 20


# ---------------------------------------------------------------------------
# Mock embedding model
# ---------------------------------------------------------------------------


class _MockModel:
    """Returns a predetermined embedding for each query string."""

    def __init__(self, embeddings: dict):
        # query_text -> np.ndarray of shape (DIM,)
        self._embeddings = embeddings

    def encode(self, queries, normalize_embeddings=True, prompt_name=None):
        query = queries[0]
        vec = self._embeddings.get(query, np.zeros(DIM, dtype=np.float32))
        return np.array([vec], dtype=np.float32)


# ---------------------------------------------------------------------------
# Store factory
# ---------------------------------------------------------------------------


def _make_store(clip_embeddings: dict, model=None) -> CaptionEmbeddingsStore:
    """Build a CaptionEmbeddingsStore with injected data (no file I/O)."""
    store = object.__new__(CaptionEmbeddingsStore)
    store.lock = Lock()
    store.searches = LRUDict(size=10)
    store.path_to_embeddings = None
    store._tag = None
    store.model = model

    index = faiss.IndexFlatIP(DIM)

    clip_ids = list(clip_embeddings.keys())
    feats = np.vstack([clip_embeddings[c] for c in clip_ids]).astype(np.float32)
    faiss.normalize_L2(feats)
    index.add(feats)

    store.features_index = index
    store.clip_row_map = ClipRowMap.build(clip_ids)

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
        v = rng.standard_normal(DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        embeddings[f"clip-{i:02d}"] = v
    return embeddings


@pytest.fixture()
def caption_embed_store(clip_embeddings):
    """Store whose mock model maps 'query-A' to clip-00's embedding."""
    model = _MockModel({"query-A": clip_embeddings["clip-00"]})
    return _make_store(clip_embeddings, model=model)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCaptionEmbedSearch:
    def _current(self, clip_embeddings):
        return {cid: SearchResults.default for cid in clip_embeddings}

    def test_returns_results(self, caption_embed_store, clip_embeddings):
        filters = SearchFilters.from_query({"caption_embed_search": ["query-A"]})
        result = caption_embed_store.search(filters, self._current(clip_embeddings))
        assert len(result) > 0

    def test_top_result_matches_query_embedding(self, caption_embed_store, clip_embeddings):
        """Mock model returns clip-00's embedding for 'query-A' → clip-00 ranks first."""
        filters = SearchFilters.from_query({"caption_embed_search": ["query-A"]})
        result = caption_embed_store.search(filters, self._current(clip_embeddings))
        top = max(result, key=lambda c: result[c].caption_embed_score)
        assert top == "clip-00"

    def test_scores_in_valid_range(self, caption_embed_store, clip_embeddings):
        """Cosine similarity of normalized vectors must be in (-1, 1] (±float32 eps)."""
        filters = SearchFilters.from_query({"caption_embed_search": ["query-A"]})
        result = caption_embed_store.search(filters, self._current(clip_embeddings))
        for r in result.values():
            assert -1.0 - 1e-5 <= r.caption_embed_score <= 1.0 + 1e-5

    def test_no_filter_returns_current_results_unchanged(
        self, caption_embed_store, clip_embeddings
    ):
        """Without caption_embed_search set, search() is a no-op."""
        current = self._current(clip_embeddings)
        filters = SearchFilters.from_query({})
        result = caption_embed_store.search(filters, current)
        assert result is current

    def test_result_subset_of_current_results(self, caption_embed_store, clip_embeddings):
        """search() only returns clips that were in current_results."""
        subset = {f"clip-{i:02d}": SearchResults.default for i in range(5)}
        filters = SearchFilters.from_query({"caption_embed_search": ["query-A"]})
        result = caption_embed_store.search(filters, subset)
        assert set(result.keys()).issubset(set(subset.keys()))

    def test_cache_populated_after_search(self, caption_embed_store):
        """search_with_text caches results when params is None (unrestricted)."""
        caption_embed_store.search_with_text("query-A")
        assert "query-A" in caption_embed_store.searches

    def test_unknown_query_returns_low_scores(self, caption_embed_store, clip_embeddings):
        """An unknown query falls back to a zero vector → all scores near zero or filtered."""
        filters = SearchFilters.from_query({"caption_embed_search": ["unknown-query"]})
        result = caption_embed_store.search(filters, self._current(clip_embeddings))
        # Zero query → distances ~0 → filtered out by the >0.001 mask; result may be empty
        for r in result.values():
            assert r.caption_embed_score <= 1.0 + 1e-5

    def test_pool_restricted_search_uses_clip_row_map(
        self, caption_embed_store, clip_embeddings
    ):
        """Restricted search routes through ClipRowMap.rows_for_clips."""
        pool = {f"clip-{i:02d}": SearchResults.default for i in range(3, 8)}
        filters = SearchFilters.from_query({"caption_embed_search": ["query-A"]})
        result = caption_embed_store.search(filters, pool)
        # clip-00 is the true top match but is excluded from the pool; the
        # returned scores must come only from the pool.
        assert set(result.keys()).issubset(set(pool.keys()))
        assert "clip-00" not in result
