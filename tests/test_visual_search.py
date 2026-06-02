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

"""Tests for CLIPEmbeddingStore.search_with_image and visual extra queries."""
import pickle
from unittest.mock import MagicMock

import numpy as np
import pytest

from sil_wheel.stores.clip_row_map import ClipRowMap
from sil_wheel.stores.search_utils import SearchFilters, SearchResults
from sil_wheel.stores.visual_embeddings_store import (
    CLIPEmbeddingStore,
    Florence2SigCLIPEmbeddingStore,
    reservoir_sample_embeddings,
)


@pytest.fixture()
def store_with_index(tmp_path):
    """CLIPEmbeddingStore with a mocked FAISS index and clip model."""
    store = CLIPEmbeddingStore.__new__(CLIPEmbeddingStore)
    from collections import defaultdict
    from threading import Lock
    from sil_wheel.stores.utils import LRUDict
    store.lock = Lock()
    store.searches = LRUDict(size=10)
    store.uploaded_image_features = LRUDict(size=32)
    store.features_index = None
    store.clips_to_index = None
    store.clip_to_faiss_ids = defaultdict(list)
    store.clip_model = None
    store.clip_preprocess = None

    # Build a tiny fake index: 3 "clips", each with a 4-d embedding
    n, dim = 3, 4
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)

    import faiss
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    store.features_index = index
    store.clips_to_index = {0: "clip-A", 1: "clip-B", 2: "clip-C"}
    for fid, cid in store.clips_to_index.items():
        store.clip_to_faiss_ids[cid].append(fid)

    return store


class TestSearchWithImage:
    def test_returns_scored_clips(self, store_with_index):
        """search_with_image returns (clip_id, score) pairs for all matched clips."""
        query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        results = store_with_index.search_with_image(query)
        clip_ids = {c for c, _ in results}
        assert "clip-A" in clip_ids

    def test_highest_score_for_matching_clip(self, store_with_index):
        """The clip whose embedding matches the query gets the highest score."""
        query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        results = dict(store_with_index.search_with_image(query))
        assert results["clip-A"] > results.get("clip-B", -1)
        assert results["clip-A"] > results.get("clip-C", -1)

    def test_uses_uploaded_image_features_in_search(self, store_with_index):
        """search() ranks the matching clip above orthogonal clips."""
        upload_id = "test-uuid-1234"
        store_with_index.uploaded_image_features[upload_id] = np.array(
            [[0.0, 1.0, 0.0, 0.0]], dtype=np.float32
        )
        filters = SearchFilters.from_query(
            {"visual_search_image_id": [upload_id]}
        )
        default = {f"clip-{c}": SearchResults() for c in ("A", "B", "C")}
        result = store_with_index.search(filters, default)
        # clip-B matches [0,1,0,0] with score 1; orthogonal clips score 0.
        assert result["clip-B"].visual_image_score == pytest.approx(1.0)
        assert result["clip-A"].visual_image_score == pytest.approx(0.0)
        assert result["clip-C"].visual_image_score == pytest.approx(0.0)

    def test_different_upload_ids_use_correct_features(self, store_with_index):
        """Each upload_id ranks its matching clip above the others."""
        store_with_index.uploaded_image_features["id-A"] = np.array(
            [[1.0, 0.0, 0.0, 0.0]], dtype=np.float32
        )
        store_with_index.uploaded_image_features["id-B"] = np.array(
            [[0.0, 1.0, 0.0, 0.0]], dtype=np.float32
        )

        # Use a fresh default per call — SearchResults instances are
        # mutated in place by with_visual_image_score.
        default_a = {f"clip-{c}": SearchResults() for c in ("A", "B", "C")}
        result_a = store_with_index.search(
            SearchFilters.from_query({"visual_search_image_id": ["id-A"]}),
            default_a,
        )
        scores_a = {k: v.visual_image_score for k, v in result_a.items()}

        default_b = {f"clip-{c}": SearchResults() for c in ("A", "B", "C")}
        result_b = store_with_index.search(
            SearchFilters.from_query({"visual_search_image_id": ["id-B"]}),
            default_b,
        )
        scores_b = {k: v.visual_image_score for k, v in result_b.items()}

        assert scores_a["clip-A"] > scores_a["clip-B"]
        assert scores_b["clip-B"] > scores_b["clip-A"]

    def test_no_image_search_without_flag(self, store_with_index):
        """Without visual_search_image_id, uploaded features are not used."""
        store_with_index.uploaded_image_features["some-id"] = np.array(
            [[1.0, 0.0, 0.0, 0.0]], dtype=np.float32
        )
        filters = SearchFilters.from_query({})
        default = {"clip-A": SearchResults()}
        result = store_with_index.search(filters, default)
        assert result["clip-A"].visual_search_score is None
        assert result["clip-A"].visual_image_score is None


def _write_pkl(path, embeddings, clip_ids):
    with open(path, "wb") as f:
        pickle.dump(
            {
                "embeddings": embeddings.astype(np.float32),
                "items": [{"clip_id": cid} for cid in clip_ids],
            },
            f,
        )


class TestReservoirSampleEmbeddings:
    def test_small_corpus_returns_every_vector(self, tmp_path):
        """When total < sample_size, reservoir holds all vectors."""
        d = 4
        p1 = tmp_path / "a.pkl"
        p2 = tmp_path / "b.pkl"
        e1 = np.eye(3, d, dtype=np.float32)
        e2 = np.eye(2, d, dtype=np.float32) * 2
        _write_pkl(p1, e1, ["x", "y", "z"])
        _write_pkl(p2, e2, ["p", "q"])

        reservoir, n_seen = reservoir_sample_embeddings(
            [p1, p2], tag="t", sample_size=100, d=d,
        )
        assert n_seen == 5
        assert reservoir.shape == (5, d)

    def test_uniform_across_shards(self, tmp_path):
        """Sampling covers all shards, not just the first one."""
        d = 5
        shards = []
        for i in range(5):
            p = tmp_path / f"shard_{i}.pkl"
            # Each shard's rows are the unit vector e_i — survives L2 norm.
            emb = np.zeros((200, d), dtype=np.float32)
            emb[:, i] = 1.0
            _write_pkl(p, emb, [f"c-{i}-{j}" for j in range(200)])
            shards.append(p)

        reservoir, n_seen = reservoir_sample_embeddings(
            shards, tag="t", sample_size=100, d=d, seed=42,
        )
        assert n_seen == 1000
        assert reservoir.shape == (100, d)
        # Every shard's axis appears in the reservoir.
        dominant_axis = reservoir.argmax(axis=1)
        assert set(dominant_axis.tolist()) == {0, 1, 2, 3, 4}

    def test_raises_when_empty(self, tmp_path):
        with pytest.raises(RuntimeError):
            reservoir_sample_embeddings(
                [], tag="t", sample_size=10, d=4,
            )


class TestSigLIPPromptTemplate:
    def _make_store(self, template):
        store = Florence2SigCLIPEmbeddingStore.__new__(
            Florence2SigCLIPEmbeddingStore
        )
        store._text_prompt_template = template
        store._device = "cpu"
        store.siglip_processor = MagicMock()
        store.siglip_processor.return_value = MagicMock()
        store.siglip_model = MagicMock()
        store.siglip_model.get_text_features.return_value = __import__(
            "torch"
        ).zeros(1, 4)
        return store

    def test_default_template_wraps_query(self):
        store = self._make_store("a photo of {text}")
        store.encode_text("a cyclist")
        call_kwargs = store.siglip_processor.call_args.kwargs
        assert call_kwargs["text"] == ["a photo of a cyclist"]

    def test_custom_template(self):
        store = self._make_store("this is a picture of {text}")
        store.encode_text("a dog")
        call_kwargs = store.siglip_processor.call_args.kwargs
        assert call_kwargs["text"] == ["this is a picture of a dog"]


def _make_store_with_index(tmp_path, tag, initial_clip_ids, dim=4):
    """Build a minimal Florence2SigCLIPEmbeddingStore wired to a flat IP
    index and a ClipRowMap with one row per provided clip_id."""
    import faiss
    from threading import Lock
    from sil_wheel.stores.utils import LRUDict

    store = Florence2SigCLIPEmbeddingStore.__new__(
        Florence2SigCLIPEmbeddingStore
    )
    store.lock = Lock()
    store.path_to_embeddings = tmp_path
    store._tag = tag
    store._nprobe = 1
    store._text_prompt_template = "{text}"
    store.searches = LRUDict(size=2)
    store.uploaded_image_features = LRUDict(size=2)

    n = len(initial_clip_ids)
    embeddings = np.eye(n, dim, dtype=np.float32)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    store.features_index = index
    store.clip_row_map = ClipRowMap(
        np.array(initial_clip_ids, dtype=object),
        np.arange(n, dtype=np.int32),
    )
    return store


class TestAppendPkl:
    def test_appends_new_clips_and_skips_duplicates(self, tmp_path):
        """New clips are appended, duplicates skipped, .npy persisted."""
        tag = "test"
        store = _make_store_with_index(tmp_path, tag, ["a", "b"], dim=4)

        shard = tmp_path / "florence2_sigclip_group_0_1.pkl"
        new_emb = np.array([
            [1.0, 0.0, 0.0, 0.0],  # duplicate clip "a" — skipped
            [0.0, 0.0, 1.0, 0.0],  # new clip "c"
            [0.0, 0.0, 0.0, 1.0],  # new clip "d"
        ], dtype=np.float32)
        _write_pkl(shard, new_emb, ["a", "c", "d"])

        store.append_pkl([shard])

        assert store.features_index.ntotal == 4
        assert set(store.clip_row_map.clip_ids.tolist()) == {"a", "b", "c", "d"}

        clip_ids_npy = tmp_path / f"visual_clip_ids_{tag}.npy"
        position_npy = tmp_path / f"visual_position_of_row_{tag}.npy"
        assert clip_ids_npy.exists()
        assert position_npy.exists()
        # Re-loadable as a ClipRowMap consistent with the FAISS index.
        reloaded = ClipRowMap(
            np.load(clip_ids_npy, allow_pickle=True),
            np.load(position_npy),
        )
        assert len(reloaded.position_of_row) == store.features_index.ntotal

    def test_no_op_when_all_clips_already_present(self, tmp_path):
        """Index and ClipRowMap are unchanged when no new clips appear."""
        tag = "test"
        store = _make_store_with_index(tmp_path, tag, ["a", "b"], dim=4)
        before_ntotal = store.features_index.ntotal
        before_clips = store.clip_row_map.clip_ids.tolist()

        shard = tmp_path / "florence2_sigclip_group_0_1.pkl"
        _write_pkl(
            shard,
            np.eye(2, 4, dtype=np.float32),
            ["a", "b"],
        )

        store.append_pkl([shard])

        assert store.features_index.ntotal == before_ntotal
        assert store.clip_row_map.clip_ids.tolist() == before_clips
        # Nothing was persisted because nothing changed.
        assert not (tmp_path / f"visual_clip_ids_{tag}.npy").exists()

    def test_accepts_non_contiguous_float64_embeddings(self, tmp_path):
        """Shards saved with non-float32 or non-contiguous arrays load
        cleanly: faiss.normalize_L2 only accepts contiguous float32, so
        the store must coerce both."""
        tag = "test"
        store = _make_store_with_index(tmp_path, tag, ["a"], dim=4)

        # float64 + non-contiguous view: a transpose then back-transpose
        # of a sliced array is the standard way to provoke this.
        base = np.zeros((4, 8), dtype=np.float64)
        base[:, ::2] = np.eye(4, 4, dtype=np.float64)
        non_contig = base[:, ::2]
        assert non_contig.dtype == np.float64
        assert not non_contig.flags["C_CONTIGUOUS"]

        shard = tmp_path / "florence2_sigclip_group_0_1.pkl"
        with open(shard, "wb") as f:
            pickle.dump(
                {
                    "embeddings": non_contig,
                    "items": [
                        {"clip_id": cid} for cid in ["c", "d", "e", "f"]
                    ],
                },
                f,
            )

        store.append_pkl([shard])
        assert store.features_index.ntotal == 5
        assert {"a", "c", "d", "e", "f"} <= set(
            store.clip_row_map.clip_ids.tolist()
        )
