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

import pickle
import random
import sys
import time
from pathlib import Path
from threading import Lock

import faiss
import numpy as np
import pandas as pd
import torch
from sil_wheel.stores.search_utils import project_starmap
from sil_wheel.stores.time_utils import Timer
from tqdm import tqdm
from sil_wheel.stores.utils import LRUDict

try:
    from sil_wheel.embeddings.cosmos_embed1 import CosmosEmbed1
except ImportError:
    print("CosmosEmbed1 not found")
    pass
from sil_wheel.search.retrieval_utils import (
    query_to_video_retrieval,
    text_to_video_retrieval,
)

# spec: (tag, nprobe)
INDEX_SPEC = {
    "FLAT": ("flat", 256),
    "IVF4096,PQ96x8": ("ivf4096_pq96x8", 256),
    "OPQ96_768,IVF4096,PQ96x8": ("opq96_768_ivf4096_pq96x8", 256),
    "IVF8192,PQ96x8": ("ivf8192_pq96x8", 512),
    "OPQ96_768,IVF8192,PQ96x8": ("opq96_768_ivf8192_pq96x8", 512),
    "IVF16384,PQ96x8": ("ivf16384_pq96x8", 1024),
    "OPQ96_768,IVF16384,PQ96x8": ("opq96_768_ivf16384_pq96x8", 1024),
    "IVF8192,PQ96x8": ("ivf8192_pq96x8_np1024", 1024),
    "IVF8192,PQ96x8": ("ivf8192_pq96x8_np2048", 2048),
    "HNSW32": ("hnsw32", 128),
    "HNSW64": ("hnsw64", 256),
}

CORPUS_RESTRICT_THRESHOLD = 2_000_000


def set_search_params(index: faiss.Index, nprobe: int) -> None:
    """Set nprobe (IVF) or efSearch (HNSW) on a FAISS index."""
    if hasattr(index, "nprobe"):
        index.nprobe = nprobe
    if hasattr(index, "hnsw"):
        try:
            index.hnsw.efSearch = max(64, nprobe)
        except Exception:
            pass

def spec_to_tag(spec):
    tag = spec.strip().lower().replace(",", "_").replace(" ", "")
    return tag.replace("/", "_").replace("-", "_")


def parse_embeddings_from_dir(
    path_to_text_to_video_embeddings: str,
    index_spec: str = "IVF4096,PQ96x8",
    nprobe: int = 256,
    keep_clips: list = [],
    mmap: bool = False,
):
    start = time.time()
    path_to_embeddings = Path(path_to_text_to_video_embeddings)
    parquet_files = list(Path(path_to_embeddings).glob("**/*.parquet"))

    tag = index_spec
    print(f"[{tag}] Found {len(parquet_files)} parquet files")

    path_to_faiss_index = (
        path_to_embeddings / f"cosmos_embeddings_{INDEX_SPEC[tag][0]}.index"
    )
    path_to_clip_index = (
        path_to_embeddings / f"cosmos_clip_to_index_{INDEX_SPEC[tag][0]}.pkl"
    )

    # If precomputed, load and return
    if path_to_faiss_index.exists() and path_to_clip_index.exists():
        flags = (
            faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY if mmap else 0
        )
        features_index = faiss.read_index(str(path_to_faiss_index), flags)
        with open(path_to_clip_index, "rb") as f:
            clip_to_index = pickle.load(f)
        clip_to_index = {sys.intern(k): v for k, v in clip_to_index.items()}
        print(
            f"[{tag}] Loaded feature index from {path_to_faiss_index} "
            f"Cosmos embeddings of size {features_index.ntotal}..."
        )
        return features_index, clip_to_index

    # Shuffle the files to do the training from multiple embeddings
    random.shuffle(parquet_files)

    d = 768
    features_index = faiss.index_factory(
        d, index_spec, faiss.METRIC_INNER_PRODUCT
    )
    set_search_params(features_index, nprobe)

    all_features = []
    clip_to_index = {}
    offset = 0
    buf_count = 0
    step = 20
    for ii in tqdm(
        range(0, len(parquet_files), step),
        desc="Parsing text-to-video embeddings",
        leave=True,
    ):
        batch_files = parquet_files[ii : ii + step]
        df = pd.read_parquet(batch_files, columns=["clip_id", "embeddings"])
        # Drop dublicates if any
        df = df.drop_duplicates(subset=["clip_id"], ignore_index=True)
        df = df[~df["clip_id"].isin(clip_to_index.keys())]

        if len(keep_clips) > 0:
            # Keep only the clips we need
            df = df[df["clip_id"].isin(keep_clips)]

        clips = df["clip_id"].to_numpy()
        features = np.vstack(df["embeddings"].values).astype(
            np.float32, copy=False
        )

        # Normalize for IP search
        faiss.normalize_L2(features)

        assert clips.shape[0] == features.shape[0]
        all_features.append(features)
        buf_count += features.shape[0]

        for i, cid in enumerate(clips):
            clip_to_index[cid] = offset + i
        offset += clips.shape[0]

        print(
            f"[{tag}] Indexed unique clips: {len(set(clip_to_index.values()))}"
        )

        if (not features_index.is_trained) and buf_count >= 1000000:
            feats = np.vstack(all_features)
            features_index.train(feats)
            features_index.add(feats)
            all_features.clear()
            buf_count = 0

        elif features_index.is_trained and buf_count >= 500000:
            feats = np.vstack(all_features)
            features_index.add(feats)
            all_features.clear()
            buf_count = 0

    # Flush any remaining
    if buf_count > 0:
        feats = np.vstack(all_features)
        if not features_index.is_trained:
            features_index.train(feats)
        features_index.add(feats)
        all_features.clear()
        buf_count = 0
    elapsed = time.time() - start
    print(f"[{tag}] Finished indexing in {elapsed:.2f} seconds")

    print(f"[{tag}] ntotal:", features_index.ntotal)
    print(f"[{tag}] type:", type(features_index))
    print(f"[{tag}] is_trained:", getattr(features_index, "is_trained", True))
    if hasattr(features_index, "nlist"):
        print(
            f"[{tag}] nlist:",
            features_index.nlist,
            "nprobe:",
            getattr(features_index, "nprobe", "-"),
        )
    print(f"[{tag}] FAISS OMP threads:", faiss.omp_get_max_threads())

    faiss.write_index(features_index, str(path_to_faiss_index))

    # Save clip_to_index mapping
    with open(path_to_clip_index, "wb") as f:
        pickle.dump(clip_to_index, f)

    return features_index, clip_to_index


def parse_embeddings_from_dir_flat(
    path_to_text_to_video_embeddings: str,
    keep_clips: list = [],
    mmap: bool = False,
):
    """
    Build/load a FAISS flat (exact) index over Cosmos text-to-video embeddings.

    This mirrors parse_embeddings_from_dir but uses an exact IndexFlatIP
    (no quantization), which is slower but maximizes recall/accuracy.

    Returns (features_index, clip_to_index).
    """
    start = time.time()
    path_to_embeddings = Path(path_to_text_to_video_embeddings)
    parquet_files = sorted(Path(path_to_embeddings).glob("**/*.parquet"))
    print(f"[FLAT] Found {len(parquet_files)} parquet files")

    # Use distinct filenames so both indexes can coexist
    path_to_faiss_index = path_to_embeddings / "cosmos_embeddings_flat.index"
    path_to_clip_index = path_to_embeddings / "cosmos_clip_to_index_flat.pkl"

    # If precomputed, load and return
    if path_to_faiss_index.exists() and path_to_clip_index.exists():
        flags = (
            faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY if mmap else 0
        )
        features_index = faiss.read_index(str(path_to_faiss_index), flags)
        with open(path_to_clip_index, "rb") as f:
            clip_to_index = pickle.load(f)
        clip_to_index = {sys.intern(k): v for k, v in clip_to_index.items()}
        print(
            f"[FLAT] Loaded feature index from {path_to_faiss_index} "
            f"Cosmos embeddings of size {features_index.ntotal}..."
        )
        return features_index, clip_to_index

    # Cosmos embeddings are 768-dim; use exact inner-product index
    d = 768
    features_index = faiss.IndexFlatIP(d)

    clip_to_index = {}
    offset = 0
    step = 20

    for ii in tqdm(
        range(0, len(parquet_files), step),
        desc="[FLAT] Parsing text-to-video embeddings",
        leave=True,
    ):
        batch_files = parquet_files[ii : ii + step]
        if len(batch_files) == 0:
            continue
        df = pd.read_parquet(batch_files, columns=["clip_id", "embeddings"])

        # Drop duplicates and those already indexed
        df = df.drop_duplicates(subset=["clip_id"], ignore_index=True)
        if len(clip_to_index) > 0:
            df = df[~df["clip_id"].isin(clip_to_index.keys())]

        if len(keep_clips) > 0:
            df = df[df["clip_id"].isin(keep_clips)]

        if len(df) == 0:
            continue

        clips = df["clip_id"].to_numpy()
        features = np.vstack(df["embeddings"].values).astype(
            np.float32, copy=False
        )

        # Normalize for inner-product search
        faiss.normalize_L2(features)

        # Add to index and record mapping
        features_index.add(features)
        for i, cid in enumerate(clips):
            clip_to_index[cid] = offset + i
        offset += clips.shape[0]

        print(len(set(clip_to_index.values())))

    elapsed = time.time() - start
    print(f"[FLAT] Finished indexing in {elapsed:.2f} seconds")

    print("[FLAT] ntotal:", features_index.ntotal)
    print("[FLAT] type:", type(features_index))
    print("[FLAT] is_trained:", getattr(features_index, "is_trained", True))
    print("[FLAT] FAISS OMP threads:", faiss.omp_get_max_threads())

    # Persist index and mapping
    faiss.write_index(features_index, str(path_to_faiss_index))
    with open(path_to_clip_index, "wb") as f:
        pickle.dump(clip_to_index, f)

    return features_index, clip_to_index


class CosmosEmbeddingsStore:
    def __init__(self, path_to_embeddings, index_spec= "IVF4096,PQ96x8", mmap=False):
        self.lock = Lock()
        self.features_index = None
        self.clips_to_index = None
        self.path_to_embeddings = path_to_embeddings
        self._index_tag = INDEX_SPEC[index_spec][0]
        nprobe = INDEX_SPEC[index_spec][1]

        if path_to_embeddings is not None:
            path_to_embeddings = Path(path_to_embeddings)
            if path_to_embeddings.exists():
                if index_spec == "FLAT":
                    features_index, clips_to_index = (
                        parse_embeddings_from_dir_flat(
                            path_to_embeddings,
                            mmap=mmap,
                        )
                    )
                else:
                    features_index, clips_to_index = parse_embeddings_from_dir(
                        path_to_embeddings,
                        index_spec=index_spec,
                        nprobe=nprobe,
                        mmap=mmap,
                    )

                self.features_index = features_index
                set_search_params(self.features_index, nprobe)
                # Some FAISS indices (e.g., IndexPreTransform) don't expose
                # make_direct_map at the top level; call it on the inner index
                # when available, and otherwise skip gracefully.
                try:
                    if hasattr(self.features_index, "make_direct_map"):
                        self.features_index.make_direct_map()
                    elif hasattr(self.features_index, "index") and hasattr(
                        self.features_index.index, "make_direct_map"
                    ):
                        self.features_index.index.make_direct_map()
                except Exception:
                    pass
                # cid -> row is handed over by the parser; also precompute
                # a compact array for the reverse direction so searches can
                # translate FAISS results in O(1) without a second full dict.
                self.clips_to_index = clips_to_index
                self.row_to_clip = np.empty(features_index.ntotal, dtype=object)
                for cid, row in clips_to_index.items():
                    self.row_to_clip[row] = cid

            print("ntotal:", self.features_index.ntotal)
            print("type:", type(self.features_index))
            print(
                "is_trained:", getattr(self.features_index, "is_trained", True)
            )
            if hasattr(self.features_index, "nlist"):
                print(
                    "nlist:",
                    self.features_index.nlist,
                    "nprobe:",
                    getattr(self.features_index, "nprobe", "-"),
                )
            print("FAISS OMP threads:", faiss.omp_get_max_threads())

            # Load the text-to-video model
            self.text_to_video_model = CosmosEmbed1("cosmos_embed1_448p")
        else:
            self.clips_to_index = {}
        self.searches = LRUDict(size=10)
        self.timers = Timer()

    def append_embeddings_parquet(self, parquet_file):
        """
        Append embeddings from a new parquet file to the existing FAISS index
        and update the clip_id -> index mapping, without retraining.

        Parquet schema must contain columns: ["clip_id", "embeddings"].

        Returns number of clips added.
        """
        if self.features_index is None or self.clips_to_index is None:
            raise ValueError(
                "Embeddings index is not initialized; cannot append."
            )

        parquet_file = Path(parquet_file)
        if not parquet_file.exists():
            raise FileNotFoundError(str(parquet_file))

        # Load and filter out duplicates
        df = pd.read_parquet(parquet_file, columns=["clip_id", "embeddings"])
        df = df.drop_duplicates(subset=["clip_id"], ignore_index=True)
        if len(self.clips_to_index) > 0:
            df = df[~df["clip_id"].isin(self.clips_to_index.keys())]

        if len(df) == 0:
            return 0

        clips = df["clip_id"].to_numpy()
        feats = np.vstack(df["embeddings"].values).astype(
            np.float32, copy=False
        )

        # Normalize for inner-product search (consistent with initial indexing)
        faiss.normalize_L2(feats)

        base = int(self.features_index.ntotal)
        self.features_index.add(feats)
        try:
            # Ensure reconstruct() works for new ids; handle PreTransform
            if hasattr(self.features_index, "make_direct_map"):
                self.features_index.make_direct_map()
            elif hasattr(self.features_index, "index") and hasattr(
                self.features_index.index, "make_direct_map"
            ):
                self.features_index.index.make_direct_map()
        except Exception:
            pass

        # Update mappings
        for i, cid in enumerate(clips):
            self.clips_to_index[cid] = base + i

        root = Path(self.path_to_embeddings)
        # Persist using the same tag as creation (flat or index spec)
        tag = self._index_tag
        faiss.write_index(
            self.features_index,
            str(root / f"cosmos_embeddings_{tag}.index"),
        )
        with open(root / f"cosmos_clip_to_index_{tag}.pkl", "wb") as f:
            pickle.dump(self.clips_to_index, f)

        # Invalidate any cached searches that depend on index contents
        self.searches = LRUDict(size=10)
        return len(clips)

    def append_multiple_embeddings_with_path(self, path_to_embed: str):
        # Grab all files containing the embeddings we wish to append
        parquet_files = sorted(Path(path_to_embed).glob("**/*.parquet"))

        # Append the files
        for pi in parquet_files:
            print(pi)
            self.append_embeddings_parquet(pi)

    def has_embeddings(self, clip_id):
        return clip_id in self.clips_to_index

    def get(self, clip_id):
        if self.features_index is not None and self.has_embeddings(clip_id):
            return self.features_index.reconstruct(self.clips_to_index[clip_id])
        else:
            return None

    def search_with_video_clip(self, search, k=8192, params=None):
        with self.lock:
            self.timers.tic()
            if params is None and search in self.searches:
                print(f"The semantic search in faiss took {self.timers.toc()}")
                return self.searches[search]

            video_ids = []
            distances = []
            query_features = self.get(clip_id=search)
            if query_features is not None:
                distances, indices = query_to_video_retrieval(
                    query_features, self.features_index, n_neighbors=k,
                    params=params,
                )
                distances = distances[0]
                indices = indices[0]

                mask = np.logical_and(distances > 0.001, indices >= 0)
                distances = distances[mask]
                indices = indices[mask]
                video_ids = self.row_to_clip[indices].tolist()
                if not isinstance(distances, list):
                    distances = distances.tolist()
            result = list(zip(video_ids, distances))
            if params is None:
                self.searches[search] = result
            print(f"The semantic search in faiss took {self.timers.toc()}")

            return result

    @torch.no_grad()
    def encode_text(self, query):
        """Encode a text query into the Cosmos video embedding space."""
        features = self.text_to_video_model.get_text_embeddings(query)
        features = torch.nn.functional.normalize(features, dim=-1)
        if features.dtype == torch.bfloat16:
            features = features.float()
        return features.cpu().numpy().astype(np.float32)

    def search_with_text(self, search, k=8192, params=None):
        with self.lock:
            self.timers.tic()
            if params is None and search in self.searches:
                print(f"The semantic search in faiss took {self.timers.toc()}")
                return self.searches[search]

            distances, indices = text_to_video_retrieval(
                search,
                self.features_index,
                self.text_to_video_model,
                n_neighbors=k,
                params=params,
            )
            distances = distances[0]
            indices = indices[0]

            mask = np.logical_and(distances > 0.001, indices >= 0)
            distances = distances[mask]
            indices = indices[mask]
            video_ids = self.row_to_clip[indices].tolist()
            result = list(zip(video_ids, distances.tolist()))
            if params is None:
                self.searches[search] = result
            print(f"The semantic search in faiss took {self.timers.toc()}")

            return result

    def _make_selector_params(self, clip_ids):
        faiss_ids = np.array(
            [self.clips_to_index[cid] for cid in clip_ids if cid in self.clips_to_index],
            dtype=np.int64,
        )
        sel = faiss.IDSelectorBatch(faiss_ids)
        index = faiss.downcast_index(self.features_index)
        if isinstance(index, faiss.IndexIVF):
            # Cap the pool-restricted nprobe at 4x baseline: scaling it up
            # with ntotal/len(faiss_ids) degenerates to a full-index scan
            # (nprobe = nlist) once the pool shrinks, which is what made
            # filtered searches 10x+ slower than unfiltered.
            nprobe = min(index.nlist, index.nprobe * 4)
            return faiss.SearchParametersIVF(sel=sel, nprobe=nprobe)
        return faiss.SearchParameters(sel=sel)

    def search(self, filters, current_results, k=114618):
        if filters.semantic_search_clipid is not None:
            # Restrict clip search to the pool already narrowed by prior filters.
            pool_params = (
                self._make_selector_params(current_results)
                if len(current_results) < CORPUS_RESTRICT_THRESHOLD else None
            )
            current_results = project_starmap(
                lambda r, s: r.with_semantic_search_clip_score(s),
                current_results,
                self.search_with_video_clip(
                    filters.semantic_search_clipid, k=k, params=pool_params
                ),
            )

        if filters.semantic_search_text is not None:
            queries = list(
                dict.fromkeys(
                    [filters.semantic_search_text]
                    + list(filters.semantic_extra_queries or [])
                )
            )
            # Restrict to whatever pool is active: prior filters, clip search, or both.
            params = (
                self._make_selector_params(current_results)
                if len(current_results) < CORPUS_RESTRICT_THRESHOLD else None
            )
            per_clip_id_score = {}
            for q in queries:
                for clip_id, score in self.search_with_text(q, k=k, params=params):
                    if clip_id not in per_clip_id_score or score > per_clip_id_score[clip_id]:
                        per_clip_id_score[clip_id] = score
            current_results = project_starmap(
                lambda r, s: r.with_semantic_search_text_score(s),
                current_results,
                list(per_clip_id_score.items()),
            )

        return current_results

    def warmup(self):
        if self.features_index is None:
            return
        self.search_with_text("warmup", k=10)
