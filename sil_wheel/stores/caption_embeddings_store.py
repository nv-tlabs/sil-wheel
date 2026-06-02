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
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sil_wheel.embeddings.sentence_transformer_loader import (
    load_sentence_transformer,
)
from sil_wheel.stores.clip_row_map import ClipRowMap
from sil_wheel.stores.search_utils import project_starmap
from sil_wheel.stores.utils import LRUDict

CORPUS_RESTRICT_THRESHOLD = 2_000_000


def set_search_params(index, nprobe):
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


def load_shard_embeddings_parquet(parquet_path, tag):
    """Load one parquet shard, L2-normalise, return (embeddings, clip_ids).

    Returns (None, None) on read/parse failure so the caller can skip.
    """
    try:
        df = pd.read_parquet(parquet_path, columns=["clip_id", "embedding"])
    except Exception as e:
        print(f"[caption/{tag}] Skipping {parquet_path.name}: {e}")
        return None, None
    clip_ids = df["clip_id"].to_numpy()
    features = np.vstack(df["embedding"].values).astype(np.float32, copy=False)
    faiss.normalize_L2(features)
    return features, clip_ids


def reservoir_sample_caption_embeddings(
    parquet_files, tag, sample_size, d, seed=0, n_workers=8,
):
    """Reservoir-sample up to `sample_size` vectors uniformly across shards.

    Single pass (Algorithm R). The sample is used to train an IVF quantizer
    on a distribution representative of the full corpus instead of whichever
    shards happen to sort first. Returns (reservoir, n_seen). The reservoir
    is truncated if the corpus has fewer than `sample_size` vectors.

    Parquet decode releases the GIL in pyarrow, so shards are decoded
    concurrently by a thread pool while the reservoir update stays
    serial. pool.map preserves submission order, so the effective stream
    order is identical to reading the shards one by one.
    """
    reservoir = np.empty((sample_size, d), dtype=np.float32)
    n_seen = 0
    rng = np.random.default_rng(seed)

    workers = max(1, min(n_workers, len(parquet_files)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(
            lambda p: load_shard_embeddings_parquet(p, tag), parquet_files,
        )
        for embeddings, _ in tqdm(
            results,
            total=len(parquet_files),
            desc=f"[caption/{tag}] Sampling for training",
        ):
            if embeddings is None:
                continue
            m = len(embeddings)
            if n_seen + m <= sample_size:
                reservoir[n_seen : n_seen + m] = embeddings
                n_seen += m
                continue
            remaining = sample_size - n_seen
            if remaining > 0:
                reservoir[n_seen:] = embeddings[:remaining]
                embeddings = embeddings[remaining:]
                n_seen = sample_size
            extra = len(embeddings)
            positions = np.arange(n_seen, n_seen + extra) + 1
            draws = rng.integers(0, positions, size=extra)
            accepted = np.nonzero(draws < sample_size)[0]
            if len(accepted) > 0:
                reservoir[draws[accepted]] = embeddings[accepted]
            n_seen += extra

    if n_seen == 0:
        raise RuntimeError(
            f"[caption/{tag}] no embeddings found across "
            f"{len(parquet_files)} shards"
        )
    if n_seen < sample_size:
        reservoir = reservoir[:n_seen]
    return reservoir, n_seen


def parse_caption_index_from_dir(path_to_embeddings, index_spec="IVF4096,PQ128x8", nprobe=256, mmap=False):
    """Load or build a FAISS index over caption embeddings from parquet files.

    Parquet schema must contain columns: ["clip_id", "embedding"].
    Multiple rows per clip_id are expected (one per caption).
    Embeddings are L2-normalized before indexing (inner-product = cosine similarity).

    Returns (features_index, index_to_clips).
    """
    path_to_embeddings = Path(path_to_embeddings)
    parquet_files = sorted(path_to_embeddings.glob("**/*.parquet"))
    tag = spec_to_tag(index_spec)

    print(f"[caption/{tag}] Found {len(parquet_files)} parquet files")

    path_to_index = path_to_embeddings / f"caption_embeddings_{tag}.index"
    path_to_map = path_to_embeddings / f"caption_index_to_clip_{tag}.pkl"

    if path_to_index.exists() and path_to_map.exists():
        flags = (
            faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY if mmap else 0
        )
        features_index = faiss.read_index(str(path_to_index), flags)
        with open(path_to_map, "rb") as f:
            index_to_clips = pickle.load(f)
        print(
            f"[caption/{tag}] Loaded index from {path_to_index} "
            f"({features_index.ntotal:,} embeddings)"
        )
        return features_index, index_to_clips

    # Detect embedding dimension from the first parquet file.
    sample = pd.read_parquet(parquet_files[0], columns=["embedding"])
    d = len(sample["embedding"].iloc[0])
    print(f"[caption/{tag}] Detected embedding dimension: {d}")

    features_index = faiss.index_factory(d, index_spec, faiss.METRIC_INNER_PRODUCT)
    set_search_params(features_index, nprobe)

    start = time.time()

    # Pass 1: sample uniformly across all shards so the IVF quantizer is
    # trained on a distribution representative of the full corpus, not
    # just whichever shards happen to sort first.
    train_sample, _ = reservoir_sample_caption_embeddings(
        parquet_files, tag, sample_size=1_000_000, d=d,
    )
    print(
        f"[caption/{tag}] Training on {len(train_sample):,} sampled vectors ..."
    )
    features_index.train(train_sample)
    del train_sample
    print(f"[caption/{tag}] Training complete")

    # Pass 2: add every shard to the trained index. Shard decode runs
    # concurrently while the serial adds/checkpoints consume results in
    # submission order.
    index_to_clips = {}
    offset = 0
    workers = max(1, min(8, len(parquet_files)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(
            lambda p: load_shard_embeddings_parquet(p, tag), parquet_files,
        )
        for ii, (features, clips) in enumerate(
            tqdm(
                results,
                total=len(parquet_files),
                desc=f"[caption/{tag}] Indexing",
            )
        ):
            if features is None:
                continue
            features_index.add(features)
            for i, cid in enumerate(clips):
                index_to_clips[offset + i] = cid
            offset += len(clips)
            print(
                f"[{tag}] Indexed unique clips: {len(set(index_to_clips.values()))}"
            )

            # Checkpoint every 5 shards so progress survives a failure.
            if ii % 5 == 0:
                faiss.write_index(features_index, str(path_to_index))
                print(f"Saved features_index to {str(path_to_index)}")
                with open(path_to_map, "wb") as f:
                    pickle.dump(index_to_clips, f)
                print(f"Saved index_to_clips to {str(path_to_map)}")

    elapsed = time.time() - start
    print(f"[caption/{tag}] Finished in {elapsed:.2f}s — ntotal: {features_index.ntotal:,}")

    faiss.write_index(features_index, str(path_to_index))
    with open(path_to_map, "wb") as f:
        pickle.dump(index_to_clips, f)

    return features_index, index_to_clips


class CaptionEmbeddingsStore:
    def __init__(
        self,
        path_to_embeddings,
        index_spec="IVF4096,PQ128x8",
        nprobe=256,
        mmap=False,
        embedding_model="Qwen/Qwen3-Embedding-8B",
    ):
        self.lock = Lock()
        self.features_index = None
        self.clip_row_map = None
        self.path_to_embeddings = path_to_embeddings
        self._tag = spec_to_tag(index_spec)
        self._embedding_model = embedding_model

        if path_to_embeddings is not None:
            path = Path(path_to_embeddings)
            if path.exists():
                features_index, index_to_clips = (
                    parse_caption_index_from_dir(path, index_spec=index_spec, nprobe=nprobe, mmap=mmap)
                )
                set_search_params(features_index, nprobe)

                # Convert the loaded {row: clip_id} dict into the compact
                # ClipRowMap and release the dict. FAISS row ids are dense
                # 0..ntotal-1 so indexed iteration reconstructs insertion order.
                clip_ids_per_row = [
                    index_to_clips[i] for i in range(features_index.ntotal)
                ]
                self.clip_row_map = ClipRowMap.build(clip_ids_per_row)
                del index_to_clips, clip_ids_per_row

                self.features_index = features_index
                print("ntotal:", self.features_index.ntotal)
                print("type:", type(self.features_index))
                print("is_trained:", getattr(self.features_index, "is_trained", True))
                if hasattr(self.features_index, "nlist"):
                    print(
                        "nlist:", self.features_index.nlist,
                        "nprobe:", getattr(self.features_index, "nprobe", "-"),
                    )

        # The query encoder must match the model that produced the parquet
        # shards,its output dimension has to equal index.d, otherwise FAISS
        # asserts at search time.
        self.model = load_sentence_transformer(
            embedding_model,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        self.searches = LRUDict(size=10)

    def warmup(self):
        if self.features_index is None:
            return
        self.search_with_text("warmup", k=10)

    def _make_selector_params(self, allowed_clip_ids):
        # Translate clip_ids to FAISS row ids for corpus-restricted search.
        faiss_ids = self.clip_row_map.rows_for_clips(allowed_clip_ids)
        sel = faiss.IDSelectorBatch(faiss_ids)
        idx = faiss.downcast_index(self.features_index)
        if isinstance(idx, faiss.IndexIVF):
            # Cap the pool-restricted nprobe at 4x baseline: scaling it up
            # with ntotal/len(faiss_ids) degenerates to a full-index scan
            # (nprobe = nlist) once the pool shrinks, which is what made
            # filtered searches 10x+ slower than unfiltered.
            nprobe = min(idx.nlist, idx.nprobe * 4)
            return faiss.SearchParametersIVF(sel=sel, nprobe=nprobe)
        return faiss.SearchParameters(sel=sel)

    @torch.no_grad()
    def encode_text(self, query):
        """Encode a text query into the caption embedding space."""
        return self.model.encode(
            [query], prompt_name="query", normalize_embeddings=True
        )[0].astype(np.float32).reshape(1, -1)

    @torch.no_grad()
    def search_with_text(self, query, k=114618, params=None):
        with self.lock:
            if params is None and query in self.searches:
                return self.searches[query]

            embedding = self.encode_text(query)

            distances, indices = self.features_index.search(embedding, k, params=params)
            # Multiple FAISS rows can map to the same clip_id (one per caption).
            # Keep only the best score per clip_id.
            #
            # FAISS inner-product search returns rows sorted by score desc,
            # so the first occurrence of each clip is its best row.
            indices = np.asarray(indices[0], dtype=np.int64)
            distances = np.asarray(distances[0], dtype=np.float32)
            mask = indices >= 0
            indices = indices[mask]
            distances = distances[mask]
            if len(indices) == 0:
                results = []
            else:
                positions = self.clip_row_map.position_of_row[indices]
                _, first = np.unique(positions, return_index=True)
                best_positions = positions[first]
                best_scores = distances[first]
                order = np.argsort(-best_scores)
                results = list(zip(
                    self.clip_row_map.clip_ids[best_positions[order]].tolist(),
                    best_scores[order].tolist(),
                ))

            if params is None:
                self.searches[query] = results
            return results

    def search(self, filters, current_results):
        if filters.caption_embed_search_text is not None:
            queries = list(
                dict.fromkeys(
                    [filters.caption_embed_search_text]
                    + list(filters.caption_embed_extra_queries or [])
                )
            )
            params = (
                self._make_selector_params(current_results)
                if len(current_results) < CORPUS_RESTRICT_THRESHOLD
                else None
            )
            per_clip_id_score = {}
            for q in queries:
                for clip_id, score in self.search_with_text(q, params=params):
                    if clip_id not in per_clip_id_score or score > per_clip_id_score[clip_id]:
                        per_clip_id_score[clip_id] = score
            current_results = project_starmap(
                lambda r, s: r.with_caption_embed_score(s),
                current_results,
                list(per_clip_id_score.items()),
            )
        return current_results

    def append_embeddings_parquet(self, parquet_file):
        """Append embeddings from a new parquet file without retraining.

        Parquet schema must contain columns: ["clip_id", "embedding"].
        Skips clip_ids already present in the index.
        Returns the number of embeddings added.
        """
        if self.features_index is None:
            raise ValueError("Index not initialized; cannot append.")

        parquet_file = Path(parquet_file)
        if not parquet_file.exists():
            raise FileNotFoundError(str(parquet_file))

        df = pd.read_parquet(parquet_file, columns=["clip_id", "embedding"])

        clips = df["clip_id"].to_numpy()
        features = np.vstack(df["embedding"].values).astype(np.float32, copy=False)
        faiss.normalize_L2(features)

        # Keep the on-disk {row: clip_id} pkl — owned by
        # parse_caption_index_from_dir — as the single source of truth.
        # Load, mutate, write back. Never touch self.clip_row_map from
        # this offline path.
        root = Path(self.path_to_embeddings)
        pkl_path = root / f"caption_index_to_clip_{self._tag}.pkl"
        with open(pkl_path, "rb") as f:
            index_to_clips = pickle.load(f)

        base = int(self.features_index.ntotal)
        self.features_index.add(features)
        for i, cid in enumerate(clips):
            index_to_clips[base + i] = cid

        faiss.write_index(
            self.features_index,
            str(root / f"caption_embeddings_{self._tag}.index"),
        )
        with open(pkl_path, "wb") as f:
            pickle.dump(index_to_clips, f)

        self.searches = LRUDict(size=10)
        return len(clips)
