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

import io
import pickle
import time
from collections import defaultdict
from pathlib import Path
from threading import Lock

import faiss
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import clip
except ImportError:
    print("Clip not found")
    pass

from transformers import AutoModel, AutoProcessor

from sil_wheel.search.features_common_utils import \
    parse_clip_embeddings_from_dir

from sil_wheel.stores.clip_row_map import ClipRowMap
from sil_wheel.stores.utils import LRUDict
from sil_wheel.stores.search_utils import project_starmap

CORPUS_RESTRICT_THRESHOLD = 2_000_000


def spec_to_tag(spec):
    # Convert an index spec to a filesystem-safe tag,
    # e.g. "IVF4096,PQ64x8" -> "ivf4096_pq64x8".
    tag = spec.strip().lower().replace(",", "_").replace(" ", "")
    return tag.replace("/", "_").replace("-", "_")


def load_shard_embeddings(pkl_path, tag):
    """Load one pkl shard, L2-normalise, return (embeddings, clip_ids).

    Returns (None, None) on read/parse failure so the caller can skip.
    """
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"[visual/{tag}] Skipping {pkl_path.name}: {e}")
        return None, None
    embeddings = data["embeddings"].astype(np.float32)
    clip_ids = [item["clip_id"] for item in data["items"]]
    faiss.normalize_L2(embeddings)
    return embeddings, clip_ids


def reservoir_sample_embeddings(pkl_files, tag, sample_size, d, seed=0):
    """Reservoir-sample up to `sample_size` vectors uniformly across shards.

    Single pass (Algorithm R). The sample is used to train an IVF quantizer
    on a distribution representative of the full corpus instead of whichever
    shards happen to sort first. Returns (reservoir, n_seen). The reservoir
    is truncated if the corpus has fewer than `sample_size` vectors.
    """
    reservoir = np.empty((sample_size, d), dtype=np.float32)
    n_seen = 0
    rng = np.random.default_rng(seed)
    for pkl_path in tqdm(
        pkl_files, desc=f"[visual/{tag}] Sampling for training"
    ):
        embeddings, _ = load_shard_embeddings(pkl_path, tag)
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
        for i in accepted:
            reservoir[int(draws[i])] = embeddings[int(i)]
        n_seen += extra

    if n_seen == 0:
        raise RuntimeError(
            f"[visual/{tag}] no embeddings found across "
            f"{len(pkl_files)} shards"
        )
    if n_seen < sample_size:
        reservoir = reservoir[:n_seen]
    return reservoir, n_seen


def set_search_params(index, nprobe):
    # Set nprobe for IVF indexes and efSearch for HNSW indexes.
    # Both control the recall/speed trade-off at query time.
    if hasattr(index, "nprobe"):
        index.nprobe = nprobe
    if hasattr(index, "hnsw"):
        try:
            index.hnsw.efSearch = max(64, nprobe)
        except Exception:
            pass


def parse_visual_index_from_pkl_dir(
    path_to_embeddings,
    index_spec="IVF4096,PQ64x8",
    nprobe=256,
    mmap=False,
):
    """Load or build a FAISS index from Florence2 SigCLIP pkl shards.

    Each pkl shard must contain:
        {"embeddings": np.ndarray (N, 768), "items":      [{"clip_id": str, ...}]}

    Each clip produces multiple embeddings (one per sampled frame and
    detected region).  All views are indexed as independent rows;
    max-score aggregation per clip_id happens at query time.

    Embeddings are L2-normalised before indexing so that inner-product
    equals cosine similarity.

    Returns (features_index, clip_row_map). The compact ClipRowMap is
    persisted alongside the index as
    ``visual_clip_ids_<tag>.npy`` (unique clip_ids, one per position)
    and ``visual_position_of_row_<tag>.npy`` (int32, one per FAISS row).
    """
    path_to_embeddings = Path(path_to_embeddings)
    pkl_files = sorted(
        path_to_embeddings.glob("**/florence2_sigclip_group_*.pkl")
    )
    tag = spec_to_tag(index_spec)

    print(f"[visual/{tag}] Found {len(pkl_files)} pkl files")

    path_to_index = (
        path_to_embeddings / f"visual_embeddings_{tag}.index"
    )
    clip_ids_npy = (
        path_to_embeddings / f"visual_clip_ids_{tag}.npy"
    )
    position_npy = (
        path_to_embeddings / f"visual_position_of_row_{tag}.npy"
    )

    if (
        path_to_index.exists()
        and clip_ids_npy.exists()
        and position_npy.exists()
    ):
        flags = (
            faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY if mmap else 0
        )
        features_index = faiss.read_index(str(path_to_index), flags)
        clip_row_map = ClipRowMap(
            np.load(clip_ids_npy, allow_pickle=True),
            np.load(position_npy),
        )
        print(
            f"[visual/{tag}] Loaded index"
            f" ({features_index.ntotal:,} embeddings,"
            f" {len(clip_row_map.clip_ids):,} clips)"
        )
        return features_index, clip_row_map

    # SigLIP base model produces 768-dimensional embeddings.
    d = 768
    features_index = faiss.index_factory(
        d, index_spec, faiss.METRIC_INNER_PRODUCT
    )
    set_search_params(features_index, nprobe)

    start = time.time()

    # Pass 1: sample uniformly across all shards so the IVF quantizer is
    # trained on a distribution representative of the full corpus, not just
    # whichever shards happen to sort first.
    train_sample, _ = reservoir_sample_embeddings(
        pkl_files, tag, sample_size=1_000_000, d=d,
    )
    print(
        f"[visual/{tag}] Training on {len(train_sample):,} sampled vectors ..."
    )
    features_index.train(train_sample)
    del train_sample
    print(f"[visual/{tag}] Training complete")

    # Pass 2: add every shard to the trained index. State is kept in the
    # compact ClipRowMap shape so we never materialise a per-row dict for
    # multi-hundred-million-row indices.
    clip_ids_unique = []
    position_of_clip_id = {}
    position_of_row_chunks = []

    for ii, pkl_path in enumerate(
        tqdm(pkl_files, desc=f"[visual/{tag}] Indexing")
    ):
        embeddings, clip_ids = load_shard_embeddings(pkl_path, tag)
        if embeddings is None:
            continue

        features_index.add(embeddings)
        positions = np.empty(len(clip_ids), dtype=np.int32)
        for i, cid in enumerate(clip_ids):
            pos = position_of_clip_id.get(cid)
            if pos is None:
                pos = len(clip_ids_unique)
                position_of_clip_id[cid] = pos
                clip_ids_unique.append(cid)
            positions[i] = pos
        position_of_row_chunks.append(positions)

        shard_clips = len(set(clip_ids))
        print(
            f"[visual/{tag}] [{ii + 1}/{len(pkl_files)}] {pkl_path}"
            f" | shard: +{len(embeddings):,} views, +{shard_clips:,} clips"
            f" | total: {features_index.ntotal:,} views,"
            f" {len(clip_ids_unique):,} clips"
        )

        # Checkpoint every 5 files so progress survives a failure.
        if ii % 5 == 0:
            _save_visual_index_artifacts(
                features_index, path_to_index,
                clip_ids_unique, clip_ids_npy,
                position_of_row_chunks, position_npy,
            )

    elapsed = time.time() - start
    print(
        f"[visual/{tag}] Finished in {elapsed:.2f}s —"
        f" ntotal: {features_index.ntotal:,},"
        f" clips: {len(clip_ids_unique):,}"
    )

    clip_ids_arr, position_of_row = _save_visual_index_artifacts(
        features_index, path_to_index,
        clip_ids_unique, clip_ids_npy,
        position_of_row_chunks, position_npy,
    )
    return features_index, ClipRowMap(clip_ids_arr, position_of_row)


def _save_visual_index_artifacts(
    features_index, path_to_index,
    clip_ids_unique, clip_ids_npy,
    position_of_row_chunks, position_npy,
):
    """Persist the FAISS index and ClipRowMap arrays atomically-enough.

    Order: arrays first, then index. If the process dies between writes
    the index will be ahead of the arrays; on the next build the arrays
    are regenerated from scratch, so this is the safer direction.
    """
    clip_ids_arr = np.array(clip_ids_unique, dtype=object)
    position_of_row = (
        np.concatenate(position_of_row_chunks)
        if position_of_row_chunks
        else np.empty(0, dtype=np.int32)
    )
    np.save(clip_ids_npy, clip_ids_arr)
    np.save(position_npy, position_of_row)
    faiss.write_index(features_index, str(path_to_index))
    return clip_ids_arr, position_of_row


class CLIPEmbeddingStore:
    def __init__(self, path_to_embeddings):
        self.lock = Lock()
        self.features_index = None
        self.clips = None
        self.clips_to_index = None
        self.clip_to_faiss_ids = defaultdict(list)
        if path_to_embeddings is not None:
            path_to_embeddings = Path(path_to_embeddings)
            if path_to_embeddings.exists():
                features_index, clips_to_index = \
                    parse_clip_embeddings_from_dir(
                        path_to_embeddings, index_factory="dotprod"
                    )
                self.features_index = features_index
                self.features_index.make_direct_map()
                self.clips_to_index = clips_to_index
                for fid, cid in clips_to_index.items():
                    self.clip_to_faiss_ids[cid].append(fid)
            # Load the CLIP model
            self.clip_model, self.clip_preprocess = clip.load(
                "ViT-B/32", device=torch.device("cuda:0")
            )
            print(
                f"Loaded {self.features_index.ntotal} embeddings for "
                f"{len(set(self.clips_to_index.values()))} clips"
            )
        self.searches = LRUDict(size=10)
        self.uploaded_image_features = LRUDict(size=32)

    def compute_image_features(self, image_bytes):
        image = self.clip_preprocess(
            Image.open(io.BytesIO(image_bytes))
        ).unsqueeze(0).to("cuda:0")
        with torch.no_grad():
            features = self.clip_model.encode_image(image)
        return torch.nn.functional.normalize(features, dim=-1).cpu().detach().numpy()

    def search_with_image(self, image_features, k=2048, params=None):
        with self.lock:
            distances, indices = self.features_index.search(image_features, k, params=params)
            distances = distances[0]
            indices = indices[0]
            mask = indices >= 0
            clip_scores = {}
            for dist, idx in zip(distances[mask].tolist(), indices[mask].tolist()):
                clip_id = self.clips_to_index[idx]
                if clip_id not in clip_scores or dist > clip_scores[clip_id]:
                    clip_scores[clip_id] = dist
            return list(clip_scores.items())

    def _make_selector_params(self, allowed_clip_ids):
        # Translate clip_ids to FAISS positions for corpus-restricted search
        allowed = set(allowed_clip_ids)
        faiss_ids = np.array(
            sum([fids for cid, fids in self.clip_to_faiss_ids.items() if cid in allowed], []),
            dtype=np.int64,
        )
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

    def search_with_text(self, search, k=2048, params=None):
        with self.lock:
            if params is None and search in self.searches:
                return self.searches[search]

            text = clip.tokenize(search).to("cuda:0")
            text_features = self.clip_model.encode_text(text)
            text_features = torch.nn.functional.normalize(
                text_features, dim=-1
            )
            text_features = text_features.cpu().detach().numpy()
            distances, indices = self.features_index.search(
                text_features, k, params=params
            )
            distances = distances[0]
            indices = indices[0]

            mask = indices >= 0
            distances = distances[mask].tolist()
            indices = indices[mask].tolist()

            clip_scores = {}
            for dist, idx in zip(distances, indices):
                clip_id = self.clips_to_index[idx]
                if clip_id not in clip_scores or dist > clip_scores[clip_id]:
                    clip_scores[clip_id] = dist

            results = list(clip_scores.items())
            if params is None:
                self.searches[search] = results
            return results

    def search(self, filters, current_results):
        if filters.visual_search_text is None and filters.visual_search_image_id is None:
            return current_results

        pool_params = (
            self._make_selector_params(current_results)
            if len(current_results) < CORPUS_RESTRICT_THRESHOLD
            else None
        )

        if filters.visual_search_text is not None:
            queries = list(dict.fromkeys(
                [filters.visual_search_text] + list(filters.visual_extra_queries or [])
            ))
            per_clip_id_score = {}
            for q in queries:
                for clip_id, score in self.search_with_text(q, params=pool_params):
                    if clip_id not in per_clip_id_score or score > per_clip_id_score[clip_id]:
                        per_clip_id_score[clip_id] = score
            current_results = project_starmap(
                lambda r, s: r.with_visual_search_score(s),
                current_results,
                list(per_clip_id_score.items()),
            )

        if filters.visual_search_image_id is not None:
            cache_key = f"image:{filters.visual_search_image_id}"
            if pool_params is None and cache_key not in self.searches:
                image_features = self.uploaded_image_features.get(filters.visual_search_image_id)
                if image_features is not None:
                    self.searches[cache_key] = self.search_with_image(image_features)
            if pool_params is not None:
                image_features = self.uploaded_image_features.get(filters.visual_search_image_id)
                results = self.search_with_image(image_features, params=pool_params) if image_features is not None else []
            else:
                results = self.searches.get(cache_key, [])
            current_results = project_starmap(
                lambda r, s: r.with_visual_image_score(s),
                current_results,
                results,
            )

        return current_results


class Florence2SigCLIPEmbeddingStore:
    def __init__(
        self,
        path_to_embeddings,
        index_spec="IVF4096,PQ64x8",
        nprobe=256,
        text_prompt_template="a photo of {text}",
        mmap=False,
        siglip_model="google/siglip2-base-patch16-224",
    ):
        self.lock = Lock()
        self.features_index = None
        self.clip_row_map = None
        self.path_to_embeddings = Path(path_to_embeddings)
        self._tag = spec_to_tag(index_spec)
        self._nprobe = nprobe
        self._text_prompt_template = text_prompt_template

        if self.path_to_embeddings is not None:
            if self.path_to_embeddings.exists():
                faiss_path = self.path_to_embeddings / f"visual_embeddings_{self._tag}.index"
                clip_ids_npy = self.path_to_embeddings / f"visual_clip_ids_{self._tag}.npy"
                position_npy = self.path_to_embeddings / f"visual_position_of_row_{self._tag}.npy"

                if clip_ids_npy.exists() and position_npy.exists():
                    if mmap:
                        self.features_index = faiss.read_index(
                            str(faiss_path),
                            faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
                        )
                    else:
                        self.features_index = faiss.read_index(
                            str(faiss_path)
                        )
                    set_search_params(self.features_index, nprobe)
                    self.clip_row_map = ClipRowMap(
                        np.load(clip_ids_npy, allow_pickle=True),
                        np.load(position_npy),
                    )
                else:
                    features_index, clip_row_map = (
                        parse_visual_index_from_pkl_dir(
                            self.path_to_embeddings,
                            index_spec=index_spec,
                            nprobe=nprobe,
                            mmap=mmap,
                        )
                    )
                    set_search_params(features_index, nprobe)
                    self.features_index = features_index
                    self.clip_row_map = clip_row_map

                n_clips = len(self.clip_row_map.clip_ids)
                print(
                    f"[visual/{self._tag}] - ntotal: {self.features_index.ntotal:,},"
                    f" clips: {n_clips:,}"
                )

        # Must use the same model variant used during extraction
        # so that text and image queries share the embedding space.
        #self._device = torch.device("cuda:0")
        self._device = torch.device("cpu")
        self.siglip_model = AutoModel.from_pretrained(
            siglip_model
        ).to(self._device)
        self.siglip_processor = AutoProcessor.from_pretrained(
            siglip_model
        )
        self.siglip_model.eval()

        # LRU cache to avoid re-encoding repeated text queries.
        self.searches = LRUDict(size=10)
        # Cache of upload_id -> image features, populated by the server
        # on /upload_image and looked up in search().
        self.uploaded_image_features = LRUDict(size=10)

    def warmup(self):
        if self.features_index is None:
            return
        self.search_with_text("warmup", k=10)

    @torch.no_grad()
    def encode_text(self, text):
        # SigLIP was trained with image-caption style prompts, so raw
        # search strings are wrapped in a template for closer alignment.
        prompt = self._text_prompt_template.format(text=text)
        inputs = self.siglip_processor(
            text=[prompt], return_tensors="pt", padding="max_length",
        ).to(self._device)
        features = self.siglip_model.get_text_features(**inputs)
        features = torch.nn.functional.normalize(features, dim=-1)
        return features.cpu().numpy().astype(np.float32)

    def compute_image_features(self, image_bytes):
        # Encode a raw image to a normalised vector for
        # image-to-video search.
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.siglip_processor(
            images=image, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            features = self.siglip_model.get_image_features(**inputs)
        features = torch.nn.functional.normalize(features, dim=-1)
        return features.cpu().numpy().astype(np.float32)

    def _make_selector_params(self, allowed_clip_ids):
        # Build a FAISS IDSelector that restricts search to rows that
        # belong to allowed_clip_ids.
        faiss_ids = self.clip_row_map.rows_for_clips(allowed_clip_ids)
        if len(faiss_ids) == 0:
            return faiss.SearchParameters(
                sel=faiss.IDSelectorBatch(np.empty(0, dtype=np.int64))
            )

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

    def _aggregate_scores(self, distances, indices):
        # Each clip has many FAISS rows (one per frame / region).
        # Keep the single highest similarity score across all rows.
        #
        # FAISS inner-product search returns rows sorted by score desc,
        # so the first occurrence of each clip is its best row. That
        # turns "max per clip" into "first per clip", which is one
        # np.unique call over the full top-k at C speed.
        indices = np.asarray(indices, dtype=np.int64)
        distances = np.asarray(distances, dtype=np.float32)
        mask = indices >= 0
        indices = indices[mask]
        distances = distances[mask]
        if len(indices) == 0:
            return []

        positions = self.clip_row_map.position_of_row[indices]
        _, first = np.unique(positions, return_index=True)
        best_positions = positions[first]
        best_scores = distances[first]
        order = np.argsort(-best_scores)
        return list(zip(
            self.clip_row_map.clip_ids[best_positions[order]].tolist(),
            best_scores[order].tolist(),
        ))

    def search_with_text(self, search, k=114618, params=None):
        with self.lock:
            if params is None and search in self.searches:
                return self.searches[search]

            query = self.encode_text(search)
            distances, indices = self.features_index.search(
                query, k, params=params
            )
            results = self._aggregate_scores(distances[0], indices[0])

            if params is None:
                self.searches[search] = results
            return results

    def search_with_image(self, image_features, k=114618, params=None):
        with self.lock:
            distances, indices = self.features_index.search(
                image_features, k, params=params
            )
            return self._aggregate_scores(distances[0], indices[0])

    def search(self, filters, current_results):
        no_text = filters.visual_search_text is None
        no_image = filters.visual_search_image_id is None
        if no_text and no_image:
            return current_results

        # Use a corpus-restricted selector when the active result set
        # is small; otherwise search the full index for efficiency.
        pool_params = (
            self._make_selector_params(current_results)
            if len(current_results) < CORPUS_RESTRICT_THRESHOLD
            else None
        )

        if filters.visual_search_text is not None:
            # De-duplicate queries (preserving order) so that the
            # query rewriter cannot inflate scores via repeats.
            queries = list(dict.fromkeys(
                [filters.visual_search_text]
                + list(filters.visual_extra_queries or [])
            ))

            # Max score over all queries that matched this clip
            per_clip_id_score = {}
            for q in queries:
                for clip_id, score in self.search_with_text(
                    q, params=pool_params
                ):
                    prev = per_clip_id_score.get(clip_id, None)
                    if prev is None or score > prev:
                        per_clip_id_score[clip_id] = score
            current_results = project_starmap(
                lambda r, s: r.with_visual_search_score(s),
                current_results,
                list(per_clip_id_score.items()),
            )

        if filters.visual_search_image_id is not None:
            img_id = filters.visual_search_image_id
            cache_key = f"image:{img_id}"
            if pool_params is None and cache_key not in self.searches:
                image_features = (
                    self.uploaded_image_features.get(img_id)
                )
                if image_features is not None:
                    self.searches[cache_key] = (
                        self.search_with_image(image_features)
                    )
            if pool_params is not None:
                image_features = self.uploaded_image_features.get(img_id)
                results = (
                    self.search_with_image(image_features, params=pool_params)
                    if image_features is not None
                    else []
                )
            else:
                results = self.searches.get(cache_key, [])
            current_results = project_starmap(
                lambda r, s: r.with_visual_image_score(s),
                current_results,
                results,
            )

        return current_results

    def append_pkl(self, pkl_paths):
        """Append embeddings for clip_ids not yet present in the index.

        Filters each shard by clip_id membership against the in-memory
        ClipRowMap: clips already indexed are skipped, new ones are added
        and their FAISS rows are recorded incrementally. The .npy
        artifacts and FAISS index are persisted once at the end so a
        partial run does not leave .npy out of sync with the index.
        """
        if self.features_index is None or self.clip_row_map is None:
            raise RuntimeError(
                f"[visual/{self._tag}] cannot append: store has no "
                "loaded index. Build the index first."
            )

        root = Path(self.path_to_embeddings)
        index_path = root / f"visual_embeddings_{self._tag}.index"
        clip_ids_npy = root / f"visual_clip_ids_{self._tag}.npy"
        position_npy = root / f"visual_position_of_row_{self._tag}.npy"

        position_of_clip_id = dict(self.clip_row_map.position_of_clip_id)
        n_existing_clips = len(self.clip_row_map.clip_ids)
        new_clip_ids = []
        new_position_chunks = []

        n_appended_rows = 0
        dirty = False

        for pkl_path in tqdm(pkl_paths):
            pkl_path = Path(pkl_path)
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                print(f"[visual/{self._tag}] Skipping {pkl_path}: {e}")
                continue

            embeddings = data["embeddings"].astype(np.float32)
            shard_clip_ids = [str(item["clip_id"]) for item in data["items"]]

            mask = np.fromiter(
                (cid not in position_of_clip_id for cid in shard_clip_ids),
                dtype=bool,
                count=len(shard_clip_ids),
            )
            n_new = int(mask.sum())
            if n_new == 0:
                continue

            embeddings = embeddings[mask]
            kept_clip_ids = [c for c, m in zip(shard_clip_ids, mask) if m]

            # Normalize the embeddings
            faiss.normalize_L2(embeddings)
            self.features_index.add(embeddings)

            positions = np.empty(len(kept_clip_ids), dtype=np.int32)
            for i, cid in enumerate(kept_clip_ids):
                pos = position_of_clip_id.get(cid)
                if pos is None:
                    pos = n_existing_clips + len(new_clip_ids)
                    position_of_clip_id[cid] = pos
                    new_clip_ids.append(cid)
                positions[i] = pos
            new_position_chunks.append(positions)

            n_appended_rows += n_new
            dirty = True

            print(
                f"[visual/{self._tag}] {pkl_path.name}"
                f" | +{n_new:,} views, +{len(set(kept_clip_ids)):,} clips"
                f" | total: {self.features_index.ntotal:,} views,"
                f" {len(position_of_clip_id):,} clips"
            )

        if not dirty:
            return

        final_clip_ids = np.concatenate([
            self.clip_row_map.clip_ids,
            np.array(new_clip_ids, dtype=object),
        ])
        final_position_of_row = np.concatenate(
            [self.clip_row_map.position_of_row] + new_position_chunks
        )

        np.save(clip_ids_npy, final_clip_ids)
        np.save(position_npy, final_position_of_row)
        faiss.write_index(self.features_index, str(index_path))

        self.clip_row_map = ClipRowMap(
            final_clip_ids, final_position_of_row
        )

        print(
            f"[visual/{self._tag}] Appended {n_appended_rows:,} views,"
            f" {len(new_clip_ids):,} new clips"
            f" | total: {self.features_index.ntotal:,} views,"
            f" {len(final_clip_ids):,} clips"
        )
