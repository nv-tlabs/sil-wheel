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

import json
import os
import pickle
import sys
import time
from pathlib import Path
from threading import Lock

import faiss
import numpy as np
from safetensors.numpy import safe_open
from sil_wheel.stores.search_utils import project_dict, project_starmap, select_key_source
from sil_wheel.stores.time_utils import Timer
from tqdm import tqdm
from sil_wheel.stores.utils import LRUDict

CORPUS_RESTRICT_THRESHOLD = 2_000_000

TRAJECTORY_EXPRESSIONS = {
    "high_curvature": "sum(curvature > 0.15) > 10",
    "stop_go": (
        "any(speed < 0.5) and "
        "any(speed > 3.0) and "
        "min(np.where(speed > 3.0)[0]) > min(np.where(speed < 0.5)[0])"
    ),
    "hard_braking": "sum(acceleration < -3.0) > 10",
    "prolonged_stop": "sum(speed < 0.5) > 150",
    "idle_to_cruise": (
        "any(speed < 0.5) and "
        "any(speed > 10.0) and "
        "min(np.where(speed > 10.0)[0]) > min(np.where(speed < 0.5)[0])"
    ),
    "high_speed_swerve": (
        "sum(curvature > 0.2) > 10 and sum(speed_kph > 50) > 10"
    ),
    "moving_ego": "sum(speed_kph > 5) > 10",
}

# Windows used by subtrajectory functions (in seconds)
WINDOWS = {
    10: [(0, 10), (5, 15), (10, 20)],
    5: [(0, 5), (5, 10), (10, 15), (15, 20)],
}

def spec_to_tag(spec: str) -> str:
    """Filename tag for a FAISS index_factory spec.
    """
    tag = spec.strip().replace(",", "_").replace(" ", "")
    return tag.replace("/", "_").replace("-", "_")


def _extract_ivf_or_none(index):
    """Return the inner IndexIVF, or None if the index is not IVF-family.

    faiss.extract_index_ivf raises RuntimeError on non-IVF indexes (e.g.
    IndexFlat) rather than returning None. Wrap it once so callers can
    branch on ``is not None``.
    """
    try:
        return faiss.extract_index_ivf(index)
    except RuntimeError:
        return None


def _build_trajectory_vec(data, trajectory_dim):
    """Centered, fixed-length full-trajectory vector.

    Returns shape (1, 2*trajectory_dim) as float32.
    Long trajectories are truncated; short ones are zero-padded.
    The first (x, y) point is subtracted from all coordinates.
    """
    T = int(data.shape[0])
    if T >= trajectory_dim:
        traj = data[:trajectory_dim, :2].astype(np.float32, copy=False)
        traj -= traj[0]
    else:
        traj = np.zeros((trajectory_dim, 2), dtype=np.float32)
        if T > 0:
            tmp = data[:T, :2].astype(np.float32, copy=False)
            tmp -= tmp[0]
            traj[:T] = tmp
    return traj.reshape(1, -1)


def _build_subtrajectory_vec(data, i0, i1, sub_traj_dim):
    """Centered sub-trajectory vector for window [i0, i1).

    Returns shape (1, 2*sub_traj_dim) as float32.
    Out-of-range or partially available windows are zero-padded.
    """
    T = int(data.shape[0])
    traj = np.zeros((sub_traj_dim, 2), dtype=np.float32)
    if T > i0:
        t = max(0, min(T, i1) - i0)
        if t > 0:
            tmp = data[i0 : i0 + t, :2].astype(np.float32, copy=False)
            tmp -= tmp[0]
            traj[:t] = tmp
    return traj.reshape(1, -1)


def parse_trajectory_data_from_dir(
    path_to_trajectory_data,
    nprobe=256,
    *,
    index_spec="OPQ121,IVF4096,PQ121x8",
):
    """Build (or load) the full-trajectory FAISS index.

    Args:
        path_to_trajectory_data: directory holding ``*/*.safetensors`` shards.
        nprobe: nprobe for IVF-family indexes; ignored for non-IVF (e.g. Flat).
        index_spec: any FAISS ``index_factory`` spec. Default is the
            production OPQ/IVF/PQ spec; pass ``"Flat"`` for exact search on
            small corpora that can't train an IVF quantizer.
    """
    start = time.time()
    base_dir = Path(path_to_trajectory_data)

    path_to_safetensors = sorted(base_dir.rglob("*/*.safetensors"))
    print(
        f"[trajectory_index] Found {len(path_to_safetensors)} safetensors "
        f"(index_spec={index_spec})"
    )

    trajectory_dim = 605
    d = 2 * trajectory_dim
    tag = spec_to_tag(index_spec)

    path_to_faiss_index = base_dir / f"trajectory_data_{tag}_p1.index"
    path_to_clip_index = base_dir / f"trajectory_clip_to_index_{tag}_p1.pkl"

    # Fast path: load an existing index for this spec.
    if path_to_faiss_index.exists() and path_to_clip_index.exists():
        features_index = faiss.read_index(str(path_to_faiss_index))
        with open(path_to_clip_index, "rb") as f:
            clip_to_index = pickle.load(f)
        clip_to_index = {sys.intern(k): v for k, v in clip_to_index.items()}
        print(
            f"[trajectory_index] Loaded feature index from {path_to_faiss_index}"
        )
        print(f"Trajectory embeddings of size {features_index.ntotal}...")
        return features_index, clip_to_index

    features_index = faiss.index_factory(d, index_spec, faiss.METRIC_L2)
    ivf = _extract_ivf_or_none(features_index)
    if ivf is not None:
        ivf.nprobe = nprobe
    TRAIN_CHUNK = 1000000

    buf = []
    clip_to_index = {}
    offset = 0
    buf_count = 0
    # Flat indexes are already trained; only IVF needs a training batch.
    needs_training = not features_index.is_trained

    ADD_CHUNK = 500000

    def finalize_batch(batch, do_train):
        """Concatenate & add (and optionally train) a batch."""
        arr = np.ascontiguousarray(
            np.concatenate(batch, axis=0), dtype=np.float32
        )
        if do_train:
            print("Training...")
            features_index.train(arr)
        features_index.add(arr)

    for path_to_data in tqdm(path_to_safetensors):
        print(path_to_data)
        with safe_open(path_to_data, framework="np") as f:
            for clip_id in tqdm(f.keys()):
                if clip_id in clip_to_index:
                    continue

                data = f.get_tensor(clip_id)  # (T, C)
                buf.append(_build_trajectory_vec(data, trajectory_dim))
                clip_to_index[clip_id] = offset
                buf_count += 1
                offset += 1

                # Train once on the first big batch, then add in chunks.
                if needs_training and buf_count >= TRAIN_CHUNK:
                    finalize_batch(buf, do_train=True)
                    needs_training = False
                    buf_count = 0
                    buf.clear()
                elif (not needs_training) and buf_count >= ADD_CHUNK:
                    finalize_batch(buf, do_train=False)
                    buf_count = 0
                    buf.clear()

                    faiss.write_index(features_index, str(path_to_faiss_index))
                    with open(path_to_clip_index, "wb") as sf:
                        pickle.dump(clip_to_index, sf)
                    print("Saved intermediate_result")

    # Flush remaining: if IVF still needs training, train on the residual.
    # For Flat, this is just an add().
    if buf_count > 0:
        finalize_batch(buf, do_train=needs_training)

    elapsed = time.time() - start
    print(f"[trajectory_index] Finished indexing in {elapsed:.2f} seconds")
    print("[trajectory_index] ntotal:", features_index.ntotal)
    print("[trajectory_index] type:", type(features_index))
    print(
        "[trajectory_index] is_trained:",
        getattr(features_index, "is_trained", True),
    )
    if hasattr(features_index, "nlist"):
        print(
            "[trajectory_index] nlist:",
            features_index.nlist,
            "nprobe:",
            getattr(features_index, "nprobe", "-"),
        )
    print("[trajectory_index] FAISS OMP threads:", faiss.omp_get_max_threads())

    faiss.write_index(features_index, str(path_to_faiss_index))

    with open(path_to_clip_index, "wb") as f:
        pickle.dump(clip_to_index, f)

    return features_index, clip_to_index


def update_index(path_to_trajectory_data, add_chunk=500000):
    """
    Incrementally add full-trajectory features to an already-trained FAISS index.
    Only clips not already present in clip_to_index are added. This function does
    not perform any training; it assumes the provided index is already trained.
    """

    base_dir = Path(path_to_trajectory_data)
    path_to_faiss_index = (
        base_dir / "trajectory_data_OPQ121_IVF4096_PQ121x8_p0.index"
    )
    path_to_clip_index = (
        base_dir / "trajectory_clip_to_index_OPQ121_IVF4096_PQ121x8_p0.pkl"
    )
    features_index = faiss.read_index(str(path_to_faiss_index))
    with open(path_to_clip_index, "rb") as f:
        clip_to_index = pickle.load(f)
    print(f"[Loaded feature index from {path_to_faiss_index} ")
    print(f"Trajectory embeddings of size {features_index.ntotal}...")
    print(f"Clips to index {len(clip_to_index)}")
    # Ensure we only add to a trained index (if the attribute exists)
    if hasattr(features_index, "is_trained") and not features_index.is_trained:
        raise ValueError(
            "features_index must be trained before calling update_index"
        )

    path_to_safetensors = sorted(base_dir.rglob("*/*.safetensors"))

    # The full-trajectory embedding packs (605 x 2) values and flattens to (1210,)
    trajectory_dim = 605

    # Offset starts from current index size
    offset = int(getattr(features_index, "ntotal", 0))

    buf = []
    buf_count = 0
    path_to_faiss_index_new = (
        base_dir / "trajectory_data_OPQ121_IVF4096_PQ121x8_p1.index"
    )
    path_to_clip_index_new = (
        base_dir / "trajectory_clip_to_index_OPQ121_IVF4096_PQ121x8_p1.pkl"
    )

    def flush_buffer(batch):
        arr = np.ascontiguousarray(
            np.concatenate(batch, axis=0), dtype=np.float32
        )
        features_index.add(arr)
        print(features_index.ntotal, len(buf))

    for path_to_data in tqdm(path_to_safetensors):
        with safe_open(path_to_data, framework="np") as f:
            for clip_id in f.keys():
                # Skip clips already indexed
                if clip_id in clip_to_index:
                    continue

                data = f.get_tensor(clip_id)  # (T, C)
                if np.isnan(data).any():
                    continue

                buf.append(_build_trajectory_vec(data, trajectory_dim))
                clip_to_index[clip_id] = offset
                buf_count += 1
                offset += 1

                if buf_count >= add_chunk:
                    flush_buffer(buf)
                    faiss.write_index(
                        features_index, str(path_to_faiss_index_new)
                    )
                    buf.clear()
                    buf_count = 0

                    with open(path_to_clip_index_new, "wb") as sf:
                        pickle.dump(clip_to_index, sf)
                    print(f"Saved intermediate_result with {features_index.ntotal}")

    # Flush any remaining buffered vectors
    if buf_count > 0:
        flush_buffer(buf)

    faiss.write_index(features_index, str(path_to_faiss_index_new))

    with open(path_to_clip_index_new, "wb") as sf:
        pickle.dump(clip_to_index, sf)

    print(f"Trajectory embeddings of size {features_index.ntotal}...")
    print(f"Clips to index {len(clip_to_index)}")

    return features_index, clip_to_index


def get_feature_embedding(clip_id, features_index, clips_to_index):
    if features_index is not None and clip_id in clips_to_index:
        ids = clips_to_index[clip_id]
        if isinstance(ids, (list, tuple)):
            query_feats = [features_index.reconstruct(int(i)) for i in ids]
            query_features = np.mean(
                np.stack(query_feats, axis=0), axis=0
            ).astype(np.float32)
        else:
            query_features = features_index.reconstruct(int(ids))
        return query_features
    else:
        return None


class LazyTrajectoryData:
    """Dict-like, mmap-backed view over per-clip trajectory tensors.

    Keeps a clip_id -> source-path index in RAM and reads individual tensors
    on demand via safetensors.
    """

    def __init__(self, paths):
        self._paths = list(paths)
        self._clip_to_path = {}
        for p in tqdm(self._paths, desc="Indexing trajectory files"):
            with safe_open(str(p), framework="np") as f:
                for k in f.keys():
                    self._clip_to_path[sys.intern(k)] = p

    def __len__(self):
        return len(self._clip_to_path)

    def __contains__(self, clip_id):
        return clip_id in self._clip_to_path

    def __iter__(self):
        return iter(self._clip_to_path)

    def keys(self):
        return self._clip_to_path.keys()

    def __getitem__(self, clip_id):
        with safe_open(str(self._clip_to_path[clip_id]), framework="np") as f:
            return f.get_tensor(clip_id)

    def items(self):
        # Iterate in file-order so each safetensors file is opened exactly once.
        for p in self._paths:
            with safe_open(str(p), framework="np") as f:
                for k in f.keys():
                    yield sys.intern(k), f.get_tensor(k)


class TrajectoryStore:
    def __init__(self, path_to_data, debug=False, *, index_spec=None):
        """
        Args:
            path_to_data: directory holding the trajectory artifacts (memmap,
                clip_to_idx, safetensors shards, FAISS indexes). Pass None to
                disable trajectory features entirely.
            debug: if True, only the first safetensors shard is indexed.
            index_spec: FAISS index_factory spec applied to all three indexes
                (full / 10s / 5s). When None (default), each index uses the
                production OPQ/IVF/PQ default for its window length. Set to
                ``"Flat"`` for exact search on small corpora.
        """
        self.lock = Lock()

        self.trajectory_data = {}
        self.searches = LRUDict(size=10)
        self.searches_shapes = LRUDict(size=10)
        # To hold the compiled predicates
        self._search_predicates = {}
        self._index_spec = index_spec

        self.timers = Timer()
        if path_to_data is not None:
            path_to_trajectory_data = sorted(
                Path(path_to_data).rglob(
                    "trajectory_data_downsampled_d5_*.safetensors"
                )
            )
            if debug:
                path_to_trajectory_data = path_to_trajectory_data[:1]
            self.trajectory_data = LazyTrajectoryData(path_to_trajectory_data)

            # Load the indexes for the full trajecetory and the sub-trajectory
            # search
            self.features_indexes = {}
            self._load_features_index(path_to_data, tag="full")
            self._load_features_index(path_to_data, tag="10s", sec=10, M=40)
            self._load_features_index(path_to_data, tag="5s", sec=5, M=20)

            # Parse the data to be visualized from the memory map
            clip_to_idx_path = Path(path_to_data) / "clip_to_idx.json"
            with open(clip_to_idx_path, "r") as f:
                self.clip_to_idx = {
                    sys.intern(k): v for k, v in json.load(f).items()
                }

            path_to_traj_mmap = Path(path_to_data) / "trajectory_data.dat"
            print(path_to_traj_mmap)
            itemsize = np.dtype(np.float32).itemsize
            rows = os.path.getsize(path_to_traj_mmap) // (itemsize * 7)
            self.traj_mmap = np.memmap(
                path_to_traj_mmap, dtype=np.float32, mode="r", shape=(rows, 7)
            )
        else:
            print("No trajectory data found")
            self.clip_to_idx = {}

    def _load_features_index(self, path_to_data, tag, sec=None, M=None):
        # When self._index_spec is None, each parse_* function falls back to
        # its own production default. Otherwise the same spec is used for all
        # three indexes (typical for "Flat" on small corpora).
        if sec is None:
            kwargs = {} if self._index_spec is None else {"index_spec": self._index_spec}
            features_index, clips_to_index = parse_trajectory_data_from_dir(
                path_to_data, **kwargs,
            )
        else:
            features_index, clips_to_index = parse_subtrajectory_data_from_dir(
                path_to_data, sec, M, index_spec=self._index_spec,
            )
        ivf = _extract_ivf_or_none(features_index)
        if ivf is None:
            print("Note: index is not IVF-family; skipping direct map.")
        else:
            ivf.make_direct_map()

        index_to_clips = {}
        for clip, index in clips_to_index.items():
            if isinstance(index, list):
                for ind in index:
                    index_to_clips[int(ind)] = clip
            else:
                index_to_clips[int(index)] = clip

        print("ntotal:", features_index.ntotal)
        print("type:", type(features_index))
        print("is_trained:", getattr(features_index, "is_trained", True))
        if hasattr(features_index, "nlist"):
            print(
                "nlist:",
                features_index.nlist,
                "nprobe:",
                getattr(features_index, "nprobe", "-"),
            )
        print("FAISS OMP threads:", faiss.omp_get_max_threads())
        self.features_indexes[tag] = {
            "feature_index": features_index,
            "clips_to_index": clips_to_index,
            "index_to_clips": index_to_clips,
        }

    def get_feature_params_index(self, tag):
        return (
            self.features_indexes[tag]["feature_index"],
            self.features_indexes[tag]["clips_to_index"],
            self.features_indexes[tag]["index_to_clips"],
        )

    def has_trajectories(self, clip_id):
        return clip_id in self.clip_to_idx

    def get_trajectory_data(self, clip_id):
        start, end = self.clip_to_idx[clip_id]
        return self.traj_mmap[start:end, :]

    def get_positions(self, clip_id):
        if clip_id not in self.clip_to_idx:
            return []

        return self.get_trajectory_data(clip_id)[:, :2].tolist()[::10]

    def get_speed(self, clip_id):
        if clip_id not in self.clip_to_idx:
            return []

        return self.get_trajectory_data(clip_id)[:, 3].tolist()

    def get_acceleration(self, clip_id):
        if clip_id not in self.clip_to_idx:
            return []

        return self.get_trajectory_data(clip_id)[:, 4].tolist()

    def get_jerk(self, clip_id):
        if clip_id not in self.clip_to_idx:
            return []

        return self.get_trajectory_data(clip_id)[:, 5].tolist()

    def get_curvature(self, clip_id):
        if clip_id not in self.clip_to_idx:
            return []

        return self.get_trajectory_data(clip_id)[:, 6].tolist()

    def _make_selector_params(self, allowed_clip_ids, tag="full"):
        clips_to_index = self.features_indexes[tag]["clips_to_index"]
        allowed = set(allowed_clip_ids)
        faiss_ids = []
        for cid in allowed:
            if cid not in clips_to_index:
                continue
            idx = clips_to_index[cid]
            if isinstance(idx, list):
                faiss_ids.extend(idx)
            else:
                faiss_ids.append(int(idx))
        faiss_ids = np.array(faiss_ids, dtype=np.int64)
        sel = faiss.IDSelectorBatch(faiss_ids)
        ivf = _extract_ivf_or_none(self.features_indexes[tag]["feature_index"])
        if ivf is not None:
            # Cap the pool-restricted nprobe at 4x baseline: scaling it up
            # with ntotal/len(faiss_ids) degenerates to a full-index scan
            # (nprobe = nlist) once the pool shrinks, which is what made
            # filtered searches 10x+ slower than unfiltered.
            nprobe = min(ivf.nlist, ivf.nprobe * 4)
            return faiss.SearchParametersIVF(sel=sel, nprobe=nprobe)
        return faiss.SearchParameters(sel=sel)

    def _resolve_tag(self, start_time, end_time):
        if start_time is None and end_time is None:
            return "full", start_time, end_time
        if end_time is None:
            end_time = 20
        if start_time is None:
            start_time = 0
        duration = end_time - start_time
        if duration <= 6.0:
            return "5s", start_time, end_time
        elif duration <= 15.0:
            return "10s", start_time, end_time
        return "full", start_time, end_time

    def search_with_video_clip(
        self, search, start_time, end_time, n_neighbors=114618, params=None
    ):
        search_tag = search
        if start_time is not None:
            search_tag += f"_start_{start_time}"
        if end_time is not None:
            search_tag += f"_end_{end_time}"

        with self.lock:
            self.timers.tic()
            if params is None and search_tag in self.searches_shapes:
                print(f"The trajectory search in faiss took {self.timers.toc()}")
                return self.searches_shapes[search_tag]

            tag, start_time, end_time = self._resolve_tag(start_time, end_time)
            print(start_time, end_time, tag)
            features_index, clips_to_index, index_to_clips = (
                self.get_feature_params_index(tag)
            )

            query_features = get_feature_embedding(
                search, features_index, clips_to_index
            )
            if query_features is None:
                return []
            if query_features is not None:
                # Namely take all neighbors
                if n_neighbors == -1:
                    n_neighbors = features_index.ntotal
                qf = np.ascontiguousarray(
                    query_features.reshape(1, -1), dtype=np.float32
                )
                distances, indices = features_index.search(qf, n_neighbors, params=params)
                distances = distances[0]
                indices = indices[0]

                mask = indices >= 0
                distances = distances[mask]
                indices = indices[mask]
                video_ids = [index_to_clips[ind] for ind in indices]
                distances = distances.tolist()

                if search not in video_ids:
                    print(f"Adding the query_clip_id {search}")
                    video_ids.insert(0, search)
                    distances.insert(0, 0.0)

            result = list(zip(video_ids, distances))
            if params is None:
                self.searches_shapes[search_tag] = result
            print(f"The trajectory search in faiss took {self.timers.toc()}")

            return result

    def _inner_search_trajectory(self, search_query, ids):
        with self.lock:
            self.timers.tic()
            ids_set = ids if isinstance(ids, set) else set(ids)
            # Search caching
            if search_query in self.searches:
                return self.searches[search_query] & ids_set

            # Compile the predicate
            _globals = {
                "__builtins__": {},
                "np": np,
                "mean": np.mean,
                "min": np.min,
                "max": np.max,
                "sum": np.sum,
                "all": np.all,
                "any": np.any,
                "len": len,
            }
            search_query = search_query.replace("\n", " ")
            fn = ""
            fn += "def predicate(speed, acceleration, jerk, curvature):\n"
            fn += "    speed_kph = speed * 3.6\n"
            fn += "    return "
            fn += search_query
            exec(fn, _globals)

            # Linear search in the stats
            video_ids_universe = set()
            for clip_id, stats in self.trajectory_data.items():
                speed = stats[:, 3]
                acceleration = stats[:, 4]
                jerk = stats[:, 5]
                curvature = stats[:, 6]
                if not _globals["predicate"](
                    speed, acceleration, jerk, curvature
                ):
                    continue
                video_ids_universe.add(clip_id)
            self.searches[search_query] = video_ids_universe
            return video_ids_universe & ids_set

    def search_trajectory(self, filters, ids):
        query = ""
        if filters.trajectory_pattern is not None:
            query = TRAJECTORY_EXPRESSIONS[filters.trajectory_pattern]
        if filters.search_speed is not None:
            if query != "":
                query += " and "
            query += filters.search_speed
        return self._inner_search_trajectory(query, ids)

    def search(self, filters, current_results):
        already_filtered = False

        if filters.trajectory_shape_clipid is not None:
            already_filtered = True
            tag, _, _ = self._resolve_tag(
                filters.trajectory_shape_start_t,
                filters.trajectory_shape_end_t,
            )
            pool_params = (
                self._make_selector_params(current_results, tag=tag)
                if len(current_results) < CORPUS_RESTRICT_THRESHOLD else None
            )
            current_results = project_starmap(
                lambda r, s: r.with_trajectory_score(s),
                current_results,
                self.search_with_video_clip(
                    filters.trajectory_shape_clipid,
                    filters.trajectory_shape_start_t,
                    filters.trajectory_shape_end_t,
                    params=pool_params,
                ),
            )

        if (
            filters.trajectory_pattern is not None
            or filters.search_speed is not None
        ):
            already_filtered = True
            current_results = project_dict(
                current_results,
                self.search_trajectory(
                    filters,
                    select_key_source(self.trajectory_data, current_results),
                ),
            )

        if not already_filtered and filters.with_ego_data is not None:
            current_results = project_dict(
                current_results, self.clip_to_idx.keys()
            )

        return current_results


def parse_trajectory_mmap(trajectory_data):
    clip_to_idx = {}
    cnt = 0
    path_to_output = f"{trajectory_data}/trajectory_data.dat"

    path_to_trajectory_data = sorted(
        Path(trajectory_data).rglob("*/*.safetensors")
    )

    json_path = f"{trajectory_data}/clip_to_idx.json"
    SAVE_EVERY = 10000

    # Precompute total number of rows across ALL safetensors.
    total_rows = 0
    for path_to_data in tqdm(path_to_trajectory_data):
        with safe_open(path_to_data, framework="np") as f:
            for k in f.keys():
                total_rows += f.get_tensor(k).shape[0]

    # Create the memmap with the correct final shape.
    fp = np.memmap(
        path_to_output, dtype="float32", mode="w+", shape=(total_rows, 7)
    )

    all_data = np.empty((0, 7), dtype=np.float32)
    old = 0

    # Iterate all safetensors and write in chunks.
    for path_to_data in path_to_trajectory_data:
        print(path_to_data)
        with safe_open(path_to_data, framework="np") as f:
            for k in tqdm(f.keys()):
                data = f.get_tensor(k)
                assert data.shape[1] == 7

                start = cnt
                end = cnt + len(data)
                clip_to_idx[k] = (start, end)
                cnt = end

                # Store chunk
                if len(clip_to_idx) % SAVE_EVERY == 0:
                    with open(json_path, "w") as jf:
                        json.dump(clip_to_idx, jf)

                all_data = np.vstack([all_data, data])

                if all_data.shape[0] > 10000:
                    new = all_data.shape[0]
                    fp[old : old + new, :] = all_data
                    fp.flush()
                    old += new
                    all_data = np.empty((0, 7), dtype=np.float32)

    # Flush any remaining rows.
    if all_data.shape[0] > 0:
        new = all_data.shape[0]
        fp[old : old + new, :] = all_data
        fp.flush()

    with open(json_path, "w") as f:
        json.dump(clip_to_idx, f)


def parse_subtrajectory_data_from_dir(
    path_to_trajectory_data, sec, M, nprobe=256, *, index_spec=None
):
    """Build (or load) a sub-trajectory FAISS index for ``sec``-second windows.

    Args:
        path_to_trajectory_data: directory holding ``*/*.safetensors`` shards.
        sec: window length in seconds (5 or 10).
        M: OPQ rotation dim / PQ subquantizer count for the default IVF spec.
            Ignored when ``index_spec`` is supplied explicitly.
        nprobe: nprobe for IVF-family indexes; ignored for non-IVF.
        index_spec: any FAISS ``index_factory`` spec. Defaults to the
            production ``OPQ{M},IVF4096,PQ{M}x8`` if None; pass ``"Flat"``
            for exact search on small corpora.
    """
    if index_spec is None:
        index_spec = f"OPQ{M},IVF4096,PQ{M}x8"

    start = time.time()
    base_dir = Path(path_to_trajectory_data)

    path_to_safetensors = sorted(base_dir.rglob("*/*.safetensors"))
    print(
        f"[trajectory_index] Found {len(path_to_safetensors)} safetensors files"
    )

    trajectory_dim = 605
    S_PER_SEC = max(1, trajectory_dim // 20)
    sub_traj_dim = min(trajectory_dim, sec * S_PER_SEC)
    d = 2 * sub_traj_dim
    tag = spec_to_tag(index_spec)
    print(f"[{sec}s] Using index_spec={index_spec} (tag={tag})")

    path_to_faiss_index = base_dir / f"trajectory_data_{sec}s_{tag}_p1.index"
    path_to_clip_index = base_dir / f"trajectory_clip_to_index_{sec}s_{tag}_p1.pkl"

    # Fast path: load existing index for this spec.
    if path_to_faiss_index.exists() and path_to_clip_index.exists():
        features_index = faiss.read_index(str(path_to_faiss_index))
        with open(path_to_clip_index, "rb") as f:
            clip_to_index = pickle.load(f)
        clip_to_index = {sys.intern(k): v for k, v in clip_to_index.items()}
        print(
            f"[trajectory_index] Loaded feature index ({sec}s) from "
            f"{path_to_faiss_index}"
        )
        print(f"[trajectory_index] ({sec}s) ntotal={features_index.ntotal}")
        return features_index, clip_to_index

    features_index = faiss.index_factory(d, index_spec, faiss.METRIC_L2)
    ivf = _extract_ivf_or_none(features_index)
    if ivf is not None:
        ivf.nprobe = nprobe

    buf = []
    clip_to_index = {}
    offset = 0
    buf_count = 0
    needs_training = not features_index.is_trained

    TRAIN_CHUNK = 1000000
    ADD_CHUNK = 500000

    def finalize_batch(batch, do_train=False):
        arr = np.ascontiguousarray(
            np.concatenate(batch, axis=0), dtype=np.float32
        )
        if do_train:
            print(f"[{sec}s] Training with {arr.shape}...")
            features_index.train(arr)
        features_index.add(arr)

    for path_to_data in tqdm(path_to_safetensors, desc=f"[{sec}s] files"):
        with safe_open(path_to_data, framework="np") as f:
            for clip_id in tqdm(f.keys()):
                if clip_id in clip_to_index:
                    continue

                data = f.get_tensor(clip_id)  # (T, C)

                for s0, s1 in WINDOWS[sec]:
                    i0 = int(s0 * S_PER_SEC)
                    i1 = int(s1 * S_PER_SEC)
                    buf.append(
                        _build_subtrajectory_vec(data, i0, i1, sub_traj_dim)
                    )
                    clip_to_index.setdefault(clip_id, []).append(offset)
                    buf_count += 1
                    offset += 1

                    if needs_training and buf_count >= TRAIN_CHUNK:
                        finalize_batch(buf, do_train=True)
                        needs_training = False
                        buf.clear()
                        buf_count = 0
                    elif (not needs_training) and buf_count >= ADD_CHUNK:
                        finalize_batch(buf, do_train=False)
                        buf.clear()
                        buf_count = 0

                        faiss.write_index(
                            features_index, str(path_to_faiss_index)
                        )
                        with open(path_to_clip_index, "wb") as sf:
                            pickle.dump(clip_to_index, sf)
                        print(f"[{sec}s] Saved intermediate result")

    # Flush remaining: train on the residual if IVF still needs training.
    # For Flat, this is just an add().
    if buf_count > 0:
        finalize_batch(buf, do_train=needs_training)

    elapsed = time.time() - start
    print(f"[trajectory_index] ({sec}s) Finished indexing in {elapsed:.2f}s")
    print(f"[trajectory_index] ({sec}s) ntotal:", features_index.ntotal)
    print(f"[trajectory_index] ({sec}s) type:", type(features_index))
    print(
        f"[trajectory_index] ({sec}s) is_trained:",
        getattr(features_index, "is_trained", True),
    )
    if hasattr(features_index, "nlist"):
        print(
            f"[trajectory_index] ({sec}s) nlist: {features_index.nlist} "
            f"nprobe: {getattr(features_index, 'nprobe', '-')}"
        )
    print(
        f"[trajectory_index] ({sec}s) FAISS OMP threads:",
        faiss.omp_get_max_threads(),
    )

    faiss.write_index(features_index, str(path_to_faiss_index))
    with open(path_to_clip_index, "wb") as f:
        pickle.dump(clip_to_index, f)

    return features_index, clip_to_index


def update_subtrajectory_index(
    path_to_trajectory_data, sec, M, add_chunk=500000
):
    """
    Incrementally add sub-trajectory features to an already-trained FAISS index.
    Only clips not already present in clip_to_index are added. This function does
    not perform any training; it assumes the provided index is already trained.

    Reads p0 files and writes p1 files.  clip_to_index values are lists of
    per-window FAISS offsets (one entry per window in WINDOWS[sec]).
    """
    base_dir = Path(path_to_trajectory_data)
    index_spec = f"OPQ{M},IVF4096,PQ{M}x8"
    tag = spec_to_tag(index_spec)

    path_to_faiss_index = (
        base_dir / f"trajectory_data_{sec}s_{tag}_p0.index"
    )
    path_to_clip_index = (
        base_dir / f"trajectory_clip_to_index_{sec}s_{tag}_p0.pkl"
    )
    path_to_faiss_index_new = (
        base_dir / f"trajectory_data_{sec}s_{tag}_p1.index"
    )
    path_to_clip_index_new = (
        base_dir / f"trajectory_clip_to_index_{sec}s_{tag}_p1.pkl"
    )

    features_index = faiss.read_index(str(path_to_faiss_index))
    with open(path_to_clip_index, "rb") as f:
        clip_to_index = pickle.load(f)

    print(f"[update_subtrajectory_index] Loaded index from {path_to_faiss_index}")
    print(f"Sub-trajectory embeddings of size {features_index.ntotal}...")
    print(f"Clips in index: {len(clip_to_index)}")

    if hasattr(features_index, "is_trained") and not features_index.is_trained:
        raise ValueError(
            "features_index must be trained before calling update_subtrajectory_index"
        )

    trajectory_dim = 605
    S_PER_SEC = max(1, trajectory_dim // 20)
    sub_traj_dim = min(trajectory_dim, sec * S_PER_SEC)

    path_to_safetensors = sorted(base_dir.rglob("*/*.safetensors"))

    # Offset starts from current index size
    offset = int(features_index.ntotal)

    buf = []
    buf_count = 0

    def flush_buffer(batch):
        arr = np.ascontiguousarray(
            np.concatenate(batch, axis=0), dtype=np.float32
        )
        features_index.add(arr)
        print(features_index.ntotal)

    for path_to_data in tqdm(path_to_safetensors):
        with safe_open(path_to_data, framework="np") as f:
            for clip_id in f.keys():
                if clip_id in clip_to_index:
                    continue

                data = f.get_tensor(clip_id)  # (T, C)
                if np.isnan(data).any():
                    continue

                for s0, s1 in WINDOWS[sec]:
                    i0 = int(s0 * S_PER_SEC)
                    i1 = int(s1 * S_PER_SEC)
                    buf.append(
                        _build_subtrajectory_vec(data, i0, i1, sub_traj_dim)
                    )
                    clip_to_index.setdefault(clip_id, []).append(offset)
                    buf_count += 1
                    offset += 1

                if buf_count >= add_chunk:
                    flush_buffer(buf)
                    faiss.write_index(
                        features_index, str(path_to_faiss_index_new)
                    )
                    buf.clear()
                    buf_count = 0

                    with open(path_to_clip_index_new, "wb") as sf:
                        pickle.dump(clip_to_index, sf)
                    print(
                        f"[{sec}s] Saved intermediate result with "
                        f"{features_index.ntotal}"
                    )

    # Flush any remaining buffered vectors
    if buf_count > 0:
        flush_buffer(buf)

    faiss.write_index(features_index, str(path_to_faiss_index_new))
    with open(path_to_clip_index_new, "wb") as sf:
        pickle.dump(clip_to_index, sf)

    print(f"Sub-trajectory embeddings of size {features_index.ntotal}...")
    print(f"Clips in index: {len(clip_to_index)}")

    return features_index, clip_to_index
