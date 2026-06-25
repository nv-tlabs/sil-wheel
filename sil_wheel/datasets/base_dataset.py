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

import fnmatch
import functools
from functools import cached_property
import io
import json
import os
import random
import tempfile

from pathlib import Path
import tarfile
import threading
from tqdm import tqdm
import queue
import zipfile

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from huggingface_hub import HfApi, snapshot_download

# List of cameras in alpamayo V2.1. data
CAMERA_NAMES = [
    # "camera_cross_left_120fov",
    # "camera_cross_right_120fov",
    "camera_front_tele_30fov",
    "camera_front_wide_120fov",
    # "camera_rear_left_70fov",
    # "camera_rear_right_70fov",
    # "camera_rear_tele_30fov"
]


def get_path_to_data(path_to_files):
    if path_to_files.endswith("json"):
        with open(path_to_files, "r") as f:
            path_to_data = json.load(f)
        path_to_data = path_to_data.values()
    elif path_to_files.endswith("txt"):
        with open(path_to_files, "r") as f:
            path_to_data = f.readlines()
        path_to_data = [pi.strip() for pi in path_to_data]
    else:
        raise Exception(f"Unsupported file type for path_to_files: {path_to_files}")

    return [pi for pi in path_to_data if pi.endswith(".mp4") or pi.endswith(".tar")]


def matches_patterns(path, allow_patterns):
    """True if path matches any glob in allow_patterns (or if no
    patterns are given). Shared by the HuggingFace tar/zip datasets."""
    if not allow_patterns:
        return True
    return any(fnmatch.fnmatch(path, p) for p in allow_patterns)


def select_hf_archives(repo_files, allow_patterns, suffix):
    """Sorted repo files ending in suffix that pass allow_patterns."""
    return sorted(
        f for f in repo_files
        if f.endswith(suffix) and matches_patterns(f, allow_patterns)
    )


def download_hf_archive_shards(
    repo_id,
    allow_patterns,
    suffix,
    process_id,
    n_processes,
    cache_dir,
    repo_files=None,
):
    """Select, shard and download this rank's HuggingFace archive shards.

    repo_files is reused when dataset_factory already listed the repo.
    Returns (local_dir, [local archive paths]).
    """
    if not repo_id:
        raise ValueError("HuggingFace dataset requires repo_id")
    if repo_files is None:
        repo_files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    shard = select_hf_archives(repo_files, allow_patterns, suffix)[
        process_id::n_processes
    ]
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=shard,
        cache_dir=cache_dir,
        max_workers=8,
    )
    return local_dir, [Path(local_dir) / f for f in shard]


class FilesDataset:
    """Simple class that goes over a list of files and returns them one by one.
    """
    def __init__(
        self,
        process_id: int = 0,
        n_processes: int = 1,
        path_to_files: str = None,
        clips_to_exclude: list[str] = None,
        camera_filter: str | None = None,
    ):
        if Path(path_to_files).is_file():
            path_to_data = get_path_to_data(path_to_files)
        elif Path(path_to_files).is_dir():
            path_to_data = sorted(Path(path_to_files).glob("**/*.mp4"))
        else:
            raise Exception(f"Invalid value for path_to_files: {path_to_files}")

        # Sort and shard (stable assignment regardless of exclusions)
        self.path_to_data = sorted(set(path_to_data))[process_id::n_processes]
        print(
            f"Process - {process_id} / {n_processes} | shard size: {len(self.path_to_data)}"
        )
        self.clips_to_paths = {}
        for pi in self.path_to_data:
            self.clips_to_paths[Path(pi).stem] = pi

        self.clips_to_exclude = clips_to_exclude
        self.camera_filter = camera_filter
        if camera_filter is not None:
            self.path_to_data = [
                p for p in self.path_to_data if camera_filter in str(p)
            ]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.path_to_data)

    @cached_property
    def get_clip_index_from_path(self):
        if "recordings" in self.path_to_data[0]:
            paths_parts = [Path(p).parts for p in self.path_to_data[:3]]
            cols = list(zip(*paths_parts))

            # Find the index where all paths have 'recordings'
            rec_idx = next(i for i, col in enumerate(cols) if all(x == "recordings"  for x in col))

            clip_id_idx = rec_idx - 1
            return clip_id_idx
        else:
            # We assume that all paths have the same format, if not this won't work
            paths_parts = [Path(p).parts for p in self.path_to_data[:3]]
            cols = list(zip(*paths_parts))

            # Find all indices where the paths differ
            diff_ind = [i for i, col in enumerate(cols) if len(set(col)) > 1]

            # Prioritize the filename (last element) if it contains "camera"
            last_idx = len(cols) - 1
            if last_idx in diff_ind and any("camera" in str(item) for item in cols[last_idx]):
                return last_idx

            # Fallback to directories that differ, excluding those with "camera"
            valid_diffs = [i for i in diff_ind if not any("camera" in str(item) for item in cols[i])]

            assert len(valid_diffs) == 1
            return valid_diffs[0]

    def __getitem__(self, idx: int):
        path_to_video = str(self.path_to_data[idx])

        if not "camera" in path_to_video:
            camera = None
            clip_id = path_to_video.split("/")[-1].split(".")[0]
        else:
            if "camera" in path_to_video.split("/")[-1]:
                clip_id_index = self.get_clip_index_from_path
                clip_id = Path(path_to_video).parts[clip_id_index]
                if "camera" in clip_id:
                    clip_id = clip_id.split(".")[0]
            else:
                clip_id = path_to_video.split("/")[-4]
            camera = path_to_video.split("/")[-1].split(".")[0]

        # In case this clip is to be excluded do not process the video
        if clip_id in self.clips_to_exclude:
            return None, clip_id, camera

        with open(path_to_video, "rb") as f:
            video_data = f.read()
        video_buffer = io.BytesIO(video_data)
        return video_buffer, clip_id, camera


class TarsDataset:
    def __init__(
        self,
        process_id: int,
        n_processes: int,
        path_to_files: str,
        clips_to_exclude: list[str] = None,
        camera_filter: str | None = None,
        cameras: list | None = None,
    ):
        with open(path_to_files, "r") as f:
            clip_to_tar = json.load(f)
        print(f"Loaded {len(clip_to_tar)} from {path_to_files}")

        tars = sorted(set([ti for ti in clip_to_tar.values()]))
        tars_to_clips = {key:[] for key in tars}
        for ci, ti in tqdm(clip_to_tar.items()):
            tars_to_clips[ti].append(ci)
        all_tars = sorted(list(tars_to_clips.keys()))

        self.tars_to_clips_chunk = {
            k: tars_to_clips[k]
            for k in all_tars[process_id::n_processes]
        }

        # Datasets with no camera dimension (e.g. PE-Video) pass
        # ``cameras=[None]`` to get exactly one row per clip; the
        # alpamayo default multiplexes each clip across every camera in
        # ``CAMERA_NAMES``.
        if cameras is not None:
            pass
        elif camera_filter:
            cameras = [camera_filter]
        else:
            cameras = CAMERA_NAMES
        self.clips = []
        for tar, clips_per_tar in self.tars_to_clips_chunk.items():
            for clip_id in clips_per_tar:
                for camera in cameras:
                    self.clips.append((tar, clip_id, camera))
        self.clips = sorted(self.clips)
        self.clips_to_exclude = set(clips_to_exclude) if clips_to_exclude else set()

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.clips)

    @functools.lru_cache(maxsize=64)
    def get_tarfile(self, tar):
        return tarfile.open(tar, "r")

    def __getitem__(self, idx: int) -> str:
        tar, clip_id, camera = self.clips[idx]
        if clip_id.split(".")[0] in self.clips_to_exclude:
            return None, clip_id.split(".")[0], camera
        tf = self.get_tarfile(tar)
        if camera is not None and f"{clip_id}.{camera}.mp4" in tf.getnames():
            video_file = tf.extractfile(f"{clip_id}.{camera}.mp4")
        else:
            video_file = tf.extractfile(f"{clip_id}")

        video_data = video_file.read()
        video_buffer = io.BytesIO(video_data)
        return video_buffer, clip_id.split(".")[0], camera


class HuggingFaceTarDataset:
    """Iterate a HuggingFace dataset that ships as tar shards.

    On construction this class first asks the HuggingFace Hub for the
    repo's file list, filters it by ``allow_patterns``, and slices the
    result by ``(process_id, n_processes)`` so each rank only fetches
    its own share of the shards. With a shared ``HF_HOME`` cache, the
    cluster collectively downloads each shard exactly once. The
    selected shards are pulled via ``snapshot_download`` (resumable,
    parallel, checksum-verified), then scanned to build a per-rank
    JSON manifest mapping each tar member name (e.g.
    ``video123.mp4``) to the path of the archive that contains it.
    Iteration is delegated to ``TarsDataset``.
    """
    def __init__(
        self,
        process_id: int = 0,
        n_processes: int = 1,
        repo_id: str = None,
        allow_patterns: list[str] = None,
        cache_dir: str | None = None,
        clips_to_exclude: list[str] = None,
        camera_filter: str | None = None,
        repo_files: list[str] | None = None,
    ):
        local_dir, local_tar_paths = download_hf_archive_shards(
            repo_id, allow_patterns, ".tar",
            process_id, n_processes, cache_dir, repo_files,
        )
        manifest_path = (
            Path(local_dir)
            / f"hf_manifest_p{process_id}_n{n_processes}.json"
        )
        # The manifest's values are the tar paths it was built from;
        # if they don't match this run's downloaded shards the cache is from
        # a different --hf-allow-patterns invocation and must be
        # regenerated. Without this check the cached manifest from a
        # narrower previous slice is reused silently.
        expected_tars = {str(p) for p in local_tar_paths}
        needs_rebuild = True
        if manifest_path.exists():
            cached_tars = set(
                json.loads(manifest_path.read_text()).values()
            )
            needs_rebuild = cached_tars != expected_tars
        if needs_rebuild:
            self._build_manifest(local_tar_paths, manifest_path)

        # The shard sharding already happened above, so TarsDataset's
        # own sharding is a no-op (process_id=0, n_processes=1).
        self._inner = TarsDataset(
            process_id=0,
            n_processes=1,
            path_to_files=str(manifest_path),
            clips_to_exclude=clips_to_exclude,
            camera_filter=camera_filter,
            cameras=[None],
        )

    @staticmethod
    def _build_manifest(tar_paths, out_path):
        """Scan the given tar archives and write a JSON manifest that
        maps each tar member name (the file path inside the archive,
        e.g. ``video123.mp4``) to the path of the archive that
        contains it. The ``.mp4`` suffix is kept on the manifest key
        because ``TarsDataset.__getitem__`` uses it to locate the
        member; the suffix is then stripped at display time, so
        callers receive the clip_id without the extension.
        """
        manifest = {}
        for tar_path in tqdm(
            sorted(tar_paths), desc="Scanning HF tar shards"
        ):
            with tarfile.open(tar_path, "r") as tf:
                for member_name in tf.getnames():
                    if not member_name.endswith(".mp4"):
                        continue
                    manifest[member_name] = str(tar_path)
        with open(out_path, "w") as f:
            json.dump(manifest, f)

    def __iter__(self):
        return iter(self._inner)

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        return self._inner[idx]


class HuggingFaceZipDataset:
    """Iterate a HuggingFace dataset that ships its videos in .zip shards.

    Datasets such as nvidia/PhysicalAI-Autonomous-Vehicles store one .zip per
    (sensor, chunk) instead of WebDataset-style .tar shards. The zip analogue of
    HuggingFaceTarDataset.

    Member names follow <clip_id>.<camera>.mp4 where clip_id is a UUID (no dots)
    and camera uses underscores, so both are recovered by splitting the filename
    on ".". A dataset whose members are bare <clip_id>.mp4 (no camera token)
    yields camera=None.
    """

    def __init__(
        self,
        process_id: int = 0,
        n_processes: int = 1,
        repo_id: str = None,
        allow_patterns: list[str] = None,
        cache_dir: str | None = None,
        clips_to_exclude: list[str] = None,
        camera_filter: str | None = None,
        repo_files: list[str] | None = None,
    ):
        local_dir, self._zip_paths = download_hf_archive_shards(
            repo_id, allow_patterns, ".zip",
            process_id, n_processes, cache_dir, repo_files,
        )
        self._clips_to_exclude = set(clips_to_exclude) if clips_to_exclude else set()
        self._camera_filter = camera_filter

        # Build the (zip_path, member_name, clip_id, camera) work list up
        # front so __len__ is exact and iteration is a simple scan. camera_filter
        # is applied here at the member level (like the tar reader), so a clip's
        # other-camera videos in the same zip are skipped.
        self._members = []
        for zip_path in tqdm(self._zip_paths, desc="Scanning HF zip shards"):
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    names = zf.namelist()
            except (zipfile.BadZipFile, FileNotFoundError) as e:
                print(f"Skipping unreadable zip {zip_path}: {e}")
                continue
            for member_name in names:
                if not member_name.endswith(".mp4"):
                    continue
                clip_id, camera = self._parse_member(member_name)
                if camera_filter and camera != camera_filter:
                    continue
                self._members.append((zip_path, member_name, clip_id, camera))
        self._members.sort(key=lambda m: (m[2], m[3] or ""))

    @staticmethod
    def _parse_member(member_name: str) -> tuple[str, str | None]:
        stem = Path(member_name).name[: -len(".mp4")]
        parts = stem.split(".")
        clip_id = parts[0]
        camera = parts[1] if len(parts) > 1 else None
        return clip_id, camera

    @functools.lru_cache(maxsize=16)
    def _open_zip(self, zip_path):
        return zipfile.ZipFile(zip_path, "r")

    def __len__(self):
        return len(self._members)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int):
        zip_path, member_name, clip_id, camera = self._members[idx]
        if clip_id in self._clips_to_exclude:
            return None, clip_id, camera
        with self._open_zip(zip_path).open(member_name) as f:
            video_buffer = io.BytesIO(f.read())
        return video_buffer, clip_id, camera


class S3ObjectFetcher:
    def __init__(self, bucket, profile, endpoint="https://s3.example.com"):
        sess = boto3.Session(
            profile_name=profile,
            region_name="us-east-1"
        )
        self.client = sess.client(
            "s3",
            config=Config(
                max_pool_connections=50,
                read_timeout=30,
                connect_timeout=5,
            ) if Config else None,
            endpoint_url=endpoint,
        )
        self.bucket = bucket

    def download_to(self, key: str, local_path: Path) -> bool:
        key = str(key).lstrip("/")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client.download_file(self.bucket, key, str(local_path))
            return True
        except ClientError as e:
            code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            print(f"S3 error fetching {key}: {code}")
            return False

    def stream_downloads(
        self,
        keys,
        tmpdir_path: Path,
        num_workers: int = 8,
        max_queue: int = 1000,
        also_download=None,
    ):
        """Producer/consumer pipeline for downloading raw videos to temporary files"""
        keys_list = list(keys)
        random.shuffle(keys_list)

        in_q = queue.Queue(maxsize=max_queue)
        out_q = queue.Queue(maxsize=max_queue)
        stop_token = object()

        # Producer thread prepares the list of keys to be processed
        def producer():
            for k in keys_list:
                in_q.put(k)
            for _ in range(num_workers):
                in_q.put(stop_token)

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        # Consumers download keys to local temp paths and push results to the queue
        def consumer():
            while True:
                k = in_q.get()
                try:
                    if k is stop_token:
                        break
                    local_path = (Path(tmpdir_path) / k).resolve()
                    ok = self.download_to(k, local_path)
                    if ok:
                        if also_download is not None:
                            for extra_key, extra_local_path in also_download(k, local_path):
                                self.download_to(extra_key, extra_local_path)
                        out_q.put(str(local_path))
                finally:
                    in_q.task_done()
            out_q.put(None)

        threads = [threading.Thread(target=consumer, daemon=True) for _ in range(num_workers)]
        for t in threads:
            t.start()

        finished = 0
        while finished < num_workers:
            item = out_q.get()
            if item is None:
                finished += 1
                continue
            yield item


class S3Mp4Dataset:
    """Dataset for S3-stored .mp4 files. One S3 key yields one video."""
    def __init__(
        self,
        process_id: int = 0,
        n_processes: int = 1,
        path_to_files: str | None = None,
        clips_to_exclude: list[str] | None = None,
        s3_bucket: str | None = None,
        s3_profile: str | None = None,
        s3_endpoint: str | None = None,
        camera_filter: str | None = None,
    ):
        raw_paths = get_path_to_data(path_to_files)
        keys_all = [p.lstrip("/") for p in raw_paths if p.endswith(".mp4")]

        self._bucket = s3_bucket
        self._exclude = set(clips_to_exclude) if clips_to_exclude else set()
        self._camera_filter = camera_filter

        # Figure out which path components form the clip_id by finding
        # directory indices that vary across keys (excluding camera parts).
        self._clip_id_indices = self._find_clip_id_indices(keys_all)

        sharded = sorted(set(keys_all))[process_id::n_processes]
        self._keys = []
        for k in sharded:
            clip_id, camera = self._infer_ids(k)
            if clip_id in self._exclude:
                continue
            if self._camera_filter and camera != self._camera_filter:
                continue
            self._keys.append(k)

        print(
            f"Process - {process_id} / {n_processes} | "
            f"mp4 shard size: {len(sharded)} | remaining after exclusions: {len(self._keys)} (S3)"
        )

        self.clips_to_paths = {Path(k).stem: k for k in self._keys}
        self._fetcher = S3ObjectFetcher(
            self._bucket,
            s3_profile,
            s3_endpoint or "https://s3.example.com",
        )
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="df_s3_mp4_")
        self._tmpdir = Path(self._tmpdir_obj.name)

    def __len__(self):
        return len(self._keys)

    def __iter__(self):
        worker_tmpdir = self._tmpdir / str(os.getpid())
        worker_tmpdir.mkdir(exist_ok=True)
        worker_tmpdir = worker_tmpdir.resolve()

        for local_path in self._fetcher.stream_downloads(
            self._keys,
            worker_tmpdir,
            num_workers=8,
            max_queue=1000,
        ):
            try:
                key = str(Path(local_path).resolve().relative_to(worker_tmpdir))
                clip_id, camera = self._infer_ids(key)
                with open(local_path, "rb") as f:
                    yield io.BytesIO(f.read()), clip_id, camera
            finally:
                Path(local_path).unlink(missing_ok=True)

    @staticmethod
    def _find_clip_id_indices(keys):
        sample = random.sample(keys, min(10, len(keys)))
        parts_list = [Path(k).parts for k in sample]
        cols = list(zip(*parts_list))
        diff_indices = [i for i, col in enumerate(cols) if len(set(col)) > 1]
        # Exclude indices whose values contain "camera" (those are camera parts)
        return [
            i for i in diff_indices
            if not any("camera" in str(v) for v in cols[i])
        ]

    def _infer_ids(self, key: str) -> tuple[str, str | None]:
        p = Path(key)
        stem = p.stem
        if "camera" in stem:
            parts = stem.split(".")
            camera = next(
                (x for x in parts if x.startswith("camera")), None
            )
            # Filename like "clip_id.camera_name.mp4"
            if len(parts) > 1:
                clip_id = parts[0]
                return clip_id, camera
            # Filename IS the camera; build clip_id from the directory
            # components that vary across the dataset.
            all_parts = p.parts
            clip_id = "_".join(all_parts[i] for i in self._clip_id_indices)
            return clip_id, camera
        return stem, None


class S3TarDataset:
    """Dataset for S3-stored .tar files containing multiple .mp4 members."""
    def __init__(
        self,
        process_id: int = 0,
        n_processes: int = 1,
        path_to_files: str | None = None,
        clips_to_exclude: list[str] | None = None,
        s3_bucket: str | None = None,
        s3_profile: str | None = None,
        s3_endpoint: str | None = None,
    ):
        raw_paths = get_path_to_data(path_to_files)
        tar_keys_all = [p.lstrip("/") for p in raw_paths if p.endswith(".tar")]

        self._bucket = s3_bucket
        self._exclude = set(clips_to_exclude) if clips_to_exclude else set()

        self._tar_keys = sorted(set(tar_keys_all))[process_id::n_processes]

        print(
            f"Process - {process_id} / {n_processes} | "
            f"tar shard size: {len(self._tar_keys)} (S3)"
        )

        self._fetcher = S3ObjectFetcher(
            self._bucket,
            s3_profile,
            s3_endpoint or "https://s3.example.com",
        )
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="df_s3_tar_")
        self._tmpdir = Path(self._tmpdir_obj.name)

    def __len__(self):
        # This is tar-level length, not clip-level length.
        return len(self._tar_keys)

    def __iter__(self):
        worker_tmpdir = self._tmpdir / str(os.getpid())
        worker_tmpdir.mkdir(exist_ok=True)

        num_tars = len(self._tar_keys)

        for tar_idx, local_tar_path in enumerate(
            self._fetcher.stream_downloads(
                self._tar_keys,
                worker_tmpdir,
                num_workers=8,
                max_queue=1000,
            )
        ):
            tar_name = Path(local_tar_path).name
            yielded_from_tar = 0

            print(f"[tar {tar_idx}/{num_tars}] Opening {tar_name}", flush=True)

            try:
                with tarfile.open(local_tar_path, "r") as tf:
                    for member in tf:
                        if not member.isfile() or not member.name.endswith(".mp4"):
                            continue

                        clip_id = Path(member.name).stem.split(".")[0]
                        if clip_id in self._exclude:
                            continue

                        video_file = tf.extractfile(member)
                        if video_file is None:
                            continue

                        yielded_from_tar += 1
                        yield io.BytesIO(video_file.read()), clip_id, None

            finally:
                print(
                    f"[tar {tar_idx}/{num_tars}] Finished {tar_name} | "
                    f"yielded {yielded_from_tar} clips",
		    flush=True
                )
                Path(local_tar_path).unlink(missing_ok=True)

def dataset_factory(
    process_id: int = 0,
    n_processes: int = 1,
    path_to_files: str = None,
    clips_to_exclude: list[str] = None,
    s3_bucket: str | None = None,
    s3_profile: str | None = None,
    s3_endpoint: str | None = None,
    hf_repo_id: str | None = None,
    hf_allow_patterns: list[str] | None = None,
    hf_cache_dir: str | None = None,
    camera_filter: str | None = None,
):
    if hf_repo_id is not None:
        # A dataset ships a single archive format, so pick the reader from the
        # first matching .zip/.tar shard (mirrors the local first_path check
        # below). repo_files is reused by the reader to avoid a second listing.
        repo_files = HfApi().list_repo_files(hf_repo_id, repo_type="dataset")
        first_archive = next(
            (f for f in repo_files
             if f.endswith((".zip", ".tar")) and matches_patterns(f, hf_allow_patterns)),
            "",
        )
        hf_cls = HuggingFaceZipDataset if first_archive.endswith(".zip") else HuggingFaceTarDataset
        return hf_cls(
            process_id=process_id,
            n_processes=n_processes,
            repo_id=hf_repo_id,
            allow_patterns=hf_allow_patterns,
            cache_dir=hf_cache_dir,
            clips_to_exclude=clips_to_exclude,
            camera_filter=camera_filter,
            repo_files=repo_files,
        )

    with open(path_to_files, "r") as f:
        first_path = f.readline().strip()

    if s3_bucket is not None:
        if first_path.endswith(".mp4"):
            return S3Mp4Dataset(
                process_id=process_id,
                n_processes=n_processes,
                path_to_files=path_to_files,
                clips_to_exclude=clips_to_exclude,
                s3_bucket=s3_bucket,
                s3_profile=s3_profile,
                s3_endpoint=s3_endpoint,
                camera_filter=camera_filter,
            )
        elif first_path.endswith(".tar"):
            return S3TarDataset(
                process_id=process_id,
                n_processes=n_processes,
                path_to_files=path_to_files,
                clips_to_exclude=clips_to_exclude,
                s3_bucket=s3_bucket,
                s3_profile=s3_profile,
                s3_endpoint=s3_endpoint,
            )
        raise Exception(
            f"Unsupported S3 data type {first_path} from path_to_files: {path_to_files}"
        )

    if first_path.endswith(".mp4"):
        return FilesDataset(
            process_id,
            n_processes,
            path_to_files,
            clips_to_exclude,
            camera_filter=camera_filter,
        )
    elif first_path.endswith(".tar"):
        return TarsDataset(
            process_id,
            n_processes,
            path_to_files,
            clips_to_exclude,
            camera_filter=camera_filter,
        )
    raise Exception(
        f"Unsupported data type {first_path} from path_to_files: {path_to_files}"
    )
