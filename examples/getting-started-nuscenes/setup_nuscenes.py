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

"""Fetch a nuScenes split and prepare it for the wheel server.

Stages, each toggleable via --skip-* flags:
  1. Download v1.0-mini (auto, ~3.88 GiB no auth) or use a pre-extracted
     v1.0-trainval / v1.0-test split placed under {workdir}/nuscenes/.
  2. Encode one CAM_FRONT MP4 per scene with ffmpeg.
  3. Run scripts/prepare_data.py.
  4. Extract Cosmos / Qwen captions / caption embeddings / Florence-2 +
     SigCLIP2 visual embeddings via scripts/extract_*.py.
  5. Build per-scene ego trajectories from nuscenes-devkit ego_pose data.
  6. Initialise SQLite stores and write config.yaml.

Launch the wheel server with:

    python scripts/launch_server.py wheel-data/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from getpass import getpass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from nuscenes.nuscenes import NuScenes
from safetensors.numpy import safe_open, save_file
from scipy.signal import savgol_filter
from tqdm import tqdm

from sil_wheel.stores.caption_embeddings_store import CaptionEmbeddingsStore
from sil_wheel.stores.cosmos_embeddings_store import CosmosEmbeddingsStore
from sil_wheel.stores.sqlite_caption_store import FTSCaptionStore
from sil_wheel.stores.trajectory_store import (
    parse_subtrajectory_data_from_dir,
    parse_trajectory_data_from_dir,
)
from sil_wheel.stores.users_data_store import UsersDataStore
from sil_wheel.stores.visual_embeddings_store import Florence2SigCLIPEmbeddingStore


log = logging.getLogger("setup_nuscenes")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


EXAMPLE_DIR = Path(__file__).resolve().parent
WHEEL_SCRIPTS = EXAMPLE_DIR.parent.parent / "scripts"

NUSCENES_MINI_URL = "https://www.nuscenes.org/data/v1.0-mini.tgz"
NUSCENES_MINI_BYTES = 4_167_696_325

DATA_SOURCE_NAME = "nuScenes"
CAMERA = "CAM_FRONT"

# SigCLIP2 variant for the visual embeddings. One value drives both extraction
# (the index) and the server's query-time encoder (written into config.yaml),
# so text and image queries share the same embedding space.
SIGLIP_MODEL = "google/siglip2-base-patch16-224"

# nuScenes is recorded in two cities; ISO 3166-1 alpha-2 codes feed wheel's
# country filter.
NUSCENES_COUNTRY = {
    "boston-seaport": "US",
    "singapore-onenorth": "SG",
    "singapore-hollandvillage": "SG",
    "singapore-queenstown": "SG",
}


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        log.error("Required binary %r not found on PATH. Install it and retry.", name)
        sys.exit(1)


def require_gpu() -> None:
    if not torch.cuda.is_available():
        log.error(
            "No CUDA GPU visible to PyTorch. The Cosmos / captioning / "
            "caption-embedding / visual-embedding stages all need a GPU; "
            "use --skip-* flags to bypass them or run on a GPU host."
        )
        sys.exit(1)
    log.info("CUDA visible: %s", torch.cuda.get_device_name(0))


def download_with_progress(url: str, dst: Path, expected_size: int | None = None) -> None:
    if dst.exists() and expected_size and dst.stat().st_size == expected_size:
        log.info("Already downloaded: %s (%d bytes)", dst, dst.stat().st_size)
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    resume_from = tmp.stat().st_size if tmp.exists() else 0

    req = urllib.request.Request(url)
    if resume_from > 0:
        req.add_header("Range", f"bytes={resume_from}-")

    log.info(
        "Downloading %s → %s%s",
        url, dst, f" (resuming at {resume_from})" if resume_from else "",
    )
    with urllib.request.urlopen(req) as resp:
        total = expected_size or (
            int(resp.headers.get("Content-Length", 0)) + resume_from
        )
        with open(tmp, "ab" if resume_from else "wb") as f:
            pbar = tqdm(total=total, initial=resume_from, unit="B", unit_scale=True)
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
            pbar.close()

    tmp.rename(dst)


def ensure_nuscenes_split(workdir: Path, version: str) -> Path:
    """Auto-download v1.0-mini; require pre-downloaded data for trainval/test.

    Trainval (~300 GiB) and test (~60 GiB) downloads need an account at
    nuscenes.org and acceptance of the Terms of Use. Place the extracted
    archives under {workdir}/nuscenes/ before running with
    --version v1.0-trainval or --version v1.0-test.
    """
    nuscenes_root = workdir / "nuscenes"
    metadata_dir = nuscenes_root / version
    if metadata_dir.exists() and (nuscenes_root / "samples").exists():
        log.info("nuScenes %s already extracted at %s", version, nuscenes_root)
        return nuscenes_root

    if version != "v1.0-mini":
        log.error(
            "%s data not found at %s. Register at "
            "https://www.nuscenes.org/nuscenes#download, accept the Terms of "
            "Use, download the metadata + blob tgz files, extract them under "
            "%s/, then re-run with --skip-download.",
            version, metadata_dir, nuscenes_root,
        )
        sys.exit(1)

    nuscenes_root.mkdir(parents=True, exist_ok=True)
    archive = workdir / "v1.0-mini.tgz"
    download_with_progress(NUSCENES_MINI_URL, archive, NUSCENES_MINI_BYTES)
    log.info("Extracting %s → %s (~4 GiB)", archive, nuscenes_root)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(nuscenes_root)

    if not metadata_dir.exists():
        log.error("Extraction failed: %s does not exist", metadata_dir)
        sys.exit(1)

    archive.unlink(missing_ok=True)
    return nuscenes_root


def enumerate_scene_frames(nusc, scene: dict) -> list[tuple[Path, int]]:
    """Walk the CAM_FRONT sample_data 'next' chain; return (path, ts_us) per frame."""
    first_sample = nusc.get("sample", scene["first_sample_token"])
    sd_token = first_sample["data"][CAMERA]
    frames = []
    while sd_token:
        sd = nusc.get("sample_data", sd_token)
        frames.append((Path(nusc.dataroot) / sd["filename"], sd["timestamp"]))
        sd_token = sd["next"]
    return frames


def encode_one(frames, output_path: Path, fps: int = 12) -> bool:
    """Encode `frames` at native 12 Hz into output_path (libx264 yuv420p)."""
    if output_path.exists():
        return True
    if not frames:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = output_path.with_suffix(".concat.txt")
    tmp_out = output_path.with_suffix(".tmp.mp4")
    duration = 1.0 / fps
    with list_path.open("w") as f:
        for path, _ in frames:
            f.write(f"file '{path.as_posix()}'\n")
            f.write(f"duration {duration:.6f}\n")
        # ffmpeg's concat demuxer requires the last file to be repeated.
        f.write(f"file '{frames[-1][0].as_posix()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-r", str(fps),
        str(tmp_out),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tmp_out.rename(output_path)
        return True
    except subprocess.CalledProcessError as e:
        log.warning("ffmpeg failed for %s: %s", output_path.name, e)
        tmp_out.unlink(missing_ok=True)
        return False
    finally:
        list_path.unlink(missing_ok=True)


def manifest_for_scene(nusc, scene: dict, raw_video_path: Path, n_frames: int) -> dict:
    log_record = nusc.get("log", scene["log_token"])
    location = log_record.get("location", "")
    return {
        "clip_id": scene["name"],
        "scene_token": scene["token"],
        "scene_description": scene["description"],
        "camera": CAMERA,
        "location": location,
        "country": NUSCENES_COUNTRY.get(location, ""),
        "raw_video_path": str(raw_video_path.resolve()),
        "n_frames": n_frames,
    }


def encode_raw_videos(nusc, raw_dir: Path, n_workers: int) -> list[dict]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for scene in nusc.scene:
        frames = enumerate_scene_frames(nusc, scene)
        if not frames:
            log.warning("Scene %s has no CAM_FRONT frames, skipping", scene["name"])
            continue
        out = raw_dir / f"{scene['name']}.mp4"
        jobs.append((scene, frames, out))

    log.info("Encoding %d front-camera videos with %d workers...", len(jobs), n_workers)
    manifest = []
    failed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {
            ex.submit(encode_one, frames, out): (scene, frames, out)
            for scene, frames, out in jobs
        }
        for fut in as_completed(futs):
            scene, frames, out = futs[fut]
            if fut.result():
                manifest.append(manifest_for_scene(nusc, scene, out, len(frames)))
            else:
                failed += 1

    manifest.sort(key=lambda r: r["clip_id"])
    log.info("Encoded %d videos (failed: %d)", len(manifest), failed)
    if not manifest:
        log.error("No videos produced; aborting.")
        sys.exit(1)
    return manifest


def reconstruct_manifest_from_disk(nusc, raw_dir: Path) -> list[dict]:
    """Rebuild the manifest from on-disk MP4s when --skip-encode is set."""
    by_name = {scene["name"]: scene for scene in nusc.scene}
    manifest = []
    for mp4 in sorted(raw_dir.glob("*.mp4")):
        scene = by_name.get(mp4.stem)
        if scene is None:
            log.warning("Unknown scene mp4 %s, skipping", mp4.name)
            continue
        manifest.append(manifest_for_scene(nusc, scene, mp4, n_frames=-1))
    return manifest


def write_video_list(paths: list[Path], dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        for p in paths:
            f.write(str(p) + "\n")
    return dst


def run_subprocess(label: str, argv: list[str]) -> None:
    log.info("[%s] %s", label, " ".join(argv))
    t0 = time.time()
    try:
        subprocess.run(argv, check=True)
    except subprocess.CalledProcessError as e:
        log.error("[%s] failed with exit code %d", label, e.returncode)
        sys.exit(e.returncode)
    log.info("[%s] done in %.1fs", label, time.time() - t0)


def run_prepare_data(raw_paths: list[Path], processed_dir: Path, video_list_path: Path) -> None:
    write_video_list(raw_paths, video_list_path)
    processed_dir.mkdir(parents=True, exist_ok=True)
    run_subprocess(
        "prepare_data",
        [
            sys.executable,
            str(WHEEL_SCRIPTS / "prepare_data.py"),
            str(video_list_path),
            str(processed_dir),
        ],
    )


def run_extract_cosmos(
    processed_paths: list[Path],
    cosmos_dir: Path,
    video_list_path: Path,
    model_type: str = "cosmos_embed1_448p",
) -> list[Path]:
    write_video_list(processed_paths, video_list_path)
    cosmos_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(cosmos_dir / "{model_type}_group_{process_id}_{n_processes}.parquet")
    run_subprocess(
        "extract_cosmos",
        [
            sys.executable,
            str(WHEEL_SCRIPTS / "extract_video_text_embeddings.py"),
            str(video_list_path),
            "--model_type", model_type,
            "--process_id", "0",
            "--n_processes", "1",
            "--output", output_template,
        ],
    )
    parquets = sorted(cosmos_dir.glob(f"{model_type}_group_*.parquet"))
    if not parquets:
        log.error("No parquet shards produced by extract_video_text_embeddings.py")
        sys.exit(1)
    return parquets


def materialize_cosmos_index(cosmos_dir: Path, index_spec: str) -> None:
    log.info("Building Cosmos FAISS index under %s (index_spec=%s)", cosmos_dir, index_spec)
    store = CosmosEmbeddingsStore(str(cosmos_dir), index_spec=index_spec)
    log.info(
        "Cosmos index ready: ntotal=%d, %d clips registered",
        store.features_index.ntotal if store.features_index is not None else 0,
        len(store.clips_to_index or {}),
    )


def run_extract_qwen_captions(
    processed_paths: list[Path],
    captions_dir: Path,
    video_list_path: Path,
    clip_duration: float = 20.0,
    gpu_memory_utilization: float = 0.7,
    enforce_eager: bool = True,
    max_model_len: int = 32768,
) -> Path:
    write_video_list(processed_paths, video_list_path)
    captions_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(captions_dir / "qwen_captions_group_{process_id}_{n_processes}.parquet")
    cmd = [
        sys.executable,
        str(WHEEL_SCRIPTS / "extract_captions.py"),
        str(video_list_path),
        "--model_family", "qwen3-vl",
        "--model_size", "4",
        "--prompt_factory_type", "yotta_prompt_long",
        "--clip_duration", str(clip_duration),
        "--clip_overlap", "0",
        "--min_duration", "5",
        "--batch_size", "1",
        "--num_workers", "1",
        "--process_id", "0",
        "--n_processes", "1",
        "--output", output_template,
        "--gpu_memory_utilization", str(gpu_memory_utilization),
        "--max_model_len", str(max_model_len),
    ]
    if enforce_eager:
        cmd.append("--enforce_eager")
    run_subprocess("extract_qwen_captions", cmd)
    parquet_path = captions_dir / "qwen_captions_group_0_1.parquet"
    if not parquet_path.exists():
        log.error("Qwen captioning produced no output at %s", parquet_path)
        sys.exit(1)
    return parquet_path


def load_captions_into_db(
    captions_db: Path,
    captions_parquet: Path,
    scene_duration_s: float = 20.0,
) -> None:
    """Insert qwen-captions parquet rows into FTSCaptionStore.

    insert_from_dataframe expects [clip_id, summary, start_time, end_time].
    The captioning script writes one sub-clip per scene when clip_duration
    matches the scene length with no overlap.
    """
    df = pd.read_parquet(captions_parquet, columns=["clip_id", "summary"])
    df = df.dropna(subset=["summary"]).copy()
    df["start_time"] = 0.0
    df["end_time"] = float(scene_duration_s)

    captions_db.parent.mkdir(parents=True, exist_ok=True)
    store = FTSCaptionStore(str(captions_db))
    model_name = "Qwen3-VL-4B"
    store.insert_from_dataframe(df, model_name, DATA_SOURCE_NAME)
    log.info("Loaded %d captions into %s (model=%s)", len(df), captions_db, model_name)


def run_extract_caption_embeddings(
    captions_parquet: Path,
    caption_embed_dir: Path,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
) -> list[Path]:
    caption_embed_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(caption_embed_dir / "group_{process_id}_{n_processes}.parquet")
    run_subprocess(
        "extract_caption_embeddings",
        [
            sys.executable,
            str(WHEEL_SCRIPTS / "extract_captions_embeddings.py"),
            str(captions_parquet),
            "--embedding_model", embedding_model,
            "--process_id", "0",
            "--n_processes", "1",
            "--batch_size", "16",
            "--output", output_template,
        ],
    )
    parquets = sorted(caption_embed_dir.glob("group_*.parquet"))
    if not parquets:
        log.error("Caption embedding extraction produced no output")
        sys.exit(1)
    return parquets


def materialize_caption_embeddings_index(
    caption_embed_dir: Path,
    embedding_model: str,
    index_spec: str = "Flat",
) -> None:
    log.info("Building caption-embedding FAISS index (index_spec=%s)", index_spec)
    store = CaptionEmbeddingsStore(
        str(caption_embed_dir),
        index_spec=index_spec,
        embedding_model=embedding_model,
    )
    log.info(
        "Caption-embedding index ready: ntotal=%d",
        store.features_index.ntotal if store.features_index is not None else 0,
    )


def run_extract_visual_embeddings(
    processed_paths: list[Path],
    visual_dir: Path,
    video_list_path: Path,
) -> list[Path]:
    write_video_list(processed_paths, video_list_path)
    visual_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(
        visual_dir / "florence2_sigclip_group_{process_id}_{n_processes}.pkl"
    )
    run_subprocess(
        "extract_visual_embeddings",
        [
            sys.executable,
            str(WHEEL_SCRIPTS / "extract_florence2_sigclip_embeddings.py"),
            str(video_list_path),
            "--siglip_model", SIGLIP_MODEL,
            "--process_id", "0",
            "--n_processes", "1",
            "--n_frames", "8",
            "--batch_size", "16",
            "--save_every", "5",
            "--output", output_template,
        ],
    )
    pkls = sorted(visual_dir.glob("florence2_sigclip_group_*.pkl"))
    if not pkls:
        log.error("Visual embedding extraction produced no output")
        sys.exit(1)
    return pkls


def materialize_visual_embeddings_index(visual_dir: Path, index_spec: str = "Flat") -> None:
    log.info("Building visual-embedding FAISS index (index_spec=%s)", index_spec)
    store = Florence2SigCLIPEmbeddingStore(
        str(visual_dir), index_spec=index_spec, siglip_model=SIGLIP_MODEL
    )
    log.info(
        "Visual-embedding index ready: ntotal=%d",
        store.features_index.ntotal if store.features_index is not None else 0,
    )


def compute_trajectory_stats(x, y, z, ts) -> np.ndarray:
    """Return shape (T, 7) [x, y, z, speed, accel, jerk, curvature] in float32.

    nuScenes ego_pose timestamps already line up with camera frame timestamps
    (each sample_data carries its own ego_pose_token), so no resampling is
    needed where sensor-rate egomotion has to be interpolated to camera-rate frames.
    """
    vx = np.gradient(x, ts)
    vy = np.gradient(y, ts)
    vz = np.gradient(z, ts)
    speed = np.sqrt(vx**2 + vy**2 + vz**2)

    # savgol window must be odd and ≤ len(speed); skip smoothing on tiny clips.
    window = min(11, len(speed) if len(speed) % 2 == 1 else len(speed) - 1)
    if window >= 5:
        speed = savgol_filter(speed, window_length=window, polyorder=3)
    accel = np.gradient(speed, ts)
    if window >= 5:
        accel = savgol_filter(accel, window_length=window, polyorder=3)
    jerk = np.gradient(accel, ts)

    ax = np.gradient(vx, ts)
    ay = np.gradient(vy, ts)
    denom = np.maximum((vx**2 + vy**2) ** 1.5, 1e-6)
    curvature = np.where(
        np.sqrt(vx**2 + vy**2) < 2 / 3.6,
        0,
        np.abs(vx * ay - vy * ax) / denom,
    )

    return np.stack([x, y, z, speed, accel, jerk, curvature], axis=1).astype(np.float32)


def extract_nuscenes_trajectories(nusc, manifest: list[dict], traj_dir: Path) -> Path:
    """Write per-scene (T, 7) ego trajectories to a single safetensors shard.

    Each CAM_FRONT sample_data carries an ego_pose_token linking to the
    interpolated ego pose at that frame's capture time. Walking the 'next'
    chain therefore yields one ego pose per camera frame at the native
    12 Hz rate — directly differentiable into the columns wheel's
    TrajectoryStore expects.
    """
    shard_dir = traj_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    out_path = shard_dir / "trajectory_data_downsampled_d5_0.safetensors"
    if out_path.exists():
        log.info("Trajectories already written at %s", out_path)
        return out_path

    by_name = {scene["name"]: scene for scene in nusc.scene}
    tensors: dict[str, np.ndarray] = {}
    for clip in manifest:
        scene = by_name.get(clip["clip_id"])
        if scene is None:
            log.warning("scene %s missing in nuscenes-devkit; skipping", clip["clip_id"])
            continue

        first_sample = nusc.get("sample", scene["first_sample_token"])
        sd_token = first_sample["data"][CAMERA]
        xs, ys, zs, tss = [], [], [], []
        first_rotation = None
        while sd_token:
            sd = nusc.get("sample_data", sd_token)
            ego = nusc.get("ego_pose", sd["ego_pose_token"])
            if first_rotation is None:
                first_rotation = ego["rotation"]
            tx, ty, tz = ego["translation"]
            xs.append(tx)
            ys.append(ty)
            zs.append(tz)
            tss.append(ego["timestamp"])
            sd_token = sd["next"]

        if len(xs) < 5:
            log.warning("scene %s has only %d frames, skipping", clip["clip_id"], len(xs))
            continue

        # ego_pose is in the global map frame (x east, y north). Rotate the path
        # into the vehicle's start heading and flip to the viewer's x-forward /
        # y-right convention (the frame extract_trajectory_stats.py produces for
        # physical_ai), so nuScenes trajectories aren't flipped relative to it.
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        w, qx, qy, qz = first_rotation  # nuScenes quaternion: [w, x, y, z]
        yaw0 = np.arctan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        c, s = np.cos(yaw0), np.sin(yaw0)
        dx, dy = xs - xs[0], ys - ys[0]
        fwd = dx * c + dy * s    # +x = initial heading (forward)
        right = dx * s - dy * c  # +y = right (viewer convention)

        tensors[scene["name"]] = compute_trajectory_stats(
            fwd,
            right,
            np.asarray(zs, dtype=np.float64),
            np.asarray(tss, dtype=np.float64) / 1e6,
        )

    if not tensors:
        log.error("No trajectories produced; aborting trajectory step.")
        sys.exit(1)

    save_file(tensors, str(out_path))
    log.info("Wrote %d trajectories to %s", len(tensors), out_path)
    return out_path


def build_trajectory_memmap_and_index(traj_dir: Path, index_spec: str = "Flat") -> None:
    """Build the memmap + FAISS indexes consumed by TrajectoryStore.

    Mirrors the snippet in docs/data-preparation.md ("Process
    Ego-trajectories"): concatenate every (T, 7) safetensors into one
    float32 memmap with a clip_id → row-range JSON, then build the three
    FAISS indexes (full / 10s / 5s).
    """
    safetensors = sorted(Path(traj_dir).rglob("*/*.safetensors"))
    if not safetensors:
        log.error("No safetensors under %s/* — did the trajectory step run?", traj_dir)
        sys.exit(1)

    total_rows = 0
    for p in safetensors:
        with safe_open(str(p), framework="np") as f:
            for k in f.keys():
                total_rows += f.get_tensor(k).shape[0]

    mmap_path = traj_dir / "trajectory_data.dat"
    fp = np.memmap(str(mmap_path), dtype="float32", mode="w+", shape=(total_rows, 7))
    clip_to_idx: dict[str, tuple[int, int]] = {}
    cnt = 0
    for p in safetensors:
        with safe_open(str(p), framework="np") as f:
            for k in f.keys():
                arr = f.get_tensor(k)
                if np.isnan(arr).any():
                    log.warning("Skipping NaN-containing trajectory for %s", k)
                    continue
                start, end = cnt, cnt + arr.shape[0]
                fp[start:end, :] = arr
                clip_to_idx[k] = (start, end)
                cnt = end
    fp.flush()
    (traj_dir / "clip_to_idx.json").write_text(json.dumps(clip_to_idx))
    log.info("memmap written: %d rows, %d clips", cnt, len(clip_to_idx))

    log.info("Building trajectory FAISS indexes (full / 10s / 5s, index_spec=%s)", index_spec)
    parse_trajectory_data_from_dir(str(traj_dir), index_spec=index_spec)
    parse_subtrajectory_data_from_dir(str(traj_dir), sec=10, M=40, index_spec=index_spec)
    parse_subtrajectory_data_from_dir(str(traj_dir), sec=5, M=20, index_spec=index_spec)
    log.info("Trajectory indexes built under %s", traj_dir)


def init_annotations_db(
    db_path: Path,
    manifest: list[dict],
    processed_paths: dict[str, Path],
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS clips (
                clip_id TEXT PRIMARY KEY,
                data_source TEXT,
                country TEXT,
                has_time INTEGER DEFAULT 0,
                has_manual_annotations INTEGER DEFAULT 0,
                has_autolabels INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS annotations (
                uid TEXT PRIMARY KEY,
                project TEXT,
                clip_id TEXT,
                key TEXT,
                value REAL,
                start_time REAL,
                end_time REAL,
                label_type TEXT
            );
            CREATE TABLE IF NOT EXISTS video_paths (
                clip_id TEXT PRIMARY KEY,
                path TEXT
            );
            CREATE TABLE IF NOT EXISTS datasets (
                name     TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                license  TEXT NOT NULL
            );
            """
        )
        with conn:
            conn.execute(
                """
                INSERT INTO datasets (name, category, license)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    category = excluded.category,
                    license  = excluded.license
                """,
                (DATA_SOURCE_NAME, "Autonomous Driving (AV)", "public"),
            )
            for clip in manifest:
                cid = clip["clip_id"]
                processed = processed_paths.get(cid)
                if processed is None or not processed.exists():
                    log.warning("Skipping DB insert for %s: processed video missing", cid)
                    continue
                conn.execute(
                    """
                    INSERT INTO clips (clip_id, data_source, country)
                    VALUES (?, ?, ?)
                    ON CONFLICT(clip_id) DO UPDATE SET
                        data_source = excluded.data_source,
                        country = excluded.country
                    """,
                    (cid, DATA_SOURCE_NAME, clip["country"]),
                )
                conn.execute(
                    """
                    INSERT INTO video_paths (clip_id, path)
                    VALUES (?, ?)
                    ON CONFLICT(clip_id) DO UPDATE SET path = excluded.path
                    """,
                    (cid, str(processed.resolve())),
                )
    finally:
        conn.close()
    log.info("Wrote %d clips to %s", len(manifest), db_path)


def init_users_db(
    db_path: Path,
    username: str,
    password: str,
    email: str | None,
    data_sources: list[str],
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = UsersDataStore(str(db_path))
    existing = store.conn.execute(
        "SELECT id FROM users WHERE username = ?", (username,)
    ).fetchone()
    if existing is None:
        store.create_user(username, password, email=email, role="admin")
        log.info("Created admin user %r in %s", username, db_path)
    else:
        log.info("Admin user %r already exists; password unchanged", username)
    for ds in data_sources:
        store.grant_datasource_to_all_users(ds)


def write_required_stubs(workdir: Path) -> dict[str, Path]:
    """Files that wheel's required stores open unconditionally on startup.

    The other store directories (cosmos_embeddings, visual_embeddings, etc.)
    are created by the extraction steps when they run, and wheel's
    launch_server.py treats absent embedding dirs as "store not configured"
    so no stubs are needed for them.
    """
    predictions_dir = workdir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # ModelsWithMetricsDataStore opens this JSON unconditionally; an empty
    # mapping is enough for the no-metrics case.
    pred_origin = predictions_dir / "video_timestamp_origin_physical_ai_90kclips_overlap_alpamayo_v2.json"
    if not pred_origin.exists():
        pred_origin.write_text("{}")

    # WMStore reads this parquet on startup and expects at least a clip_id
    # column. An empty frame is fine when the wm filters aren't used.
    wm_stats = workdir / "wm_stats.parquet"
    if not wm_stats.exists():
        pd.DataFrame({"clip_id": []}).to_parquet(wm_stats)

    # AutolabelsDataStore takes a parsed dict; launch_server.py reads the JSON.
    clips_to_apis = workdir / "clips_to_apis.json"
    if not clips_to_apis.exists():
        clips_to_apis.write_text("{}")

    return {
        "predictions": predictions_dir,
        "wm_stats": wm_stats,
        "clips_to_apis": clips_to_apis,
    }


def write_config(
    workdir: Path,
    stub_paths: dict[str, Path],
    host: str,
    port: int,
    cosmos_index_spec: str,
    caption_embed_model: str,
    trajectory_populated: bool,
) -> Path:
    trajectory_store: dict = {"trajectory_dir": None}
    if trajectory_populated:
        trajectory_store = {
            "trajectory_dir": str((workdir / "trajectory_data").resolve()),
            "index_spec": "Flat",
        }

    config = {
        "datastores": {
            "annotations_db": str((workdir / "annotations.db").resolve()),
            "captions_db": str((workdir / "captions.db").resolve()),
            "users_db": str((workdir / "users.db").resolve()),
            "trajectory_store": trajectory_store,
            "cosmos_embed_store": {
                "embeddings_dir": str((workdir / "cosmos_embeddings").resolve()),
                "index_spec": cosmos_index_spec,
            },
            "visual_embed_store": {
                "embeddings_dir": str((workdir / "visual_embeddings").resolve()),
                "index_spec": "Flat",
                "siglip_model": SIGLIP_MODEL,
            },
            "caption_embed_store": {
                "embeddings_dir": str((workdir / "caption_embeddings").resolve()),
                "index_spec": "Flat",
                "embedding_model": caption_embed_model,
            },
            "wm_store": {"data_file": str(stub_paths["wm_stats"].resolve())},
            "predictions_store": {
                "predictions_dir": str(stub_paths["predictions"].resolve()),
            },
            "classifier_search": {
                "classifier_dir": str((workdir / "classifiers").resolve()),
            },
            "cluster_search": {
                "clustering_dir": str((workdir / "clustering").resolve()),
            },
            "clip_list_search": {
                "clip_lists_dir": str((workdir / "clip_lists").resolve()),
            },
            "bev_store": {
                "s3_bucket": "_local_disabled_",
                "metrics_index_dir": str((workdir / "bev_index").resolve()),
            },
        },
        "clips_to_sil_apis": str(stub_paths["clips_to_apis"].resolve()),
        "server": {
            "bindto": f"{host}:{port}",
            "debug": False,
            "llm_provider": "local",
        },
    }

    config_path = workdir / "config.yaml"
    with config_path.open("w") as f:
        f.write("# Auto-generated by getting-started-nuscenes/setup_nuscenes.py\n")
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    log.info("Wrote config to %s", config_path)
    return config_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workdir", default="./wheel-data", type=Path,
        help="Where to download nuScenes and write all generated artifacts.",
    )
    parser.add_argument(
        "--version",
        choices=["v1.0-mini", "v1.0-trainval", "v1.0-test"],
        default="v1.0-mini",
        help="Which nuScenes split to process. Mini auto-downloads (~4 GiB); "
             "trainval (~300 GiB) and test (~60 GiB) require pre-downloaded "
             "archives extracted under {workdir}/nuscenes/ (registration "
             "at nuscenes.org required).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8012, type=int)
    parser.add_argument("--admin-user", default="admin")
    parser.add_argument("--admin-password", default=None)
    parser.add_argument("--admin-email", default=None)
    parser.add_argument(
        "--n-encode-workers", default=max(2, (os.cpu_count() or 4) // 2), type=int,
        help="Parallel ffmpeg jobs for raw video encoding.",
    )
    parser.add_argument(
        "--cosmos-index-spec", default="FLAT",
        help="FAISS index spec for the Cosmos store. FLAT for small corpora; "
             "switch to OPQ/IVF/PQ at full-corpus scale.",
    )
    parser.add_argument(
        "--gpu-memory-utilization", default=0.7, type=float,
        help="vLLM GPU memory fraction for the captioning model.",
    )
    parser.add_argument(
        "--no-enforce-eager", action="store_true",
        help="Let vLLM capture cudagraphs (faster runtime, ~3 GiB peak that "
             "OOMs a 4090 at 0.7 fraction).",
    )
    parser.add_argument(
        "--max-model-len", default=32768, type=int,
        help="vLLM max sequence length. Default 128k inflates the KV cache by ~5 GiB.",
    )
    parser.add_argument(
        "--caption-embed-model", default="Qwen/Qwen3-Embedding-0.6B",
        help="SentenceTransformer for caption embeddings. The same model is "
             "loaded by the wheel server at query time; they must match.",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-encode", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-cosmos", action="store_true")
    parser.add_argument("--skip-captions", action="store_true")
    parser.add_argument("--skip-caption-embeddings", action="store_true")
    parser.add_argument("--skip-visual-embeddings", action="store_true")
    parser.add_argument("--skip-trajectory", action="store_true")
    args = parser.parse_args()

    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    require_binary("ffmpeg")
    require_binary("ffprobe")
    if not WHEEL_SCRIPTS.exists():
        log.error(
            "scripts/ not found at %s. Are you running this from a partial wheel checkout?",
            WHEEL_SCRIPTS,
        )
        sys.exit(1)

    needs_gpu = not (
        args.skip_cosmos
        and args.skip_captions
        and args.skip_caption_embeddings
        and args.skip_visual_embeddings
    )
    if needs_gpu:
        require_gpu()

    if args.skip_download:
        nuscenes_root = workdir / "nuscenes"
        if not (nuscenes_root / args.version).exists():
            log.error("--skip-download set but %s/%s missing", nuscenes_root, args.version)
            sys.exit(1)
    else:
        nuscenes_root = ensure_nuscenes_split(workdir, args.version)

    log.info("Loading nuScenes %s metadata from %s", args.version, nuscenes_root)
    nusc = NuScenes(version=args.version, dataroot=str(nuscenes_root), verbose=False)
    log.info("Loaded %d scenes", len(nusc.scene))

    raw_dir = workdir / "raw_videos"
    if args.skip_encode and any(raw_dir.glob("*.mp4")):
        log.info("--skip-encode set; reusing existing raw videos under %s", raw_dir)
        manifest = reconstruct_manifest_from_disk(nusc, raw_dir)
    else:
        manifest = encode_raw_videos(nusc, raw_dir, n_workers=args.n_encode_workers)
    (workdir / "clip_manifest.json").write_text(json.dumps(manifest, indent=2))

    processed_dir = workdir / "processed_videos"
    raw_paths = [Path(c["raw_video_path"]) for c in manifest]
    if args.skip_prepare and any(processed_dir.glob("*.mp4")):
        log.info("--skip-prepare set; reusing existing processed videos under %s", processed_dir)
    else:
        run_prepare_data(raw_paths, processed_dir, workdir / "video_paths_raw.txt")

    processed_paths: dict[str, Path] = {}
    for clip in manifest:
        cid = clip["clip_id"]
        processed = processed_dir / f"{cid}.mp4"
        if processed.exists():
            processed_paths[cid] = processed
        else:
            log.warning("processed video for %s missing at %s", cid, processed)
    if not processed_paths:
        log.error("No processed videos were produced; aborting.")
        sys.exit(1)
    sorted_processed = sorted(processed_paths.values())

    if not args.skip_cosmos:
        cosmos_dir = workdir / "cosmos_embeddings"
        run_extract_cosmos(
            sorted_processed, cosmos_dir,
            workdir / "video_paths_processed.txt",
        )
        materialize_cosmos_index(cosmos_dir, args.cosmos_index_spec)

    captions_db = workdir / "captions.db"
    captions_dir = workdir / "captions"
    captions_parquet = captions_dir / "qwen_captions_group_0_1.parquet"
    if not args.skip_captions:
        captions_parquet = run_extract_qwen_captions(
            sorted_processed, captions_dir,
            workdir / "video_paths_for_captions.txt",
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=not args.no_enforce_eager,
            max_model_len=args.max_model_len,
        )
        load_captions_into_db(captions_db, captions_parquet)

    if not args.skip_caption_embeddings and captions_parquet.exists():
        caption_embed_dir = workdir / "caption_embeddings"
        run_extract_caption_embeddings(
            captions_parquet, caption_embed_dir,
            embedding_model=args.caption_embed_model,
        )
        materialize_caption_embeddings_index(caption_embed_dir, args.caption_embed_model)

    if not args.skip_visual_embeddings:
        visual_dir = workdir / "visual_embeddings"
        run_extract_visual_embeddings(
            sorted_processed, visual_dir,
            workdir / "video_paths_for_visual.txt",
        )
        materialize_visual_embeddings_index(visual_dir)

    trajectory_populated = False
    if not args.skip_trajectory:
        traj_dir = workdir / "trajectory_data"
        extract_nuscenes_trajectories(nusc, manifest, traj_dir)
        build_trajectory_memmap_and_index(traj_dir)
        trajectory_populated = True

    init_annotations_db(workdir / "annotations.db", manifest, processed_paths)

    password = args.admin_password
    if password is None:
        if sys.stdin.isatty():
            password = getpass(f"Password for admin user '{args.admin_user}': ") or "admin"
        else:
            password = "admin"
            log.info("No --admin-password and stdin is not a tty; defaulting to 'admin'")
    init_users_db(
        workdir / "users.db",
        username=args.admin_user,
        password=password,
        email=args.admin_email,
        data_sources=[DATA_SOURCE_NAME],
    )

    stubs = write_required_stubs(workdir)
    config_path = write_config(
        workdir, stubs, args.host, args.port,
        cosmos_index_spec=args.cosmos_index_spec,
        caption_embed_model=args.caption_embed_model,
        trajectory_populated=trajectory_populated,
    )

    print()
    print("=" * 70)
    print("Setup complete.")
    print()
    print(f"  Encoded clips: {len(processed_paths)}")
    print(f"  Workdir:       {workdir}")
    print(f"  Config:        {config_path}")
    print(f"  Admin user:    {args.admin_user}")
    print(f"  Admin pass:    {password}")
    print()
    print("Launch the wheel server:")
    print()
    print(f"    python {WHEEL_SCRIPTS / 'launch_server.py'} {config_path}")
    print()
    print(f"Then open http://{args.host}:{args.port}/ and log in.")
    print("=" * 70)


if __name__ == "__main__":
    main()
