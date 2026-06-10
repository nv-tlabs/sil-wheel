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

"""Fetch a slice of the Physical AI Autonomous Vehicles dataset and prepare
it for the wheel server.

The dataset (https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
ships its videos as per-chunk .zip shards — one ~2 GiB zip per (camera, chunk),
3146 chunks per camera, ~300k clips total. Processing all of it is impractical
for a getting-started walkthrough, so this script lets you pick a single camera
and a handful of chunks (--chunks) and, optionally, cap the number of clips that
flow through the GPU stages (--max-clips).

It runs the same end-to-end pipeline as the nuScenes example, but drives the
existing scripts/ against the HuggingFace zip shards instead of a local nuScenes
split:

  1. Download + compress the selected camera chunks with
     scripts/prepare_data.py --hf-repo-id ... (uses the new HuggingFaceZipDataset
     reader under the hood).
  2. Extract Cosmos / Qwen captions / caption embeddings / Florence-2 +
     SigCLIP2 visual embeddings via scripts/extract_*.py.
  3. Build per-clip ego trajectories from the egomotion.offline parquet files
     via scripts/extract_trajectory_stats.py (which auto-detects the physical_ai
     source from the .egomotion.offline.parquet filename).
  4. Initialise the SQLite stores and write config.yaml.

Launch the wheel server with:

    python scripts/launch_server.py wheel-data-physical-ai/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
import subprocess
import sys
import time
import zipfile
from getpass import getpass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from huggingface_hub import snapshot_download
from safetensors.numpy import safe_open

from sil_wheel.stores.caption_embeddings_store import CaptionEmbeddingsStore
from sil_wheel.stores.cosmos_embeddings_store import CosmosEmbeddingsStore
from sil_wheel.stores.sqlite_caption_store import FTSCaptionStore
from sil_wheel.stores.trajectory_store import (
    parse_subtrajectory_data_from_dir,
    parse_trajectory_data_from_dir,
)
from sil_wheel.stores.users_data_store import UsersDataStore
from sil_wheel.stores.visual_embeddings_store import Florence2SigCLIPEmbeddingStore


log = logging.getLogger("setup_physical_ai")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


EXAMPLE_DIR = Path(__file__).resolve().parent
WHEEL_SCRIPTS = EXAMPLE_DIR.parent.parent / "scripts"

REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
DATA_SOURCE_NAME = "PhysicalAI-AV"

# SigCLIP2 variant for the visual embeddings. One value drives both extraction
# (the index) and the server's query-time encoder (written into config.yaml),
# so text and image queries share the same embedding space.
SIGLIP_MODEL = "google/siglip2-base-patch16-224"

# The seven cameras shipped by the dataset. The forward-facing wide camera is
# the closest analogue to nuScenes' CAM_FRONT and the natural default.
CAMERA_CHOICES = [
    "camera_front_wide_120fov",
    "camera_front_tele_30fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
]
DEFAULT_CAMERA = "camera_front_wide_120fov"

# Clips are ~20 s long (~202 egomotion samples at ~10 Hz).
CLIP_DURATION_S = 20.0


def parse_chunks(spec: str) -> list[int]:
    """Parse a chunk spec like '0', '0,1,2' or '0-3,7' into a
    sorted, de-duplicated list of chunk indices."""
    chunks = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            chunks.update(range(int(lo), int(hi) + 1))
        else:
            chunks.add(int(token))
    if not chunks:
        log.error("No chunks parsed from --chunks %r", spec)
        sys.exit(1)
    return sorted(chunks)


def run_subprocess(label: str, argv: list[str]) -> None:
    log.info("[%s] %s", label, " ".join(argv))
    t0 = time.time()
    try:
        subprocess.run(argv, check=True)
    except subprocess.CalledProcessError as e:
        log.error("[%s] failed with exit code %d", label, e.returncode)
        sys.exit(e.returncode)
    log.info("[%s] done in %.1fs", label, time.time() - t0)


def write_video_list(paths: list[Path], dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        for p in paths:
            f.write(str(p) + "\n")
    return dst


def run_prepare_data_hf(
    processed_dir: Path,
    camera: str,
    chunks: list[int],
    cache_dir: Path | None,
) -> None:
    """Drive scripts/prepare_data.py against the HuggingFace zip shards.

    prepare_data.py downloads the matching camera chunk zips, extracts each
    <clip_id>.<camera>.mp4 member, and compresses it into processed_dir as
    <clip_id>.mp4.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = [
        f"camera/{camera}/{camera}.chunk_{c:04d}.zip" for c in chunks
    ]
    argv = [
        sys.executable,
        str(WHEEL_SCRIPTS / "prepare_data.py"),
        str(processed_dir),
        "--hf-repo-id", REPO_ID,
        "--hf-allow-patterns", *allow_patterns,
    ]
    if cache_dir is not None:
        argv += ["--hf-cache-dir", str(cache_dir)]
    run_subprocess("prepare_data", argv)


def select_processed_clips(processed_dir: Path, max_clips: int | None) -> list[str]:
    clip_ids = sorted(p.stem for p in processed_dir.glob("*.mp4"))
    if not clip_ids:
        log.error("No processed videos found under %s; aborting.", processed_dir)
        sys.exit(1)
    if max_clips is not None and len(clip_ids) > max_clips:
        log.info(
            "Capping to the first %d of %d processed clips (--max-clips).",
            max_clips, len(clip_ids),
        )
        clip_ids = clip_ids[:max_clips]
    return clip_ids


def run_extract_cosmos(
    processed_paths: list[Path],
    cosmos_dir: Path,
    video_list_path: Path,
    model_type: str = "cosmos_embed1_448p",
) -> None:
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
    model_size: int = 3,
    clip_duration: float = CLIP_DURATION_S,
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
    model_size: int = 3,
    scene_duration_s: float = CLIP_DURATION_S,
) -> None:
    """Insert qwen-captions parquet rows into FTSCaptionStore.

    insert_from_dataframe expects [clip_id, summary, start_time, end_time].
    The captioning script writes one sub-clip per clip when clip_duration
    matches the clip length with no overlap.
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
) -> None:
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
) -> None:
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


def materialize_visual_embeddings_index(visual_dir: Path, index_spec: str = "Flat") -> None:
    log.info("Building visual-embedding FAISS index (index_spec=%s)", index_spec)
    store = Florence2SigCLIPEmbeddingStore(
        str(visual_dir), index_spec=index_spec, siglip_model=SIGLIP_MODEL
    )
    log.info(
        "Visual-embedding index ready: ntotal=%d",
        store.features_index.ntotal if store.features_index is not None else 0,
    )


def _extract_zip_members(zip_path: Path, suffix: str, clip_ids: set[str], rename) -> int:
    """Extract every <clip>.<suffix> member for a selected clip into the
    destination given by rename(clip_id) -> Path. Returns the count written.
    """
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if not member.endswith(suffix):
                continue
            clip_id = Path(member).name.split(".")[0]
            if clip_id not in clip_ids:
                continue
            dst = rename(clip_id)
            if not dst.exists():
                dst.write_bytes(zf.read(member))
            n += 1
    return n


def download_egomotion(
    chunks: list[int],
    clip_ids: set[str],
    egomotion_dir: Path,
    cache_dir: Path | None,
    camera: str,
) -> list[Path]:
    """Download the ego trajectory inputs for the selected clips.

    For each kept clip two files are written into egomotion_dir:

    * <clip_id>.egomotion.offline.parquet — ego x/y/z at ~10 Hz, from
      labels/egomotion.offline/egomotion.offline.chunk_NNNN.zip.
    * <clip_id>.timestamps.parquet — camera frame times at ~30 Hz, from the
      already-downloaded camera/<camera>/<camera>.chunk_NNNN.zip.

    extract_trajectory_stats.py resamples the ego trajectory onto those frame
    timestamps (paired by the .timestamps.parquet filename), so it emits one
    trajectory row per video frame rather than per 10 Hz ego sample.
    """
    egomotion_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = (
        [f"labels/egomotion.offline/egomotion.offline.chunk_{c:04d}.zip" for c in chunks]
        + [f"camera/{camera}/{camera}.chunk_{c:04d}.zip" for c in chunks]
    )
    log.info(
        "Downloading egomotion + %s frame timestamps for %d chunk(s) from %s",
        camera, len(chunks), REPO_ID,
    )
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        cache_dir=str(cache_dir) if cache_dir else None,
        max_workers=8,
    )

    written = []
    for c in chunks:
        ego_zip = Path(local_dir) / f"labels/egomotion.offline/egomotion.offline.chunk_{c:04d}.zip"
        if not ego_zip.exists():
            log.warning("egomotion chunk zip missing: %s", ego_zip)
            continue
        _extract_zip_members(
            ego_zip, ".egomotion.offline.parquet", clip_ids,
            lambda cid: egomotion_dir / f"{cid}.egomotion.offline.parquet",
        )
        written.extend(
            egomotion_dir / f"{cid}.egomotion.offline.parquet" for cid in clip_ids
            if (egomotion_dir / f"{cid}.egomotion.offline.parquet").exists()
        )

        # Extract the camera frame timestamps for the same clips.
        cam_zip = Path(local_dir) / f"camera/{camera}/{camera}.chunk_{c:04d}.zip"
        if cam_zip.exists():
            _extract_zip_members(
                cam_zip, ".timestamps.parquet", clip_ids,
                lambda cid: egomotion_dir / f"{cid}.timestamps.parquet",
            )
        else:
            log.warning(
                "camera chunk zip missing, frame timestamps unavailable: %s", cam_zip,
            )

    written = sorted(set(written))
    if not written:
        log.warning(
            "No egomotion parquet matched the %d selected clips; trajectory "
            "search will be unavailable.", len(clip_ids),
        )
    else:
        log.info("Extracted %d egomotion parquet files to %s", len(written), egomotion_dir)
    return written


def run_extract_trajectories(
    egomotion_paths: list[Path],
    traj_dir: Path,
    video_list_path: Path,
) -> Path | None:
    """Run scripts/extract_trajectory_stats.py over the egomotion parquets.

    The egomotion files are named <clip_id>.egomotion.offline.parquet, which
    extract_trajectory_stats.py auto-detects as the physical_ai source. The
    script writes trajectory_stats_smoothed_0.safetensors; we rename it to the
    trajectory_data_downsampled_d5_* pattern that the server's LazyTrajectoryData
    viewer globs for, and which the memmap/index builder also picks up.
    """
    if not egomotion_paths:
        return None
    shard_dir = traj_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    write_video_list(sorted(egomotion_paths), video_list_path)
    run_subprocess(
        "extract_trajectories",
        [
            sys.executable,
            str(WHEEL_SCRIPTS / "extract_trajectory_stats.py"),
            str(video_list_path),
            str(shard_dir),
            "0",
        ],
    )
    produced = shard_dir / "trajectory_stats_smoothed_0.safetensors"
    if not produced.exists():
        log.error("Trajectory extraction produced no output at %s", produced)
        return None
    renamed = shard_dir / "trajectory_data_downsampled_d5_0.safetensors"
    produced.replace(renamed)
    log.info("Trajectory shard ready at %s", renamed)
    return renamed


def build_trajectory_memmap_and_index(traj_dir: Path, index_spec: str = "Flat") -> bool:
    """Build the memmap + FAISS indexes consumed by TrajectoryStore.

    Concatenate every (T, 7) safetensors into one float32 memmap with a
    clip_id -> row-range JSON, then build the three FAISS indexes
    (full / 10s / 5s). Mirrors docs/data-preparation.md.
    """
    safetensors = sorted(Path(traj_dir).rglob("*/*.safetensors"))
    if not safetensors:
        log.warning("No safetensors under %s/* — skipping trajectory index.", traj_dir)
        return False

    total_rows = 0
    for p in safetensors:
        with safe_open(str(p), framework="np") as f:
            for k in f.keys():
                total_rows += f.get_tensor(k).shape[0]

    mmap_path = traj_dir / "trajectory_data.dat"
    fp = np.memmap(str(mmap_path), dtype="float32", mode="w+", shape=(total_rows, 7))
    clip_to_idx = {}
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
    return True


def init_annotations_db(
    db_path: Path,
    clip_ids: list[str],
    processed_paths: dict[str, Path],
    camera: str,
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
                # Gated under the NVIDIA AV Dataset License -> "licensed" (🔒).
                (DATA_SOURCE_NAME, "Autonomous Driving (AV)", "licensed"),
            )
            inserted = 0
            for cid in clip_ids:
                processed = processed_paths.get(cid)
                if processed is None or not processed.exists():
                    log.warning("Skipping DB insert for %s: processed video missing", cid)
                    continue
                conn.execute(
                    """
                    INSERT INTO clips (clip_id, data_source, country)
                    VALUES (?, ?, ?)
                    ON CONFLICT(clip_id) DO UPDATE SET
                        data_source = excluded.data_source
                    """,
                    (cid, DATA_SOURCE_NAME, ""),
                )
                conn.execute(
                    """
                    INSERT INTO video_paths (clip_id, path)
                    VALUES (?, ?)
                    ON CONFLICT(clip_id) DO UPDATE SET path = excluded.path
                    """,
                    (cid, str(processed.resolve())),
                )
                inserted += 1
    finally:
        conn.close()
    log.info("Wrote %d clips to %s", inserted, db_path)


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
    """Files that wheel's required stores open unconditionally on startup."""
    predictions_dir = workdir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    pred_origin = predictions_dir / "video_timestamp_origin_physical_ai_90kclips_overlap_alpamayo_v2.json"
    if not pred_origin.exists():
        pred_origin.write_text("{}")

    wm_stats = workdir / "wm_stats.parquet"
    if not wm_stats.exists():
        pd.DataFrame({"clip_id": []}).to_parquet(wm_stats)

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
    trajectory_store = {"trajectory_dir": None}
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
        f.write("# Auto-generated by getting-started-physical-ai-autonomous-vehicles/setup_physical_ai.py\n")
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    log.info("Wrote config to %s", config_path)
    return config_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workdir", default="./wheel-data-physical-ai", type=Path,
        help="Where to write all generated artifacts.",
    )
    parser.add_argument(
        "--camera", default=DEFAULT_CAMERA, choices=CAMERA_CHOICES,
        help="Which camera to host (one camera per run; default %(default)s).",
    )
    parser.add_argument(
        "--chunks", default="0", type=str,
        help="Which dataset chunks to process. Comma-separated indices and/or "
             "ranges, e.g. '0', '0,1,2' or '0-3,7'. Each camera chunk zip is "
             "~2 GiB and holds ~96 clips. Default: 0.",
    )
    parser.add_argument(
        "--max-clips", default=None, type=int,
        help="Cap the number of clips fed to the GPU stages / DB (handy for a "
             "quick smoke test). The full chunk zip is still downloaded.",
    )
    parser.add_argument(
        "--hf-cache-dir", default=None, type=Path,
        help="HuggingFace cache directory (defaults to $HF_HOME). The raw "
             "chunk zips land here; point it at a disk with room to spare.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8012, type=int)
    parser.add_argument("--admin-user", default="admin")
    parser.add_argument("--admin-password", default=None)
    parser.add_argument("--admin-email", default=None)
    parser.add_argument(
        "--cosmos-index-spec", default="FLAT",
        help="FAISS index spec for the Cosmos store. FLAT for small corpora.",
    )
    parser.add_argument(
        "--qwen-model-size", default=4, type=int,
        help="Unused: captioning is hardcoded to Qwen3-VL-4B.",
    )
    parser.add_argument(
        "--gpu-memory-utilization", default=0.7, type=float,
        help="vLLM GPU memory fraction for the captioning model.",
    )
    parser.add_argument(
        "--no-enforce-eager", action="store_true",
        help="Let vLLM capture cudagraphs (faster runtime, more VRAM).",
    )
    parser.add_argument(
        "--max-model-len", default=32768, type=int,
        help="vLLM max sequence length.",
    )
    parser.add_argument(
        "--caption-embed-model", default="Qwen/Qwen3-Embedding-0.6B",
        help="SentenceTransformer for caption embeddings. Must match the model "
             "the wheel server loads at query time.",
    )
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-cosmos", action="store_true")
    parser.add_argument("--skip-captions", action="store_true")
    parser.add_argument("--skip-caption-embeddings", action="store_true")
    parser.add_argument("--skip-visual-embeddings", action="store_true")
    parser.add_argument("--skip-trajectory", action="store_true")
    args = parser.parse_args()

    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    chunks = parse_chunks(args.chunks)

    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            log.error("Required binary %r not found on PATH. Install it and retry.", binary)
            sys.exit(1)
    if not WHEEL_SCRIPTS.exists():
        log.error("scripts/ not found at %s.", WHEEL_SCRIPTS)
        sys.exit(1)

    needs_gpu = not (
        args.skip_cosmos
        and args.skip_captions
        and args.skip_caption_embeddings
        and args.skip_visual_embeddings
    )
    if needs_gpu and not torch.cuda.is_available():
        log.error(
            "No CUDA GPU visible to PyTorch. The Cosmos / captioning / "
            "caption-embedding / visual-embedding stages all need a GPU; "
            "use --skip-* flags to bypass them or run on a GPU host."
        )
        sys.exit(1)

    log.info(
        "Camera: %s | chunks: %s | max-clips: %s",
        args.camera, chunks, args.max_clips,
    )

    processed_dir = workdir / "processed_videos"
    if args.skip_prepare and any(processed_dir.glob("*.mp4")):
        log.info("--skip-prepare set; reusing existing processed videos under %s", processed_dir)
    else:
        run_prepare_data_hf(processed_dir, args.camera, chunks, args.hf_cache_dir)

    clip_ids = select_processed_clips(processed_dir, args.max_clips)
    processed_paths = {cid: processed_dir / f"{cid}.mp4" for cid in clip_ids}
    sorted_processed = [processed_paths[c] for c in clip_ids]
    log.info("Selected %d clips for downstream processing.", len(clip_ids))
    (workdir / "clip_manifest.json").write_text(
        json.dumps(
            [{"clip_id": c, "camera": args.camera} for c in clip_ids], indent=2
        )
    )

    if not args.skip_cosmos:
        cosmos_dir = workdir / "cosmos_embeddings"
        run_extract_cosmos(sorted_processed, cosmos_dir, workdir / "video_paths_processed.txt")
        materialize_cosmos_index(cosmos_dir, args.cosmos_index_spec)

    captions_db = workdir / "captions.db"
    captions_dir = workdir / "captions"
    captions_parquet = captions_dir / "qwen_captions_group_0_1.parquet"
    if not args.skip_captions:
        captions_parquet = run_extract_qwen_captions(
            sorted_processed, captions_dir,
            workdir / "video_paths_for_captions.txt",
            model_size=args.qwen_model_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=not args.no_enforce_eager,
            max_model_len=args.max_model_len,
        )
        load_captions_into_db(captions_db, captions_parquet, model_size=args.qwen_model_size)

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
            sorted_processed, visual_dir, workdir / "video_paths_for_visual.txt",
        )
        materialize_visual_embeddings_index(visual_dir)

    trajectory_populated = False
    if not args.skip_trajectory:
        traj_dir = workdir / "trajectory_data"
        egomotion_paths = download_egomotion(
            chunks, set(clip_ids), workdir / "egomotion", args.hf_cache_dir,
            args.camera,
        )
        shard = run_extract_trajectories(
            egomotion_paths, traj_dir, workdir / "video_paths_egomotion.txt",
        )
        if shard is not None:
            trajectory_populated = build_trajectory_memmap_and_index(traj_dir)

    init_annotations_db(workdir / "annotations.db", clip_ids, processed_paths, args.camera)

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
    print(f"  Camera:        {args.camera}")
    print(f"  Chunks:        {chunks}")
    print(f"  Clips:         {len(clip_ids)}")
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
