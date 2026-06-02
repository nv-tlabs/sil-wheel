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

import argparse
import json
import logging
import os
import subprocess
import shutil
import tempfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path

from tqdm import tqdm

from arguments import add_hf_dataset_args
from sil_wheel.datasets.base_dataset import dataset_factory

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def get_video_duration(input_path) -> float | None:
    """
    Return duration in seconds using ffprobe, or None if unavailable.
    Prefers container duration; falls back to max stream duration.
    """
    try:
        if shutil.which("ffprobe") is None:
            print("ffprobe not found on PATH")
            return None

        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=duration,codec_type",
            "-of", "json",
            str(input_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(proc.stdout)

        dur = None
        fmt = info.get("format", {})
        if isinstance(fmt, dict) and fmt.get("duration") not in (None, "N/A"):
            try:
                dur = float(fmt["duration"])
            except (TypeError, ValueError):
                dur = None

        if not dur or dur <= 0:
            streams = info.get("streams", []) or []
            candidates = []
            for s in streams:
                d = s.get("duration")
                if d and d != "N/A":
                    try:
                        candidates.append(float(d))
                    except (TypeError, ValueError):
                        pass
            if candidates:
                dur = max(candidates)

        return dur if (dur and dur > 0) else None

    except Exception:
        return None


def compress_video_to_target_size(
    input_path, output_path, target_size_mb=0.5, resolution=640, crf=28
):
    target_size_bytes = target_size_mb * 1024 * 1024
    output_path = Path(output_path)
    temp_output = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)

    try:
        if os.path.getsize(input_path) <= target_size_bytes:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, output_path)
            return str(output_path)
    except OSError:
        return None

    def run_ffmpeg(cmd):
        try:
            subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
            if temp_output.exists():
                shutil.move(temp_output, output_path)
                return str(output_path)
            else:
                temp_output.unlink(missing_ok=True)
                return None
        except subprocess.CalledProcessError:
            temp_output.unlink(missing_ok=True)
            return None

    # First try CRF-based
    cmd_crf = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", f"scale={resolution}:-2",
        "-c:v", "libx264",
        "-preset", "faster",
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", "96k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(temp_output),
    ]
    result = run_ffmpeg(cmd_crf)
    if result:
        Path(input_path).unlink(missing_ok=True)
        return result

    # Fallback: Bitrate-based if CRF overshoots
    duration = get_video_duration(input_path)
    if duration is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(input_path), str(output_path))
        return str(output_path)

    target_bitrate_kbps = int((target_size_bytes * 8) / duration / 1000)
    if target_bitrate_kbps < 100:
        print(f"Skipping {input_path}: bitrate too low for {duration:.2f}s video")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(input_path), str(output_path))
        return str(output_path)

    cmd_bitrate = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", f"scale={resolution}:-2",
        "-c:v", "libx264",
        "-b:v", f"{target_bitrate_kbps}k",
        "-maxrate", f"{target_bitrate_kbps}k",
        "-bufsize", f"{2 * target_bitrate_kbps}k",
        "-c:a", "aac",
        "-b:a", "96k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(temp_output),
    ]
    result = run_ffmpeg(cmd_bitrate)
    if result:
        Path(input_path).unlink(missing_ok=True)
        return result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(input_path), str(output_path))
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download, extract and compress videos to a target size"
    )
    parser.add_argument(
        "path_to_data",
        nargs="?",
        default=None,
        help="Path to a .txt or .json file listing S3 keys or local "
             "paths (.mp4 or .tar). Optional when --hf-repo-id is set.",
    )
    parser.add_argument(
        "output_directory",
        help="Path to output directory",
    )
    parser.add_argument(
        "--process_id",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--bucket",
        default=None,
        type=str,
        help="S3 bucket name containing the videos (enables S3 mode)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        type=str,
        help="AWS profile name for credentials (e.g., sil-wheel)",
    )
    parser.add_argument(
        "--endpoint",
        default="https://s3.example.com",
        type=str,
        help="S3 endpoint URL (default https://s3.example.com)",
    )
    parser.add_argument(
        "--camera",
        default=None,
        type=str,
        help="Only process clips from this camera (e.g. camera_front_wide_120fov). "
             "Defaults to all cameras.",
    )
    add_hf_dataset_args(parser)
    args = parser.parse_args()

    if args.hf_repo_id is None and args.path_to_data is None:
        parser.error("path_to_data is required unless --hf-repo-id is set")
    if args.hf_repo_id is not None and args.bucket is not None:
        parser.error("--hf-repo-id and --bucket are mutually exclusive")

    log.info(
        "Process %d / %d | output: %s | camera: %s",
        args.process_id, args.n_processes, args.output_directory,
        args.camera or "all",
    )

    os.makedirs(args.output_directory, exist_ok=True)

    # Build skip set from already-processed output files
    processed_clips = {
        Path(f).stem for f in Path(args.output_directory).glob("*.mp4")
    }
    log.info("Found %d already-processed clips in output dir, will skip them", len(processed_clips))

    dataset = dataset_factory(
        process_id=args.process_id,
        n_processes=args.n_processes,
        path_to_files=args.path_to_data,
        clips_to_exclude=processed_clips,
        s3_bucket=args.bucket,
        s3_profile=args.profile,
        s3_endpoint=args.endpoint,
        hf_repo_id=args.hf_repo_id,
        hf_allow_patterns=args.hf_allow_patterns,
        hf_cache_dir=args.hf_cache_dir,
        camera_filter=args.camera,
    )

    log.info(
        "Process %d / %d assigned %d items from the dataset",
        args.process_id, args.n_processes, len(dataset),
    )

    n_submitted = 0
    n_skipped = 0
    n_failed = 0
    n_saved = 0

    with ThreadPoolExecutor(max_workers=16) as executor:
        pending = {}
        for video_buffer, clip_id, camera in tqdm(dataset):
            if video_buffer is None:
                n_skipped += 1
                continue

            dst_path = Path(args.output_directory) / f"{clip_id}.mp4"
            if dst_path.exists():
                n_skipped += 1
                continue

            # Write BytesIO to a temp file so ffmpeg can read it
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(video_buffer.read())
                tmp_path = tmp.name

            future = executor.submit(
                compress_video_to_target_size, tmp_path, str(dst_path)
            )
            pending[future] = clip_id
            n_submitted += 1

            # Eagerly drain completed futures
            done, _ = wait(pending.keys(), timeout=0, return_when=FIRST_COMPLETED)
            for f in done:
                cid = pending.pop(f)
                if f.result() is None:
                    log.warning("Compression failed for %s", cid)
                    n_failed += 1
                else:
                    n_saved += 1

        # Drain remaining in-flight futures
        for future in as_completed(list(pending.keys())):
            cid = pending.pop(future)
            if future.result() is None:
                log.warning("Compression failed for %s", cid)
                n_failed += 1
            else:
                n_saved += 1

    log.info(
        "Process %d / %d done — submitted: %d | saved: %d | skipped: %d | failed: %d",
        args.process_id, args.n_processes,
        n_submitted, n_saved, n_skipped, n_failed,
    )
