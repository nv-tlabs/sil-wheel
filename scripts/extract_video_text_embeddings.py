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
import time
from typing import BinaryIO, Optional, Union

from decord import VideoReader
from decord._ffi.base import DECORDError
from tqdm import tqdm
import pandas as pd
from pathlib import Path

import numpy as np
import torch

from arguments import add_hf_dataset_args
from sil_wheel.datasets.base_dataset import dataset_factory
from sil_wheel.embeddings import get_embedding_model


def extract_frames_btchw(
    video_source: Union[str, BinaryIO],
    fps: float = 1.0,
    max_frames: int = 64,
    n_frames: Optional[int] = None,
):
    """Extract frames from a video and return as BTCHW uint8 numpy array.

    Args:
        video_source: file path string or file-like object.
        fps: target sampling rate (frames per second).
        max_frames: maximum number of frames to extract.
        n_frames: if set, extract exactly this many equidistant frames (overrides fps).

    Returns:
        np.ndarray of shape (1, T, C, H, W) uint8, or None on failure.
    """
    try:
        vr = VideoReader(video_source)
    except (RuntimeError, DECORDError) as e:
        print("Failed to open video %s: %s", video_source, e)
        return None

    total = len(vr)
    if total == 0:
        return None

    if n_frames is not None:
        if n_frames > total:
            return None
        num = n_frames
    else:
        video_fps = vr.get_avg_fps()
        duration = total / video_fps if video_fps > 0 else total
        num = min(int(duration * fps), max_frames, total)
        num = max(1, num)

    indices = np.linspace(0, total - 1, num, dtype=int).tolist()
    try:
        frames = vr.get_batch(indices).asnumpy()
    except (RuntimeError, DECORDError) as e:
        print("Failed to decode frames from %s: %s", video_source, e)
        return None

    batch = np.transpose(np.expand_dims(frames, 0), (0, 1, 4, 2, 3))
    return batch


def flush_batch(pending, buffered_rows, model):
    """Run one batched forward pass and append per-clip rows to buffered_rows.

    Mutates pending (cleared) and buffered_rows (extended) in place.
    """
    if not pending:
        return
    # The cosmos processor requires a uniform BTCHW tensor, but clips can
    # have different native H,W (cropped/letterboxed variants). Group by
    # shape and run one forward pass per group.
    groups = {}
    for idx, (frames, _, _) in enumerate(pending):
        groups.setdefault(frames.shape, []).append(idx)

    feats_by_idx = {}
    for shape, indices in groups.items():
        frames_batch = np.concatenate(
            [pending[i][0] for i in indices], axis=0
        )
        feats = model.get_video_embeddings(frames_batch)
        for j, idx in enumerate(indices):
            feats_by_idx[idx] = feats[j]

    for idx, (_, clip_id, camera) in enumerate(pending):
        buffered_rows.append({
            "clip_id": clip_id,
            "camera_id": camera if camera is not None else "N/A",
            "embeddings": feats_by_idx[idx].tolist(),
        })
    pending.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract video-to-text embeddings")
    parser.add_argument(
        "path_to_data",
        nargs="?",
        default=None,
        help="Path to data. Optional when --hf-repo-id is set.",
    )
    parser.add_argument(
        "--output",
        default="/lustre/fsw/portfolios/nvr/users/dpaschalidou/datasets/{model}_embeddings/{split}/group_{process_id}_{n_processes}.parquet"
    )
    parser.add_argument(
        "--model_assets_dir",
        default="/lustre/fsw/portfolios/nvr/users/dpaschalidou/internvideo2-stage2_1b-224p-f4",
        help="Path to the directory containing the internvideo2 weights"
    )
    parser.add_argument(
        "--n_frames",
        default=8,
        type=int,
        help="Number of frames to use for extracting the video features"
    )
    parser.add_argument(
        "--fps",
        default=1,
        type=float,
        help="Target frame sampling rate for Qwen3-VL-Embed (frames per second). "
             "A 10s video at fps=1.0 yields 10 frames, capped by --n_frames."
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Number of clips per model forward pass. Higher values amortize "
             "GPU launch overhead and improve throughput for cosmos_embed1_*. "
             "Qwen3-VL-Embed wrappers accept B>1 but loop internally, so "
             "raising --batch_size for those models does not improve throughput.",
    )
    parser.add_argument(
        "--process_id",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--n_processes",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--model_type",
        choices=[
            "cosmos_embed1_224p",
            "cosmos_embed1_336p",
            "cosmos_embed1_448p",
            "qwen3_vl_embed_2b",
            "qwen3_vl_embed_8b",
            "pe_core_b16_224p",
            "pe_core_l14_336p",
            "pe_core_g14_448p",
        ],
        default="cosmos_embed1_448p",
        help="The embedding model to be used"
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
        help="AWS profile name for credentials (e.g., sil-gws-data)",
    )
    parser.add_argument(
        "--endpoint",
        default="https://pdx.s8k.io",
        type=str,
        help="S3 endpoint URL (default https://pdx.s8k.io)",
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

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Continue from previous run if there is one.
    path_to_output = args.output.format(
        process_id=args.process_id,
        n_processes=args.n_processes,
        model_type=args.model_type
    )

    if Path(path_to_output).is_file():
        data = pd.read_parquet(path_to_output)
        print(f"Loading {len(set(data['clip_id']))} items previously saved at {path_to_output}")
        processed_clips = set(sorted(data["clip_id"]))
    else:
        # Get all files in the output folder
        parquet_files = sorted(Path(
            Path(path_to_output).parent
        ).glob(f"**/{args.model_type}_*.parquet"))
        processed_clips = set()
        for pi in tqdm(parquet_files):
            data = pd.read_parquet(pi)
            processed_clips |= set(sorted(data["clip_id"]))
        print(f"Previously processed {len(processed_clips)} clips")

        data = pd.DataFrame(columns=["clip_id", "embeddings", "camera_id"])

    # Load the model
    model = get_embedding_model(model_type=args.model_type)

    # Make sure that we also pass the clips that we already processed
    dataset = dataset_factory(
        args.process_id,
        args.n_processes,
        args.path_to_data,
        clips_to_exclude=processed_clips,
        s3_bucket=args.bucket,
        s3_profile=args.profile,
        s3_endpoint=args.endpoint,
        hf_repo_id=args.hf_repo_id,
        hf_allow_patterns=args.hf_allow_patterns,
        hf_cache_dir=args.hf_cache_dir,
        camera_filter=args.camera,
    )

    start = time.time()
    save_every = 100
    counter = 0
    buffered_rows = []
    pending = []
    for video_buffer, clip_id, camera in tqdm(dataset, total=None, unit="clip"):
        # If it was previsouly computed move to the next one
        if clip_id in processed_clips:
            continue

        frames = extract_frames_btchw(
            video_buffer, fps=args.fps, n_frames=args.n_frames
        )
        if frames is None:
            continue

        pending.append((frames, clip_id, camera))
        if len(pending) < args.batch_size:
            continue

        n_in_batch = len(pending)
        flush_batch(pending, buffered_rows, model)
        counter += n_in_batch

        if counter >= save_every:
            data = pd.concat(
                [data, pd.DataFrame(buffered_rows)], ignore_index=True
            )
            buffered_rows = []
            data.to_parquet(path_to_output)
            print(f"Output with {len(set(data['clip_id']))} items saved at {path_to_output}")
            elapsed = time.time() - start
            print(f"Processing {counter} elements took {elapsed:.4f}s")

            # Reset counter and timer
            counter = 0
            start = time.time()

    flush_batch(pending, buffered_rows, model)
    if buffered_rows:
        data = pd.concat(
            [data, pd.DataFrame(buffered_rows)], ignore_index=True
        )
    # Remove duplicates if any
    data = data.drop_duplicates(subset="clip_id", keep="first")
    if data.empty:
        print(f"No new clips to process; leaving {path_to_output} untouched")
    else:
        data.to_parquet(path_to_output)
        print(f"Output with {len(set(data['clip_id']))} clips saved at {path_to_output}")
    elapsed = time.time() - start
    print(f"Processing {counter} elements took {elapsed:.4f}s")
