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
import io
import math
import time
from typing import Any, Optional
import os

import cv2
import decord
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import AutoProcessor

# vLLM's flashinfer top-k/top-p sampler JIT-compiles against the CUDA toolkit and fails on
# hosts whose system nvcc is older than the wheels (e.g. CUDA 12.0); fall back to the native
# sampler. Export VLLM_USE_FLASHINFER_SAMPLER=1 to force flashinfer on a matching toolchain.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

from caption_prompting import prompt_factory

from arguments import add_hf_dataset_args
from sil_wheel.datasets.base_dataset import dataset_factory


MODEL_IDS = {
    ("qwen2.5-vl", 3):  "Qwen/Qwen2.5-VL-3B-Instruct",
    ("qwen2.5-vl", 7):  "Qwen/Qwen2.5-VL-7B-Instruct",
    ("qwen2.5-vl", 32): "Qwen/Qwen2.5-VL-32B-Instruct",
    ("qwen2.5-vl", 72): "Qwen/Qwen2.5-VL-72B-Instruct",
    ("qwen3-vl", 2):    "Qwen/Qwen3-VL-2B-Instruct",
    ("qwen3-vl", 4):    "Qwen/Qwen3-VL-4B-Instruct",
    ("qwen3-vl", 8):    "Qwen/Qwen3-VL-8B-Instruct",
    ("qwen3-vl", 30):   "Qwen/Qwen3-VL-30B-A3B-Instruct",
    ("qwen3-vl", 235):  "Qwen/Qwen3-VL-235B-A22B-Instruct",
}


class SubClipDataset(IterableDataset):
    """Wrapper class to produce subclips that will be then captioned"""
    def __init__(
        self,
        base_dataset,
        clip_duration: float,
        clip_overlap: float,
        min_duration: float,
        new_size: tuple[int, int] | None,
        processed_clips: set[str] | None = None,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.clip_duration = clip_duration
        self.clip_overlap = clip_overlap
        self.min_duration = min_duration
        self.new_size = new_size
        self.processed_clips = processed_clips

    def __iter__(self):
        wi = get_worker_info()
        num_workers = wi.num_workers if wi else 1
        worker_id = wi.id if wi else 0

        for i, (video_buffer, clip_id, camera) in enumerate(self.base_dataset):
            if (i % num_workers) != worker_id:
                continue

            if clip_id.startswith("anonymized"):
                clip_id = clip_id.split("_", 1)[1]

            if clip_id in self.processed_clips:
                continue

            clips = parse_videos_to_frames(
                video_buffer,
                clip_duration=self.clip_duration,
                clip_overlap=self.clip_overlap,
                min_duration=self.min_duration,
                new_size=self.new_size,
                target_fps=2.0,
            )

            for sub_i, frames in enumerate(clips or []):
                clip_key = f"{clip_id}_{camera}_clip{sub_i}" if camera else f"{clip_id}_clip{sub_i}"
                start_time = sub_i * self.clip_duration
                end_time = start_time + self.clip_duration
                yield clip_key, {
                    "frames": frames,
                    "clip_id": clip_id,
                    "start_time": start_time,
                    "end_time": end_time,
                }

def collate_fn(batch):
     return {k: v for (k, v) in batch}


def preprocess(
    processor,
    text: list[str],
    video_with_captions: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Prepares messages and corresponding text prompts for a batch of video inputs.

    Args:
        processor: A Hugging Face processor object that applies chat templates.
        text: A list of textual prompts (one for each video in the batch).
        video_with_captions: A dictionary where each key is a video ID and each value is a dict
            containing "frames" (np.ndarray of shape (T, H, W, 3) uint8) and metadata like "clip_id"

    Returns:
        messages: A list of message dicts formatted for the model input.
        texts: A list of serialized chat templates to be passed to the model.
    """
    # Construct user messages, combining video and corresponding text
    vals = [v for _, v in video_with_captions.items()]
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": [Image.fromarray(f) for f in value["frames"]],
                "fps": 2.0,
            },
            {"type": "text", "text": txt},
        ],
    } for txt, value in zip(text, vals)]

    # Serialize messages using the processor's chat template
    texts = [
        processor.apply_chat_template(
            [mi],
            add_generation_prompt=True,
            tokenize=False
        )
        for mi in messages
    ]
    return messages, texts


def parse_videos_to_frames(
    video_buffer: io.BytesIO,
    clip_duration: Optional[float] = None,
    clip_overlap: float = 1.0,
    min_duration: float = 5.0,
    new_size: tuple[int] | None = None,
    target_fps: float = 2.0,
) -> list[np.ndarray]:
    """Decode a video buffer into one or more frame arrays at target_fps.

    Args:
        video_buffer: Input video buffer as bytes or BytesIO.
        clip_duration: Optional duration (in seconds) for slicing into multiple clips.
        clip_overlap: Overlap between consecutive clips (in seconds), i.e.
                     how much each consecutive clip overlaps with the previous one.
        min_duration: Minimum duration for a clip to be valid.

    Returns:
        List of (T, H, W, 3) uint8 frame arrays — one per subclip.
    """
    try:
        vr = decord.VideoReader(video_buffer)
    except Exception as e:
        return []

    fps = float(vr.get_avg_fps())
    total_frames = len(vr)
    if total_frames <= 0 or fps <= 0:
        return []

    duration = total_frames / fps
    if not clip_duration:
        clip_duration = duration

    step = clip_duration - clip_overlap
    if step <= 0:
        return []

    # Compute time windows in seconds (avoid round() which can cause drift / 0 step)
    n_steps = int(math.ceil(max(duration - min_duration, 0.0) / step)) + 1

    clips = []
    # stride for downsampling: pick frames so output is ~target_fps
    stride = max(int(round(fps / target_fps)), 1)

    for k in range(n_steps):
        start_time = k * step
        end_time = min(start_time + clip_duration, duration)
        sub_dur = end_time - start_time
        if sub_dur < min_duration:
            continue

        start_idx = int(start_time * fps)
        end_idx = min(int(end_time * fps), total_frames)

        frame_indices = np.arange(start_idx, end_idx, stride, dtype=np.int64)
        if frame_indices.size == 0:
            continue

        try:
            frames = vr.get_batch(frame_indices.tolist()).asnumpy()
            if new_size is not None:
                w, h = new_size
                # Faster than manual prealloc+loop in Python
                frames = np.stack(
                    [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames],
                    axis=0,
                )
        except Exception as e:
            print(f"Failed to read frames {start_idx}-{end_idx}: {e}")
            continue

        clips.append(frames)

    return clips

@torch.no_grad()
def process_video_batch(
    video_data_batch: dict[str, dict[str, str]],
    processor,
    model,
    data: pd.DataFrame,
    prompt_factory_type: str,
    max_new_tokens: int = 1024,
) -> pd.DataFrame:
    """
    Process a batch of video clips by generating multi-step captions using a vision-language model.

    Args:
        video_data_batch: Dictionary containing video paths and metadata.
        processor: Pretrained processor for tokenizing and formatting inputs.
        model: Vision-language model used for inference.
        data: Current dataframe storing all processed clip results.

    Returns:
        Updated dataframe 
    """

    prompt_start = time.time()
    prompts = prompt_factory(prompt_factory_type)
    all_captions = [[] for _ in range(len(video_data_batch))]  # Store step-by-step captions
    vision_cache = []  # Cache visual tokens for reuse

    sampling = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=max_new_tokens
    )

    for step, prompt in enumerate(prompts):
        # Fill each prompt with captions generated so far
        formatted_prompts = [
            prompt.format(*captions) for captions in all_captions
        ]
        # Build model-ready inputs
        messages, texts = preprocess(processor, formatted_prompts, video_data_batch)

        if not vision_cache:
            # Return video kwargs so vLLM can re-use them
            _, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            vision_cache = (videos, video_kwargs)

        videos, video_kwargs = vision_cache
        def _slice_mm_kwargs(mm_kw, i):
            if not mm_kw:
                return {}
            out = {}
            for k, v in mm_kw.items():
                # If this key is batched (list/tuple/ndarray), pick the i-th entry.
                if isinstance(v, (list, tuple, np.ndarray)):
                    out[k] = v[i] if len(v) > 1 else (v[0] if len(v) == 1 else v)
                else:
                    out[k] = v
            return out

        requests = []
        for i, (text, vid) in enumerate(zip(texts, videos)):
            per_kwargs = _slice_mm_kwargs(video_kwargs, i)
            frames = np.asarray(vid)
            fps = float(per_kwargs.get("fps", 2.0))
            n_frames = int(frames.shape[0])
            metadata = {
                "fps": fps,
                "total_num_frames": n_frames,
                "duration": n_frames / fps,
                "video_backend": "decord",
                "frames_indices": list(range(n_frames)),
            }
            per_kwargs.setdefault("do_sample_frames", False)
            requests.append({
                "prompt": text,
                "multi_modal_data": {"video": (frames, metadata)},
                "mm_processor_kwargs": per_kwargs,
            })


        # Generate with vLLM (returns a list of RequestOutput)
        try:
            responses = model.generate(requests, sampling)
        except ValueError as e:
            print("\n[WARNING] Skipping batch due to processor error:")
            print(e)
            print("Batch clip_keys:", list(video_data_batch.keys()))
            return data

        # Extract plain text for each item
        step_captions = [r.outputs[0].text for r in responses]

        # Append current step's caption to each video's running caption list
        for i, caption in enumerate(step_captions):
            all_captions[i].append(caption)

    # Finalize and store results
    rows = []
    for (ci, vdata), captions in zip(video_data_batch.items(), all_captions):
        if not captions:
            continue
        summary = captions[-1].replace("\n", "")
        # print(f"{ci}: {summary}\n")

        rows.append({
            "clip_key": ci,
            "clip_id": vdata["clip_id"],
            "captions": captions,
            "summary": summary,
        })

    if rows:
        data = pd.concat([data, pd.DataFrame(rows)], ignore_index=True)

    elapsed = time.time() - prompt_start
    unique_clips = len(set([r["clip_id"] for r in rows])) if rows else 0
    print(f"Generating {len(all_captions)} captions for {unique_clips} clips with Qwen took {elapsed:.4f}s")

    return data

def atomic_save_parquet(data: pd.DataFrame, path: str) -> pd.DataFrame:
    """Dedup on clip_key, then write to a sibling tmp file and rename.
    """
    data = data.drop_duplicates(subset="clip_key", keep="first")
    tmp_path = f"{path}.tmp"
    data.to_parquet(tmp_path)
    os.replace(tmp_path, path)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract captions for AV videos using qwen"
    )
    parser.add_argument(
        "path_to_data",
        nargs="?",
        default=None,
        help="Path to data. Optional when --hf-repo-id is set.",
    )
    parser.add_argument(
        "--model_family",
        choices=["qwen2.5-vl", "qwen3-vl"],
        default="qwen2.5-vl",
        help="Qwen VL family to use"
    )
    parser.add_argument(
        "--model_size",
        type=int,
        default=7,
        help="Model size in B. Valid combos with --model_family come from "
             "MODEL_IDS (e.g. qwen2.5-vl: 3/7/32/72; qwen3-vl: 2/4/8/30/235)."
    )
    parser.add_argument(
        "--prompt_factory_type",
        choices=[
            "yotta_prompt_long",
            "video_caption_dense",
            "reason_prompt",
        ],
        default="yotta_prompt_long",
        help="The prompt to be used for captioning"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="The size of the input to the VLM"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers"
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
        "--output",
        default="/path/to/datasets/alpamayo-v2.1/qwen_captions/group_{process_id}_{n_processes}.parquet"
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=5.0,
        help="Minimum duration for a clip to be valid"
    )
    parser.add_argument(
        "--clip_overlap",
        type=float,
        default=0.0,
        help="How much each consecutive clip overalps with the previous one"
    )
    parser.add_argument(
        "--clip_duration",
        type=float,
        default=20.0,
        help="The clip duration"
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
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory fraction. Lower this on shared GPUs where other "
             "processes already hold some VRAM (default: 0.9).",
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Skip vLLM cudagraph capture. Slightly slower at runtime but "
             "avoids the ~3 GiB peak that triggers OOM on a near-full GPU.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Cap the model context length so the KV cache fits on smaller GPUs. "
             "20s @ 2fps with 800x600 frames needs ~32k; default 128k uses 5+ GiB more.",
    )

    args = parser.parse_args()

    if args.hf_repo_id is None and args.path_to_data is None:
        parser.error("path_to_data is required unless --hf-repo-id is set")
    if args.hf_repo_id is not None and args.bucket is not None:
        parser.error("--hf-repo-id and --bucket are mutually exclusive")

    try:
        model_id = MODEL_IDS[(args.model_family, args.model_size)]
    except KeyError:
        valid = sorted(s for (f, s) in MODEL_IDS if f == args.model_family)
        parser.error(
            f"--model_size {args.model_size} is not valid for "
            f"--model_family {args.model_family}. Valid sizes: {valid}"
        )

    llm_kwargs = dict(
        model=model_id,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        limit_mm_per_prompt={"image": 10, "video": 10},
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    model = LLM(**llm_kwargs)
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    processor.tokenizer.padding_side ="left"

    path_to_output = args.output.format(
        process_id=args.process_id, n_processes=args.n_processes
    )
    print(path_to_output)
    if Path(path_to_output).is_file():
        data = pd.read_parquet(
            path_to_output,
            columns=["clip_key", "clip_id", "captions", "summary"]
        )
        processed_clips = set(list(data["clip_id"]))
        print(
            f"Loading {len(processed_clips)} clips previously "
            f"saved at {path_to_output}"
        )
    else:
        parquet_files = sorted(
            Path(path_to_output).parent.glob("**/*.parquet")
        )
        processed_clips = set()
        for pi in tqdm(parquet_files, desc="scanning prior shards"):
            try:
                df = pd.read_parquet(pi, columns=["clip_id"])
                processed_clips |= set(df["clip_id"])
            except Exception as e:
                print(f"[warn] skipping {pi}: {e}")
        print(
            f"Previously processed {len(processed_clips)} clips "
            f"across all shards"
        )
        data = pd.DataFrame(
            columns=["clip_key", "clip_id", "captions", "summary"]
        )

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
    new_size = (800, 600)

    # Initialize structures to store data
    save_every = 10
    counter = 0

    subclip_ds = SubClipDataset(
        base_dataset=dataset,
        clip_duration=args.clip_duration,
        clip_overlap=args.clip_overlap,
        min_duration=args.min_duration,
        new_size=new_size,
        processed_clips=processed_clips,
    )

    loader = DataLoader(
        subclip_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
        persistent_workers=True if args.num_workers > 1 else False,
    )

    start = time.time()
    t_a = time.time()

    for video_data_batch in tqdm(loader, desc="Batches", leave=True):
        data = process_video_batch(
            video_data_batch,
            processor,
            model,
            data,
            prompt_factory_type=args.prompt_factory_type,
            max_new_tokens=1024,
        )
        counter += 1
        N = len(set(data["clip_id"]))
        print(f"Computed captions for {N} videos in {time.time() - t_a:.3f}s")

        if counter == save_every:
            path_to_output = args.output.format(
                process_id=args.process_id, n_processes=args.n_processes
            )
            data = atomic_save_parquet(data, path_to_output)
            print(f"Saving {len(data)} items at {path_to_output} took {time.time() - start:.3f}s")
            counter = 0

        t_a = time.time()

    if data.empty:
        print(f"No new clips to process; leaving {path_to_output} untouched")
    else:
        data = atomic_save_parquet(data, path_to_output)
        print(f"Output with {len(set(data['clip_id']))} clips saved at {path_to_output}")
