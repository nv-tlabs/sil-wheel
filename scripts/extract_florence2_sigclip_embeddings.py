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
import pickle
import time
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from decord import VideoReader
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
)

from arguments import add_hf_dataset_args
from sil_wheel.datasets.base_dataset import dataset_factory, S3TarDataset


def l2_normalize(x: np.ndarray, eps: float = 1e-12):
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(denom, eps, None)


def sample_frame_indices(num_frames: int, n_samples: int):
    if num_frames <= 0:
        return np.array([], dtype=np.int64)
    n_samples = min(n_samples, num_frames)
    return np.linspace(
        0, num_frames - 1, n_samples, dtype=np.int64
    )


def pil_from_array(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(frame).convert("RGB")


def clamp_box_xyxy(box, width: int, height: int):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width))
    y2 = max(0, min(int(round(y2)), height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def expand_box_xyxy(box, width: int, height: int, scale: float = 1.1):
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale
    expanded = [
        cx - bw / 2,
        cy - bh / 2,
        cx + bw / 2,
        cy + bh / 2,
    ]
    return clamp_box_xyxy(expanded, width, height)


def crop_image(img: Image.Image, box_xyxy):
    return img.crop(tuple(box_xyxy))


class Florence2RegionProposer:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        task_prompt: str = "<OD>",
        max_regions_per_frame: int = 8,
        min_region_size: int = 32,
    ):
        self.device = device
        self.task_prompt = task_prompt
        self.max_regions_per_frame = max_regions_per_frame
        self.min_region_size = min_region_size

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        # transformers ≥ 4.57 dispatches via _check_and_adjust_attn_implementation
        # which requires custom-code models to declare _supports_sdpa. Florence-2's
        # remote modeling code doesn't, and crashes on AutoModelForCausalLM. Pass
        # attn_implementation="eager" to skip the dispatch entirely.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, img: Image.Image) -> list[dict]:
        inputs = self.processor(
            text=self.task_prompt,
            images=img,
            return_tensors="pt",
        ).to(self.device)

        # use_cache=False sidesteps a transformers ≥ 4.45 incompatibility:
        # Florence-2's remote modeling_florence2.py expects past_key_values to
        # be the legacy tuple-of-tuples; new transformers passes a Cache object
        # and prepare_inputs_for_generation crashes on .shape[2]. Skipping the
        # cache costs a little speed for one-shot generation.
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=False,
        )

        transition_scores = self.model.compute_transition_scores(
            sequences=generated_ids.sequences,
            scores=generated_ids.scores,
            beam_indices=generated_ids.beam_indices,
        )

        parsed_answer = self.processor.post_process_generation(
            sequence=generated_ids.sequences[0],
            transition_beam_score=transition_scores[0],
            task=self.task_prompt,
            image_size=(img.width, img.height),
        )

        task_result = parsed_answer.get(self.task_prompt, {})
        boxes = task_result.get("bboxes", [])
        labels = task_result.get("labels", [])
        scores = task_result.get("scores", [])

        regions = []
        for i, box in enumerate(boxes):
            box = expand_box_xyxy(
                box,
                width=img.width,
                height=img.height,
                scale=1.1,
            )
            if box is None:
                continue

            x1, y1, x2, y2 = box
            if (x2 - x1) < self.min_region_size or (y2 - y1) < self.min_region_size:
                continue

            label = labels[i] if i < len(labels) else "__detected_region__"
            score = scores[i] if i < len(scores) else 0.0

            regions.append(
                {
                    "bbox_xyxy": box,
                    "label": str(label),
                    "confidence": float(score),
                }
            )

        return regions[:self.max_regions_per_frame]


class SigLIPEngine:
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_name, use_fast=True
        )
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image], batch_size: int = 32):
        if len(images) == 0:
            raise ValueError("encode_images received an empty list")

        all_features = []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**inputs)
            all_features.append(features.cpu().numpy().astype(np.float32))

        return np.concatenate(all_features, axis=0)


def load_sampled_frames(
    video_bytes: BinaryIO,
    n_frames: int,
) -> tuple[list[Image.Image] | None, np.ndarray | None]:
    try:
        reader = VideoReader(video_bytes)
    except RuntimeError:
        return None, None

    if len(reader) == 0:
        return None, None

    key_indices = np.array(reader.get_key_indices())
    if len(key_indices) >= n_frames:
        frame_indices = key_indices[
            np.linspace(0, len(key_indices) - 1, n_frames, dtype=int)
        ]
    else:
        frame_indices = sample_frame_indices(len(reader), n_frames)

    frames = reader.get_batch(frame_indices.tolist()).asnumpy()
    pil_frames = [pil_from_array(frame) for frame in frames]
    return pil_frames, frame_indices


def build_views_for_clip(
    frames: list[Image.Image],
    frame_indices: np.ndarray,
    proposer: Florence2RegionProposer | None,
    include_full_frame: bool = True,
) -> tuple[list[Image.Image], list[dict]]:
    views = []
    metadata = []

    for img, absolute_frame_idx in zip(frames, frame_indices):
        if include_full_frame:
            views.append(img)
            metadata.append(
                {
                    "frame_index": int(absolute_frame_idx),
                    "bbox_xyxy": [0, 0, img.width, img.height],
                    "label": "__full_frame__",
                }
            )

        if proposer is None:
            continue

        regions = proposer.detect(img)
        for region in regions:
            crop = crop_image(img, region["bbox_xyxy"])
            views.append(crop)
            metadata.append(
                {
                    "frame_index": int(absolute_frame_idx),
                    "bbox_xyxy": region["bbox_xyxy"],
                    "label": region["label"],
                }
            )

    return views, metadata


def save_checkpoint(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_checkpoint(path: str) -> dict | None:
    if not Path(path).is_file():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(
        "Extract per-view embeddings using Florence-2 <OD> proposals and SigLIP embeddings"
    )
    parser.add_argument(
        "path_to_data",
        nargs="?",
        default=None,
        help="Path to data. Optional when --hf-repo-id is set.",
    )
    parser.add_argument(
        "--process_id",
        type=int,
        default=0
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=1
    )

    parser.add_argument(
        "--n_frames",
        type=int,
        default=8
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=30
    )

    parser.add_argument(
        "--florence_model",
        default="microsoft/Florence-2-base",
    )
    parser.add_argument(
        "--florence_task",
        default="<OD>",
        help='Florence-2 task prompt. For generic object detection, use "<OD>".',
    )

    parser.add_argument("--max_regions_per_frame", type=int, default=8)
    parser.add_argument("--min_region_size", type=int, default=32)

    parser.add_argument(
        "--siglip_model",
        default="google/siglip-base-patch16-224",
        choices=[
            "google/siglip-base-patch16-224",
            "google/siglip2-base-patch16-224",
        ],
    )

    parser.add_argument("--no_full_frame", action="store_true")

    parser.add_argument(
        "--output",
        default=(
            "/path/to/visual_embeddings/"
            "florence2_siglip_features_group_{process_id}_{n_processes}.pkl"
        ),
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
    add_hf_dataset_args(parser)

    args = parser.parse_args()

    if args.hf_repo_id is None and args.path_to_data is None:
        parser.error("path_to_data is required unless --hf-repo-id is set")
    if args.hf_repo_id is not None and args.bucket is not None:
        parser.error("--hf-repo-id and --bucket are mutually exclusive")

    output_path = args.output.format(
        process_id=args.process_id,
        n_processes=args.n_processes,
    )

    ckpt = load_checkpoint(output_path)
    if ckpt is None:
        embeddings_shards = []
        items = []
        seen_clip_ids = set()
        sibling_ckpts = sorted(Path(output_path).parent.glob("**/*.pkl"))
        for pi in tqdm(sibling_ckpts, desc="scanning prior shards"):
            try:
                other = load_checkpoint(str(pi))
                if other is None:
                    continue
                seen_clip_ids |= {
                    it["clip_id"] for it in other.get("items", [])
                }
            except Exception as e:
                print(f"[warn] skipping {pi}: {e}")
        if seen_clip_ids:
            print(
                f"Previously processed {len(seen_clip_ids)} clips "
                f"across all shards"
            )
    else:
        embeddings_existing = ckpt["embeddings"]
        embeddings_shards = [embeddings_existing] if len(embeddings_existing) > 0 else []
        items = ckpt["items"]
        seen_clip_ids = {item["clip_id"] for item in items}
        print(
            f"Loaded checkpoint from {output_path} "
            f"with {len(items)} items across {len(seen_clip_ids)} clips"
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    embedding_engine = SigLIPEngine(
        args.siglip_model,
        device=device,
    )

    proposer = Florence2RegionProposer(
        model_name=args.florence_model,
        device=device,
        task_prompt=args.florence_task,
        max_regions_per_frame=args.max_regions_per_frame,
        min_region_size=args.min_region_size,
    )

    dataset = dataset_factory(
        args.process_id,
        args.n_processes,
        args.path_to_data,
        clips_to_exclude=seen_clip_ids,
        s3_bucket=args.bucket,
        s3_profile=args.profile,
        s3_endpoint=args.endpoint,
        hf_repo_id=args.hf_repo_id,
        hf_allow_patterns=args.hf_allow_patterns,
        hf_cache_dir=args.hf_cache_dir,
    )

    processed_since_save = 0
    processed_total = 0
    start = time.time()

    progress_total = None if isinstance(dataset, S3TarDataset) else len(dataset)
    for video_buffer, clip_id, camera in tqdm(dataset, total=progress_total, desc="Clips"):
        if clip_id in seen_clip_ids:
            continue

        try:
            frames, frame_indices = load_sampled_frames(
                video_buffer, n_frames=args.n_frames
            )
            if frames is None:
                continue

            views, view_metadata = build_views_for_clip(
                frames=frames,
                frame_indices=frame_indices,
                proposer=proposer,
                include_full_frame=not args.no_full_frame,
            )
            if len(views) == 0:
                continue

            features = embedding_engine.encode_images(
                views, batch_size=args.batch_size
            )
            features = l2_normalize(features).astype(np.float32)
        except Exception as e:
            print(f"Skipping clip {clip_id}: {e}", flush=True)
            continue

        embeddings_shards.append(features)

        for meta in view_metadata:
            items.append(
                {
                    "clip_id": clip_id,
                    "camera": camera,
                    **meta,
                }
            )

        seen_clip_ids.add(clip_id)
        processed_since_save += 1
        processed_total += 1

        if processed_since_save >= args.save_every:
            embeddings = (
                np.concatenate(embeddings_shards, axis=0)
                if len(embeddings_shards) > 0
                else np.empty((0, 0), dtype=np.float32)
            )

            if len(items) != len(embeddings):
                raise RuntimeError(
                    f"Mismatch between embeddings and items during checkpoint: "
                    f"{len(embeddings)} embeddings vs {len(items)} items"
                )

            save_checkpoint(
                output_path,
                {"embeddings": embeddings, "items": items},
            )
            elapsed = time.time() - start
            print(
                f"Saved checkpoint to {output_path} | "
                f"processed {processed_since_save} clips since last save | "
                f"{processed_total} total clips | "
                f"{elapsed:.2f}s since last save",
		flush=True
            )

            processed_since_save = 0
            start = time.time()
            embeddings_shards = [embeddings]

    embeddings = (
        np.concatenate(embeddings_shards, axis=0)
        if len(embeddings_shards) > 0
        else np.empty((0, 0), dtype=np.float32)
    )

    if len(items) != len(embeddings):
        raise RuntimeError(
            f"Mismatch between embeddings and items: "
            f"{len(embeddings)} embeddings vs {len(items)} metadata items"
        )

    save_checkpoint(
        output_path,
        {"embeddings": embeddings, "items": items},
    )
    elapsed = time.time() - start
    print(f"Final save to {output_path}")
    print(f"Processed {processed_since_save} clips in {elapsed:.2f}s")
    print(f"Saved {len(embeddings)} embedding rows")


if __name__ == "__main__":
    main()
