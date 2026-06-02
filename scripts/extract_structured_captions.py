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

"""Extract structured captions for AV video clips using a tool-calling API."""
import argparse
import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import openai
import pandas as pd
from tqdm import tqdm

from sil_wheel.captions import get_caption_mode
from arguments import add_hf_dataset_args
from sil_wheel.datasets.base_dataset import dataset_factory



def encode_buffer(video_buffer) -> str:
    """Base64-encode a BytesIO video buffer into a data URL."""
    video_buffer.seek(0)
    b64 = base64.b64encode(video_buffer.read()).decode("utf-8")
    return f"data:video/mp4;base64,{b64}"


def run_caption(client, model, b64_url, mode):
    """Call the API with tool-use and return (raw, processed) dicts."""
    user_text = mode["camera_prefix"] + mode["user_prompt"]
    messages = [
        {"role": "system", "content": mode["system_prompt"]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": b64_url}},
            ],
        },
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=mode["tools"],
        tool_choice="required",
    )
    tool_call = response.choices[0].message.tool_calls[0]
    raw = json.loads(tool_call.function.arguments)
    processed = mode["process_caption"](raw)
    return raw, processed


def build_row(clip_id, camera, caption_mode_name, model, processed, elapsed):
    """Build a flat dict row from processed caption output."""
    row = {
        "clip_id": clip_id,
        "camera": camera,
        "caption_mode": caption_mode_name,
        "model": model,
        "elapsed_s": round(elapsed, 2),
    }
    for key, val in processed.items():
        row[key] = (
            json.dumps(val, default=str)
            if isinstance(val, (dict, list))
            else val
        )
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract structured captions via tool-calling API"
    )
    parser.add_argument(
        "path_to_data",
        nargs="?",
        default=None,
        help="Path to explicit file list (.json or .txt). "
             "Optional when --hf-repo-id is set.",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="S3 bucket name (enables S3 mode)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="AWS profile name for credentials (e.g., sil-wheel)",
    )
    parser.add_argument(
        "--endpoint",
        default="https://s3.example.com",
        help="S3 endpoint URL (default: https://s3.example.com)",
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
        "--output",
        default=(
            "/path/to/datasets/"
            "structured_captions/{caption_mode}/"
            "group_{process_id}_{n_processes}.parquet"
        ),
        help="Output parquet path. Supports {caption_mode}, "
             "{process_id}, {n_processes} placeholders.",
    )
    parser.add_argument(
        "--caption_mode",
        choices=["sil_av_benchmark", "comprehensive_v3"],
        default="sil_av_benchmark",
        help="Caption mode to use",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model identifier passed to the API",
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="API key (falls back to OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--base_url",
        default="https://api.openai.com/v1",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent API calls",
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="Only process clips from this camera (e.g. camera_front_wide_120fov). "
             "Defaults to all cameras.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-caption clips that already have results",
    )
    add_hf_dataset_args(parser)
    args = parser.parse_args()

    if args.hf_repo_id is None and args.path_to_data is None:
        parser.error("path_to_data is required unless --hf-repo-id is set")
    if args.hf_repo_id is not None and args.bucket is not None:
        parser.error("--hf-repo-id and --bucket are mutually exclusive")

    api_key = args.api_key or __import__("os").environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY or pass --api_key")
        sys.exit(1)

    path_to_output = args.output.format(
        caption_mode=args.caption_mode,
        process_id=args.process_id,
        n_processes=args.n_processes,
    )
    Path(path_to_output).parent.mkdir(parents=True, exist_ok=True)

    mode = get_caption_mode(args.caption_mode)
    columns = ["clip_id", "camera", "caption_mode", "model", "elapsed_s"]
    columns += mode["caption_keys"]

    if Path(path_to_output).is_file() and not args.force:
        data = pd.read_parquet(path_to_output)
        processed_clips = set(data["clip_id"])
        print(
            f"Resuming: {len(processed_clips)} clips already done "
            f"({path_to_output})"
        )
    else:
        data = pd.DataFrame(columns=columns)
        processed_clips = set()

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

    client = openai.OpenAI(api_key=api_key, base_url=args.base_url)

    save_every = 50
    results_buffer = []
    pending = {}  # future -> (clip_id, camera, t0)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for video_buffer, clip_id, camera in tqdm(dataset):
            if clip_id in processed_clips:
                continue

            try:
                b64_url = encode_buffer(video_buffer)
            except Exception as e:
                print("Encode failed for %s: %s", clip_id, e)
                continue

            t0 = time.time()
            future = executor.submit(
                run_caption, client, args.model, b64_url, mode
            )
            pending[future] = (clip_id, camera, t0)

            if len(pending) >= args.workers:
                wait(pending.keys(), return_when=FIRST_COMPLETED)

            # Collect all completed futures
            done = [f for f in list(pending) if f.done()]
            for f in done:
                cid, cam, t_start = pending.pop(f)
                elapsed = time.time() - t_start
                try:
                    _, processed = f.result()
                    results_buffer.append(
                        build_row(
                            cid, cam, args.caption_mode, args.model,
                            processed, elapsed,
                        )
                    )
                    #print(f"OK  {cid}  ({elapsed:.1f}s)")
                except Exception as e:
                    pass
                    #print(f"FAIL {cid}: {e}")

            if len(results_buffer) >= save_every:
                data = pd.concat(
                    [data, pd.DataFrame(results_buffer)], ignore_index=True
                )
                results_buffer.clear()
                data.to_parquet(path_to_output)
                print(f"Saved {len(data)} clips → {path_to_output}")

        # Drain any remaining in-flight requests
        wait(pending.keys())
        for f in list(pending):
            cid, cam, t_start = pending.pop(f)
            elapsed = time.time() - t_start
            try:
                _, processed = f.result()
                results_buffer.append(
                    build_row(
                        cid, cam, args.caption_mode, args.model,
                        processed, elapsed,
                    )
                )
                print(f"OK  {cid}  ({elapsed:.1f}s)")
            except Exception as e:
                print(f"FAIL {cid}: {e}")

    if results_buffer:
        data = pd.concat(
            [data, pd.DataFrame(results_buffer)], ignore_index=True
        )

    data = data.drop_duplicates(subset="clip_id", keep="first")
    data.to_parquet(path_to_output)
    print(f"Done. {len(data)} clips saved → {path_to_output}")
