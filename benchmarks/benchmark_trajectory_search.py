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

"""Benchmark TrajectoryStore expression-based and FAISS shape search.

Usage (run from repo root):
    python -m benchmarks.benchmark_trajectory_search \\
        --data-dir /path/to/trajectory_data \\
        --repeat 3 \\
        --clip-id <clip_id_for_shape_search>
"""
import argparse
import statistics
import time
from pathlib import Path
from typing import Iterable, List, Tuple

from benchmarks import path_size, write_markdown
from sil_wheel.stores.trajectory_store import TRAJECTORY_EXPRESSIONS, TrajectoryStore

CUSTOM_SPEED_EXPR = "sum(speed_kph > 80) > 10"

DEFAULT_CLIP_IDS = [
    "e38d0087-7c1b-46df-875f-bf21fba71f86",
    "1d9eb540-5ece-44ae-8a60-ea67f4709225",
    "b4517e9d-4be0-482f-94e9-b22acb195ae0",
    "00000000-0000-0000-0000-000000000000",  # artificial / not-found sentinel
]


def time_fn(fn, repeat: int) -> Tuple[float, int]:
    """Warmup once then time `repeat` calls. Returns (mean_seconds, last_count)."""
    fn()  # warmup
    samples: List[float] = []
    count = 0
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - t0)
        count = len(result) if hasattr(result, "__len__") else 0
    return statistics.mean(samples), count


def format_row(cells: Iterable[str], widths: List[int]) -> str:
    return " | ".join(c.ljust(w) for c, w in zip(cells, widths))


def main() -> None:
    args = parse_args()
    parser = argparse.ArgumentParser(
        description="Benchmark TrajectoryStore search latency."
    )
    parser.add_argument(
        "data_dir",
        help="Path to trajectory data directory.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        help=(
            "Named trajectory pattern to benchmark. Can be repeated. "
            "Defaults to all TRAJECTORY_EXPRESSIONS keys."
        ),
    )
    parser.add_argument(
        "--clip-id",
        action="append",
        dest="clip_ids",
        help="Clip ID for FAISS shape search. Can be repeated.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of timed runs per query (default: 3).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write results to this markdown file (optional).",
    )
    args = p.parse_args()

    data_dir = args.data_dir
    if not Path(data_dir).exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    print(f"Loading TrajectoryStore from {data_dir} ...")
    store = TrajectoryStore(data_dir)

    all_ids = set(store.trajectory_data.keys())
    print(f"Loaded {len(all_ids)} clips with trajectory data.")

    patterns = args.patterns or list(TRAJECTORY_EXPRESSIONS.keys())
    clip_ids: List[str] = args.clip_ids or DEFAULT_CLIP_IDS

    # --- Index / safetensors summary ---
    n_safetensors_clips = len(store.trajectory_data)
    index_ntotals = {
        tag: store.features_indexes[tag]["feature_index"].ntotal
        for tag in store.features_indexes
    }
    print(f"Safetensors clips (expression search): {n_safetensors_clips:,}")
    for tag, ntotal in index_ntotals.items():
        print(f"FAISS index rows [{tag}] (shape search): {ntotal:,}")

    headers = ["type", "pattern/clip_id", "seconds", "count"]
    widths = [max(len(h), 10) for h in headers]
    rows_data = []

    # --- Named expression searches ---
    for pattern in patterns:
        if pattern not in TRAJECTORY_EXPRESSIONS:
            print(f"  Warning: unknown pattern {pattern!r}, skipping.")
            continue
        expr = TRAJECTORY_EXPRESSIONS[pattern]
        # Clear cache so we get a real search
        store.searches.clear()
        seconds, count = time_fn(
            lambda e=expr: store._inner_search_trajectory(e, all_ids),
            args.repeat,
        )
        rows_data.append(("expression", pattern, seconds, count))

    # --- Custom speed expression ---
    store.searches.clear()
    seconds, count = time_fn(
        lambda: store._inner_search_trajectory(CUSTOM_SPEED_EXPR, all_ids),
        args.repeat,
    )
    rows_data.append(("expression", "custom_speed_kph>80", seconds, count))

    # --- Cache hit check ---
    t0 = time.perf_counter()
    _ = store._inner_search_trajectory(CUSTOM_SPEED_EXPR, all_ids)
    cache_hit_s = time.perf_counter() - t0
    rows_data.append(("expression (cache)", "custom_speed_kph>80", cache_hit_s, -1))

    # --- FAISS shape search ---
    for clip_id in clip_ids:
        seconds, count = time_fn(
            lambda cid=clip_id: store.search_with_video_clip(cid, None, None),
            args.repeat,
        )
        rows_data.append(("shape", clip_id, seconds, count))

    # --- Print table ---
    rows_str = [
        (t, q, f"{s:.4f}", str(c) if c >= 0 else "cached")
        for t, q, s, c in rows_data
    ]
    for row in rows_str:
        widths = [max(w, len(c)) for w, c in zip(widths, row)]

    print(format_row(headers, widths))
    print(format_row(["-" * len(h) for h in headers], widths))
    for row in rows_str:
        print(format_row(row, widths))

    if args.output:
        md_metadata: dict = {
            "data dir": data_dir,
            "size": path_size(data_dir),
            "safetensors clips": f"{n_safetensors_clips:,}",
        }
        for tag, ntotal in index_ntotals.items():
            md_metadata[f"index rows [{tag}]"] = f"{ntotal:,}"
        write_markdown(
            args.output,
            "Trajectory Search Benchmark",
            headers,
            rows_str,
            metadata=md_metadata,
        )


if __name__ == "__main__":
    main()
