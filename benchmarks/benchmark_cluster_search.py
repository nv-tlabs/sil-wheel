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

"""Benchmark the cluster_clips_and_select pipeline end-to-end.

Calls cluster_clips_and_select.main() directly with varying parameter
combinations to measure total wall-clock time per configuration.

Usage (run from repo root):
    python -m benchmarks.benchmark_cluster_search \\
        /path/to/embeddings \\
        --n-clusters 100 500 1000 \\
        --max-clips 10000 50000 \\
        --n-iter 20 \\
        --n-redo 1
"""
import argparse
import json
import os
import pickle
import random
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Tuple

import psutil

from cluster_clips_and_select import main as cluster_clips_main, spec_to_tag
from benchmarks import path_size, write_markdown


def load_clip_to_index(path_to_embeddings: str) -> dict:
    base = Path(path_to_embeddings)
    tag = spec_to_tag("IVF4096,PQ96x8")
    path_to_clip_index = base / f"cosmos_clip_to_index_{tag}.pkl"
    if not path_to_clip_index.exists():
        raise SystemExit(f"clip_to_index not found: {path_to_clip_index}")
    with open(path_to_clip_index, "rb") as f:
        clip_to_index = pickle.load(f)
    print(f"clip_to_index: {len(clip_to_index)} entries")
    return clip_to_index


def make_clip_ids_file(
    clip_to_index: dict, max_clips: int, seed: int, tmp_dir: str
) -> str:
    """Write a sampled clip_ids JSON and return its path."""
    all_ids = list(clip_to_index.keys())
    sampled = random.Random(seed).sample(all_ids, min(max_clips, len(all_ids)))
    path = Path(tmp_dir) / f"clip_ids_{len(sampled)}.json"
    with open(path, "w") as f:
        json.dump(sampled, f)
    return str(path)


class PeakMemoryMonitor:
    """Poll process RSS from a background thread to capture peak memory."""

    def __init__(self, interval: float = 0.05):
        self._interval = interval
        self._proc = psutil.Process(os.getpid())
        self._peak_bytes = 0
        self._stop = threading.Event()
        self._thread = None

    def _monitor(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                rss = self._proc.memory_info().rss
                if rss > self._peak_bytes:
                    self._peak_bytes = rss
            except psutil.NoSuchProcess:
                break

    def __enter__(self):
        self._peak_bytes = self._proc.memory_info().rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()

    @property
    def peak_gb(self) -> float:
        return self._peak_bytes / 1024**3


def run_full_pipeline(
    path_to_embeddings: str,
    clip_ids_file: str,
    n_clusters: int,
    n_iter: int,
    n_redo: int,
    spherical: bool,
    max_points_per_centroid: int,
    seed: int,
    output_dir: str,
) -> Tuple[float, float]:
    """Call cluster_clips_and_select.main() and return (total_sec, peak_mem_gb)."""
    argv = [
        output_dir,
        path_to_embeddings,
        str(n_clusters),
        "--path_to_clip_ids", clip_ids_file,
        "--n_iter", str(n_iter),
        "--n_redo", str(n_redo),
        "--seed", str(seed),
        "--max_points_per_centroid", str(max_points_per_centroid),
    ]
    if spherical:
        argv.append("--spherical_kmeans")

    with PeakMemoryMonitor() as mem:
        t0 = time.perf_counter()
        cluster_clips_main(argv)
        total_sec = time.perf_counter() - t0

    return total_sec, mem.peak_gb


def format_row(cells: List[str], widths: List[int]) -> str:
    return " | ".join(c.ljust(w) for c, w in zip(cells, widths))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark cluster_clips_and_select.py end-to-end across "
            "different parameter combinations."
        )
    )
    p.add_argument(
        "path_to_embeddings",
        help="Directory containing the Cosmos FAISS index and clip_to_index pickle.",
    )
    p.add_argument(
        "--n-clusters",
        type=int,
        nargs="+",
        default=[100, 500],
        metavar="K",
        help="Number of clusters to benchmark (default: 100 500).",
    )
    p.add_argument(
        "--max-clips",
        type=int,
        nargs="+",
        default=[50_000],
        metavar="N",
        help="Number of clips to cluster (default: 50000). "
        "Multiple values produce a cartesian product with --n-clusters.",
    )
    p.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="K-means iterations (default: 20).",
    )
    p.add_argument(
        "--n-redo",
        type=int,
        default=1,
        help="K-means restarts / nredo (default: 1).",
    )
    p.add_argument(
        "--spherical",
        action="store_true",
        help="Use spherical K-means (passes --spherical_kmeans to the script).",
    )
    p.add_argument(
        "--max-points-per-centroid",
        type=int,
        default=256,
        help="FAISS max_points_per_centroid (default: 256).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Write results to this markdown file (optional).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    clip_to_index = load_clip_to_index(args.path_to_embeddings)

    headers = [
        "max_clips",
        "n_clusters",
        "n_iter",
        "n_redo",
        "spherical",
        "max_pts_per_centroid",
        "total_sec",
        "peak_mem_gb",
    ]
    widths = [max(len(h), 12) for h in headers]
    rows_str = []

    with tempfile.TemporaryDirectory() as tmp_root:
        # Generate clip_ids files once per unique max_clips value
        clip_ids_files = {
            max_clips: make_clip_ids_file(
                clip_to_index, max_clips, args.seed, tmp_root
            )
            for max_clips in sorted(set(args.max_clips))
        }

        for max_clips in sorted(set(args.max_clips)):
            clip_ids_file = clip_ids_files[max_clips]
            actual_clips = min(max_clips, len(clip_to_index))

            for n_clusters in args.n_clusters:
                print(
                    f"\n--- max_clips={actual_clips}, n_clusters={n_clusters}, "
                    f"n_iter={args.n_iter}, n_redo={args.n_redo}, "
                    f"spherical={args.spherical}, "
                    f"max_pts_per_centroid={args.max_points_per_centroid} ---"
                )
                with tempfile.TemporaryDirectory() as out_dir:
                    total_sec, peak_mem_gb = run_full_pipeline(
                        path_to_embeddings=args.path_to_embeddings,
                        clip_ids_file=clip_ids_file,
                        n_clusters=n_clusters,
                        n_iter=args.n_iter,
                        n_redo=args.n_redo,
                        spherical=args.spherical,
                        max_points_per_centroid=args.max_points_per_centroid,
                        seed=args.seed,
                        output_dir=out_dir,
                    )

                print(f"Total time: {total_sec:.3f}s  Peak RSS: {peak_mem_gb:.3f} GB")

                row = (
                    str(actual_clips),
                    str(n_clusters),
                    str(args.n_iter),
                    str(args.n_redo),
                    str(args.spherical),
                    str(args.max_points_per_centroid),
                    f"{total_sec:.3f}",
                    f"{peak_mem_gb:.3f}",
                )
                rows_str.append(row)
                for i, cell in enumerate(row):
                    widths[i] = max(widths[i], len(cell))

    print()
    print(format_row(headers, widths))
    print(format_row(["-" * w for w in widths], widths))
    for row in rows_str:
        print(format_row(row, widths))

    if args.output:
        write_markdown(
            args.output,
            "Cluster Search Benchmark",
            headers,
            rows_str,
            metadata={
                "embeddings dir": args.path_to_embeddings,
                "size": path_size(args.path_to_embeddings),
            },
        )


if __name__ == "__main__":
    main()
