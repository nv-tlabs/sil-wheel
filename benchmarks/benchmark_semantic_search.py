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

"""Benchmark CosmosEmbeddingsStore.search() for text→video and clip→video retrieval.

Usage (run from repo root):
    python -m benchmarks.benchmark_semantic_search \\
        /path/to/embeddings \\
        --query "pedestrian crossing" \\
        --clip-id <clip_id> \\
        --ks 1024 4096 8192 16384
"""
import argparse
import time
from pathlib import Path
from typing import Iterable, List, Tuple

from sil_wheel.stores.cosmos_embeddings_store import CosmosEmbeddingsStore, INDEX_SPEC
from sil_wheel.stores.search_utils import SearchFilters, SearchResults
from benchmarks import path_size, write_markdown

DEFAULT_CLIP_IDS = [
    "cdcfb35f-0031-4e41-8d43-8c729ccf6326",
    "ee76a44e-0087-4afd-be52-401eab2205ae",
    "a1b2c3d4-e5f6-7890-abcd-ef1234567890",  # arbitrary placeholder
]

DEFAULT_TEXT_QUERIES = [
    # Maneuvers
    "pedestrian crossing",
    "sharp left turn",
    "sharp right turn",
    "lane change on highway",
    "u-turn in residential area",
    "parallel parking",
    "merging onto highway",
    "roundabout navigation",
    # Traffic and road conditions
    "red traffic light",
    "stop sign",
    "construction zone with lane shifts",
    "road works ahead",
    # Environment and weather
    "highway driving",
    "rain",
    "heavy rain reduces visibility",
    "night driving",
    "fog",
    "snow",
    # Road users
    "cyclist on the road",
    "emergency vehicle with sirens",
    "school zone with children",
    # Ego-vehicle events
    "hard braking",
    "stop and go traffic",
    # Rare / edge cases
    "vehicle driving the wrong way",
    "animal crossing the road",
    "debris on the road",
    "flooded road",
    "fallen tree blocking the road",
    "car accident",
    "level crossing with train",
    "tunnel entrance",
    "icy road",
    "glare from the sun",
]


def score_stats(result: dict, attr: str) -> Tuple[int, float, float]:
    scores = [
        getattr(r, attr) for r in result.values() if getattr(r, attr) is not None
    ]
    if not scores:
        return 0, float("nan"), float("nan")
    return len(scores), max(scores), min(scores)


def format_row(cells: Iterable[str], widths: List[int]) -> str:
    return " | ".join(c.ljust(w) for c, w in zip(cells, widths))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark CosmosEmbeddingsStore.search() latency."
    )
    p.add_argument(
        "embeddings_dir",
        help="Directory containing Cosmos embedding parquet files and FAISS index.",
    )
    p.add_argument(
        "--query",
        action="append",
        dest="queries",
        help=(
            "Text query to benchmark. Can be repeated. "
            "Defaults to a built-in set if omitted."
        ),
    )
    p.add_argument(
        "--clip-id",
        action="append",
        dest="clip_ids",
        help=(
            "Clip ID for video→video search. Can be repeated. "
            "Defaults to a built-in set if omitted."
        ),
    )
    p.add_argument(
        "--index-spec",
        default="IVF4096,PQ96x8",
        choices=list(INDEX_SPEC.keys()),
        help="FAISS index spec to load (default: IVF4096,PQ96x8).",
    )
    p.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384],
        metavar="K",
        help="FAISS neighbor counts to sweep (default: 1024 2048 4096 8192 16384).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Write results to this markdown file (optional).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    embeddings_dir = args.embeddings_dir
    if not Path(embeddings_dir).exists():
        raise SystemExit(f"Embeddings directory not found: {embeddings_dir}")

    print(f"Loading CosmosEmbeddingsStore from {embeddings_dir} ...")
    store = CosmosEmbeddingsStore(embeddings_dir, args.index_spec)

    # Build current_results from every clip in the index — mirrors what the
    # server does: search() intersects the FAISS top-k with the caller's set.
    current_results = {
        clip_id: SearchResults.default for clip_id in store.clips_to_index
    }
    print(f"Index contains {len(current_results):,} clips.")

    text_queries: List[str] = args.queries or DEFAULT_TEXT_QUERIES
    clip_ids: List[str] = args.clip_ids or DEFAULT_CLIP_IDS

    ks: List[int] = sorted(set(args.ks))

    # Warm up the embedding model once before any timing.
    if text_queries:
        store.search(
            SearchFilters.from_query({"semantic_search_text": [text_queries[0]]}),
            current_results,
            k=ks[0],
        )
        store.searches.clear()

    headers = ["type", "query / clip_id", "k", "seconds", "count", "max_score", "min_score"]
    widths = [max(len(h), 10) for h in headers]
    rows_data = []

    for k in ks:
        print(f"\n--- k={k} ---")
        # --- Text queries ---
        for q in text_queries:
            filters = SearchFilters.from_query({"semantic_search_text": [q]})
            store.searches.clear()
            t0 = time.perf_counter()
            result = store.search(filters, current_results, k=k)
            seconds = time.perf_counter() - t0
            count, max_score, min_score = score_stats(result, "semantic_search_text_score")
            rows_data.append(("text→video", q, k, seconds, count, max_score, min_score))

        # --- Clip→video queries ---
        for clip_id in clip_ids:
            if not store.has_embeddings(clip_id):
                print(f"  Warning: clip_id {clip_id!r} not in index, skipping.")
                continue
            filters = SearchFilters.from_query({"semantic_search_clipid": [clip_id]})
            store.searches.clear()
            t0 = time.perf_counter()
            result = store.search(filters, current_results, k=k)
            seconds = time.perf_counter() - t0
            count, max_score, min_score = score_stats(result, "semantic_search_clip_score")
            rows_data.append(("clip→video", clip_id, k, seconds, count, max_score, min_score))

    # --- Print table ---
    rows_str = [
        (t, q, str(k), f"{s:.4f}", str(c), f"{mx:.4f}", f"{mn:.4f}")
        for t, q, k, s, c, mx, mn in rows_data
    ]
    for row in rows_str:
        widths = [max(w, len(c)) for w, c in zip(widths, row)]

    print(format_row(headers, widths))
    print(format_row(["-" * len(h) for h in headers], widths))
    for row in rows_str:
        print(format_row(row, widths))

    if args.output:
        write_markdown(
            args.output,
            "Semantic Search Benchmark",
            headers,
            rows_str,
            metadata={
                "embeddings dir": embeddings_dir,
                "size": path_size(embeddings_dir),
                "index spec": args.index_spec,
                "ks": " ".join(str(k) for k in ks),
                "index size": str(len(current_results)),
            },
        )


if __name__ == "__main__":
    main()
