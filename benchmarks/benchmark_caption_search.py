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
import random
import sqlite3
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


from benchmarks import path_size, write_markdown
from sil_wheel.stores.sqlite_caption_store import FTSCaptionStore


def list_distinct_data_sources(
    db_path: str, limit: int = 50
) -> List[Tuple[str, int]]:
    """Return up to `limit` distinct data_sources with rough counts (desc)."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            """
            SELECT data_source, COUNT(*) as cnt
            FROM captions
            WHERE data_source IS NOT NULL AND data_source != ''
            GROUP BY data_source
            ORDER BY cnt DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return [(r[0], int(r[1])) for r in rows]
    finally:
        conn.close()


def time_once(
    store: FTSCaptionStore,
    query: str,
    data_sources: Optional[List[str]],
    limit: int,
) -> Tuple[float, int]:
    """Time a single _inner_search run, returning (seconds, result_count)."""
    t0 = time.perf_counter()
    clip_ids = store._inner_search(
        query, limit=limit, data_sources=data_sources
    )
    dt_s = time.perf_counter() - t0
    return dt_s, len(clip_ids)


def time_multi(
    store: FTSCaptionStore,
    queries: List[str],
    data_sources: Optional[List[str]],
    limit: int,
) -> Tuple[float, int]:
    """Time a multi-query OR search (rewrite-style union), returning (seconds, result_count)."""
    t0 = time.perf_counter()
    clip_ids = store._inner_search(
        queries, limit=limit, data_sources=data_sources
    )
    dt_s = time.perf_counter() - t0
    return dt_s, len(clip_ids)


# Representative rewrite bundles — each tuple mimics what the query rewriter
# would produce for a single user query.
DEFAULT_REWRITE_BUNDLES = [
    (
        "pedestrian crossing",
        [
            "pedestrian crossing",
            "person crossing",
            "crossing pedestrian",
            "street crossing",
            "walking across",
        ],
    ),
    (
        "sharp left turn",
        [
            "sharp left turn",
            "left turn",
            "turning left",
            "hard left",
            "steering left",
        ],
    ),
    (
        "highway driving",
        ["highway driving", "motorway", "freeway", "high speed", "expressway"],
    ),
    (
        "emergency vehicle",
        ["emergency vehicle", "ambulance", "fire truck", "police car", "siren"],
    ),
    (
        "construction zone",
        [
            "construction zone",
            "road works",
            "lane shift",
            "work zone",
            "road construction",
        ],
    ),
]

DEFAULT_QUERIES = [
    # Common words (high hit-count)
    "turn",
    "brake",
    "stop",
    "intersection",
    "pedestrian",
    "highway",
    # Phrases (quoted for FTS5 exact phrase)
    '"red light"',
    '"left turn"',
    '"4-way stop intersection"',
    '"nudge left"',
    # Rare / low-frequency words
    "nudge",
    "sway",
    "fishtail",
    "hydroplane",
    "jackknife",
    "chicane",
    "pothole",
    "glare",
    "tailgating",
    "contraflow",
]


def format_row(cells: Iterable[str], widths: List[int]) -> str:
    return " | ".join(c.ljust(w) for c, w in zip(cells, widths))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FTS5 caption searches with and without dataset filters."
        )
    )
    parser.add_argument("db", help="Path to the captions SQLite DB file.")
    parser.add_argument(
        "--query",
        action="append",
        help=(
            "Query string for FTS MATCH. Can be repeated. "
            "If omitted, uses a default set."
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help=(
            "Dataset/data_source value to filter by. Can be repeated. "
            "If omitted, prints top options and benchmarks only unfiltered."
        ),
    )
    parser.add_argument(
        "--dataset_combo",
        action="append",
        nargs="+",
        help=(
            "Two or more data_source values to filter by simultaneously (OR logic), "
            "benchmarked as a single combined filter. Can be repeated for multiple "
            "combos, e.g. --dataset_combo DS1 DS2 --dataset_combo DS3 DS4."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Result limit used in queries (default: 5000).",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List top data_sources in the DB and exit.",
    )
    parser.add_argument(
        "--top-datasets",
        type=int,
        default=0,
        help=(
            "If set > 0 and --dataset not provided, randomly sample N datasets "
            "to include in filtered benchmarks."
        ),
    )
    parser.add_argument(
        "--rewrite-bundle",
        action="append",
        nargs="+",
        help=(
            "Benchmark multi-query union search (rewriter output simulation). "
            "Pass the label as the first token and rewrites as the rest, e.g. "
            "--rewrite-bundle 'pedestrian' 'person crossing' 'street crossing'. "
            "Can be repeated. Defaults to a built-in set if omitted."
        ),
    )
    parser.add_argument(
        "--sweep-limits",
        action="store_true",
        help=(
            "Run each query across multiple limit values and report how "
            "latency and result count scale with limit."
        ),
    )
    parser.add_argument(
        "--limits",
        nargs="+",
        type=int,
        default=[5_000, 10_000, 30_000, 50_000, 100_000, 200_000, 500_000],
        help=(
            "Limit values to sweep when --sweep-limits is set "
            "(default: 5000 10000 30000 50000 100000 200000 500000)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write results to this markdown file (optional).",
    )
    args = parser.parse_args()

    db_path = args.db
    if not Path(db_path).exists():
        raise SystemExit(f"DB file not found: {db_path}")

    if args.list_datasets:
        top = list_distinct_data_sources(db_path, limit=100)
        print("Top data_source values:")
        for ds, cnt in top:
            print(f"  {ds}  ({cnt:,} rows)")
        return

    queries = args.query or DEFAULT_QUERIES

    # datasets_to_benchmark: None (unfiltered) and each provided dataset as a
    # single-element list
    datasets_to_benchmark: List[Optional[List[str]]] = [None]
    ds_labels: List[str] = ["<all>"]
    datasets = args.dataset
    if not datasets and args.top_datasets > 0:
        all_ds = list_distinct_data_sources(db_path, limit=10)
        all_names = [ds for ds, _ in all_ds]
        datasets = random.sample(
            all_names, min(args.top_datasets, len(all_names))
        )
        print(f"Randomly chose {datasets}")

    if datasets:
        for ds in datasets:
            if ds is None or ds == "":
                continue
            datasets_to_benchmark.append([ds])
            ds_labels.append(ds)

    if args.dataset_combo:
        for combo in args.dataset_combo:
            combo = [ds for ds in combo if ds]
            if not combo:
                continue
            datasets_to_benchmark.append(combo)
            ds_labels.append(" + ".join(combo))

    store = FTSCaptionStore(db_path)

    # Warmup (use a dedicated query so the cache is not seeded for benchmarked queries)
    time_once(store, "warmup", None, args.limit)

    # Print header
    headers = ["query", "dataset", "seconds", "count", "limit"]
    widths = [max(len(h), 10) for h in headers]
    print(format_row(headers, widths))
    print(format_row(["-" * len(h) for h in headers], widths))

    all_rows = []
    for q in queries:
        for ds_label, ds in zip(ds_labels, datasets_to_benchmark):
            seconds, count = time_once(store, q, ds, args.limit)
            row = [
                q,
                ds_label,
                f"{seconds:.3f}",
                str(count),
                str(args.limit),
            ]
            widths = [max(w, len(c)) for w, c in zip(widths, row)]
            print(format_row(row, widths))
            all_rows.append(tuple(row))

    rewrite_bundles = args.rewrite_bundle or None
    if rewrite_bundles is None:
        bundles = DEFAULT_REWRITE_BUNDLES
    else:
        # First token is the label; the rest are the rewrite queries.
        bundles = [(b[0], b) for b in rewrite_bundles if b]

    if bundles:
        print()
        rewrite_headers = [
            "label",
            "rewrites",
            "n_rewrites",
            "dataset",
            "base_secs",
            "base_count",
            "rewrite_secs",
            "rewrite_count",
        ]
        rewrite_widths = [max(len(h), 10) for h in rewrite_headers]
        rewrite_rows = []

        print(format_row(rewrite_headers, rewrite_widths))
        print(
            format_row(["-" * len(h) for h in rewrite_headers], rewrite_widths)
        )

        for label, bundle_queries in bundles:
            rewrites_str = r" \| ".join(bundle_queries)
            for ds_label, ds in zip(ds_labels, datasets_to_benchmark):
                base_secs, base_count = time_once(store, label, ds, args.limit)
                rw_secs, rw_count = time_multi(
                    store, list(bundle_queries), ds, args.limit
                )
                row = [
                    label,
                    rewrites_str,
                    str(len(bundle_queries)),
                    ds_label,
                    f"{base_secs:.3f}",
                    str(base_count),
                    f"{rw_secs:.3f}",
                    str(rw_count),
                ]
                rewrite_widths = [
                    max(w, len(c)) for w, c in zip(rewrite_widths, row)
                ]
                print(format_row(row, rewrite_widths))
                rewrite_rows.append(tuple(row))

    sweep_rows = []
    if args.sweep_limits:
        print()
        sweep_headers = ["query", "dataset", "limit", "seconds", "count"]
        sweep_widths = [max(len(h), 10) for h in sweep_headers]

        print(format_row(sweep_headers, sweep_widths))
        print(format_row(["-" * len(h) for h in sweep_headers], sweep_widths))

        for q in queries:
            for ds_label, ds in zip(ds_labels, datasets_to_benchmark):
                for lim in sorted(args.limits):
                    store.searches.clear()
                    seconds, count = time_once(store, q, ds, lim)
                    row = [q, ds_label, str(lim), f"{seconds:.3f}", str(count)]
                    sweep_widths = [
                        max(w, len(c)) for w, c in zip(sweep_widths, row)
                    ]
                    print(format_row(row, sweep_widths))
                    sweep_rows.append(tuple(row))

    if args.output:
        write_markdown(
            args.output,
            "Caption Search Benchmark",
            headers,
            all_rows,
            metadata={"db": db_path, "db size": path_size(db_path)},
        )
        if bundles:
            write_markdown(
                args.output,
                "Caption Rewrite-Bundle Benchmark",
                rewrite_headers,
                rewrite_rows,
                metadata={
                    "db": db_path,
                    "db size": path_size(db_path),
                    "rewrites column": "queries in each bundle separated by \\|",
                },
                append=True,
            )
        if sweep_rows:
            write_markdown(
                args.output,
                "Caption Search Limit-Sweep Benchmark",
                sweep_headers,
                sweep_rows,
                metadata={"db": db_path, "db size": path_size(db_path)},
                append=True,
            )


if __name__ == "__main__":
    main()
