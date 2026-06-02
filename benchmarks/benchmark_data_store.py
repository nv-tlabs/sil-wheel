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

"""Benchmark SQLiteDataStore query performance.

Usage (run from repo root):
    python -m benchmarks.benchmark_data_store --db /path/to/annotations.db \\
        --data-source "AV-V2.1_train" --project "Alpamayo" --runs 5
"""
import argparse
import statistics
from pathlib import Path
from timeit import default_timer as timer
from typing import Callable, Dict, List, Tuple

from benchmarks import path_size, write_markdown
from sil_wheel.stores.search_utils import SearchFilters
from sil_wheel.stores.sqlite_data_store import SQLiteDataStore


def benchmark_fn(fn: Callable, runs: int) -> Tuple[float, float, float, int]:
    """Warmup once, then time `runs` calls. Returns (min, mean, max, result_count)."""
    result = fn()  # warmup — also captures result for counting
    try:
        count = len(result) if result is not None else 0
    except TypeError:
        count = 1
    times = []
    for _ in range(runs):
        t0 = timer()
        fn()
        times.append(timer() - t0)
    return min(times), statistics.mean(times), max(times), count


def format_table(
    rows: List[Tuple[str, float, float, float, int]], title: str
) -> None:
    print(f"\n{title}")
    col_w = max(len(r[0]) for r in rows)
    header = (
        f"{'Query':<{col_w}}  {'Min (s)':>10}  {'Mean (s)':>10}"
        f"  {'Max (s)':>10}  {'Results':>9}"
    )
    print(header)
    print("-" * len(header))
    for name, mn, mean, mx, count in rows:
        print(
            f"{name:<{col_w}}  {mn:>10.6f}  {mean:>10.6f}"
            f"  {mx:>10.6f}  {count:>9}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark SQLiteDataStore query latency."
    )
    parser.add_argument(
        "db",
        help="Path to the annotations SQLite DB file."
    )
    parser.add_argument(
        "--data-source",
        default=None,
        help="Data source name to use in filtered queries.",
    )
    parser.add_argument(
        "--project",
        default="Alpamayo",
        help="Project name used for annotation queries (default: Alpamayo).",
    )
    parser.add_argument(
        "--annotation",
        default=None,
        help="Annotation label key to use in filtered queries.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of timed runs per query (default: 5).",
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

    print(f"Loading SQLiteDataStore from {db_path} ...")
    store = SQLiteDataStore(db_path)

    default_results = store.default_results
    project_source = [args.project]

    # Pick a representative clip id and data source for single-item queries
    sample_clip_ids = list(default_results.keys())[:10]
    single_clip_id = sample_clip_ids[0] if sample_clip_ids else None

    data_source = args.data_source
    if data_source is None and store.data_source_options:
        data_source = store.data_source_options[0]

    annotation = args.annotation
    opts = store.options(project_source)
    if annotation is None:
        annotation = opts[0] if opts else "unknown_label"
    second_annotation = next((o for o in opts if o != annotation), None)
    available_countries = list(store.country_to_clip_ids.keys())
    metric_names = store.metric_names(project_source)

    # Build filter helpers
    def make_filters(query_dict):
        query_dict.setdefault("project_source", [args.project])
        return SearchFilters.from_query(query_dict)

    # --- Define benchmark queries ---
    benchmarks: Dict[str, Callable] = {}

    # 1. Default results (all clips with video paths)
    benchmarks["default_results"] = lambda: store.default_results

    # 2. Data source reverse-index lookup
    if data_source:
        benchmarks["get_clip_ids_for_data_sources"] = (
            lambda ds=data_source: store.get_clip_ids_for_data_sources(
                tuple([ds])
            )
        )

    # 3. search() with data-source filter
    if data_source:
        filters_ds = make_filters({"data_source": [data_source]})
        benchmarks["search_data_source_filter"] = (
            lambda f=filters_ds: store.search(f, default_results)
        )

    # 4. search() with annotation filter
    if annotation:
        filters_ann = make_filters({"filter": [annotation]})
        benchmarks["search_annotation_filter"] = (
            lambda f=filters_ann: store.search(f, default_results)
        )

    # 5. search() with combined filter (data source + annotation)
    if data_source and annotation:
        filters_combined = make_filters(
            {"data_source": [data_source], "filter": [annotation]}
        )
        benchmarks["search_combined_filter"] = (
            lambda f=filters_combined: store.search(f, default_results)
        )

    # 6. get_clips_dict() — hydrate up to 10 clips
    if sample_clip_ids and project_source:
        benchmarks["get_clips_dict_10"] = (
            lambda ids=sample_clip_ids, ps=project_source: store.get_clips_dict(
                ids, ps
            )
        )

    # 7. get() — single clip lookup
    if single_clip_id:
        benchmarks["get_single_clip"] = (
            lambda cid=single_clip_id, ps=project_source: store.get(
                cid, ps
            )
        )

    # 8. get_clip_ids_without_annotations()
    benchmarks["get_clip_ids_without_annotations"] = (
        lambda: store.get_clip_ids_without_annotations()
    )

    # 9. search() with AND filter mode (project_dict_all code path)
    if annotation and second_annotation:
        filters_and = make_filters(
            {
                "filter": [f"{annotation}||{second_annotation}"],
                "filter_mode": ["all"],
            }
        )
        benchmarks["search_annotation_and_filter"] = (
            lambda f=filters_and: store.search(f, default_results)
        )

    # 10. search() with labels_to_exclude (exclude_dict_any code path)
    if annotation:
        filters_excl = make_filters({"labels_to_exclude": [annotation]})
        benchmarks["search_exclude_labels"] = (
            lambda f=filters_excl: store.search(f, default_results)
        )

    # 11. search() with without_ann=True
    filters_without = make_filters({"without_ann": ["true"]})
    benchmarks["search_without_annotations"] = (
        lambda f=filters_without: store.search(f, default_results)
    )

    # 12. get_clips_dict() — hydrate 100 clips
    sample_100 = list(default_results.keys())[:100]
    if len(sample_100) > 10:
        benchmarks["get_clips_dict_100"] = (
            lambda ids=sample_100, ps=project_source: store.get_clips_dict(
                ids, ps
            )
        )

    # 13. options() — label dropdown population on page load
    benchmarks["options_lookup"] = (
        lambda ps=project_source: store.options(ps)
    )

    # 14. search() with country filter (country_to_clip_ids lookup)
    if available_countries:
        country = available_countries[0]
        filters_country = make_filters({"search_country": [country]})
        benchmarks["search_country_filter"] = (
            lambda f=filters_country: store.search(f, default_results)
        )

    # 15. search() with times_filter=True (clips that have time metadata)
    filters_times_with = make_filters({"times": ["true"]})
    benchmarks["search_times_with"] = (
        lambda f=filters_times_with: store.search(f, default_results)
    )

    # 16. search() with times_filter=False (clips without time metadata)
    filters_times_without = make_filters({"times": ["false"]})
    benchmarks["search_times_without"] = (
        lambda f=filters_times_without: store.search(f, default_results)
    )

    # 17. search() with label_types filter (manual annotations only)
    filters_label_type = make_filters({"label_types": ["manual"]})
    benchmarks["search_label_type"] = (
        lambda f=filters_label_type: store.search(f, default_results)
    )

    # 18. search() with search_clipid (single clip ID lookup)
    if single_clip_id:
        filters_clipid = make_filters({"search_clipid": [single_clip_id]})
        benchmarks["search_clipid"] = (
            lambda f=filters_clipid: store.search(f, default_results)
        )

    # 19. search() with numeric_filter (range-based, project_starmap code path)
    if metric_names:
        metric = metric_names[0]
        filters_numeric = make_filters(
            {"numeric_filter": [f"{metric},0,1,asc"]}
        )
        benchmarks["search_numeric_filter"] = (
            lambda f=filters_numeric: store.search(f, default_results)
        )

    # 20. search() multi-label OR (3+ labels, stresses project_dict_any)
    if len(opts) >= 3:
        multi_label = "||".join(opts[:3])
        filters_multi_or = make_filters({"filter": [multi_label]})
        benchmarks["search_annotation_multi_or"] = (
            lambda f=filters_multi_or: store.search(f, default_results)
        )

    # 21. search() annotation + country stacked (two independent index lookups)
    if annotation and available_countries:
        filters_ann_country = make_filters(
            {"filter": [annotation], "search_country": [available_countries[0]]}
        )
        benchmarks["search_annotation_and_country"] = (
            lambda f=filters_ann_country: store.search(f, default_results)
        )

    # 22. search() annotation + exclude stacked (filter in, then filter out)
    if annotation and second_annotation and second_annotation != annotation:
        filters_ann_excl = make_filters(
            {"filter": [annotation], "labels_to_exclude": [second_annotation]}
        )
        benchmarks["search_annotation_and_exclude"] = (
            lambda f=filters_ann_excl: store.search(f, default_results)
        )

    # --- Run benchmarks ---
    rows = []
    for name, fn in benchmarks.items():
        mn, mean, mx, count = benchmark_fn(fn, args.runs)
        rows.append((name, mn, mean, mx, count))
        print(f"  {name}: mean={mean:.4f}s  results={count}")

    format_table(rows, f"Benchmark Results ({args.runs} runs each):")

    if args.output:
        headers = ["Query", "Min (s)", "Mean (s)", "Max (s)", "Results"]
        rows_str = [
            (name, f"{mn:.6f}", f"{mean:.6f}", f"{mx:.6f}", str(count))
            for name, mn, mean, mx, count in rows
        ]
        write_markdown(
            args.output,
            "Data Store Benchmark",
            headers,
            rows_str,
            metadata={"db": db_path, "db size": path_size(db_path)},
        )

    # --- DB stats ---
    store._get_table_stats()


if __name__ == "__main__":
    main()
