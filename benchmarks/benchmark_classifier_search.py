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

"""Benchmark ClassifierSearch training and expression-based filter latency.

Benchmarks three expression types per label:
  - greater-than : p > 0.5
  - less-than    : p < 0.5
  - range        : 0.3 < p < 0.7

For each expression, reports one cold run (empty filter cache) and one
immediate warm run (filter cache populated by the cold run).
Both cosmos and caption embed types are benchmarked automatically.

If --database is provided, training time is also benchmarked by exporting
annotations from the DB and calling train_classifier.py as a subprocess,
mirroring how the server triggers training.

Usage (run from repo root):
    python -m benchmarks.benchmark_classifier_search \\
        /path/to/classifiers \\
        /path/to/cosmos_embeddings \\
        /path/to/caption_embeddings \\
        --output benchmarks/classifier_search.md

    # With training benchmark:
    python -m benchmarks.benchmark_classifier_search \\
        /path/to/classifiers \\
        /path/to/cosmos_embeddings \\
        /path/to/caption_embeddings \\
        --database /path/to/annotations.db \\
        --output benchmarks/classifier_search.md
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, List

from benchmarks import path_size, write_markdown
from sil_wheel.stores.classifier_search import ClassifierSearch
from sil_wheel.stores.sqlite_data_store import SQLiteDataStore

_THRESHOLD = 0.5


def format_row(cells: Iterable[str], widths: List[int]) -> str:
    return " | ".join(c.ljust(w) for c, w in zip(cells, widths))


def _make_expressions():
    t = _THRESHOLD
    lo = round(max(0.0, t - 0.2), 4)
    hi = round(min(1.0, t + 0.2), 4)
    return {
        "gt": f"p > {t}",
        "lt": f"p < {t}",
        "range": f"{lo} < p < {hi}",
    }


def _benchmark_training(
    classifier_dir: str,
    label: str,
    embed_dir: str,
    datastore: SQLiteDataStore,
    n_negative_samples: int,
    embed_type: str,
) -> float:
    """Export annotations from the DB and run train_classifier.py, returning
    elapsed wall time in seconds."""
    train_script = str(
        Path(__file__).resolve().parents[2] / "train_classifier.py"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False, suffix=".json"
    ) as tmp_fp:
        datastore.export_to_json(tmp_fp, keys=[label])
        path_to_annotations = tmp_fp.name

    cmd = [
        sys.executable,
        train_script,
        classifier_dir,
        path_to_annotations,
        label,
        embed_dir,
        "--n_negative_samples",
        str(n_negative_samples),
        "--embed_type", embed_type,
    ]
    try:
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True)
        return time.perf_counter() - t0
    finally:
        os.unlink(path_to_annotations)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ClassifierSearch training and filter_clips latency."
    )
    parser.add_argument(
        "classifier_dir",
        help="Root classifier directory (contains cosmos/ and caption/ subdirs).",
    )
    parser.add_argument(
        "cosmos_embed_dir",
        help="Path to cosmos embeddings directory.",
    )
    parser.add_argument(
        "caption_embed_dir",
        help="Path to caption embeddings directory.",
    )
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        help="Label name to benchmark. Can be repeated.",
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Path to SQLite annotations DB. Required for training benchmark.",
    )
    parser.add_argument(
        "--n-negative-samples",
        type=int,
        default=100,
        help="Negative samples passed to train_classifier.py (default: 100).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write results to this markdown file (optional).",
    )
    args = parser.parse_args()
    classifier_dir = args.classifier_dir
    if not Path(classifier_dir).exists():
        raise SystemExit(f"Classifier dir not found: {classifier_dir}")

    embed_dirs = {"cosmos": args.cosmos_embed_dir, "caption": args.caption_embed_dir}
    run_training = args.database is not None
    store = ClassifierSearch(classifier_dir)
    embed_types = ["cosmos", "caption"]

    # Build {embed_type: [labels]} — use --label for all types if provided,
    # otherwise auto-detect from each embed_type subdirectory.
    labels_by_type = {}
    for embed_type in embed_types:
        if args.labels:
            labels_by_type[embed_type] = args.labels
        else:
            subdir = Path(classifier_dir) / embed_type
            labels_by_type[embed_type] = [
                d for d in os.listdir(subdir)
                if (subdir / d / "predicted_scores.json").exists()
            ] if subdir.exists() else []

    # --- Training benchmark (sample up to 3 labels per embed type) ---
    train_rows_data = []
    if run_training:
        import random
        datastore = SQLiteDataStore(args.database)
        for embed_type in embed_types:
            labels = labels_by_type[embed_type]
            if not labels:
                continue
            train_labels = random.sample(labels, min(3, len(labels)))
            print(f"Benchmarking training ({embed_type}) on {train_labels}...")
            for label in train_labels:
                print(f"  Training {label!r}...")
                elapsed = _benchmark_training(
                    classifier_dir,
                    label,
                    embed_dirs[embed_type],
                    datastore,
                    args.n_negative_samples,
                    embed_type,
                )
                train_rows_data.append((embed_type, label, elapsed))
                store.invalidate_cache(embed_type, label)

    # --- Search benchmark ---
    expressions = _make_expressions()
    search_headers = ["embed_type", "label", "expression", "cold (s)", "warm (s)", "count"]
    search_widths = [max(len(h), 10) for h in search_headers]
    search_rows_data = []

    for embed_type in embed_types:
        for label in labels_by_type[embed_type]:
            clip_ids, _ = store.load_scores(embed_type, label)
            if clip_ids is None:
                print(f"  Skipping {embed_type}/{label!r}: no scores found.")
                continue
            all_ids = clip_ids.tolist()

            for expr in expressions.values():
                store.invalidate_cache(embed_type, label)
                fn = lambda l=label, e=expr, ids=all_ids, et=embed_type: store.filter_clips(ids, l, e, et)

                t0 = time.perf_counter()
                result = fn()
                cold_s = time.perf_counter() - t0
                count = len(result)

                t0 = time.perf_counter()
                fn()
                warm_s = time.perf_counter() - t0

                search_rows_data.append((embed_type, label, expr, cold_s, warm_s, count))

    # --- Print training table ---
    if train_rows_data:
        train_headers = ["embed_type", "label", "train time (s)"]
        train_widths = [max(len(h), 10) for h in train_headers]
        train_rows_str = [(et, lbl, f"{t:.4f}") for et, lbl, t in train_rows_data]
        for row in train_rows_str:
            train_widths = [max(w, len(c)) for w, c in zip(train_widths, row)]

        print("\nTraining")
        print(format_row(train_headers, train_widths))
        print(format_row(["-" * len(h) for h in train_headers], train_widths))
        for row in train_rows_str:
            print(format_row(row, train_widths))

    # --- Print search table ---
    search_rows_str = [
        (et, lbl, expr, f"{cold:.4f}", f"{warm:.4f}", str(c))
        for et, lbl, expr, cold, warm, c in search_rows_data
    ]
    for row in search_rows_str:
        search_widths = [max(w, len(c)) for w, c in zip(search_widths, row)]

    print("\nSearch")
    print(format_row(search_headers, search_widths))
    print(format_row(["-" * len(h) for h in search_headers], search_widths))
    for row in search_rows_str:
        print(format_row(row, search_widths))

    # --- Write markdown ---
    if args.output:
        metadata = {
            "classifier dir": classifier_dir,
            "size": path_size(classifier_dir),
        }
        if train_rows_data:
            write_markdown(
                args.output,
                "Classifier Training Benchmark",
                train_headers,
                train_rows_str,
                metadata=metadata,
            )
            write_markdown(
                args.output,
                "Classifier Search Benchmark",
                search_headers,
                search_rows_str,
                metadata=metadata,
                append=True,
            )
        else:
            write_markdown(
                args.output,
                "Classifier Search Benchmark",
                search_headers,
                search_rows_str,
                metadata=metadata,
            )


if __name__ == "__main__":
    main()
