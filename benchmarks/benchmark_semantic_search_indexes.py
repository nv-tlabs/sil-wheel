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

"""Benchmark semantic search (text→video) across multiple FAISS index specs
and data slices.

Measures query time, result count, max_score, min_score, precision, recall
and precision-recall curves at different thresholds for every combination of:
  - FAISS index spec (same set as benchmark_faiss_indexes_cosmos_embed.py)
  - Data slice: full index, per data_source (e.g. MADS / Nexar),
    annotation-based subsets (e.g. night scenes, people holding signs),
    and their pairwise intersections.

Usage (run from repo root):
    python -m benchmarks.benchmark_semantic_search_indexes EMBEDDINGS_DIR \\
        --annotations-db /path/to/annotations.db \\
        --output results.md

    # Custom annotation slices
    python -m benchmarks.benchmark_semantic_search_indexes EMBEDDINGS_DIR \\
        --annotations-db /path/to/annotations.db \\
        --annotation-slices 'night:%night%' 'sign:%sign%' 'rain:%rain%' \\
        --output results.md
"""
import argparse
import gc
import math
import pickle
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np

from benchmarks import path_size, write_markdown
from sil_wheel.stores.cosmos_embeddings_store import CosmosEmbeddingsStore


INDEX_CONFIGS = [
    # (spec, tag, nprobe)
    ("FLAT",                           "flat",                       256),
    ("IVF4096,PQ96x8",                 "ivf4096_pq96x8",             256),
    ("OPQ96_768,IVF4096,PQ96x8",       "opq96_768_ivf4096_pq96x8",  256),
    ("IVF8192,PQ96x8",                 "ivf8192_pq96x8",             512),
    ("IVF8192,PQ96x8",                 "ivf8192_pq96x8_np1024",      1024),
    ("IVF8192,PQ96x8",                 "ivf8192_pq96x8_np2048",      2048),
    ("OPQ96_768,IVF8192,PQ96x8",       "opq96_768_ivf8192_pq96x8",  512),
    ("IVF16384,PQ96x8",                "ivf16384_pq96x8",            1024),
    ("OPQ96_768,IVF16384,PQ96x8",      "opq96_768_ivf16384_pq96x8", 1024),
    ("HNSW32",                         "hnsw32",                     128),
    ("HNSW64",                         "hnsw64",                     256),
]

DEFAULT_QUERIES = [
    # Pedestrians & vulnerable road users
    "pedestrian crossing the street",
    "wheelchair user crossing road",
    "bicyclist passing by parked cars",
    "child near a crosswalk",
    "dog on the sidewalk",
    # Traffic scenarios
    "car turning left at an intersection",
    "traffic cones on the street",
    "construction zone with cones and barriers",
    # Environment / time of day
    "nighttime street with headlights",
    "rainy day pedestrians with umbrellas",
    "night",
    # Edge-case / AV-specific
    "officer holding a sign",
    "officer signaling to stop",
    "officer waving to go",
]

DEFAULT_DATA_SOURCES = ["MADS", "Nexar"]

PR_CURVE_KS = [10, 50, 100, 200, 500, 1000, 4096]

# (slice_name, SQL LIKE pattern against annotations.key)
DEFAULT_ANNOTATION_SLICES = [
    ("night", "%night%"),
    ("sign", "%sign%"),
]


@dataclass
class DataSlice:
    name: str
    description: str
    clip_ids: Set[str]


def verify_shared_clip_ids(
    emb_dir: Path,
    configs: List[Tuple[str, str, int]],
) -> Set[str]:
    """Load each index's clip_to_index pkl and verify they all share the same clip IDs.

    Raises SystemExit if any mismatch is found. Returns the shared clip ID set.
    """
    reference_ids = None
    reference_tag = None
    for _, tag, _ in configs:
        map_path = emb_dir / f"cosmos_clip_to_index_{tag}.pkl"
        if not map_path.exists():
            continue
        with open(map_path, "rb") as f:
            clip_to_index = pickle.load(f)
        ids = set(clip_to_index.keys())
        if reference_ids is None:
            reference_ids = ids
            reference_tag = tag
        elif ids != reference_ids:
            only_ref = len(reference_ids - ids)
            only_cur = len(ids - reference_ids)
            raise SystemExit(
                f"Clip ID mismatch between '{reference_tag}' and '{tag}': "
                f"{only_ref} only in {reference_tag}, {only_cur} only in {tag}. "
                "Rebuild indexes using the same shared embeddings parquet."
            )
    if reference_ids is None:
        raise SystemExit(
            "No index pkl files found under "
            f"{emb_dir}. Run benchmark_faiss_indexes_cosmos_embed.py first."
        )
    print(f"[verify] All indexes share the same {len(reference_ids):,} clip IDs.")
    return reference_ids


def load_annotation_slices(
    db_path: str,
    all_clip_ids: Set[str],
    data_sources: List[str],
    annotation_specs: List[Tuple[str, str]],
) -> List[DataSlice]:
    """Query annotations.db and build DataSlice objects.

    Creates:
      - One slice per *data_source* (from the ``clips`` table).
      - One slice per *annotation_spec* (clips whose annotation key matches
        the LIKE pattern in the ``annotations`` table).
      - Pairwise intersections: data_source × annotation_spec.

    All clip ID sets are intersected with *all_clip_ids* so only clips that
    are actually present in the FAISS index are included.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    slices = []

    ds_sets = {}
    for ds in data_sources:
        cur = conn.execute(
            "SELECT clip_id FROM clips WHERE data_source LIKE ?",
            (f"%{ds}%",),
        )
        ids = {r["clip_id"] for r in cur} & all_clip_ids
        ds_sets[ds] = ids
        slices.append(DataSlice(
            name=ds.lower(),
            description=f"data_source LIKE '%{ds}%'",
            clip_ids=ids,
        ))
        print(f"    slice '{ds.lower()}': {len(ids):,} clips")

    ann_sets = {}
    for slice_name, pattern in annotation_specs:
        cur = conn.execute(
            "SELECT DISTINCT clip_id FROM annotations WHERE key LIKE ?",
            (pattern,),
        )
        ids = {r["clip_id"] for r in cur} & all_clip_ids
        ann_sets[slice_name] = ids
        slices.append(DataSlice(
            name=slice_name,
            description=f"annotation key LIKE '{pattern}'",
            clip_ids=ids,
        ))
        print(f"    slice '{slice_name}': {len(ids):,} clips")

    for ds, ds_ids in ds_sets.items():
        for ann_name, ann_ids in ann_sets.items():
            ids = ds_ids & ann_ids
            slices.append(DataSlice(
                name=f"{ds.lower()}_{ann_name}",
                description=(
                    f"data_source LIKE '%{ds}%' "
                    f"AND annotation key LIKE '{ann_name}'"
                ),
                clip_ids=ids,
            ))
            print(f"    slice '{ds.lower()}_{ann_name}': {len(ids):,} clips")

    conn.close()
    return slices


def fmt_score(s: float) -> str:
    return "nan" if math.isnan(s) else f"{s:.4f}"


def fmt_recall(r: float) -> str:
    return "n/a" if math.isnan(r) else f"{r:.3f}"


def recall(baseline, tested) -> float:
    """Recall = |baseline ∩ tested| / |baseline|, or NaN when either is empty."""
    if not baseline or not tested:
        return float("nan")
    return len(set(baseline) & set(tested)) / len(baseline)


def run_text_query(store, query, slice_ids, k):
    """Run *query* against *store*, filtered to *slice_ids*.

    Clears the LRU cache before each call so every timing reflects a real
    FAISS search. Returns (seconds, count, max_score, min_score, top_k_clip_ids).
    """
    store.searches.clear()
    t0 = time.perf_counter()
    raw = store.search_with_text(query, k=k)
    seconds = time.perf_counter() - t0
    # Filter results to the current slice and extract scores.
    hits = [(cid, s) for cid, s in raw if cid in slice_ids]
    scores = [s for _, s in hits]
    count = len(scores)
    max_s = max(scores) if scores else float("nan")
    min_s = min(scores) if scores else float("nan")
    top_k = [cid for cid, _ in hits[:k]]
    return seconds, count, max_s, min_s, top_k


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark semantic search (text→video) across multiple FAISS "
            "index specs and data slices. "
            "Reports query time, result count, max_score, min_score."
        )
    )
    p.add_argument(
        "embeddings_dir",
        help=(
            "Directory containing Cosmos embedding parquet files "
            "and pre-built FAISS indexes."
        ),
    )
    p.add_argument(
        "--annotations-db",
        default=None,
        metavar="PATH",
        help="Path to annotations.db (SQLite) for data-slice benchmarks.",
    )
    p.add_argument(
        "--data-sources",
        nargs="*",
        default=DEFAULT_DATA_SOURCES,
        metavar="NAME",
        help=(
            "Data-source names to create per-source slices from "
            "(matched with LIKE %%name%%). Default: MADS Nexar."
        ),
    )
    p.add_argument(
        "--annotation-slices",
        nargs="*",
        default=None,
        metavar="NAME:PATTERN",
        help=(
            "Annotation-key slices as 'name:%%sql_pattern%%' pairs. "
            "Default: 'night:%%night%%' 'sign:%%sign%%'."
        ),
    )
    p.add_argument(
        "--k",
        type=int,
        default=2048,
        help="Number of FAISS neighbors to retrieve per query (default: 2048).",
    )
    p.add_argument(
        "--queries",
        nargs="*",
        default=None,
        metavar="QUERY",
        help="Text queries to benchmark. Defaults to the built-in set.",
    )
    p.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Write results to this markdown file (optional).",
    )
    return p.parse_args()


def print_table(headers, rows) -> None:
    col_w = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_w[i] = max(col_w[i], len(cell))
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"

    def fmt_row(cells):
        return (
            "| "
            + " | ".join(str(c).ljust(w) for c, w in zip(cells, col_w))
            + " |"
        )

    print(sep)
    print(fmt_row(tuple(headers)))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print(sep)


def main() -> None:
    args = parse_args()
    emb_dir = Path(args.embeddings_dir)
    if not emb_dir.exists():
        raise SystemExit(f"Embeddings directory not found: {emb_dir}")

    queries = args.queries or DEFAULT_QUERIES

    if args.annotation_slices:
        ann_specs = []
        for spec_str in args.annotation_slices:
            name, _, pattern = spec_str.partition(":")
            if not pattern:
                raise SystemExit(
                    f"Bad --annotation-slices value: {spec_str!r}. "
                    "Expected 'name:%%pattern%%'."
                )
            ann_specs.append((name.strip(), pattern.strip()))
    else:
        ann_specs = DEFAULT_ANNOTATION_SLICES

    if not args.annotations_db:
        print(
            "\nWARNING: --annotations-db not provided. "
            "Data-source slices (e.g. MADS, Nexar) and annotation-based "
            "slices (e.g. night, sign) will be skipped.\n"
            "To include them, pass: --annotations-db /path/to/annotations.db\n"
        )

    shared_clip_ids = verify_shared_clip_ids(emb_dir, INDEX_CONFIGS)

    slices = [DataSlice(
        name="all",
        description="All clips in the index",
        clip_ids=shared_clip_ids,
    )]
    if args.annotations_db:
        db_path = args.annotations_db
        if not Path(db_path).exists():
            print(
                f"Warning: annotations DB not found at {db_path!r}. "
                "Skipping annotation/data-source slices."
            )
        else:
            print(f"Loading slices from {db_path} ...")
            slices.extend(
                load_annotation_slices(
                    db_path, shared_clip_ids, args.data_sources, ann_specs
                )
            )

    headers = [
        "index", "nprobe", "slice", "n_slice", "query",
        "seconds", "count", "max_score", "min_score", "recall",
    ]
    all_rows = []
    # FLAT results serve as ground truth for recall; keyed by (slice_name, query).
    flat_baselines = {}
    # Per (index_tag, slice_name): list of (baseline_top_k, tested_top_k) pairs,
    # one per query — used to compute the PR curve after all indexes are done.
    pr_data = defaultdict(list)

    for spec, tag, nprobe in INDEX_CONFIGS:
        print(f"\n{'='*64}")
        print(f"[index] {spec}  (tag={tag}, nprobe={nprobe})")
        print("=" * 64)

        store = CosmosEmbeddingsStore(
            path_to_embeddings=str(emb_dir),
            index_spec=spec,
        )
        if hasattr(store.features_index, "nprobe"):
            store.features_index.nprobe = nprobe
        if hasattr(store.features_index, "hnsw"):
            try:
                store.features_index.hnsw.efSearch = max(64, nprobe)
            except Exception:
                pass

        # Warm up the embedding model and FAISS before timing.
        try:
            store.search_with_text("warmup", k=4096 if spec == "FLAT" else args.k)
        except Exception:
            pass
        store.searches.clear()

        for sl in slices:
            if not sl.clip_ids:
                print(f"  Slice '{sl.name}': empty — skipping.")
                continue

            n = len(sl.clip_ids)
            print(f"\n  [{tag}] slice='{sl.name}' ({n:,} clips) — {len(queries)} queries")

            for q in queries:
                sec, cnt, mx, mn, top_k = run_text_query(
                    store, q, sl.clip_ids, 4096 if spec == "FLAT" else args.k
                )

                if spec == "FLAT":
                    if not top_k:
                        continue
                    # Store FLAT result as ground truth for this (slice, query).
                    flat_baselines[(sl.name, q)] = top_k
                    recall_str = "baseline"
                else:
                    baseline = flat_baselines.get((sl.name, q), [])
                    if not baseline:
                        continue
                    recall_str = fmt_recall(recall(baseline, top_k))
                    pr_data[(tag, sl.name)].append((baseline, top_k))

                row = (
                    tag, str(nprobe), sl.name, str(n), q,
                    f"{sec:.4f}", str(cnt),
                    fmt_score(mx), fmt_score(mn),
                    recall_str,
                )
                all_rows.append(row)
                print(
                    f"    '{q[:44]}'"
                    f" → {sec:.3f}s  count={cnt}"
                    f"  max={fmt_score(mx)}  recall={recall_str}"
                )

        del store
        gc.collect()

    groups = defaultdict(lambda: {"secs": [], "counts": [], "recalls": []})
    n_slice_map = {}
    for row in all_rows:
        tag_, nprobe_, sl_, n_sl_, _, sec_, cnt_, _, _, recall_ = row
        key = (tag_, nprobe_, sl_)
        n_slice_map[key] = n_sl_
        groups[key]["secs"].append(float(sec_))
        groups[key]["counts"].append(int(cnt_))
        if recall_ not in ("baseline", "n/a"):
            groups[key]["recalls"].append(float(recall_))

    summary_headers = [
        "index", "nprobe", "slice", "n_slice",
        "mean_s", "mean_count", "mean_recall", "valid_q",
    ]
    n_q = len(queries)
    summary_rows = []
    for key, g in groups.items():
        tag_, nprobe_, sl_ = key
        mean_s = float(np.mean(g["secs"])) if g["secs"] else float("nan")
        mean_c = float(np.mean(g["counts"])) if g["counts"] else float("nan")
        mean_r = float(np.mean(g["recalls"])) if g["recalls"] else float("nan")
        summary_rows.append((
            tag_, nprobe_, sl_, n_slice_map[key],
            f"{mean_s:.4f}",
            f"{mean_c:.0f}",
            fmt_recall(mean_r),
            f"{len(g['recalls'])}/{n_q}",
        ))

    print(f"\n{'='*64}\nSUMMARY (per index × slice, averaged over {n_q} queries)\n{'='*64}")
    print_table(summary_headers, summary_rows)

    print(f"\n{'='*64}\nDETAIL ({len(all_rows)} rows)\n{'='*64}")
    print_table(headers, all_rows)

    # --- PR curve: precision@k and recall@k swept over k, averaged over queries ---
    # Only computed for the 'all' slice; small slices are too sparse (FLAT baseline
    # often has <10 results, so recall saturates immediately at k=10).
    curve_ks = [k for k in PR_CURVE_KS if k <= args.k]
    if args.k not in curve_ks:
        curve_ks.append(args.k)

    # Lookup from index tag to its nprobe string, for the table output.
    tag_to_nprobe = {t: str(np_) for _, t, np_ in INDEX_CONFIGS}
    pr_headers = ["index", "nprobe", "k", "precision", "recall"]
    pr_rows = []
    for (tag, sl_name), pairs in pr_data.items():
        if sl_name != "all":
            continue
        for k in curve_ks:
            # For each query, count how many of the FLAT ground-truth top-K
            # results appear in the approximate index's top-k.
            # precision@k = hits/k, recall@k = hits/K, then average over queries.
            precisions, recalls = [], []
            for baseline, tested in pairs:
                K = len(baseline)
                # Get the number of unique elements from baseline that also
                # appear in top-k of the tested
                hits = len(set(baseline) & set(tested[:k]))
                precisions.append(hits / k if k > 0 else 0.0)
                recalls.append(hits / K if K > 0 else 0.0)
            avg_p = float(np.mean(precisions)) if precisions else float("nan")
            avg_r = float(np.mean(recalls)) if recalls else float("nan")
            pr_rows.append((
                tag, tag_to_nprobe.get(tag, "?"), str(k),
                f"{avg_p:.3f}", f"{avg_r:.3f}",
            ))

    if pr_rows:
        print(f"\n{'='*64}\nPR CURVE — 'all' slice, averaged over {len(queries)} queries\n{'='*64}")
        print_table(pr_headers, pr_rows)

    if args.output:
        metadata = {
            "embeddings_dir": str(emb_dir),
            "embeddings_size": path_size(str(emb_dir)),
            "k": str(args.k),
            "n_queries": str(len(queries)),
            "n_indexes": str(len(INDEX_CONFIGS)),
            "annotations_db": str(args.annotations_db or "n/a"),
        }
        write_markdown(
            args.output,
            "Semantic Search Benchmark — Summary",
            summary_headers,
            summary_rows,
            metadata=metadata,
        )
        write_markdown(
            args.output,
            "Semantic Search Benchmark — Detail",
            headers,
            all_rows,
            metadata=metadata,
            append=True,
        )
        if pr_rows:
            write_markdown(
                args.output,
                "Semantic Search Benchmark — PR Curve (all slice)",
                pr_headers,
                pr_rows,
                metadata=metadata,
                append=True,
            )


if __name__ == "__main__":
    main()
