#!/usr/bin/env python3
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

"""
Offline analyzer: compute per–data-source dataset statistics from SQLite DBs.

Inputs
- Annotations DB (SQLite) with tables matching `sqlite_data_store.py`:
  - clips(clip_id TEXT PRIMARY KEY, data_source TEXT, country TEXT,
          has_time INTEGER, has_manual_annotations INTEGER, has_autolabels INTEGER)
  - annotations(uid TEXT PRIMARY KEY, project TEXT, clip_id TEXT, key TEXT,
                value REAL, start_time REAL, end_time REAL, label_type TEXT)
  - video_paths(clip_id TEXT PRIMARY KEY, path TEXT)
- Captions DB (optional, SQLite) with tables: captions, captions_fts.

Outputs
- One JSON summary per data source in the output_dir. The schema matches
  the structure consumed by the Data Stats UI (features, per_clip_avg, percentiles).

Example
  python analyze_data_stats.py \
    --db /path/to/annotations.db \
    --captions-db /path/to/captions.db \
    --output_dir /path/to/out \
    --datasource some_dataset --datasource another_ds

Notes
- The script relies only on DB contents; it does not touch videos.
- Data source filtering handles comma-separated `clips.data_source` values.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

from sil_wheel.stores.sqlite_caption_store import FTSCaptionStore

# Prefer using the in-process stores to reuse their reverse indexes and pragmas
from sil_wheel.stores.sqlite_data_store import SQLiteDataStore


def slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("_").lower()


def connect_sqlite(path: Optional[str]) -> Optional[sqlite3.Connection]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SQLite file not found: {path}")
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    return conn


def ds_membership_clause(alias: str = "clips") -> str:
    """SQL clause for exact membership of a token inside a comma-separated field.

    Mirrors server usage in upsert/select logic:
      instr(',' || data_source || ',', ',' || ? || ',') > 0
    """
    return f"instr(',' || {alias}.data_source || ',', ',' || ? || ',') > 0"


def query_data_sources(conn: sqlite3.Connection) -> list[str]:
    ds = set()
    cur = conn.execute("SELECT data_source FROM clips")
    for row in cur:
        val = row["data_source"]
        if not val:
            continue
        for item in str(val).split(","):
            item = item.strip()
            if item:
                ds.add(item)
    return sorted(ds)


def percentile(arr: list[float], q: float) -> Optional[float]:
    if not arr:
        return None
    if q <= 0:
        return float(min(arr))
    if q >= 100:
        return float(max(arr))
    s = sorted(arr)
    k = (len(s) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    d0 = s[int(f)] * (c - k)
    d1 = s[int(c)] * (k - f)
    return float(d0 + d1)


def basic_stats(values: Iterable[float]) -> dict:
    vals = [
        float(v) for v in values if v is not None and math.isfinite(float(v))
    ]
    if not vals:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
        }
    n = len(vals)
    mu = sum(vals) / n
    var = sum((v - mu) ** 2 for v in vals) / n if n > 1 else 0.0
    std = math.sqrt(var)
    return {
        "mean": mu,
        "std": std,
        "min": min(vals),
        "max": max(vals),
        "median": percentile(vals, 50),
    }


def _maybe_matplotlib_save_barplot(
    data: dict[str, int],
    title: str,
    out_png: Path,
    fallback_svg: Path,
    top_k: int = 30,
) -> Path:
    """Save barplot as PNG if matplotlib is available, else as simple SVG.

    Returns the written file path.
    """
    # Keep only top_k by count
    items = sorted(data.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    labels = [k for k, _ in items]
    counts = [v for _, v in items]

    try:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt

        # Horizontal bar plot for better readability with many labels
        # Compute a reasonable figure size based on number of items
        fig_h = max(2.8, min(18, 0.35 * len(items) + 1.5))
        # Grow width slightly with magnitude of counts to keep value labels inside
        order = len(str(max(counts) if counts else 1))
        fig_w = min(24, 8 + max(0, order - 3) * 1.2)
        plt.figure(figsize=(fig_w, fig_h))
        max_count = max(counts) if counts else 1
        y_pos = list(range(len(items)))
        bars = plt.barh(y_pos, counts, color="#76B900")
        plt.title(title)
        plt.xlabel("# Annotations")
        plt.yticks(y_pos, labels)
        # Ensure there is room for right-side padding (5%) if we have outside labels
        plt.xlim(0, max_count * 1.10)
        # Add value labels at end of each bar
        for y, b in zip(y_pos, bars):
            w = b.get_width()
            label = f"{int(w):,}"
            # Place inside if the bar is wide enough, otherwise just outside
            if w >= max_count * 0.12:
                plt.text(
                    w - max_count * 0.01,
                    y,
                    label,
                    va="center",
                    ha="right",
                    fontsize=9,
                    color="#ffffff",
                )
            else:
                plt.text(
                    min(w + max_count * 0.02, max_count * 1.08),
                    y,
                    label,
                    va="center",
                    ha="left",
                    fontsize=9,
                    color="#111",
                )
        # Put the largest at the top
        plt.gca().invert_yaxis()
        plt.tight_layout(rect=(0.12, 0.02, 0.98, 0.98))
        plt.savefig(out_png, dpi=150)
        plt.close()
        return out_png
    except Exception:
        # Fallback: minimal inline SVG bar chart
        # Dimensions for horizontal bars
        bar_h = 18
        gap = 8
        label_w = 140  # left margin for labels
        right_margin = 20
        height = 30 + (bar_h + gap) * len(items)
        max_v = max(counts) if counts else 1
        # Dynamic width: leave 12% for right padding
        width = max(360, label_w + 10 + 640)

        def x(v):
            # map value to width portion (simple linear scale)
            usable = int((width - label_w - right_margin) * 0.88)
            return int((v / max_v) * usable)

        svg_parts = [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
            "<style> .lbl{font:10px sans-serif} .ttl{font:12px sans-serif;font-weight:bold} </style>",
            f"<text x='{width//2}' y='14' text-anchor='middle' class='ttl'>{title}</text>",
        ]
        # Draw horizontal bars: largest at top
        for i, (lab, val) in enumerate(items):
            y_top = 26 + i * (bar_h + gap)
            # label at left
            short = (lab[:22] + "…") if len(lab) > 23 else lab
            svg_parts.append(
                f"<text x='{label_w-6}' y='{y_top + bar_h/2 + 3}' class='lbl' text-anchor='end'>{short}</text>"
            )
            # bar
            bw = x(val)
            svg_parts.append(
                f"<rect x='{label_w}' y='{y_top}' width='{bw}' height='{bar_h}' fill='#76B900' />"
            )
            # count label: inside if bar wide enough, else just to the right (clamped to viewport)
            lbl = f"{val:,}"
            if bw >= 0.12 * (width - label_w - right_margin):
                tx = label_w + bw - 6
                anchor = "end"
                fill = "#ffffff"
            else:
                tx = min(label_w + bw + 8, width - right_margin - 6)
                anchor = "start"
                fill = "#111111"
            svg_parts.append(
                f"<text x='{tx}' y='{y_top + bar_h/2 + 3}' class='lbl' text-anchor='{anchor}' fill='{fill}'>{lbl}</text>"
            )
        svg_parts.append("</svg>")
        fallback_svg.write_text("\n".join(svg_parts), encoding="utf-8")
        return fallback_svg


def compute_for_datasource(
    ds: str,
    ann_store: SQLiteDataStore,
    cap_store: Optional[FTSCaptionStore] = None,
    clip_limit: Optional[int] = None,
) -> dict:
    t0 = time.perf_counter()
    # 1) Collect clips for this data source (via reverse index in SQLiteDataStore)
    ds_all = list(ann_store.data_source_to_clip_ids.get(ds, set()))
    clip_ids = ds_all if not clip_limit else ds_all[:clip_limit]
    n_clips = len(clip_ids)
    t1 = time.perf_counter()
    print(f"    [data_stats] {ds}: collected {n_clips} clips in {t1 - t0:.2f}s")

    ann_conn = ann_store.conn

    # 2) Per-clip annotation counts and label frequencies via JOINs
    ann_per_clip_manual = Counter()
    ann_per_clip_autolabel = Counter()
    ann_durations = []  # across all timed annotations
    ann_per_label = Counter()

    if clip_ids:
        # Per-clip counts via reverse index in SQLiteDataStore (aggregate across projects)
        t2 = time.perf_counter()
        ds_set = set(clip_ids)
        # Merge labeltype_to_clip_ids across projects
        for proj, lt_map in ann_store.labeltype_to_clip_ids.items():
            # manual
            man = lt_map.get("manual")
            if man:
                for cid, cnt in man.items():
                    if cid in ds_set:
                        ann_per_clip_manual[cid] += int(cnt)
            # autolabel
            auto = lt_map.get("autolabel")
            if auto:
                for cid, cnt in auto.items():
                    if cid in ds_set:
                        ann_per_clip_autolabel[cid] += int(cnt)
        t3 = time.perf_counter()
        print(f"    [data_stats] {ds}: per-clip ann counts in {t3 - t2:.2f}s")

        # Label frequencies across dataset
        t4 = time.perf_counter()
        for proj, key_map in ann_store.key_to_clip_ids.items():
            for key, counter in key_map.items():
                # sum counts for clips in this dataset
                s = 0
                for cid, cnt in counter.items():
                    if cid in ds_set:
                        s += int(cnt)
                if s:
                    ann_per_label[key] += s
        t5 = time.perf_counter()
        print(
            f"    [data_stats] {ds}: label freq for {len(ann_per_label)} labels in {t5 - t4:.2f}s"
        )

        # Timed durations distribution will be accumulated in the chunked pass below

    ann_per_clip_total = Counter(
        {
            cid: ann_per_clip_manual.get(cid, 0)
            + ann_per_clip_autolabel.get(cid, 0)
            for cid in clip_ids
        }
    )

    # 3) Caption stats (optional)
    cap_per_clip = Counter()
    caption_lengths = []  # words per caption
    caption_durations = []  # seconds per caption if start/end provided
    # overlap metric: annotations with captions (only timed)
    timed_ann_total = 0
    timed_ann_with_caption = 0
    cap_conn = cap_store.conn if cap_store else None
    if cap_conn is not None and clip_ids:
        # Load all captions for this dataset via data_source filter (fast, indexed)
        t8 = time.perf_counter()
        cap_intervals = defaultdict(list)
        timed_ann_total = 0
        timed_ann_with_caption = 0
        q_caps_ds = "SELECT clip_id, caption, start_time, end_time FROM captions WHERE data_source = ?"
        for row in cap_conn.execute(q_caps_ds, (ds,)):
            cid = row["clip_id"]
            cap_per_clip[cid] += 1
            txt = (row["caption"] or "").strip()
            if txt:
                caption_lengths.append(len(txt.split()))
            st = row["start_time"]
            et = row["end_time"]
            if st is not None and et is not None and et > st:
                try:
                    caption_durations.append(float(et) - float(st))
                    cap_intervals[cid].append((float(st), float(et)))
                except Exception:
                    pass

        # Now iterate annotations in chunks and check overlap against cap_intervals
        CH = 2000
        for i in range(0, len(clip_ids), CH):
            sub = clip_ids[i : i + CH]
            holders = ",".join(["?"] * len(sub))
            q_ann = (
                f"SELECT clip_id, start_time, end_time FROM annotations "
                f"WHERE clip_id IN ({holders}) AND start_time != -1 AND end_time != -1 AND end_time > start_time"
            )
            for row in ann_conn.execute(q_ann, sub):
                cid = row["clip_id"]
                st = float(row["start_time"])
                et = float(row["end_time"])
                ann_durations.append(et - st)
                timed_ann_total += 1
                for cst, cet in cap_intervals.get(cid, []):
                    if cst <= et and cet >= st:
                        timed_ann_with_caption += 1
                        break
        t9 = time.perf_counter()
        print(f"    [data_stats] {ds}: captions+overlap in {t9 - t8:.2f}s")

    # 4) Assemble stats
    def series_from_counter(counter: Counter, ids: list[str]) -> list[float]:
        return [float(counter.get(cid, 0)) for cid in ids]

    features = {
        # Annotation aggregates
        "annotations_per_clip": basic_stats(
            series_from_counter(ann_per_clip_total, clip_ids)
        ),
        "manual_annotations_per_clip": basic_stats(
            series_from_counter(ann_per_clip_manual, clip_ids)
        ),
        "autolabel_annotations_per_clip": basic_stats(
            series_from_counter(ann_per_clip_autolabel, clip_ids)
        ),
        "annotation_duration_sec": basic_stats(ann_durations),
    }

    # Optional caption aggregates
    if cap_conn is not None:
        features.update(
            {
                "caption_count_per_clip": basic_stats(
                    series_from_counter(cap_per_clip, clip_ids)
                ),
                "caption_words_per_caption": basic_stats(caption_lengths),
                "caption_duration_sec": basic_stats(caption_durations),
                "annotations_with_captions_count": float(
                    timed_ann_with_caption
                ),
                "annotations_with_captions_ratio": (
                    (float(timed_ann_with_caption) / float(timed_ann_total))
                    if timed_ann_total > 0
                    else None
                ),
            }
        )

    # Per-clip aggregates (for UI sections)
    per_clip_avg = {
        k: {
            "mean": v.get("mean"),
            "max": None,
            "min": None,
            "std": v.get("std"),
            "median": v.get("median"),
        }
        for k, v in features.items()
        if k.endswith("_per_clip")
    }
    per_clip_max = {
        k: {
            "max": v.get("max"),
            "mean": v.get("mean"),
            "min": v.get("min"),
            "std": v.get("std"),
            "median": v.get("median"),
        }
        for k, v in features.items()
        if k.endswith("_per_clip")
    }

    # Percentiles for selected distributions
    percentiles = {}
    if ann_durations:
        percentiles["annotation_duration_sec"] = {
            "p10": percentile(ann_durations, 10),
            "p25": percentile(ann_durations, 25),
            "p50": percentile(ann_durations, 50),
            "p75": percentile(ann_durations, 75),
            "p90": percentile(ann_durations, 90),
            "p95": percentile(ann_durations, 95),
        }
    if cap_conn is not None and caption_lengths:
        percentiles["caption_words_per_caption"] = {
            "p10": percentile(caption_lengths, 10),
            "p25": percentile(caption_lengths, 25),
            "p50": percentile(caption_lengths, 50),
            "p75": percentile(caption_lengths, 75),
            "p90": percentile(caption_lengths, 90),
            "p95": percentile(caption_lengths, 95),
        }
    if cap_conn is not None and caption_durations:
        percentiles["caption_duration_sec"] = {
            "p10": percentile(caption_durations, 10),
            "p25": percentile(caption_durations, 25),
            "p50": percentile(caption_durations, 50),
            "p75": percentile(caption_durations, 75),
            "p90": percentile(caption_durations, 90),
            "p95": percentile(caption_durations, 95),
        }

    result = {
        "dataset": ds,
        "n_clips_sampled": n_clips,
        "features": features,
        "per_clip_avg": per_clip_avg or None,
        "per_clip_max": per_clip_max or None,
        "percentiles": percentiles or None,
    }

    # Write barplot artifact for labels
    if ann_per_label:
        # Create artifacts filenames
        slug = slugify(ds)
        title = f"Annotations per label — {ds}"
        out_png = Path("labels_barplot_" + slug + ".png")
        out_svg = Path("labels_barplot_" + slug + ".svg")
        written = _maybe_matplotlib_save_barplot(
            dict(ann_per_label), title, out_png, out_svg
        )
        result.setdefault("artifacts", {})["labels_barplot"] = str(written)

    return result


def compute_all_datasets(
    ann_store: SQLiteDataStore,
    cap_store: Optional[FTSCaptionStore] = None,
    clip_limit: Optional[int] = None,
):
    t0 = time.perf_counter()
    datasets = list(ann_store.data_source_options)
    # 1) Build per-dataset clip sets from a single scan of clips (respect clip_limit per dataset)
    ds_set = set(datasets)
    ds_clips = {ds: [] for ds in datasets}
    clip_to_ds = defaultdict(list)
    limits = {ds: (clip_limit if clip_limit else None) for ds in datasets}
    with ann_store.lock, ann_store.conn:
        cur = ann_store.conn.execute("SELECT clip_id, data_source FROM clips")
        for row in cur:
            cid = row["clip_id"]
            ds_field = row["data_source"] or ""
            if not ds_field:
                continue
            for token in ds_field.split(","):
                ds = token.strip()
                if not ds or ds not in ds_set:
                    continue
                # Enforce per-dataset clip limit if provided
                lim = limits.get(ds)
                if lim is not None and len(ds_clips[ds]) >= lim:
                    continue
                ds_clips[ds].append(cid)
                clip_to_ds[cid].append(ds)
    t1 = time.perf_counter()
    print(
        f"[data_stats] built ds clip sets for {len(datasets)} datasets in {t1 - t0:.2f}s"
    )
    # Precompute set membership for fast checks downstream
    ds_clips_set = {ds: set(clips) for ds, clips in ds_clips.items()}

    # 2) Annotation aggregates using reverse indices (no DB hits)
    ann_per_clip_manual = {ds: Counter() for ds in datasets}
    ann_per_clip_autolabel = {ds: Counter() for ds in datasets}
    ann_per_label = {ds: Counter() for ds in datasets}

    for proj, lt_map in ann_store.labeltype_to_clip_ids.items():
        man = lt_map.get("manual") or {}
        for cid, cnt in man.items():
            for ds in clip_to_ds.get(cid, ()):
                ann_per_clip_manual[ds][cid] += int(cnt)
        auto = lt_map.get("autolabel") or {}
        for cid, cnt in auto.items():
            for ds in clip_to_ds.get(cid, ()):
                ann_per_clip_autolabel[ds][cid] += int(cnt)

    for proj, key_map in ann_store.key_to_clip_ids.items():
        for key, counter in key_map.items():
            s_per_ds = defaultdict(int)
            for cid, cnt in counter.items():
                for ds in clip_to_ds.get(cid, ()):
                    s_per_ds[ds] += int(cnt)
            for ds, s in s_per_ds.items():
                if s:
                    ann_per_label[ds][key] += s
    t2 = time.perf_counter()
    print(f"[data_stats] aggregated annotation counts in {t2 - t1:.2f}s")

    # 3) Captions: single pass to build per-dataset metrics and intervals
    cap_per_clip = {ds: Counter() for ds in datasets}
    caption_lengths = {ds: [] for ds in datasets}
    caption_durations = {ds: [] for ds in datasets}
    cap_intervals = {ds: defaultdict(list) for ds in datasets}
    if cap_store is not None:
        cap_conn = cap_store.conn
        t3 = time.perf_counter()
        q_caps_all = "SELECT clip_id, caption, start_time, end_time, data_source FROM captions"
        for row in cap_conn.execute(q_caps_all):
            ds = row["data_source"]
            if not ds or ds not in ds_clips:
                continue
            cid = row["clip_id"]
            if cid not in ds_clips_set[ds]:
                continue
            cap_per_clip[ds][cid] += 1
            txt = (row["caption"] or "").strip()
            if txt:
                caption_lengths[ds].append(len(txt.split()))
            st = row["start_time"]
            et = row["end_time"]
            if st is not None and et is not None and et > st:
                try:
                    caption_durations[ds].append(float(et) - float(st))
                    cap_intervals[ds][cid].append((float(st), float(et)))
                except Exception:
                    pass
        t4 = time.perf_counter()
        print(f"[data_stats] scanned captions once in {t4 - t3:.2f}s")

        # Fast overlap counts per dataset via a single SQL join by attaching captions DB
        try:
            ann_store.conn.execute(
                f"ATTACH DATABASE '{cap_store.db_path}' AS cap"
            )
            q_overlap = (
                "SELECT cp.data_source AS ds, COUNT(*) AS n "
                "FROM annotations a "
                "JOIN cap.captions cp ON cp.clip_id = a.clip_id "
                "WHERE a.start_time != -1 AND a.end_time != -1 AND a.end_time > a.start_time "
                "  AND cp.start_time IS NOT NULL AND cp.end_time IS NOT NULL AND cp.end_time > cp.start_time "
                "  AND cp.start_time <= a.end_time AND cp.end_time >= a.start_time "
                "GROUP BY ds"
            )
            for row in ann_store.conn.execute(q_overlap):
                ds = row["ds"]
                if ds in cap_intervals:  # only datasets we saw
                    # Initialize counter; we'll overwrite later when computing summaries
                    # Store directly into a side map to avoid dependence on cap_intervals
                    cap_intervals.setdefault(
                        ds, defaultdict(list)
                    )  # ensure key exists
                    # keep result in a dedicated dict for overlap
            # We'll fill timed_ann_with_caption using the overlap results below
            overlap_counts = {}
            for row in ann_store.conn.execute(q_overlap):
                overlap_counts[row["ds"]] = int(row["n"] or 0)
        except Exception as e:
            print(
                f"[data_stats] overlap join failed, falling back to Python: {e}"
            )
            overlap_counts = None

    # 4) Timed annotations: single pass; accumulate durations and overlaps per dataset
    ann_conn = ann_store.conn
    timed_ann_total = {ds: 0 for ds in datasets}
    timed_ann_with_caption = {ds: 0 for ds in datasets}
    ann_durations = {ds: [] for ds in datasets}
    t5 = time.perf_counter()
    q_ann_all = (
        "SELECT clip_id, start_time, end_time FROM annotations "
        "WHERE start_time != -1 AND end_time != -1 AND end_time > start_time"
    )
    for row in ann_conn.execute(q_ann_all):
        cid = row["clip_id"]
        st = float(row["start_time"])
        et = float(row["end_time"])
        dss = clip_to_ds.get(cid)
        if not dss:
            continue
        for ds in dss:
            if cid not in ds_clips_set[ds]:
                continue
            ann_durations[ds].append(et - st)
            timed_ann_total[ds] += 1
            if cap_store is not None and overlap_counts is None:
                for cst, cet in cap_intervals[ds].get(cid, []):
                    if cst <= et and cet >= st:
                        timed_ann_with_caption[ds] += 1
                        break
    t6 = time.perf_counter()
    print(f"[data_stats] scanned timed annotations once in {t6 - t5:.2f}s")

    # If overlap computed via SQL, inject counts now
    if (
        cap_store is not None
        and "overlap_counts" in locals()
        and overlap_counts is not None
    ):
        for ds, n in overlap_counts.items():
            if ds in timed_ann_with_caption:
                timed_ann_with_caption[ds] = n

    # 5) Build per-dataset summaries
    summaries = {}
    for ds in datasets:
        clips = ds_clips[ds]
        # per-clip totals
        total_counts = Counter(
            {
                cid: ann_per_clip_manual[ds].get(cid, 0)
                + ann_per_clip_autolabel[ds].get(cid, 0)
                for cid in clips
            }
        )
        features = {
            "annotations_per_clip": basic_stats(
                [float(total_counts.get(cid, 0)) for cid in clips]
            ),
            "manual_annotations_per_clip": basic_stats(
                [float(ann_per_clip_manual[ds].get(cid, 0)) for cid in clips]
            ),
            "autolabel_annotations_per_clip": basic_stats(
                [float(ann_per_clip_autolabel[ds].get(cid, 0)) for cid in clips]
            ),
            "annotation_duration_sec": basic_stats(ann_durations[ds]),
        }
        if cap_store is not None:
            features.update(
                {
                    "caption_count_per_clip": basic_stats(
                        [float(cap_per_clip[ds].get(cid, 0)) for cid in clips]
                    ),
                    "caption_words_per_caption": basic_stats(
                        caption_lengths[ds]
                    ),
                    "caption_duration_sec": basic_stats(caption_durations[ds]),
                    "annotations_with_captions_count": float(
                        timed_ann_with_caption[ds]
                    ),
                    "annotations_with_captions_ratio": (
                        (
                            float(timed_ann_with_caption[ds])
                            / float(timed_ann_total[ds])
                        )
                        if timed_ann_total[ds] > 0
                        else None
                    ),
                }
            )
        # Percentiles
        percentiles = {}
        if ann_durations[ds]:
            percentiles["annotation_duration_sec"] = {
                "p10": percentile(ann_durations[ds], 10),
                "p25": percentile(ann_durations[ds], 25),
                "p50": percentile(ann_durations[ds], 50),
                "p75": percentile(ann_durations[ds], 75),
                "p90": percentile(ann_durations[ds], 90),
                "p95": percentile(ann_durations[ds], 95),
            }
        if cap_store is not None and caption_lengths[ds]:
            percentiles["caption_words_per_caption"] = {
                "p10": percentile(caption_lengths[ds], 10),
                "p25": percentile(caption_lengths[ds], 25),
                "p50": percentile(caption_lengths[ds], 50),
                "p75": percentile(caption_lengths[ds], 75),
                "p90": percentile(caption_lengths[ds], 90),
                "p95": percentile(caption_lengths[ds], 95),
            }
        if cap_store is not None and caption_durations[ds]:
            percentiles["caption_duration_sec"] = {
                "p10": percentile(caption_durations[ds], 10),
                "p25": percentile(caption_durations[ds], 25),
                "p50": percentile(caption_durations[ds], 50),
                "p75": percentile(caption_durations[ds], 75),
                "p90": percentile(caption_durations[ds], 90),
                "p95": percentile(caption_durations[ds], 95),
            }

        result = {
            "dataset": ds,
            "n_clips_sampled": len(clips),
            "features": features,
            "per_clip_avg": {
                k: {
                    "mean": v.get("mean"),
                    "max": None,
                    "min": None,
                    "std": v.get("std"),
                    "median": v.get("median"),
                }
                for k, v in features.items()
                if k.endswith("_per_clip")
            },
            "per_clip_max": {
                k: {
                    "max": v.get("max"),
                    "mean": v.get("mean"),
                    "min": v.get("min"),
                    "std": v.get("std"),
                    "median": v.get("median"),
                }
                for k, v in features.items()
                if k.endswith("_per_clip")
            },
            "percentiles": percentiles or None,
        }
        if ann_per_label[ds]:
            slug = slugify(ds)
            title = f"Annotations per label — {ds}"
            out_png = Path("labels_barplot_" + slug + ".png")
            out_svg = Path("labels_barplot_" + slug + ".svg")
            written = _maybe_matplotlib_save_barplot(
                dict(ann_per_label[ds]), title, out_png, out_svg
            )
            result.setdefault("artifacts", {})["labels_barplot"] = str(written)
        summaries[ds] = result

    t7 = time.perf_counter()
    print(
        f"[data_stats] built per-dataset summaries in {t7 - t6:.2f}s (total {t7 - t0:.2f}s)"
    )
    return summaries


def main():
    ap = argparse.ArgumentParser(
        description="Compute per–data-source statistics from SQLite DBs."
    )
    ap.add_argument(
        "--db",
        default="/path/to/wheel-data/annotations_latest_schema.db",
        help="Path to annotations SQLite DB (default: %(default)s)",
    )
    ap.add_argument(
        "--captions-db",
        default="/path/to/wheel-data/captions_latest_schema_dev.db",
        help="Optional path to captions SQLite DB (default: %(default)s)",
    )
    ap.add_argument(
        "--output_dir", required=True, help="Directory to write JSON summaries"
    )
    ap.add_argument(
        "--datasource",
        action="append",
        default=None,
        help="Data source(s) to analyze; if omitted, analyze all present in DB",
    )
    ap.add_argument(
        "--clip-limit",
        type=int,
        default=None,
        help="Optional max clips per data source (for quick runs)",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build stores (reuse reverse indices from SQLiteDataStore; optional captions store)
    ann_store = SQLiteDataStore(args.db)
    # PRAGMA tuning for faster full-table scans
    try:
        with ann_store.conn:
            ann_store.conn.execute("PRAGMA journal_mode=WAL;")
            ann_store.conn.execute("PRAGMA synchronous=NORMAL;")
            ann_store.conn.execute(
                "PRAGMA cache_size=-200000;"
            )  # ~200k pages (~200MB if 1k/page)
            ann_store.conn.execute("PRAGMA temp_store=MEMORY;")
            ann_store.conn.execute("PRAGMA mmap_size=536870912;")  # 512MB
    except Exception:
        pass

    cap_store = (
        FTSCaptionStore(args.captions_db) if args.captions_db else None
    )
    if cap_store is not None:
        try:
            with cap_store.conn:
                cap_store.conn.execute("PRAGMA journal_mode=WAL;")
                cap_store.conn.execute("PRAGMA synchronous=NORMAL;")
                cap_store.conn.execute("PRAGMA cache_size=-200000;")
                cap_store.conn.execute("PRAGMA temp_store=MEMORY;")
                cap_store.conn.execute("PRAGMA mmap_size=536870912;")
        except Exception:
            pass

    t_start = time.perf_counter()
    summaries = compute_all_datasets(
        ann_store, cap_store, clip_limit=args.clip_limit
    )
    for ds, summary in summaries.items():
        slug = slugify(ds)
        out_path = out_dir / f"data_stats_summary_{slug}.json"
        # Relativize artifact paths into output dir and move files if created in CWD
        arts = summary.get("artifacts", {})
        relocated = {}
        for k, v in arts.items():
            try:
                p = Path(v)
                if p.exists():
                    target = out_dir / p.name
                    if str(p.resolve()) != str(target.resolve()):
                        try:
                            data = p.read_bytes()
                            target.write_bytes(data)
                            try:
                                p.unlink()
                            except Exception:
                                pass
                        except Exception:
                            target = p
                    relocated[k] = str(target)
            except Exception:
                pass
        if relocated:
            summary["artifacts"] = relocated

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False)
        print(f"  Wrote {out_path}")

    total_elapsed = time.perf_counter() - t_start
    print(
        f"Completed {len(summaries)} data source(s) in {total_elapsed:.2f}s (all datasources, single scans)"
    )

    # Close DB handles gracefully
    try:
        ann_store.conn.close()
    except Exception:
        pass
    if cap_store is not None:
        try:
            cap_store.conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
