#!/usr/bin/env python
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

"""UMAP scatter colored by the recursive level-1 taxonomy branch.

Joins a flat clustering run's ``umap.json`` (2D coords per sampled clip) with a
recursive-k-means taxonomy's ``hier_assignments.parquet`` (clip -> path), colors
each point by its level-1 branch, and labels each branch at its centroid with the
branch's topic keywords. Shows how the recursive top-level split lays out on the
embedding map. The flat run and the taxonomy must be over the same embedding/pool.

    python make_umap_hier.py --run-dir <flat_run> --hier-dir <hier> --out umap_hier.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True, help="flat run with umap.json")
    ap.add_argument("--hier-dir", type=Path, required=True,
                    help="taxonomy dir with hier_assignments.parquet + hier_topics.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    umap = json.loads((args.run_dir / "umap.json").read_text())
    rows = []
    for cid, pts in umap["clips"].items():
        ids = umap["clip_ids"].get(cid, [])
        for (x, y), clip in zip(pts, ids):
            rows.append((str(clip), x, y))
    pts = pd.DataFrame(rows, columns=["clip_id", "x", "y"])

    ha = pd.read_parquet(args.hier_dir / "hier_assignments.parquet", columns=["clip_id", "path"])
    ha["l1"] = ha["path"].astype(str).str.split(".").str[0]
    pts = pts.merge(ha[["clip_id", "l1"]], on="clip_id", how="inner")

    topics = json.loads((args.hier_dir / "hier_topics.json").read_text())

    def kws(node):
        v = topics.get(node, {})
        return v.get("keywords", v) if isinstance(v, dict) else v

    l1s = sorted(pts["l1"].dropna().unique(), key=lambda s: int(s))
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(13, 9))
    for i, l1 in enumerate(l1s):
        sub = pts[pts["l1"] == l1]
        ax.scatter(sub["x"], sub["y"], s=5, color=cmap(i % 10), alpha=0.5, linewidths=0)
    for i, l1 in enumerate(l1s):
        sub = pts[pts["l1"] == l1]
        cx, cy = sub["x"].median(), sub["y"].median()
        label = ", ".join(kws(l1)[:3])
        ax.text(cx, cy, label, fontsize=10, fontweight="bold", ha="center", va="center",
                color="black",
                bbox=dict(boxstyle="round,pad=0.25", fc=cmap(i % 10), ec="none", alpha=0.85))

    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Cosmos-Embed1 PhysicalAI UMAP, colored by recursive level-1 taxonomy branch "
                 f"({len(l1s)} branches)", fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print("wrote", args.out, "| points:", len(pts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
