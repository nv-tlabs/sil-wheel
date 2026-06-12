#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Compose the per-embedding overlay maps + topic tables into one 2x3 figure.

Row 1: the three precomputed UMAP overlay maps (``overlay_map_<name>.png`` from
``make_cluster_overlay_table.py --map-only``). Row 2: the three topic tables,
rendered natively (colour swatch + cluster id + LLM theme), using the SAME
distinct-cluster selection and Vivid colours as the maps so the two rows align.

    python make_overlay_grid.py --clustering-dir CLR --maps-dir results/... \
        --runs k50_cosmos:Cosmos-Embed1 k50_caption:"Caption (Qwen3-Emb-8B)" \
               k50_visual:"Visual (Florence-2/SigLIP)" --out overlay_grid.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

import figstyle  # noqa: F401  (NVIDIA Sans)
from make_cluster_overlay_table import _distinct_clusters

_PAL = ["#E58606", "#5D69B1", "#52BCA3", "#99C945", "#CC61B0", "#24796C",
        "#DAA51B", "#2F8AC4", "#764E9F", "#ED645A", "#CC3A8E", "#A5AA99"]


def _theme(td):
    return td.get("description") or ", ".join(td.get("keywords", [])[:5])


def main(argv=None) -> int:
    import json
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clustering-dir", type=Path, required=True)
    ap.add_argument("--maps-dir", type=Path, required=True, help="dir with overlay_map_<name>.png")
    ap.add_argument("--runs", nargs="+", required=True, help="runid:Title per column (name from runid)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-frac", type=float, default=0.5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    cols = []
    for spec in args.runs:
        rid, title = spec.split(":", 1)
        name = rid.split("_")[-1]
        cols.append((rid, name, title))

    fig = plt.figure(figsize=(18, 10.5), dpi=150)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1.0], hspace=0.04, wspace=0.03,
                          left=0.005, right=0.995, top=0.95, bottom=0.01)

    for j, (rid, name, title) in enumerate(cols):
        run = args.clustering_dir / rid
        topics = json.loads((run / "cluster_topics.json").read_text())["topics"]
        df = pd.read_parquet(run / "cluster_assignments.parquet", columns=["clip_id", "cluster_id"])
        sizes = df.groupby("cluster_id").size()
        clusters = _distinct_clusters(run, sizes, args.k, args.min_frac)
        color = {c: _PAL[i % len(_PAL)] for i, c in enumerate(clusters)}

        # row 1: the overlay map
        ax = fig.add_subplot(gs[0, j])
        img = mpimg.imread(str(args.maps_dir / f"overlay_map_{name}.png"))
        ax.imshow(img); ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=6)

        # row 2: the topic table (one line per cluster)
        axt = fig.add_subplot(gs[1, j]); axt.axis("off")
        n = len(clusters)
        for i, c in enumerate(clusters):
            y = 1 - (i + 0.5) / n
            axt.add_patch(plt.Rectangle((0.01, y - 0.018), 0.022, 0.036, color=color[c],
                                        transform=axt.transAxes, clip_on=False))
            axt.text(0.05, y, f"C{c}", transform=axt.transAxes, fontsize=10,
                     fontweight="bold", color=color[c], va="center")
            axt.text(0.115, y, _theme(topics[str(c)]), transform=axt.transAxes,
                     fontsize=10, color="#1a1a1a", va="center")
            axt.text(0.99, y, f"{sizes[c]:,}", transform=axt.transAxes, fontsize=8.5,
                     color="#888888", va="center", ha="right")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
