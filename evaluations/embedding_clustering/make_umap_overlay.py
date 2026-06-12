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

"""UMAP clustering view with representative video frames overlaid.

Plots the 2D UMAP scatter of a clustering run (from its ``umap.json``), then pins
the centroid-clip thumbnail of selected clusters at each cluster's UMAP centroid,
framed in the cluster's color. Gives a "what does this region of the embedding
look like" view of a clustering.

    python make_umap_overlay.py --run-dir <clustering>/<run_id> \
        --grid scenario_grid_5x3_cdn.json --col "Cosmos-Embed1" \
        --shots ./screenshots --out umap_overlay.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

ACCENTS = ["#E45756", "#4C78A8", "#54A24B", "#F58518", "#B279A2",
           "#9D755D", "#72B7B2", "#EECA3B"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--grid", type=Path, required=True,
                    help="scenario grid JSON: {col: [{cluster, clip_id, topic}, ...]}")
    ap.add_argument("--col", required=True, help="embedding-label key in the grid JSON")
    ap.add_argument("--shots", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--thumb-w", type=int, default=190)
    args = ap.parse_args(argv)

    umap = json.loads((args.run_dir / "umap.json").read_text())
    clips, centroids = umap["clips"], umap["centroids"]
    cells = json.loads(args.grid.read_text())[args.col]
    color_of = {str(c["cluster"]): ACCENTS[k % len(ACCENTS)] for k, c in enumerate(cells)}

    fig, ax = plt.subplots(figsize=(13, 9))
    # faint background: every sampled clip
    for cid, pts in clips.items():
        if not pts:
            continue
        a = np.asarray(pts)
        if cid in color_of:
            ax.scatter(a[:, 0], a[:, 1], s=10, c=color_of[cid], alpha=0.85, linewidths=0, zorder=2)
        else:
            ax.scatter(a[:, 0], a[:, 1], s=3, c="#d9d9d9", alpha=0.30, linewidths=0, zorder=1)

    # overlay the centroid clip thumbnail of each highlighted cluster
    for k, c in enumerate(cells):
        cid = str(c["cluster"])
        xy = centroids.get(cid)
        if not xy:
            continue
        col = color_of[cid]
        im = Image.open(args.shots / f"{c['clip_id']}.png").convert("RGB")
        im.thumbnail((args.thumb_w, args.thumb_w))
        ab = AnnotationBbox(OffsetImage(np.asarray(im), zoom=1.0), xy, frameon=True,
                            pad=0.15, zorder=5,
                            bboxprops=dict(edgecolor=col, linewidth=2.5))
        ax.add_artist(ab)
        topic = c["topic"] if isinstance(c["topic"], str) else ", ".join(c["topic"][:2])
        ax.annotate(topic[:32], xy, xytext=(0, -args.thumb_w * 0.32), textcoords="offset points",
                    ha="center", va="top", fontsize=9.5, color=col, fontweight="bold", zorder=6)

    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("UMAP of the Cosmos-Embed1 PhysicalAI clustering, with the centroid clip of "
                 "five distinct clusters overlaid ($k$=1000)", fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
