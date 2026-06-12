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

"""Assemble the scenario example grid: columns = embeddings, rows = each
embedding's top distinct clusters, one centroid-clip frame per cell.

Rendered with matplotlib so the typography matches the other §4.4 figures
(``make_umap_overview`` / ``make_topic_focus``): colored bold column headers and
a per-cell topic caption. Reads the grid spec written by the centroid-clip
selection (``--grid`` JSON: ``{embedding_label: [{cluster, topic, clip_id}, ...]}``)
and the ``<clip_id>.png`` screenshots in ``--shots``. Frames are used at their
original framing; ``--crop-bottom`` is available but defaults to 0 (no crop).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PALETTE = {
    "Cosmos-Embed1": "#4C78A8",
    "Caption (Qwen3-Emb-8B)": "#F58518",
    "Florence-2/SigLIP": "#54A24B",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--shots", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--crop-bottom", type=float, default=0.0,
                    help="fraction cropped from the bottom (0 = original framing)")
    args = ap.parse_args(argv)

    grid = json.loads(args.grid.read_text())
    cols = list(grid)
    nrows = max(len(v) for v in grid.values())
    ncols = len(cols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.6, nrows * 2.7), squeeze=False)
    for j, col in enumerate(cols):
        for i in range(nrows):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i >= len(grid[col]):
                ax.axis("off")
                continue
            cell = grid[col][i]
            im = Image.open(args.shots / f"{cell['clip_id']}.png").convert("RGB")
            if args.crop_bottom > 0:
                w, h = im.size
                im = im.crop((0, 0, w, int((1 - args.crop_bottom) * h)))
            ax.imshow(np.asarray(im))
            topic = cell["topic"] if isinstance(cell["topic"], str) \
                else ", ".join(cell["topic"][:3])
            ax.set_xlabel(topic[:52], fontsize=9.5, color="#222222", labelpad=4)
        axes[0][j].set_title(col, fontsize=14, fontweight="bold",
                             color=PALETTE.get(col, "#333333"), pad=10)

    fig.suptitle("Representative clips for each embedding's most distinctive clusters "
                 "(PhysicalAI, $k$=1000)", fontsize=13, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
