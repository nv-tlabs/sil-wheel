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

"""UMAP overview of the spherical k-means clusters.

Renders a (pools x embeddings) grid of UMAP scatters, points colored by cluster
id, from the ``umap.json`` each clustering run writes. Pools (rows), embeddings
(columns), labels, and run ids all come from a ``fig_runs.json`` so the figure
is fully data-driven (see fig_runs.example.json for the schema).

    python make_umap_overview.py \
        --clustering-dir /path/to/clustering --fig-runs fig_runs.json \
        --out umap_overview.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle  # noqa: F401  (registers NVIDIA Sans, sets it as default)


def _umap_xy_by_cluster(run_dir: Path):
    data = json.loads((run_dir / "umap.json").read_text())
    xs, ys, cids = [], [], []
    for cid, pts in data["clips"].items():
        for x, y in pts:
            xs.append(x)
            ys.append(y)
            cids.append(int(cid))
    return np.array(xs), np.array(ys), np.array(cids)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clustering-dir", type=Path, required=True,
                    help="dir holding the per-run clustering output subdirs")
    ap.add_argument("--fig-runs", type=Path, required=True,
                    help="JSON mapping pools/embeddings to run ids (see example)")
    ap.add_argument("--out", type=Path, default=Path("umap_overview.png"))
    args = ap.parse_args(argv)

    spec = json.loads(args.fig_runs.read_text())
    embeds = spec["embeds"]
    pools = spec["pools"]

    fig, axes = plt.subplots(len(pools), len(embeds), figsize=(15, 10), squeeze=False)
    for i, pool in enumerate(pools):
        for j, emb in enumerate(embeds):
            ax = axes[i][j]
            run_id = pool["runs"][emb["key"]]
            x, y, cids = _umap_xy_by_cluster(args.clustering_dir / run_id)
            ax.scatter(x, y, c=cids % 20, cmap="tab20", s=1.5, alpha=0.4, linewidths=0)
            if i == 0:
                ax.set_title(emb["label"], fontsize=12)
            if j == 0:
                ax.set_ylabel(f"{pool['label']}\n({pool['n']:,} clips)", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.suptitle("UMAP overview of spherical $k$-means clusters "
                 "($k$=1000; up to 50k clips/panel shown, colored by cluster)",
                 fontsize=14, y=0.99)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
