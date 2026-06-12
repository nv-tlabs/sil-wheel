#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Category-composition bar chart for the overlay figure's shown clusters.

For each embedding we take the 10 most-distinct k=50 clusters (the ones shown in
the overlay), categorise every word of their LLM topic phrases as appearance,
activity, or objects via a fixed lexicon (shared road/place terms excluded), and
plot the per-category share. This is the quantitative version of the colour
coding: the per-phrase colours are mixed, but the aggregate shares separate.
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

import figstyle  # noqa: F401
import make_taxonomy_compare as mt
from make_cluster_overlay_table import _distinct_clusters


def _theme(td):
    return td.get("description") or ", ".join(td.get("keywords", [])[:5])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clustering-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args(argv)

    cols = [("k50_cosmos", "Cosmos-Embed1"), ("k50_caption", "Caption (Qwen3-Emb-8B)"),
            ("k50_visual", "Visual (Florence-2/SigLIP)")]
    cat = {mt.APPEAR: 0, mt.ACTIVITY: 1, mt.OBJECT: 2}
    emb_col = ["#E58606", "#5D69B1", "#52BCA3"]
    comp = []
    for rid, _ in cols:
        run = args.clustering_dir / rid
        topics = json.loads((run / "cluster_topics.json").read_text())["topics"]
        sizes = pd.read_parquet(run / "cluster_assignments.parquet",
                                columns=["clip_id", "cluster_id"]).groupby("cluster_id").size()
        clusters = _distinct_clusters(run, sizes, args.k, 0.5)
        cnt = [0, 0, 0]
        for c in clusters:
            for w in _theme(topics[str(c)]).split():
                i = cat.get(mt._color(w))
                if i is not None:
                    cnt[i] += 1
        s = sum(cnt) or 1
        comp.append([c / s * 100 for c in cnt])

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=160)
    x = np.arange(3); w = 0.26
    for k, (_, name) in enumerate(cols):
        ax.bar(x + (k - 1) * w, comp[k], w, label=name, color=emb_col[k],
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["appearance /\nlighting / weather", "activity /\nsignals / agents",
                        "objects /\nscenery"], fontsize=10)
    ax.set_ylabel("share of topic words (%)", fontsize=10)
    ax.set_ylim(0, 65)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=9, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8); ax.set_axisbelow(True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
