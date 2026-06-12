#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Hierarchical drill-down with arrows.

One row per embedding: a high-level UMAP (coloured by the 10 level-1 branches,
with one branch highlighted) and an arrow into a zoom UMAP that re-clusters that
branch into its level-2 sub-clusters. For each embedding we pick the level-1
branch most aligned with its focus (Cosmos appearance, caption activity, visual
objects), so the figure shows that hierarchical clustering refines each
embedding along its own axis into finer, interpretable sub-clusters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.patches import ConnectionPatch

import figstyle  # noqa: F401
from cluster_select import dense_xy as _dense_xy
from topic_lexicon import APPEAR, ACTIVITY, OBJECT
from topic_lexicon import categorize as _color

_PAL = ["#E58606", "#5D69B1", "#52BCA3", "#99C945", "#CC61B0", "#24796C",
        "#DAA51B", "#2F8AC4", "#764E9F", "#ED645A", "#CC3A8E", "#A5AA99"]
_HALO = [pe.withStroke(linewidth=3.0, foreground="white")]
_FOCUSCAT = {"cosmos": (APPEAR, "appearance"), "caption": (ACTIVITY, "activity"),
             "visual": (OBJECT, "objects")}


def _phrase(v, n=3):
    return v.get("description") or ", ".join(v.get("keywords", [])[:n])


def _pick_focus(t, l1s, cat):
    best, bestc = l1s[0], -1
    for p in l1s:
        cnt = 0
        for q in [p] + [x for x in t if x.startswith(p + ".") and t[x]["depth"] == 2]:
            for w in _phrase(t[q], 4).split():
                if _color(w) == cat:
                    cnt += 1
        if cnt > bestc:
            bestc, best = cnt, p
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clustering-dir", type=Path, required=True)
    ap.add_argument("--hier-base", type=Path, required=True)
    ap.add_argument("--npz-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--label-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    rows = [("k50_cosmos", "pai_cosmos", "cosmos.npz", "cosmos", "Cosmos-Embed1", False),
            ("k50_caption", "pai_caption", "caption.npz", "caption", "Caption (Qwen3-Emb-8B)", False),
            ("k50_visual", "pai_visual", "visual.npz", "visual", "Florence-2/SigLIP", True)]
    rng = np.random.default_rng(args.seed)
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 14.5), gridspec_kw=dict(width_ratios=[1, 1.05]))

    for r, (flat, hierd, npz, key, name, center) in enumerate(rows):
        t = json.loads((args.hier_base / hierd / "hier_topics.json").read_text())
        ha = pd.read_parquet(args.hier_base / hierd / "hier_assignments.parquet",
                             columns=["clip_id", "path"])
        ha["clip_id"] = ha["clip_id"].astype(str)
        ha["l1"] = ha["path"].astype(str).str.split(".").str[0]
        uj = json.loads((args.clustering_dir / flat / "umap.json").read_text())
        pts = []
        for cid, ps in uj["clips"].items():
            for (x, y), clip in zip(ps, uj["clip_ids"].get(cid, [])):
                pts.append((str(clip), x, y))
        ov = pd.DataFrame(pts, columns=["clip_id", "x", "y"]).merge(
            ha[["clip_id", "l1"]], on="clip_id", how="inner")
        l1s = sorted(ov["l1"].dropna().unique(), key=lambda s: int(s))
        cat, catname = _FOCUSCAT[key]
        focus = _pick_focus(t, l1s, cat)
        fcol = _PAL[l1s.index(focus) % len(_PAL)]

        # --- overview ---
        axO = axes[r, 0]; axO.set_facecolor("#f6f6f9")
        bg = ov[ov["l1"] != focus]; fg = ov[ov["l1"] == focus]
        axO.scatter(bg["x"], bg["y"], s=3, color="#d7d7de", alpha=0.5, linewidths=0, rasterized=True)
        axO.scatter(fg["x"], fg["y"], s=6, color=fcol, alpha=0.85, linewidths=0, rasterized=True)
        fx, fy = _dense_xy(fg["x"].values, fg["y"].values)
        axO.text(fx, fy, f"C{focus}", fontsize=11, fontweight="bold", ha="center", va="center",
                 color=fcol, zorder=6, path_effects=_HALO)
        axO.set_title(f"{name}: {len(l1s)} level-1 clusters", fontsize=12, fontweight="bold")
        axO.set_xticks([]); axO.set_yticks([])
        for s in axO.spines.values():
            s.set_visible(False)

        # --- zoom: re-cluster the focus branch ---
        d = np.load(args.npz_dir / npz, allow_pickle=True)
        idx = {str(c): i for i, c in enumerate(d["clip_ids"])}
        emb = np.asarray(d["embeddings"], dtype=np.float32)
        if center:
            emb = emb - emb.mean(0, keepdims=True)
            nrm = np.linalg.norm(emb, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
            emb = emb / nrm
        clips = ha[(ha["l1"] == focus) & (ha["clip_id"].isin(idx))]
        if len(clips) > args.sample:
            clips = clips.iloc[rng.choice(len(clips), args.sample, replace=False)]
        rowi = np.array([idx[c] for c in clips["clip_id"]], dtype=np.int64)
        coords = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=args.seed).fit_transform(
            np.ascontiguousarray(emb[rowi]))
        l2 = clips["path"].astype(str).values
        cats2 = sorted(set(l2), key=lambda s: [int(x) for x in s.split(".")])
        axZ = axes[r, 1]; axZ.set_facecolor("#f6f6f9")
        for ci, c in enumerate(cats2):
            m = l2 == c
            axZ.scatter(coords[m, 0], coords[m, 1], s=7, color=_PAL[ci % len(_PAL)],
                        alpha=0.7, linewidths=0, rasterized=True)
        order = sorted(range(len(cats2)), key=lambda ci: -(l2 == cats2[ci]).sum())
        for ci in order[:args.label_k]:
            c = cats2[ci]; m = l2 == c
            lx, ly = _dense_xy(coords[m, 0], coords[m, 1])
            axZ.text(lx, ly, "\n".join(__import__("textwrap").wrap(_phrase(t[c], 2), 16)),
                     fontsize=7.5, ha="center", va="center", color="#1a1a1a", zorder=6,
                     path_effects=_HALO)
        axZ.set_title("C%s “%s”  (%d sub-clusters)" % (focus, _phrase(t[focus]), len(cats2)),
                      fontsize=10.5, color=fcol, fontweight="bold")
        axZ.set_xticks([]); axZ.set_yticks([])
        for s in axZ.spines.values():
            s.set_visible(False)

        con = ConnectionPatch(xyA=(fx, fy), coordsA=axO.transData, xyB=(0.02, 0.5),
                              coordsB=axZ.transAxes, arrowstyle="-|>", mutation_scale=22,
                              lw=2.0, color=fcol, alpha=0.85, zorder=20)
        fig.add_artist(con)

    fig.tight_layout()   # no suptitle; the figure caption serves as the title
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
