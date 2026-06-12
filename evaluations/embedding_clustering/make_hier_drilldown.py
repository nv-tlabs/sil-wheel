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

"""Hierarchical drill-down figure: 10 level-1 clusters -> pick a few branches ->
re-UMAP each branch's clips colored by its level-2 sub-clusters.

Top-left panel is the overview (the whole pool's UMAP colored by the 10 level-1
taxonomy branches, with the picked branches highlighted). Each remaining panel
zooms into one picked branch: its member clips are re-embedded with UMAP and
colored by that branch's level-2 sub-clusters, with sub-topic labels, so the
scene type gets progressively finer while the hierarchy stays visible.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

import figstyle  # noqa: F401  (registers NVIDIA Sans, sets it as default)

# Plotly "Vivid" qualitative palette (px.colors.qualitative.Vivid); cycled for clusters
_PAL = ["#E58606", "#5D69B1", "#52BCA3", "#99C945", "#CC61B0", "#24796C",
        "#DAA51B", "#2F8AC4", "#764E9F", "#ED645A", "#CC3A8E", "#A5AA99"]
_HALO = [pe.withStroke(linewidth=3.2, foreground="white")]   # white outline for label text


def _label(topics, node, n_kw=3, wrap=None):
    """LLM one-phrase description if present, else top keywords. Optionally wrapped."""
    v = topics.get(str(node), {})
    if isinstance(v, dict):
        s = v.get("description") or ", ".join(v.get("keywords", [])[:n_kw])
    else:
        s = ", ".join(v[:n_kw]) if v else ""
    s = s or str(node)
    return "\n".join(textwrap.wrap(s, wrap)) if wrap else s


def _dense_xy(xs, ys, bins=24):
    """Label anchor = mean of points in the densest 2D-histogram bin (avoids
    dropping a label in the empty gap between two sub-blobs that a median hits)."""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if len(xs) < 5:
        return float(np.median(xs)), float(np.median(ys))
    H, xe, ye = np.histogram2d(xs, ys, bins=bins)
    bx, by = np.unravel_index(int(np.argmax(H)), H.shape)
    m = (xs >= xe[bx]) & (xs <= xe[bx + 1]) & (ys >= ye[by]) & (ys <= ye[by + 1])
    return (float(xs[m].mean()), float(ys[m].mean())) if m.any() \
        else (float(np.median(xs)), float(np.median(ys)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True, help="flat run with umap.json (overview)")
    ap.add_argument("--hier-dir", type=Path, required=True)
    ap.add_argument("--npz", type=Path, required=True, help="npz (clip_ids + embeddings)")
    ap.add_argument("--picks", nargs="+", default=None,
                    help="level-1 ids to drill into (default: the N largest branches)")
    ap.add_argument("--n-branches", type=int, default=3, help="branches to drill if --picks unset")
    ap.add_argument("--label-k", type=int, default=6,
                    help="level-2 sub-clusters to label per panel (largest by size)")
    ap.add_argument("--embed-name", default="Cosmos-Embed1", help="encoder name for titles")
    ap.add_argument("--center", action="store_true",
                    help="mean-center+renorm the zoom embeddings (match a centered hier run)")
    ap.add_argument("--sample", type=int, default=5000, help="max clips per branch for the zoom UMAP")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    topics = json.loads((args.hier_dir / "hier_topics.json").read_text())
    ha = pd.read_parquet(args.hier_dir / "hier_assignments.parquet", columns=["clip_id", "path"])
    ha["clip_id"] = ha["clip_id"].astype(str)
    ha["l1"] = ha["path"].astype(str).str.split(".").str[0]

    # overview coords from the flat run's umap.json, joined to level-1
    umap_json = json.loads((args.run_dir / "umap.json").read_text())
    rows = []
    for cid, pts in umap_json["clips"].items():
        for (x, y), clip in zip(pts, umap_json["clip_ids"].get(cid, [])):
            rows.append((str(clip), x, y))
    ov = pd.DataFrame(rows, columns=["clip_id", "x", "y"]).merge(
        ha[["clip_id", "l1"]], on="clip_id", how="inner")
    l1s = sorted(ov["l1"].dropna().unique(), key=lambda s: int(s))

    # embeddings for the zoom panels
    d = np.load(args.npz, allow_pickle=True)
    idx = {str(c): i for i, c in enumerate(d["clip_ids"])}
    emb = np.asarray(d["embeddings"], dtype=np.float32)
    if args.center:                       # match a centered hier run (e.g. visual)
        emb = emb - emb.mean(axis=0, keepdims=True)
        nrm = np.linalg.norm(emb, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
        emb = emb / nrm
    rng = np.random.default_rng(args.seed)

    # default picks = the N largest level-1 branches
    picks = args.picks or [str(c) for c in ha["l1"].value_counts().index[:args.n_branches]]
    picks = [p for p in picks if p in set(l1s)]
    n_panels = 1 + len(picks)
    ncol = 2
    nrow = (n_panels + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 6.6, nrow * 5.5),
                             constrained_layout=True)
    axes = axes.ravel()
    pick_color = {p: _PAL[l1s.index(p) % len(_PAL)] for p in picks}

    def _style(ax):
        ax.set_facecolor("#f6f6f9")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # --- overview: faint grey backdrop, picked branches glow + colour ---
    ax = axes[0]; _style(ax)
    bg = ov[~ov["l1"].isin(picks)]
    ax.scatter(bg["x"], bg["y"], s=3, color="#dadae2", alpha=0.5, linewidths=0,
               zorder=1, rasterized=True)
    for p in picks:
        sub = ov[ov["l1"] == p]
        ax.scatter(sub["x"], sub["y"], s=34, color=pick_color[p], alpha=0.05,
                   linewidths=0, zorder=2, rasterized=True)          # soft glow
        ax.scatter(sub["x"], sub["y"], s=6, color=pick_color[p], alpha=0.8,
                   linewidths=0, zorder=3, rasterized=True)
    for p in picks:
        sub = ov[ov["l1"] == p]
        lx, ly = _dense_xy(sub["x"].values, sub["y"].values)
        ax.text(lx, ly, _label(topics, p, wrap=22), fontsize=9.5, fontweight="bold",
                ha="center", va="center", color=pick_color[p], zorder=6, path_effects=_HALO)
    ax.set_title(f"{args.embed_name}: {len(l1s)} level-1 clusters, {len(picks)} drilled below",
                 fontsize=12.5, fontweight="semibold", color="#222222")

    # --- zoom panels: each picked branch re-UMAP'd, coloured by level-2 ---
    for k, p in enumerate(picks):
        ax = axes[k + 1]; _style(ax)
        clips = ha[ha["l1"] == p].copy()
        clips = clips[clips["clip_id"].isin(idx)]
        if len(clips) > args.sample:
            clips = clips.iloc[rng.choice(len(clips), args.sample, replace=False)]
        rowi = np.array([idx[c] for c in clips["clip_id"]], dtype=np.int64)
        sub_emb = np.ascontiguousarray(emb[rowi], dtype=np.float32)
        coords = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                           random_state=args.seed).fit_transform(sub_emb)
        l2 = clips["path"].astype(str).values
        cats = sorted(set(l2), key=lambda s: [int(t) for t in s.split(".")])
        ccolor = {cat: _PAL[ci % len(_PAL)] for ci, cat in enumerate(cats)}
        for cat in cats:
            m = l2 == cat
            ax.scatter(coords[m, 0], coords[m, 1], s=7, color=ccolor[cat], alpha=0.65,
                       linewidths=0, rasterized=True)
        sizes2 = {cat: int((l2 == cat).sum()) for cat in cats}
        for cat in sorted(cats, key=lambda c: -sizes2[c])[:args.label_k]:
            m = l2 == cat
            lx, ly = _dense_xy(coords[m, 0], coords[m, 1])
            ax.text(lx, ly, _label(topics, cat, n_kw=2, wrap=18), fontsize=7.5,
                    ha="center", va="center", color="#1a1a1a", zorder=6, path_effects=_HALO)
        ttl = f"“{_label(topics, p)}”  ({len(cats)} sub-clusters)"
        ax.set_title("\n".join(textwrap.wrap(ttl, 48)), fontsize=10.5,
                     color=pick_color[p], fontweight="bold")

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{args.embed_name} hierarchical drill-down: level-1 branches refine "
                 "into level-2 scene types", fontsize=13.5, color="#333333", fontweight="semibold")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
