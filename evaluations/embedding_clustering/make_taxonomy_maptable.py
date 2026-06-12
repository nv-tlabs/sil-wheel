#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Condensed hierarchy view: a level-1 overview UMAP (map) + a LaTeX taxonomy
table (level-1 branch -> its level-2 children), the map+table analogue of the
overlay figure. Replaces the busy 4-panel drill-down with one map and one table.

Reads the flat run's ``umap.json`` for 2D coordinates and the hierarchical run's
``hier_topics.json`` / ``hier_assignments.parquet`` for the taxonomy. Writes a
tight-cropped overview PNG (branches coloured + C-id labels) and, with
``--emit-tex``, an ``\\input``-able tabular (swatch + C-id + level-1 phrase, then
its top level-2 child phrases).

    python make_taxonomy_maptable.py --run-dir CLR/k50_cosmos \
        --hier-dir hier/pai_cosmos --embed-name "Cosmos-Embed1" \
        --out overview_cosmos.png --emit-tex tables/taxonomy_cosmos.tex
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

import figstyle  # noqa: F401  (NVIDIA Sans)
from make_taxonomy_compare import NEUTRAL as _NEU, _color as _cat

_PAL = ["#E58606", "#5D69B1", "#52BCA3", "#99C945", "#CC61B0", "#24796C",
        "#DAA51B", "#2F8AC4", "#764E9F", "#ED645A", "#CC3A8E", "#A5AA99"]


def _cw(w):  # colour a word by its category (shared scaffolding stays black)
    c = _cat(w)
    return _esc(w) if c == _NEU else \
        r"\textcolor[HTML]{%s}{\textbf{%s}}" % (c.lstrip("#").upper(), _esc(w))


def _cphrase(s):
    return " ".join(_cw(w) for w in s.split())
_HALO = [pe.withStroke(linewidth=3.0, foreground="white")]


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _esc(s):
    return s.replace("&", r"\&").replace("_", r"\_").replace("#", r"\#").replace("%", r"\%")


def _phrase(v, n=3):
    return v.get("description") or ", ".join(v.get("keywords", [])[:n])


def _dense_xy(xs, ys, bins=24):
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
    ap.add_argument("--run-dir", type=Path, required=True, help="flat run with umap.json")
    ap.add_argument("--hier-dir", type=Path, required=True)
    ap.add_argument("--embed-name", default="")
    ap.add_argument("--child-k", type=int, default=3, help="level-2 children listed per branch")
    ap.add_argument("--out", type=Path, required=True, help="overview map PNG")
    ap.add_argument("--emit-tex", type=Path, default=None, help="taxonomy tabular path")
    args = ap.parse_args(argv)

    topics = json.loads((args.hier_dir / "hier_topics.json").read_text())
    ha = pd.read_parquet(args.hier_dir / "hier_assignments.parquet", columns=["clip_id", "path"])
    ha["clip_id"] = ha["clip_id"].astype(str)
    ha["l1"] = ha["path"].astype(str).str.split(".").str[0]

    umap_json = json.loads((args.run_dir / "umap.json").read_text())
    rows = []
    for cid, pts in umap_json["clips"].items():
        for (x, y), clip in zip(pts, umap_json["clip_ids"].get(cid, [])):
            rows.append((str(clip), x, y))
    ov = pd.DataFrame(rows, columns=["clip_id", "x", "y"]).merge(
        ha[["clip_id", "l1"]], on="clip_id", how="inner")
    l1s = sorted(ov["l1"].dropna().unique(), key=lambda s: int(s))
    color = {l1: _PAL[i % len(_PAL)] for i, l1 in enumerate(l1s)}

    # --- map: all level-1 branches coloured, labelled by C-id ---
    fig = plt.figure(figsize=(6.2, 6.0), dpi=160)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.93])
    ax.set_facecolor("#f6f6f9")
    for l1 in l1s:
        s = ov[ov["l1"] == l1]
        ax.scatter(s["x"], s["y"], s=5, color=color[l1], alpha=0.65, linewidths=0, rasterized=True)
    for l1 in l1s:
        s = ov[ov["l1"] == l1]
        lx, ly = _dense_xy(s["x"].values, s["y"].values)
        ax.text(lx, ly, f"C{l1}", fontsize=11, fontweight="bold", ha="center", va="center",
                color=color[l1], zorder=6, path_effects=_HALO)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if args.embed_name:
        ax.set_title(f"{args.embed_name}: {len(l1s)} level-1 branches",
                     fontsize=13, fontweight="bold")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", args.out)

    # --- taxonomy table: branches by size, each with its top level-2 children ---
    if args.emit_tex:
        size = {l1: int(topics[l1]["size"]) for l1 in l1s}
        out = [r"\begin{tabular}{@{}l p{0.86\linewidth}@{}}", r"\toprule"]
        first = True
        for l1 in sorted(l1s, key=lambda p: -size[p]):
            if not first:
                out.append(r"\addlinespace[2.5pt]")
            first = False
            r, g, b = _rgb(color[l1])
            sw = r"\textcolor[rgb]{%.3f,%.3f,%.3f}{\rule{1.1ex}{1.1ex}}" % (r, g, b)
            out.append(r"%s~\textbf{C%s} & %s \\" % (sw, l1, _cphrase(_phrase(topics[l1]))))
            kids = sorted((q for q in topics if q.startswith(l1 + ".") and topics[q]["depth"] == 2),
                          key=lambda q: -topics[q]["size"])[:args.child_k]
            for q in kids:
                out.append(r" & \quad\textendash\ %s \\" % _cphrase(_phrase(topics[q], 2)))
        out += [r"\bottomrule", r"\end{tabular}"]
        args.emit_tex.parent.mkdir(parents=True, exist_ok=True)
        args.emit_tex.write_text("\n".join(out) + "\n")
        print("wrote", args.emit_tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
