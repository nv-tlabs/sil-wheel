#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Single-figure comparison of the three embeddings' level-1 taxonomies.

Three columns (one per embedding), each listing its 10 level-1 branch topic
phrases. Words are coloured by category -- appearance/lighting/weather,
activity/agents, objects/scenery -- so each embedding's differing FOCUS is
visible at a glance (which colour dominates a column). Road/place scaffolding
(street, highway, intersection, ...) stays neutral grey, since it is shared.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle  # noqa: F401  (NVIDIA Sans)

APPEAR = "#2166AC"; ACTIVITY = "#C2451E"; OBJECT = "#1B7837"; NEUTRAL = "#7a7a7a"

# Only each embedding's SIGNATURE vocabulary is coloured; words shared across all
# three (night, illuminated, intersection, street, highway, residential, ...) stay
# neutral so the distinctive focus of each column actually stands out.
_CAT = {
    APPEAR: {"sunny", "sunlit", "daytime", "overcast", "cloudy", "cloud", "clouds",
             "wet", "rainy", "rain", "snow", "snowy", "dusk", "sunset", "twilight",
             "gray", "grey", "gloomy", "clear", "sky", "shadows", "shadow", "blue"},
    ACTIVITY: {"traffic", "light", "lights", "signal", "signals", "red", "roundabout",
               "roundabouts", "stop", "stopping", "stopped", "waiting", "yield",
               "yielding", "accelerating", "turning", "flowing", "changing", "crossing",
               "pedestrians", "cyclists"},
    OBJECT: {"parked", "cars", "trees", "tree-lined", "bare", "forest", "fields",
             "field", "concrete", "buildings", "building", "houses", "house", "village",
             "tunnel", "tunnels", "construction", "sedan", "suv", "suvs", "trucks",
             "truck", "semi", "vehicles", "pole", "poles"},
}
_WORD2COL = {w: c for c, ws in _CAT.items() for w in ws}


def _phrase(v, n=3):
    return v.get("description") or ", ".join(v.get("keywords", [])[:n])


def _color(word):
    return _WORD2COL.get(re.sub(r"[^a-z-]", "", word.lower()), NEUTRAL)


def _esc(s):
    return s.replace("&", r"\&").replace("_", r"\_").replace("#", r"\#").replace("%", r"\%")


def _emit_tex(data, path):
    """Colour-coded 3-column LaTeX table (C-id linked to the maps)."""
    def tw(w):
        c = _color(w)
        return _esc(w) if c == NEUTRAL else \
            r"\textcolor[HTML]{%s}{\textbf{%s}}" % (c.lstrip("#").upper(), _esc(w))
    items = [it for _, it in data]
    n = max(len(it) for it in items)
    out = [r"\begin{tabular}{@{}p{0.30\linewidth} p{0.30\linewidth} p{0.30\linewidth}@{}}", r"\toprule"]
    for i in range(n):
        cells = []
        for it in items:
            cid, ph = it[i] if i < len(it) else ("", "")
            cells.append((r"\textbf{C%s}~%s" % (cid, " ".join(tw(w) for w in ph.split()))) if ph else "")
        out.append(" & ".join(cells) + r" \\[3pt]")
    out += [r"\bottomrule", r"\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    print("wrote", path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hier-base", type=Path, required=True, help="dir holding pai_<emb>/")
    ap.add_argument("--out", type=Path, default=None, help="optional PNG render")
    ap.add_argument("--emit-tex", type=Path, default=None, help="LaTeX colour-coded table")
    ap.add_argument("--bars", type=Path, default=None,
                    help="grouped bar chart of topic-word category composition per embedding")
    args = ap.parse_args(argv)

    cols = [("pai_cosmos", "Cosmos-Embed1"), ("pai_caption", "Caption (Qwen3-Emb-8B)"),
            ("pai_visual", "Visual (Florence-2/SigLIP)")]
    data = []
    for d, name in cols:
        t = json.loads((args.hier_base / d / "hier_topics.json").read_text())
        l1 = sorted((p for p, v in t.items() if v["depth"] == 1), key=lambda p: -t[p]["size"])
        data.append((name, [(p, _phrase(t[p])) for p in l1]))

    fig = plt.figure(figsize=(15, 6.6), dpi=160)
    axes = [fig.add_axes([0.01 + j * 0.331, 0.02, 0.32, 0.88]) for j in range(3)]
    fig.canvas.draw(); rnd = fig.canvas.get_renderer()

    for ax, (name, phrases) in zip(axes, data):
        ax.axis("off")
        ax.text(0.5, 1.0, name, ha="center", va="top", fontsize=13, fontweight="bold",
                transform=ax.transAxes)
        y = 0.94
        for cid, ph in phrases:
            x = 0.0
            ct = ax.text(x, y, f"C{cid} ", color="#444444", fontsize=9.3, fontweight="bold",
                         va="top", transform=ax.transAxes)
            x += ct.get_window_extent(rnd).width / ax.bbox.width
            for word in ph.split():
                col = _color(word)
                bold = col != NEUTRAL
                while True:
                    txt = ax.text(x, y, word + " ", color=col, fontsize=9.3,
                                  fontweight="bold" if bold else "normal",
                                  va="top", transform=ax.transAxes)
                    w = txt.get_window_extent(rnd).width / ax.bbox.width
                    if x > 0.001 and x + w > 0.99:
                        txt.remove(); x = 0.0; y -= 0.038
                        continue
                    break
                x += w
            y -= 0.038 + 0.018          # next phrase: line height + gap

    # legend (drawn swatches; NVIDIA Sans has no U+25A0 glyph)
    for i, (c, lab) in enumerate([(APPEAR, "appearance / lighting / weather"),
                                  (ACTIVITY, "activity / signals / agents"),
                                  (OBJECT, "objects / scenery")]):
        x = 0.06 + i * 0.31
        fig.add_artist(plt.Rectangle((x, 0.96), 0.012, 0.022, color=c,
                                     transform=fig.transFigure, clip_on=False))
        fig.text(x + 0.018, 0.971, lab, color=c, fontsize=11, fontweight="bold",
                 ha="left", va="center")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=160, bbox_inches="tight", pad_inches=0.06)
        print("wrote", args.out)
    plt.close(fig)
    if args.emit_tex:
        _emit_tex(data, args.emit_tex)

    if args.bars:
        import numpy as np
        cat_idx = {APPEAR: 0, ACTIVITY: 1, OBJECT: 2}
        emb_col = ["#E58606", "#5D69B1", "#52BCA3"]
        comp = []
        for d, _ in cols:
            t = json.loads((args.hier_base / d / "hier_topics.json").read_text())
            cnt = [0, 0, 0]
            for v in t.values():
                if v["depth"] in (1, 2):
                    for w in _phrase(v, 4).split():
                        i = cat_idx.get(_color(w))
                        if i is not None:
                            cnt[i] += 1
            s = sum(cnt) or 1
            comp.append([c / s * 100 for c in cnt])
        fig2, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
        x = np.arange(3); w = 0.26
        for k, (_, name) in enumerate(cols):
            ax.bar(x + (k - 1) * w, comp[k], w, label=name, color=emb_col[k],
                   edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["appearance /\nlighting / weather", "activity /\nsignals / agents",
                            "objects /\nscenery"], fontsize=10)
        ax.set_ylabel("share of topic words (\\%)".replace("\\", ""), fontsize=10)
        ax.set_ylim(0, 60)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=9, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
        ax.grid(axis="y", color="#e8e8e8", linewidth=0.8); ax.set_axisbelow(True)
        args.bars.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(args.bars, dpi=160, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig2)
        print("wrote", args.bars)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
