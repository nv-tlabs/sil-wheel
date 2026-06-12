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

"""Interactive hierarchical-taxonomy visualization (plotly sunburst + treemap).

Reads a recursive-k-means run's ``hier_topics.json`` ({path: {keywords,
description, size, depth}}) and renders the level-1 -> level-2 topic taxonomy as
a sunburst (rings) and a treemap (nested boxes). Wedge/box size = cluster size;
on-figure label = the topic description (or top keywords); hover = TF-IDF
keywords. Writes a self-contained interactive HTML and, if kaleido is available,
a static PNG/PDF.

    python make_hier_viz.py --hier-dir ./hier/full_cosmos --title "Full / Cosmos-Embed1" --png
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import plotly.graph_objects as go

# Plotly "Vivid" qualitative palette for the level-1 families (matches the
# overlay + drill-down figures).
PALETTE = [
    "#E58606", "#5D69B1", "#52BCA3", "#99C945", "#CC61B0", "#24796C",
    "#DAA51B", "#2F8AC4", "#764E9F", "#ED645A", "#CC3A8E", "#A5AA99",
]


# Stop-word-ish fragments the TF-IDF keyword extractor leaves behind; we drop
# phrases made entirely of these so on-figure labels stay informative.
_STOP = {
    "the", "and", "a", "an", "of", "to", "in", "on", "with", "are", "is", "be",
    "ego", "two", "this", "that", "it", "its", "their", "depicts", "scene",
    "video", "shows", "showing", "vehicle", "vehicles",
}


def _clean_keywords(kws: list[str], k: int) -> list[str]:
    """Pick up to ``k`` informative, non-redundant keywords for a label.

    Drops empty/stop-word-only fragments and any phrase that is a substring of
    (or superset of) one already chosen, so we don't print "intersection" and
    "the intersection" side by side.
    """
    out: list[str] = []
    for w in kws:
        w = (w or "").strip()
        if not w:
            continue
        wl = w.lower()
        if all(tok in _STOP for tok in wl.split()):
            continue
        if any(wl in o.lower() or o.lower() in wl for o in out):
            continue
        out.append(w)
        if len(out) >= k:
            break
    return out


def _label(v: dict, *, k: int, width: int, use_desc: bool = True) -> str:
    """Readable, non-truncated on-figure label.

    Level-1 headers prefer the curated LLM ``description`` (room for a phrase);
    the dense level-2 boxes set ``use_desc=False`` and use short keywords, since
    a full phrase will not fit. Long text is *wrapped* (HTML ``<br>``), not
    clipped, so nothing is trimmed.
    """
    desc = (v.get("description") or "").strip() if use_desc else ""
    text = desc or " · ".join(_clean_keywords(v.get("keywords", []), k))
    if not text:
        return "—"
    return "<br>".join(textwrap.wrap(text, width=width, break_long_words=False))


def _hex_to_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def _build(hier_dir: Path):
    topics = json.loads((hier_dir / "hier_topics.json").read_text())
    l1 = sorted((p for p, v in topics.items() if v["depth"] == 1), key=lambda p: int(p))
    color_of = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(l1)}

    root_id = hier_dir.name
    root_size = sum(topics[p]["size"] for p in l1)

    # Only label the largest few level-2 wedges per branch so text can size up;
    # the other wedges still show (colour + hover), just without a cramped label.
    from collections import defaultdict
    _by_parent = defaultdict(list)
    for path, v in topics.items():
        if v["depth"] == 2:
            _by_parent[path.split(".")[0]].append((path, v["size"]))
    label_l2 = {p for items in _by_parent.values()
                for p, _ in sorted(items, key=lambda x: -x[1])[:5]}

    ids, labels, parents, values, colors, custom = [], [], [], [], [], []
    # root
    ids.append(root_id)
    labels.append(f"<b>{root_id}</b>")
    parents.append("")
    values.append(root_size)
    colors.append("rgba(0,0,0,0)")
    custom.append([f"{root_size:,} clips", ""])

    for path, v in topics.items():
        depth = v["depth"]
        l1key = path.split(".")[0]
        base = color_of.get(l1key, "#888888")
        # The level-1 ring has room for 3 wrapped keywords; the dense level-2
        # ring (often 100 wedges) gets a tighter 2-keyword label so text stays
        # legible instead of overlapping. Full keyword list is always on hover.
        if depth == 1:
            on_fig = _label(v, k=3, width=18, use_desc=True)    # LLM phrase header
        elif path in label_l2:
            on_fig = _label(v, k=2, width=13, use_desc=False)   # short keywords (largest wedges)
        else:
            on_fig = ""                                          # tiny wedge: colour + hover only
        kw = ", ".join(v.get("keywords", [])[:8])
        ids.append(path)
        labels.append(on_fig)
        parents.append(l1key if depth == 2 else root_id)
        values.append(v["size"])
        colors.append(base if depth == 1 else _hex_to_rgba(base, 0.55))
        custom.append([f"{v['size']:,} clips", kw])

    return ids, labels, parents, values, colors, custom, len(l1), root_size


def _fig(kind, ids, labels, parents, values, colors, custom, n_l1, root_size, *, title):
    common = dict(
        ids=ids, labels=labels, parents=parents, values=values,
        marker=dict(colors=colors, line=dict(color="white", width=1)),
        customdata=custom,
        hovertemplate="<b>%{label}</b><br>%{customdata[0]}<br>"
                      "<i>%{customdata[1]}</i><extra></extra>",
        branchvalues="total",
    )
    layout = dict(
        title=dict(text=f"<b>{title}</b> — cluster-topic taxonomy "
                        f"({n_l1} level-1 topics, {len(ids)-1-n_l1} level-2 subtopics, "
                        f"{root_size:,} clips)", x=0.5, font=dict(size=18)),
        margin=dict(t=64, l=8, r=8, b=8),
        font=dict(family="NVIDIA Sans, Helvetica, Arial", size=15),
        paper_bgcolor="white",
    )
    if kind == "sunburst":
        # Radial text reads outward along each wedge and fits far more
        # characters in a thin level-2 wedge than tangential. Only the largest
        # few level-2 wedges per branch carry a label (set in ``_build``), so we
        # can pin a readable floor and hide (not shrink) the rest.
        trace = go.Sunburst(insidetextorientation="radial", textfont=dict(size=16), **common)
        layout["uniformtext"] = dict(minsize=11, mode="hide")
    else:
        # Treemap boxes have room to wrap full labels; pin a readable floor and
        # hide (don't clip) the rare box too small for even 9pt text.
        trace = go.Treemap(tiling=dict(pad=3), textposition="middle center", **common)
        layout["uniformtext"] = dict(minsize=9, mode="hide")
    fig = go.Figure(trace)
    fig.update_layout(**layout)
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hier-dir", type=Path, required=True,
                    help="dir holding hier_topics.json (from run_hier_cluster.py)")
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    ap.add_argument("--title", default=None)
    ap.add_argument("--png", action="store_true", help="also export static PNG/PDF (needs kaleido)")
    args = ap.parse_args(argv)

    hier_dir = args.hier_dir
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    title = args.title or hier_dir.name
    stem = f"hier_{hier_dir.name}"

    data = _build(hier_dir)
    for kind in ("sunburst", "treemap"):
        fig = _fig(kind, *data, title=title)
        html = out / f"{stem}_{kind}.html"
        fig.write_html(str(html), include_plotlyjs=True, full_html=True)
        print("wrote", html, flush=True)
        if args.png:
            try:
                # Square canvas for the sunburst (more arc-length per wedge =
                # more room for the dense level-2 labels); wider for the treemap.
                w, h = (1800, 1800) if kind == "sunburst" else (1800, 1500)
                fig.write_image(str(out / f"{stem}_{kind}.png"), width=w, height=h, scale=2)
                print("wrote", out / f"{stem}_{kind}.png", flush=True)
            except Exception as e:
                print(f"[png] skipped ({kind}): {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
