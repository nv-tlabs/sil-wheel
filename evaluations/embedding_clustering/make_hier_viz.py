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
from pathlib import Path

import plotly.graph_objects as go

# Clean qualitative palette for the level-1 families (extended Plotly/D3).
PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B",
    "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#1F77B4", "#2CA02C",
]


def _short(text: str, n: int = 28) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


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
        desc = v.get("description") or ", ".join(v.get("keywords", [])[:3])
        kw = ", ".join(v.get("keywords", [])[:8])
        ids.append(path)
        labels.append(_short(desc))
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
    trace = (go.Sunburst(insidetextorientation="radial", **common)
             if kind == "sunburst" else go.Treemap(tiling=dict(pad=2), **common))
    fig = go.Figure(trace)
    fig.update_layout(
        title=dict(text=f"<b>{title}</b> — cluster-topic taxonomy "
                        f"({n_l1} top topics → {len(ids)-1-n_l1} subtopics, "
                        f"{root_size:,} clips)", x=0.5, font=dict(size=16)),
        margin=dict(t=56, l=8, r=8, b=8), font=dict(family="Helvetica, Arial", size=12),
        paper_bgcolor="white",
    )
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
                fig.write_image(str(out / f"{stem}_{kind}.png"), width=1100, height=900, scale=2)
                print("wrote", out / f"{stem}_{kind}.png", flush=True)
            except Exception as e:
                print(f"[png] skipped ({kind}): {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
