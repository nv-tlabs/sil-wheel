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

"""Figures and tables for the embedding-space analysis. Subcommands:

* ``themes``        -- LLM one-phrase name per cluster (backfilled into cluster_topics.json).
* ``umap-overview`` -- (pools x embeddings) grid of UMAP scatters, coloured by cluster.
* ``overlay-maps``  -- one embedding's UMAP with its distinct clusters pinned to a clip.
* ``tables``        -- the colour-coded topic table and the distinctive-terms table.
* ``hierarchical``  -- drill one branch per embedding into its sub-clusters.

The genuinely heavy/optional deps (``boto3``+``decord`` for overlay-maps, ``umap``
for hierarchical) are imported inside their subcommand, so e.g. ``tables`` and
``themes`` run without them.

    python make_figures.py overlay-maps --run-dir ./clustering/k50_cosmos --map-only --out o.png
    python make_figures.py tables --clustering-dir ./clustering --what both
"""

import argparse
import io
import json
import os
import sqlite3
import textwrap
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError

import boto3
import decord
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import ConnectionPatch
from PIL import Image, ImageFilter

import figlib
from figlib import ACTIVITY, APPEAR, NEUTRAL, OBJECT
from figlib import categorize as _color
from figlib import dense_xy as _dense_xy
from figlib import distinct_clusters as _distinct_clusters
from figlib import distinctive_terms, latex_escape, topic_profiles

matplotlib.use("Agg")
figlib.use_nvidia_style()

# Plotly "Vivid" qualitative palette, shared by every figure.
_PAL = [
    "#E58606",
    "#5D69B1",
    "#52BCA3",
    "#99C945",
    "#CC61B0",
    "#24796C",
    "#DAA51B",
    "#2F8AC4",
    "#764E9F",
    "#ED645A",
    "#CC3A8E",
    "#A5AA99",
]


# themes -- offline LLM cluster naming (backfill)

_THEME_SYSTEM = (
    "You are given TF-IDF keyword terms from a cluster of driving-scene video "
    "captions. Reply with ONE short noun phrase (4-9 words) naming the common "
    "SCENE or SCENARIO (road type, setting, weather, lighting, time of day, or "
    "traffic situation). Do NOT mention the camera or viewpoint: never use "
    "'first-person', 'first person view', 'dashcam', 'POV', 'point of view', "
    "'ego vehicle', or 'driving perspective'. Output only the phrase: no quotes, "
    "no leading 'A '/'The ', no trailing punctuation."
)


def _summarize(keywords, base, model, key, retries=6):
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": _THEME_SYSTEM},
                {"role": "user", "content": "Keywords: " + ", ".join(keywords)},
            ],
            "max_tokens": 1024,
            "temperature": 0.2,
        }
    ).encode()
    url = base.rstrip("/") + "/chat/completions"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            content = d["choices"][0]["message"].get("content")
            if content and content.strip():
                return content.strip().strip('"').strip("'").rstrip(".")
            if (
                attempt < retries - 1
            ):  # null content (reasoning-model truncation) -> retry
                time.sleep(1.0 + attempt)
                continue
            return ""
        except HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2**attempt)
                continue
            raise
        except (URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(2**attempt)
                continue
            raise
    return ""


def cmd_themes(args):
    key = os.environ.get("NV_INFERENCE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not (key and args.base_url and args.model):
        raise SystemExit("need a key (NV_INFERENCE_API_KEY) + --base-url + --model")

    for rid in args.runs:
        p = args.clustering_dir / rid / args.topics_name
        tj = json.loads(p.read_text())
        flat = "topics" not in tj  # hier_topics.json is a flat {node: {...}} dict
        topics = tj if flat else tj["topics"]
        cids = [
            c for c, v in topics.items() if isinstance(v, dict) and v.get("keywords")
        ]

        def work(c):
            try:
                return c, _summarize(
                    topics[c]["keywords"][: args.top_k], args.base_url, args.model, key
                )
            except Exception as e:
                print(
                    f"  {rid} {c} failed: {type(e).__name__} {str(e)[:80]}", flush=True
                )
                return c, ""

        n_ok = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for c, desc in ex.map(work, cids):
                if desc:
                    topics[c]["description"] = desc
                    n_ok += 1
        if not flat:
            tj["topics"] = topics
        p.write_text(json.dumps(tj))
        print(f"{rid}: {n_ok}/{len(cids)} nodes themed", flush=True)
    return 0


# umap-overview -- (pools x embeddings) grid of cluster scatters


def _umap_xy_by_cluster(run_dir):
    data = json.loads((run_dir / "umap.json").read_text())
    xs, ys, cids = [], [], []
    for cid, pts in data["clips"].items():
        for x, y in pts:
            xs.append(x)
            ys.append(y)
            cids.append(int(cid))
    return np.array(xs), np.array(ys), np.array(cids)


def cmd_umap_overview(args):
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
    fig.suptitle(
        "UMAP overview of spherical $k$-means clusters "
        "($k$=1000; up to 50k clips/panel shown, colored by cluster)",
        fontsize=14,
        y=0.99,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print("wrote", args.out)
    return 0


# overlay-maps -- UMAP with the distinct clusters pinned to a clip


def _sharpness(fr):
    """Focus/contrast score (variance of a discrete Laplacian on a downscaled gray
    frame). High = sharp & content-rich; motion-blurred / washed frames score low."""
    g = fr[..., :3].mean(2)[::4, ::4]
    lap = g[2:, 1:-1] + g[:-2, 1:-1] + g[1:-1, 2:] + g[1:-1, :-2] - 4 * g[1:-1, 1:-1]
    return float(lap.var())


def _load_best(s3, bucket, clip, camera, positions=(0.35, 0.5, 0.65)):
    """Download a clip once, scan a few temporal positions, return the sharpest raw
    frame so a blurred mid-frame isn't picked by default."""
    key = f"anonymized/{clip[:4]}/{clip}.{camera}.mp4"
    obj = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    vr = decord.VideoReader(io.BytesIO(obj))
    n = len(vr)
    best, score = None, -1.0
    for p in positions:
        fr = vr[min(n - 1, int(n * p))].asnumpy()
        sc = _sharpness(fr)
        if sc > score:
            best, score = fr, sc
    return best, score


def _process(fr, blur_bottom=0.13, thumb=170):
    """Use the FULL (uncropped) frame; blur the hood / maker-emblem band; thumbnail."""
    im = Image.fromarray(fr)
    if blur_bottom > 0:
        ww, hh = im.size
        y0 = int((1 - blur_bottom) * hh)
        band = im.crop((0, y0, ww, hh)).filter(ImageFilter.GaussianBlur(12))
        im.paste(band, (0, y0))
    im.thumbnail((thumb, thumb))
    return np.asarray(im)


def _rep_clips(sorted_df, topics, clusters, captions_db, cand_pool):
    """Clip to display per cluster. Without a DB: nearest-to-centroid. With a DB:
    among the nearest ``cand_pool``, the clip whose caption best matches the cluster
    theme keywords -- so the frame depicts the theme, not the generic centroid."""
    near = {
        c: sorted_df[sorted_df.cluster_id == c]["clip_id"].astype(str).tolist()
        for c in clusters
    }
    if not captions_db or not Path(captions_db).exists():
        return {c: v[0] for c, v in near.items()}, {c: 0.0 for c in clusters}
    con = sqlite3.connect(str(captions_db))
    out, sc_out = {}, {}
    for c in clusters:
        cand = near[c][:cand_pool]
        kws = [k.lower() for k in topics.get(str(c), {}).get("keywords", [])[:8]]
        rows = con.execute(
            "select clip_id, caption from captions where clip_id in (%s)"
            % ",".join("?" * len(cand)),
            cand,
        ).fetchall()
        caps = {r[0]: (r[1] or "").lower() for r in rows}

        def score(clip):
            cap = caps.get(clip, "")
            s = 0.0
            for kw in kws:
                if not kw:
                    continue
                if kw in cap:
                    s += 2.0
                else:
                    s += 0.7 * sum(
                        1 for tok in kw.split() if len(tok) > 2 and tok in cap
                    )
            return s

        best, bkey = cand[0], (-1.0, 1)
        for rank, clip in enumerate(cand):
            key = (score(clip), -rank)  # best theme match, tie-break nearer centroid
            if key > bkey:
                bkey, best = key, clip
        out[c], sc_out[c] = best, bkey[0]
    con.close()
    return out, sc_out


def cmd_overlay_maps(args):

    umap_data = json.loads((args.run_dir / "umap.json").read_text())
    topics = json.loads((args.run_dir / "cluster_topics.json").read_text()).get(
        "topics", {}
    )
    df = pd.read_parquet(
        args.run_dir / "cluster_assignments.parquet",
        columns=["clip_id", "cluster_id", "distance"],
    )
    sizes = df.groupby("cluster_id").size()
    sorted_df = df.sort_values(["cluster_id", "distance"])

    if args.select == "distinct":
        clusters = _distinct_clusters(args.run_dir, sizes, args.k, args.min_frac)
    else:
        clusters = sorted((int(c) for c in sizes.index), key=lambda c: -sizes[c])[
            : args.k
        ]
    color = {c: mcolors.to_rgb(_PAL[i % len(_PAL)]) for i, c in enumerate(clusters)}

    show_clip, match = _rep_clips(
        sorted_df, topics, clusters, args.captions_db, args.cand_pool
    )
    print("  showing clusters:", clusters)

    s3 = boto3.Session(profile_name=args.profile).client(
        "s3", endpoint_url=args.endpoint, region_name="us-east-1"
    )

    def _one(c):
        try:
            fr, _ = _load_best(s3, args.bucket, show_clip[c], args.camera)
            return c, _process(fr)
        except Exception as e:
            print(
                f"  frame fail c{c} ({show_clip[c][:8]}): {type(e).__name__} {str(e)[:60]}",
                flush=True,
            )
            return c, None

    frames = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for c, fr in ex.map(_one, clusters):
            if fr is not None:
                frames[c] = fr

    # draw-dpi == save-dpi so measured px == saved px. Geometry is fixed up front
    # (add_axes / subplots_adjust, no tight_layout) so the de-overlap reads a
    # transData that won't shift before saving.
    if args.map_only:
        # no title/table in map-only mode: let the map fill the figure (tight margins)
        fig = plt.figure(figsize=(9, 8.4), dpi=140)
        axL = fig.add_axes([0.004, 0.004, 0.992, 0.992])
        axR = None
    else:
        fig = plt.figure(figsize=(17, 8), dpi=140)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.85, 1.4], wspace=0.02)
        axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        fig.subplots_adjust(left=0.012, right=0.99, top=0.93, bottom=0.02, wspace=0.02)

    # left: UMAP scatter + de-overlapped centroid frames with leader lines
    zoom = 0.50  # larger frames fill the UMAP's empty space (less whitespace)
    allpts, colpts = [], []
    for cid, pts in umap_data["clips"].items():
        if not pts:
            continue
        a = np.asarray(pts)
        allpts.append(a)
        ci = int(cid)
        if ci in color:
            colpts.append(a)
            axL.scatter(
                a[:, 0],
                a[:, 1],
                s=8,
                color=color[ci],
                alpha=0.7,
                linewidths=0,
                zorder=2,
            )
        else:
            axL.scatter(
                a[:, 0],
                a[:, 1],
                s=3,
                color="#dddddd",
                alpha=0.3,
                linewidths=0,
                zorder=1,
            )
    allpts = np.vstack(allpts)
    # map-only frames the view on the highlighted clusters (+small pad for the pinned
    # frames) instead of the whole sparse background cloud, then crops tight; the faint
    # background points outside the window are simply clipped.
    base = np.vstack(colpts) if (args.map_only and colpts) else allpts
    pad_frac = 0.12 if args.map_only else 0.30
    lo, hi = base.min(0), base.max(0)
    pad = (hi - lo) * pad_frac
    axL.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    axL.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    axL.set_xticks([])
    axL.set_yticks([])
    for s in axL.spines.values():
        s.set_visible(False)
    if args.map_only:
        axL.patch.set_visible(
            False
        )  # drop the white axes rectangle from the tight bbox
    if not args.map_only:
        axL.set_title("Cluster map (representative clip per cluster)", fontsize=12)

    # de-overlap the pinned frames: render a probe box, MEASURE its real pixel bbox,
    # repel anchors in display space until every pair clears that box, then VERIFY.
    shown = [c for c in clusters if c in frames and umap_data["centroids"].get(str(c))]
    if shown:
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        probe = AnnotationBbox(
            OffsetImage(frames[shown[0]], zoom=zoom),
            (0.5, 0.5),
            xycoords="axes fraction",
            frameon=True,
            pad=0.06,
            bboxprops=dict(linewidth=1.6),
        )
        axL.add_artist(probe)
        fig.canvas.draw()
        pb = probe.get_window_extent(rend)
        bw, bh = pb.width, pb.height  # true rendered box size, px
        probe.remove()
        print(
            f"  measured frame box: {bw:.0f}x{bh:.0f}px (zoom={zoom}, dpi={fig.get_dpi():.0f})"
        )

        cen = np.array([umap_data["centroids"][str(c)] for c in shown], dtype=float)
        disp = axL.transData.transform(cen)
        wx, wy = bw + 14, bh + 24  # centre sep per axis (extra y for the top label)
        ang = np.linspace(0, 2 * np.pi, len(shown), endpoint=False)  # tie-break dirs
        n = len(shown)
        for _ in range(3000):  # damped Jacobi: stable, no axis-flip oscillation
            force = np.zeros_like(disp)
            any_ov = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx, dy = disp[i] - disp[j]
                    ox, oy = wx - abs(dx), wy - abs(dy)
                    if ox > 0 and oy > 0:  # boxes overlap on both axes
                        any_ov = True
                        pen = min(ox, oy) + 1.0
                        dist = float(np.hypot(dx, dy))
                        ux, uy = (
                            (dx / dist, dy / dist)
                            if dist > 1e-6
                            else (float(np.cos(ang[i])), float(np.sin(ang[i])))
                        )
                        force[i, 0] += ux * pen
                        force[i, 1] += uy * pen
                        force[j, 0] -= ux * pen
                        force[j, 1] -= uy * pen
            if not any_ov:
                break
            disp += 0.5 * force
        newpos = axL.transData.inverted().transform(disp)
        for k, c in enumerate(shown):
            tc, fp = cen[k], tuple(newpos[k])
            axL.scatter(
                [tc[0]],
                [tc[1]],
                s=26,
                color=color[c],
                edgecolor="white",
                linewidths=0.6,
                zorder=4,
            )
            ab = AnnotationBbox(
                OffsetImage(frames[c], zoom=zoom),
                tc,
                xybox=fp,
                xycoords="data",
                boxcoords="data",
                frameon=True,
                pad=0.06,
                zorder=5,
                bboxprops=dict(edgecolor=color[c], linewidth=1.6),
                arrowprops=dict(arrowstyle="-", color=color[c], lw=0.8, alpha=0.7),
            )
            axL.add_artist(ab)
            # cluster id just OUTSIDE the frame's top-centre rim: cluster-coloured
            # text with a thin white outline, so it never covers content.
            hh = bh / 2 * 72.0 / fig.get_dpi()
            axL.annotate(
                f"C{c}",
                fp,
                xytext=(0, hh + 1.0),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                color=color[c],
                zorder=6,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
            )
        # numeric verification on the actual IMAGE rectangles (bw x bh).
        fig.canvas.draw()
        ctr = axL.transData.transform(newpos)
        bad = []
        for i in range(len(shown)):
            for j in range(i + 1, len(shown)):
                gx = bw - abs(ctr[i][0] - ctr[j][0])
                gy = bh - abs(ctr[i][1] - ctr[j][1])
                if gx > 0 and gy > 0:
                    bad.append((shown[i], shown[j], round(gx, 1), round(gy, 1)))
        print("  overlap-check:", "OK (0 overlaps)" if not bad else f"FAIL {bad}")

    def _cluster_theme(c):
        td = topics.get(str(c), {})
        return td.get("description") or ", ".join(td.get("keywords", [])[:5])

    # right: per-cluster LLM theme phrase (skipped in map-only mode)
    if axR is not None:
        axR.axis("off")
        axR.set_title(f"Cluster themes ({len(clusters)} most distinct)", fontsize=12)
        n = len(clusters)
        for i, c in enumerate(clusters):
            y = 1 - (i + 0.5) / n
            axR.add_patch(
                plt.Rectangle(
                    (0.0, y + 0.0),
                    0.022,
                    0.024,
                    color=color[c],
                    transform=axR.transAxes,
                    clip_on=False,
                )
            )
            axR.text(
                0.04,
                y + 0.006,
                f"C{c} (n={sizes[c]:,})",
                transform=axR.transAxes,
                fontsize=9.5,
                fontweight="bold",
                va="bottom",
            )
            axR.text(
                0.04,
                y - 0.004,
                _cluster_theme(c),
                transform=axR.transAxes,
                fontsize=8.5,
                color="#222222",
                va="top",
            )

    if args.title:
        fig.suptitle(args.title, fontsize=13, y=0.985)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.map_only:  # trim margins (positions already fixed)
        fig.savefig(args.out, dpi=140, bbox_inches="tight", pad_inches=0.01)
    else:
        fig.savefig(args.out, dpi=140)
    plt.close(fig)

    if args.emit_tex:
        rows = [r"\begin{tabular}{@{}l l@{}}"]  # natural-width: themes stay on one line
        for c in clusters:
            r, g, b = color[c][:3]
            sw = r"\textcolor[rgb]{%.3f,%.3f,%.3f}{\rule{1.3ex}{1.3ex}}" % (r, g, b)
            n_fmt = f"{sizes[c]:,}".replace(",", "{,}")
            theme = _cluster_theme(c).replace("&", r"\&").replace("_", r"\_")
            rows.append(
                r"%s~\textbf{C%d}\,{\scriptsize(n=%s)} & %s \\" % (sw, c, n_fmt, theme)
            )
        rows.append(r"\end{tabular}")
        args.emit_tex.parent.mkdir(parents=True, exist_ok=True)
        args.emit_tex.write_text("\n".join(rows) + "\n")
        print("wrote", args.emit_tex)

    print("wrote", args.out, "| frames:", len(frames), "/", len(clusters))
    return 0


# tables -- colour-coded topic table + distinctive-terms table

# run-id -> display name (k50_<key> subdirs under --clustering-dir)
_TBL_COLS = [
    ("k50_cosmos", "Cosmos-Embed1"),
    ("k50_caption", "Caption (Qwen3-Embedding-8B)"),
    ("k50_visual", "Visual (Florence-2/SigLIP)"),
]
_FOCUS = {
    APPEAR: ("weather \\& sky", "2166AC"),
    ACTIVITY: ("maneuvers \\& actions", "C2451E"),
    OBJECT: ("objects \\& vehicles", "1B7837"),
}


def _theme(td):
    return td.get("description") or ", ".join(td.get("keywords", [])[:5])


def _texword(w):
    c = _color(w)
    return (
        latex_escape(w)
        if c == NEUTRAL
        else r"\textcolor[HTML]{%s}{\textbf{%s}}"
        % (c.lstrip("#").upper(), latex_escape(w))
    )


def _emit_topics(clustering_dir, out, k, map_width, topics_width, maps_prefix):
    """3-row x 2-col figure interior: per embedding, overlay map (left) + topics (right)."""
    percol = []
    for rid, _ in _TBL_COLS:
        run = clustering_dir / rid
        topics = json.loads((run / "cluster_topics.json").read_text())["topics"]
        df = pd.read_parquet(
            run / "cluster_assignments.parquet", columns=["clip_id", "cluster_id"]
        )
        sizes = df.groupby("cluster_id").size()
        clusters = _distinct_clusters(run, sizes, k, 0.5)
        cells = []
        for i, c in enumerate(clusters):
            r, g, b = mcolors.to_rgb(_PAL[i % len(_PAL)])
            sw = r"\textcolor[rgb]{%.3f,%.3f,%.3f}{\rule{1.1ex}{1.1ex}}" % (r, g, b)
            th = _theme(topics[str(c)])
            cells.append(
                r"%s~\textbf{C%d}~%s"
                % (sw, c, " ".join(_texword(w) for w in th.split()))
            )
        percol.append(cells)

    keys = [rid.split("_", 1)[-1] for rid, _ in _TBL_COLS]  # k50_cosmos -> cosmos
    lines = [
        "% Overlay figure interior: 3 rows x 2 cols, one row per embedding.",
        "% Left: UMAP overlay map. Right: that embedding's colour-coded cluster topics.",
        "% Generated by make_figures.py tables; \\input from the paper.",
    ]
    for j, (key, (_, name)) in enumerate(zip(keys, _TBL_COLS)):
        lines.append(
            r"\begin{minipage}[c]{%.2f\linewidth}\centering"
            r"\includegraphics[width=\linewidth]{%s%s.png}\end{minipage}\hfill"
            % (map_width, maps_prefix, key)
        )
        lines.append(
            r"\begin{minipage}[c]{%.2f\linewidth}\footnotesize\raggedright\textbf{%s}\\[2pt]"
            % (topics_width, name)
        )
        cells = percol[j]
        for i, cell in enumerate(cells):
            lines.append(cell + (r"\\" if i < len(cells) - 1 else ""))
        lines.append(r"\end{minipage}" + (r"\\[5pt]" if j < len(_TBL_COLS) - 1 else ""))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)


def _emit_distinctive(clustering_dir, out, topn):
    """One row per embedding: name + coloured focus label + its distinctive terms."""
    runs = {
        name: json.loads((clustering_dir / rid / "cluster_topics.json").read_text())[
            "topics"
        ]
        for rid, name in _TBL_COLS
    }
    prof = topic_profiles(runs)
    embs = [name for _, name in _TBL_COLS]
    terms = {e: distinctive_terms(prof, e, embs, topn) for e in embs}
    focus = {}
    for e in embs:
        cnt = Counter()
        for w in terms[e]:
            for tok in w.split():
                if _color(tok) in (APPEAR, ACTIVITY, OBJECT):
                    cnt[_color(tok)] += 1
        focus[e] = _FOCUS[cnt.most_common(1)[0][0]] if cnt else ("", "000000")

    for e in embs:
        print(f"{e:30} [{focus[e][0]}]: {', '.join(terms[e])}")

    rows = [r"\begin{tabular}{@{}l l p{0.52\linewidth}@{}}", r"\toprule"]
    for _, n in _TBL_COLS:
        rows.append(
            r"\textbf{%s} & \textcolor[HTML]{%s}{\textbf{%s}} & %s \\[1pt]"
            % (n, focus[n][1], focus[n][0], ", ".join(terms[n]))
        )
    rows += [r"\bottomrule", r"\end{tabular}"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n")
    print("wrote", out)


def cmd_tables(args):
    if args.what in ("topics", "both"):
        _emit_topics(
            args.clustering_dir,
            args.topics_out,
            args.k,
            args.map_width,
            args.topics_width,
            args.maps_prefix,
        )
    if args.what in ("distinctive", "both"):
        _emit_distinctive(args.clustering_dir, args.distinctive_out, args.topn)
    return 0


# hierarchical -- drill one branch per embedding into its sub-clusters

_FOCUSCAT = {
    "cosmos": (APPEAR, "appearance"),
    "caption": (ACTIVITY, "activity"),
    "visual": (OBJECT, "objects"),
}


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


def cmd_hierarchical(args):

    rows = [
        ("k50_cosmos", "pai_cosmos", "cosmos.npz", "cosmos", "Cosmos-Embed1", False),
        (
            "k50_caption",
            "pai_caption",
            "caption.npz",
            "caption",
            "Caption (Qwen3-Emb-8B)",
            False,
        ),
        ("k50_visual", "pai_visual", "visual.npz", "visual", "Florence-2/SigLIP", True),
    ]
    rng = np.random.default_rng(args.seed)
    fig, axes = plt.subplots(
        3, 2, figsize=(12.5, 14.5), gridspec_kw=dict(width_ratios=[1, 1.05])
    )
    arrows = []  # (axO, axZ, focus-box text, colour); arrows drawn after layout

    for r, (flat, hierd, npz, key, name, center) in enumerate(rows):
        t = json.loads((args.hier_base / hierd / "hier_topics.json").read_text())
        ha = pd.read_parquet(
            args.hier_base / hierd / "hier_assignments.parquet",
            columns=["clip_id", "path"],
        )
        ha["clip_id"] = ha["clip_id"].astype(str)
        ha["l1"] = ha["path"].astype(str).str.split(".").str[0]
        uj = json.loads((args.clustering_dir / flat / "umap.json").read_text())
        pts = []
        for cid, ps in uj["clips"].items():
            for (x, y), clip in zip(ps, uj["clip_ids"].get(cid, [])):
                pts.append((str(clip), x, y))
        ov = pd.DataFrame(pts, columns=["clip_id", "x", "y"]).merge(
            ha[["clip_id", "l1"]], on="clip_id", how="inner"
        )
        l1s = sorted(ov["l1"].dropna().unique(), key=lambda s: int(s))
        cat, catname = _FOCUSCAT[key]
        focus = _pick_focus(t, l1s, cat)
        fcol = _PAL[l1s.index(focus) % len(_PAL)]

        # --- overview ---
        axO = axes[r, 0]
        axO.set_facecolor("#f6f6f9")
        bg = ov[ov["l1"] != focus]
        fg = ov[ov["l1"] == focus]
        axO.scatter(
            bg["x"],
            bg["y"],
            s=3,
            color="#d7d7de",
            alpha=0.5,
            linewidths=0,
            rasterized=True,
        )
        axO.scatter(
            fg["x"], fg["y"], s=6, color=fcol, alpha=0.85, linewidths=0, rasterized=True
        )
        fx, fy = _dense_xy(fg["x"].values, fg["y"].values)
        # focus cluster id + theme on the left, in a white box with the focus-colour
        # edge so it ties to the arrow and the zoom-panel title
        flabel = "C%s  %s" % (focus, _phrase(t[focus], 3))
        ftxt = axO.text(
            fx,
            fy,
            "\n".join(textwrap.wrap(flabel, 18)),
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="center",
            color="#1a1a1a",
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.85,
                edgecolor=fcol,
                linewidth=1.0,
            ),
        )
        axO.set_title(
            f"{name}: {len(l1s)} level-1 clusters", fontsize=12, fontweight="bold"
        )
        axO.set_xticks([])
        axO.set_yticks([])
        for s in axO.spines.values():
            s.set_visible(False)

        # --- zoom: re-cluster the focus branch ---
        d = np.load(args.npz_dir / npz, allow_pickle=True)
        idx = {str(c): i for i, c in enumerate(d["clip_ids"])}
        emb = np.asarray(d["embeddings"], dtype=np.float32)
        if center:
            emb = emb - emb.mean(0, keepdims=True)
            nrm = np.linalg.norm(emb, axis=1, keepdims=True)
            nrm[nrm == 0] = 1.0
            emb = emb / nrm
        clips = ha[(ha["l1"] == focus) & (ha["clip_id"].isin(idx))]
        if len(clips) > args.sample:
            clips = clips.iloc[rng.choice(len(clips), args.sample, replace=False)]
        rowi = np.array([idx[c] for c in clips["clip_id"]], dtype=np.int64)
        coords = umap.UMAP(
            n_neighbors=15, min_dist=0.1, random_state=args.seed
        ).fit_transform(np.ascontiguousarray(emb[rowi]))
        l2 = clips["path"].astype(str).values
        cats2 = sorted(set(l2), key=lambda s: [int(x) for x in s.split(".")])
        axZ = axes[r, 1]
        axZ.set_facecolor("#f6f6f9")
        for ci, c in enumerate(cats2):
            m = l2 == c
            axZ.scatter(
                coords[m, 0],
                coords[m, 1],
                s=7,
                color=_PAL[ci % len(_PAL)],
                alpha=0.7,
                linewidths=0,
                rasterized=True,
            )
        order = sorted(range(len(cats2)), key=lambda ci: -(l2 == cats2[ci]).sum())
        for ci in order[: args.label_k]:
            c = cats2[ci]
            m = l2 == c
            lx, ly = _dense_xy(coords[m, 0], coords[m, 1])
            # small white box behind the topic, thin edge in the sub-cluster's colour
            axZ.text(
                lx,
                ly,
                "\n".join(textwrap.wrap(_phrase(t[c], 2), 16)),
                fontsize=7.5,
                ha="center",
                va="center",
                color="#1a1a1a",
                zorder=6,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.85,
                    edgecolor=_PAL[ci % len(_PAL)],
                    linewidth=0.8,
                ),
            )
        axZ.set_title(
            "C%s “%s”  (%d sub-clusters)" % (focus, _phrase(t[focus]), len(cats2)),
            fontsize=10.5,
            color=fcol,
            fontweight="bold",
        )
        axZ.set_xticks([])
        axZ.set_yticks([])
        for s in axZ.spines.values():
            s.set_visible(False)

        arrows.append((axO, axZ, ftxt, fcol))

    fig.tight_layout()  # no suptitle; the figure caption serves as the title
    # draw the focus->zoom arrows last, anchored at the RIGHT RIM of each white box
    # (measured after layout) so the arrow leaves the label cleanly
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for axO, axZ, ftxt, fcol in arrows:
        bb = ftxt.get_bbox_patch().get_window_extent(rend)  # box incl. pad, display px
        start = axO.transData.inverted().transform((bb.x1, 0.5 * (bb.y0 + bb.y1)))
        con = ConnectionPatch(
            xyA=start,
            coordsA=axO.transData,
            xyB=(0.0, 0.5),
            coordsB=axZ.transAxes,
            arrowstyle="-|>",
            mutation_scale=22,
            lw=2.0,
            color=fcol,
            alpha=0.85,
            zorder=20,
        )
        fig.add_artist(con)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print("wrote", args.out)
    return 0


# CLI


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("themes", help="LLM one-phrase name per cluster (backfill)")
    p.add_argument("--clustering-dir", type=Path, required=True)
    p.add_argument(
        "--runs", nargs="+", required=True, help="run-id subdirs to backfill"
    )
    p.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL"))
    p.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    p.add_argument(
        "--workers", type=int, default=2, help="concurrency (gateway rate-limits)"
    )
    p.add_argument("--top-k", type=int, default=10, help="keywords sent per cluster")
    p.add_argument(
        "--topics-name",
        default="cluster_topics.json",
        help="topics file per run dir (use hier_topics.json for hierarchical runs)",
    )
    p.set_defaults(func=cmd_themes)

    p = sub.add_parser(
        "umap-overview", help="(pools x embeddings) grid of cluster scatters"
    )
    p.add_argument("--clustering-dir", type=Path, required=True)
    p.add_argument(
        "--fig-runs", type=Path, required=True, help="JSON from prep.py fig-runs"
    )
    p.add_argument("--out", type=Path, default=Path("umap_overview.png"))
    p.set_defaults(func=cmd_umap_overview)

    p = sub.add_parser(
        "overlay-maps", help="one embedding's UMAP with distinct clusters pinned"
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="")
    p.add_argument("--k", type=int, default=10, help="clusters shown in the map")
    p.add_argument(
        "--select",
        choices=["size", "distinct"],
        default="distinct",
        help="which clusters to show: largest, or the most mutually distinct",
    )
    p.add_argument(
        "--min-frac",
        type=float,
        default=0.5,
        help="--select distinct: drop clusters smaller than this * mean size",
    )
    p.add_argument(
        "--captions-db",
        type=Path,
        default=None,
        help="captions DB; show the near-centroid clip whose caption best "
        "matches the cluster theme (not the geometric centroid)",
    )
    p.add_argument(
        "--cand-pool",
        type=int,
        default=300,
        help="near-centroid clips scanned for the theme-matching frame",
    )
    p.add_argument(
        "--map-only",
        action="store_true",
        help="render only the cluster map (no theme table)",
    )
    p.add_argument(
        "--emit-tex",
        type=Path,
        default=None,
        help="also write the themes as a LaTeX tabular to this path",
    )
    p.add_argument("--camera", default="camera_front_wide_120fov")
    p.add_argument("--bucket", default="physical_ai_av")
    p.add_argument("--profile", default="avfoundation")
    p.add_argument("--endpoint", default="https://pdx.s8k.io")
    p.add_argument("--workers", type=int, default=6)
    p.set_defaults(func=cmd_overlay_maps)

    p = sub.add_parser(
        "tables", help="colour-coded topic table + distinctive-terms table"
    )
    p.add_argument("--clustering-dir", type=Path, required=True)
    p.add_argument("--what", choices=["topics", "distinctive", "both"], default="both")
    p.add_argument(
        "--topics-out", type=Path, default=Path("tables/emb_cluster_topics.tex")
    )
    p.add_argument(
        "--distinctive-out", type=Path, default=Path("tables/emb_distinctive_terms.tex")
    )
    p.add_argument(
        "--k", type=int, default=10, help="clusters shown per embedding (topics)"
    )
    p.add_argument(
        "--topn", type=int, default=8, help="distinctive terms per embedding"
    )
    p.add_argument(
        "--maps-prefix",
        default="figures/overlay_map_",
        help="prefix for the per-embedding map PNG (<prefix><key>.png)",
    )
    p.add_argument(
        "--map-width", type=float, default=0.45, help="map minipage width (\\linewidth)"
    )
    p.add_argument(
        "--topics-width",
        type=float,
        default=0.53,
        help="topics minipage width (\\linewidth)",
    )
    p.set_defaults(func=cmd_tables)

    p = sub.add_parser(
        "hierarchical", help="drill one branch per embedding into its sub-clusters"
    )
    p.add_argument("--clustering-dir", type=Path, required=True)
    p.add_argument("--hier-base", type=Path, required=True)
    p.add_argument("--npz-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sample", type=int, default=4000)
    p.add_argument("--label-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=1234)
    p.set_defaults(func=cmd_hierarchical)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
