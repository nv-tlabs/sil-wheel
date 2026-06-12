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

"""Per-run cluster overview: UMAP overlay (left) + top-N topic table (right).

For one clustering run, the left panel is the run's UMAP scatter with the
centroid clip of each shown cluster pinned at its UMAP position; the right panel
is a table of those clusters' top topic terms. Centroid clip = the member nearest
the centroid (first row of the distance-sorted cluster_assignments.parquet).
Frames are fetched from S3 (clip_id -> anonymized/<clip>.<camera>.mp4) and a
mid-frame is extracted with decord; the bottom band (hood / maker emblem) is
blurred.
"""
from __future__ import annotations

import argparse
import io
import json
import sqlite3
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import decord
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

import figstyle  # noqa: F401  (registers NVIDIA Sans, sets it as default)
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageFilter


def _sharpness(fr):
    """Focus/contrast score (variance of a discrete Laplacian on a downscaled gray
    frame). High = sharp & content-rich; motion-blurred / washed frames score low.
    Night frames score low in absolute terms but compare fairly within a cluster."""
    g = fr[..., :3].mean(2)[::4, ::4]
    lap = (g[2:, 1:-1] + g[:-2, 1:-1] + g[1:-1, 2:] + g[1:-1, :-2] - 4 * g[1:-1, 1:-1])
    return float(lap.var())


def _load_best(s3, bucket, clip, camera, positions=(0.35, 0.5, 0.65)):
    """Download a clip once, scan a few temporal positions, return (sharpest raw
    frame, its score) so a blurred mid-frame doesn't get picked by default."""
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


# The k-most-distinct cluster selection now lives in cluster_select; it is
# re-exported here under its original private name so this module's CLI and the
# superseded scripts that import it from here keep working unchanged.
from cluster_select import distinct_clusters as _distinct_clusters  # noqa: E402,F401


def _rep_clips(sorted_df, topics, clusters, captions_db, cand_pool):
    """Clip to display per cluster. Without a DB: nearest-to-centroid. With a DB:
    among the nearest `cand_pool`, the clip whose caption best matches the cluster
    theme keywords -- so the frame depicts the theme, not the generic centroid."""
    near = {c: sorted_df[sorted_df.cluster_id == c]["clip_id"].astype(str).tolist()
            for c in clusters}
    if not captions_db or not Path(captions_db).exists():
        return {c: v[0] for c, v in near.items()}, {c: 0.0 for c in clusters}
    con = sqlite3.connect(str(captions_db))
    out, sc_out = {}, {}
    for c in clusters:
        cand = near[c][:cand_pool]
        kws = [k.lower() for k in topics.get(str(c), {}).get("keywords", [])[:8]]
        rows = con.execute(
            "select clip_id, caption from captions where clip_id in (%s)"
            % ",".join("?" * len(cand)), cand).fetchall()
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
                    s += 0.7 * sum(1 for tok in kw.split() if len(tok) > 2 and tok in cap)
            return s

        best, bkey = cand[0], (-1.0, 1)
        for rank, clip in enumerate(cand):
            key = (score(clip), -rank)        # best theme match, tie-break nearer centroid
            if key > bkey:
                bkey, best = key, clip
        out[c], sc_out[c] = best, bkey[0]
    con.close()
    return out, sc_out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--k", type=int, default=10, help="clusters shown in the map")
    ap.add_argument("--select", choices=["size", "distinct"], default="distinct",
                    help="which clusters to show: largest, or the most mutually distinct")
    ap.add_argument("--min-frac", type=float, default=0.5,
                    help="--select distinct: drop clusters smaller than this * mean size")
    ap.add_argument("--captions-db", type=Path, default=None,
                    help="qwen3.5 captions DB; with it, show the near-centroid clip whose "
                         "caption best matches the cluster theme (not the geometric centroid)")
    ap.add_argument("--cand-pool", type=int, default=300,
                    help="near-centroid clips scanned for the theme-matching frame")
    ap.add_argument("--map-only", action="store_true",
                    help="render only the cluster map (no theme table); for LaTeX side-by-side")
    ap.add_argument("--emit-tex", type=Path, default=None,
                    help="also write the themes as a LaTeX tabular to this path")
    ap.add_argument("--camera", default="camera_front_wide_120fov")
    ap.add_argument("--bucket", default="physical_ai_av")
    ap.add_argument("--profile", default="avfoundation")
    ap.add_argument("--endpoint", default="https://pdx.s8k.io")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)

    umap = json.loads((args.run_dir / "umap.json").read_text())
    topics = json.loads((args.run_dir / "cluster_topics.json").read_text()).get("topics", {})
    df = pd.read_parquet(args.run_dir / "cluster_assignments.parquet",
                         columns=["clip_id", "cluster_id", "distance"])
    sizes = df.groupby("cluster_id").size()
    sorted_df = df.sort_values(["cluster_id", "distance"])

    if args.select == "distinct":
        clusters = _distinct_clusters(args.run_dir, sizes, args.k, args.min_frac)
    else:
        clusters = sorted((int(c) for c in sizes.index), key=lambda c: -sizes[c])[:args.k]
    # Plotly "Vivid" qualitative palette (matches the drill-down figures)
    _PAL = ["#E58606", "#5D69B1", "#52BCA3", "#99C945", "#CC61B0", "#24796C",
            "#DAA51B", "#2F8AC4", "#764E9F", "#ED645A", "#CC3A8E", "#A5AA99"]
    color = {c: mcolors.to_rgb(_PAL[i % len(_PAL)]) for i, c in enumerate(clusters)}

    show_clip, match = _rep_clips(sorted_df, topics, clusters, args.captions_db, args.cand_pool)
    print("  showing clusters:", clusters)

    s3 = boto3.Session(profile_name=args.profile).client(
        "s3", endpoint_url=args.endpoint, region_name="us-east-1")

    def _one(c):
        try:
            fr, _ = _load_best(s3, args.bucket, show_clip[c], args.camera)
            return c, _process(fr)
        except Exception as e:
            print(f"  frame fail c{c} ({show_clip[c][:8]}): {type(e).__name__} {str(e)[:60]}", flush=True)
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
        fig = plt.figure(figsize=(9, 8.4), dpi=140)
        axL = fig.add_axes([0.01, 0.01, 0.98, 0.92])
        axR = None
    else:
        fig = plt.figure(figsize=(17, 8), dpi=140)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.85, 1.4], wspace=0.02)
        axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        fig.subplots_adjust(left=0.012, right=0.99, top=0.93, bottom=0.02, wspace=0.02)

    # left: UMAP scatter + de-overlapped centroid frames with leader lines
    zoom, thumb = 0.42, 170
    allpts = []
    for cid, pts in umap["clips"].items():
        if not pts:
            continue
        a = np.asarray(pts); allpts.append(a); ci = int(cid)
        if ci in color:
            axL.scatter(a[:, 0], a[:, 1], s=8, color=color[ci], alpha=0.7, linewidths=0, zorder=2)
        else:
            axL.scatter(a[:, 0], a[:, 1], s=3, color="#dddddd", alpha=0.3, linewidths=0, zorder=1)
    allpts = np.vstack(allpts)
    lo, hi = allpts.min(0), allpts.max(0); pad = (hi - lo) * 0.30
    axL.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); axL.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    axL.set_xticks([]); axL.set_yticks([])
    for s in axL.spines.values():
        s.set_visible(False)
    if not args.map_only:
        axL.set_title("Cluster map (representative clip per cluster)", fontsize=12)

    # de-overlap the pinned frames. Do NOT assume the frame size: render a probe
    # box, MEASURE its real pixel bbox (image + pad + border), repel anchors in
    # display space until every pair clears that measured box on both axes, then
    # numerically VERIFY no two final boxes overlap.
    shown = [c for c in clusters if c in frames and umap["centroids"].get(str(c))]
    if shown:
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        probe = AnnotationBbox(OffsetImage(frames[shown[0]], zoom=zoom), (0.5, 0.5),
                               xycoords="axes fraction", frameon=True, pad=0.06,
                               bboxprops=dict(linewidth=1.6))
        axL.add_artist(probe); fig.canvas.draw()
        pb = probe.get_window_extent(rend)
        bw, bh = pb.width, pb.height        # true rendered box size, px
        probe.remove()
        print(f"  measured frame box: {bw:.0f}x{bh:.0f}px (zoom={zoom}, dpi={fig.get_dpi():.0f})")

        cen = np.array([umap["centroids"][str(c)] for c in shown], dtype=float)
        disp = axL.transData.transform(cen)
        wx, wy = bw + 14, bh + 24           # centre sep per axis (extra y for the top label)
        ang = np.linspace(0, 2 * np.pi, len(shown), endpoint=False)  # tie-break dirs
        n = len(shown)
        for _ in range(3000):               # damped Jacobi: stable, no axis-flip oscillation
            force = np.zeros_like(disp)
            any_ov = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx, dy = disp[i] - disp[j]
                    ox, oy = wx - abs(dx), wy - abs(dy)
                    if ox > 0 and oy > 0:               # boxes overlap on both axes
                        any_ov = True
                        pen = min(ox, oy) + 1.0
                        dist = float(np.hypot(dx, dy))
                        ux, uy = (dx / dist, dy / dist) if dist > 1e-6 \
                            else (float(np.cos(ang[i])), float(np.sin(ang[i])))
                        force[i, 0] += ux * pen; force[i, 1] += uy * pen
                        force[j, 0] -= ux * pen; force[j, 1] -= uy * pen
            if not any_ov:
                break
            disp += 0.5 * force
        newpos = axL.transData.inverted().transform(disp)
        placed = []
        for k, c in enumerate(shown):
            tc, fp = cen[k], tuple(newpos[k])
            axL.scatter([tc[0]], [tc[1]], s=26, color=color[c],
                        edgecolor="white", linewidths=0.6, zorder=4)
            ab = AnnotationBbox(OffsetImage(frames[c], zoom=zoom), tc, xybox=fp,
                                xycoords="data", boxcoords="data", frameon=True, pad=0.06,
                                zorder=5, bboxprops=dict(edgecolor=color[c], linewidth=1.6),
                                arrowprops=dict(arrowstyle="-", color=color[c], lw=0.8, alpha=0.7))
            axL.add_artist(ab); placed.append((c, ab))
            # cluster id just OUTSIDE the frame's top-centre rim: cluster-coloured
            # text with a thin white outline (no badge), so it never covers content.
            hh = bh / 2 * 72.0 / fig.get_dpi()
            axL.annotate(f"C{c}", fp, xytext=(0, hh + 1.0), textcoords="offset points",
                         ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                         color=color[c], zorder=6,
                         path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
        # numeric verification on the actual IMAGE rectangles (bw x bh centred at
        # the rendered box centre). NOT get_window_extent -- that includes the
        # leader arrow and reports false overlaps.
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

    def _theme(c):
        td = topics.get(str(c), {})
        return td.get("description") or ", ".join(td.get("keywords", [])[:5])

    # right: per-cluster LLM theme phrase (skipped in map-only mode)
    if axR is not None:
        axR.axis("off")
        axR.set_title(f"Cluster themes ({len(clusters)} most distinct)", fontsize=12)
        n = len(clusters)
        for i, c in enumerate(clusters):
            y = 1 - (i + 0.5) / n
            axR.add_patch(plt.Rectangle((0.0, y + 0.0), 0.022, 0.024, color=color[c],
                                        transform=axR.transAxes, clip_on=False))
            axR.text(0.04, y + 0.006, f"C{c} (n={sizes[c]:,})", transform=axR.transAxes,
                     fontsize=9.5, fontweight="bold", va="bottom")
            axR.text(0.04, y - 0.004, _theme(c), transform=axR.transAxes, fontsize=8.5,
                     color="#222222", va="top")        # one line; panel widened to fit

    if args.title:
        fig.suptitle(args.title, fontsize=13, y=0.985)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.map_only:                       # trim margins (positions already fixed)
        fig.savefig(args.out, dpi=140, bbox_inches="tight", pad_inches=0.05)
    else:
        fig.savefig(args.out, dpi=140)
    plt.close(fig)

    if args.emit_tex:
        rows = [r"\begin{tabular}{@{}l l@{}}"]   # natural-width: themes stay on one line
        for c in clusters:
            r, g, b = color[c][:3]
            sw = r"\textcolor[rgb]{%.3f,%.3f,%.3f}{\rule{1.3ex}{1.3ex}}" % (r, g, b)
            n_fmt = f"{sizes[c]:,}".replace(",", "{,}")
            theme = _theme(c).replace("&", r"\&").replace("_", r"\_")
            rows.append(r"%s~\textbf{C%d}\,{\scriptsize(n=%s)} & %s \\" % (sw, c, n_fmt, theme))
        rows.append(r"\end{tabular}")
        args.emit_tex.parent.mkdir(parents=True, exist_ok=True)
        args.emit_tex.write_text("\n".join(rows) + "\n")
        print("wrote", args.emit_tex)

    print("wrote", args.out, "| frames:", len(frames), "/", len(clusters))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
