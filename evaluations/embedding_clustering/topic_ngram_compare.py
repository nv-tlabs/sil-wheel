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

"""Side-by-side comparison of cluster topic n-grams across embeddings.

For one pool (a row of fig_runs.json), reads each embedding's clustering run and
aggregates the per-cluster TF-IDF topic keywords -- restricted to n-grams of at
least ``--min-n`` tokens -- by the fraction of clusters each n-gram labels. It
then reports, per embedding:

* the most *distinctive* n-grams (concentrated in this embedding's clusters but
  dispersed across the others; weighted log-odds), and
* the n-grams *shared* by all embeddings (the common AV scene vocabulary).

Output is a plain markdown table + a LaTeX table (``--out``) + JSON -- a simple
chart comparison, no plot, to show how the embeddings' extracted cluster topics
differ. Topics come from the same caption corpus for every embedding, so the
contrast reflects how differently each embedding groups the clips.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def _profile(run_dir: Path, min_n: int, top_k: int) -> tuple[dict, int]:
    """n-gram -> fraction of clusters whose top-`top_k` keywords contain it."""
    topics = json.loads((run_dir / "cluster_topics.json").read_text()).get("topics", {})
    counts: Counter = Counter()
    n_clusters = 0
    for v in topics.values():
        grams = {k for k in v.get("keywords", [])[:top_k] if len(k.split()) >= min_n}
        if not grams:
            continue
        n_clusters += 1
        counts.update(grams)
    return {g: c / max(n_clusters, 1) for g, c in counts.items()}, n_clusters


def _distinctive(profiles: dict, key: str, keys: list[str], topn: int,
                 floor: float = 0.01) -> list[tuple[str, float, float]]:
    others = [k for k in keys if k != key]
    eps = 1e-3
    scored = []
    for g, pe in profiles[key].items():
        if pe < floor:
            continue
        bg = sum(profiles[o].get(g, 0.0) for o in others) / max(len(others), 1)
        scored.append((g, pe, pe * math.log((pe + eps) / (bg + eps))))
    scored.sort(key=lambda x: x[2], reverse=True)
    return [(g, pe, s) for g, pe, s in scored[:topn]]


def _latex(pool_label, n_clips, embeds, keys, distinctive, topn) -> str:
    cols = "l" * len(keys)
    head = " & ".join(f"\\textbf{{{e['label']}}}" for e in embeds) + " \\\\"
    lines = [
        "\\begin{table}[!t]", "\\centering \\small",
        f"  \\begin{{tabular}}{{{cols}}}", "    \\toprule", "    " + head, "    \\midrule",
    ]
    for i in range(topn):
        row = " & ".join(
            (distinctive[k][i][0] if i < len(distinctive[k]) else "") for k in keys
        )
        lines.append("    " + row + " \\\\")
    lines += [
        "    \\bottomrule", "  \\end{tabular}",
        f"  \\caption{{\\textbf{{Most distinctive cluster-topic {min(2, 9)}-grams per embedding}} "
        f"on the {pool_label} pool ({n_clips:,} clips, $k\\!=\\!1{{,}}000$). Each column lists the "
        "caption $n$-grams ($n\\ge2$) that concentrate in that embedding's clusters but disperse "
        "across the other two; topics are drawn from the same caption corpus, so the contrast "
        "reflects how the embeddings group clips.}}",
        "  \\label{tab::emb_cluster_topics}", "\\end{table}",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clustering-dir", type=Path, required=True)
    ap.add_argument("--fig-runs", type=Path, required=True)
    ap.add_argument("--pool", default=None, help="pool-label substring (default: first pool)")
    ap.add_argument("--min-n", type=int, default=2, help="minimum tokens per n-gram")
    ap.add_argument("--top-k", type=int, default=15, help="keywords/cluster considered")
    ap.add_argument("--topn", type=int, default=12, help="rows per embedding column")
    ap.add_argument("--out", type=Path, default=None, help="write LaTeX table + JSON here (stem)")
    args = ap.parse_args(argv)

    spec = json.loads(args.fig_runs.read_text())
    embeds = spec["embeds"]
    pools = spec["pools"]
    pool = pools[0]
    if args.pool:
        pool = next(p for p in pools if args.pool.lower() in p["label"].lower())
    keys = [e["key"] for e in embeds]
    lbl = {e["key"]: e["label"] for e in embeds}

    profiles, ncl = {}, {}
    for e in embeds:
        profiles[e["key"]], ncl[e["key"]] = _profile(
            args.clustering_dir / pool["runs"][e["key"]], args.min_n, args.top_k
        )
    distinctive = {k: _distinctive(profiles, k, keys, args.topn) for k in keys}

    allw = set().union(*[set(profiles[k]) for k in keys])
    shared = sorted(
        ((w, min(profiles[k].get(w, 0.0) for k in keys)) for w in allw),
        key=lambda x: -x[1],
    )
    shared = [(w, s) for w, s in shared if s > 0][: args.topn]

    print(f"\n## Distinctive >={args.min_n}-gram cluster topics -- "
          f"{pool['label']} ({pool['n']:,} clips)\n")
    print("| " + " | ".join(lbl[k] for k in keys) + " |")
    print("|" + "|".join(["---"] * len(keys)) + "|")
    for i in range(args.topn):
        cells = []
        for k in keys:
            if i < len(distinctive[k]):
                g, pe, _ = distinctive[k][i]
                cells.append(f"{g} ({pe*100:.0f}%)")
            else:
                cells.append("")
        print("| " + " | ".join(cells) + " |")
    print("\n**Shared across all three** (common AV vocabulary): "
          + ", ".join(w for w, _ in shared))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tex = _latex(pool["label"], pool["n"], embeds, keys, distinctive, args.topn)
        args.out.with_suffix(".tex").write_text(tex)
        args.out.with_suffix(".json").write_text(json.dumps({
            "pool": pool["label"], "n_clips": pool["n"], "min_n": args.min_n,
            "distinctive": {k: [(g, round(pe, 3)) for g, pe, _ in distinctive[k]] for k in keys},
            "shared": [(w, round(s, 3)) for w, s in shared],
        }, indent=2))
        print(f"\nwrote {args.out.with_suffix('.tex')} + .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
