#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Contrastive distinctive-terms table for the overlay comparison.

For each embedding we score every cluster-topic term by how much it concentrates
in that embedding's clusters relative to the other two (weighted log-odds) and
list the top distinctive terms. Unlike the full topic phrases (which overlap
because all embeddings describe the same scenes), these terms separate the
embeddings: appearance/weather vs. maneuvers/actions vs. objects/vehicles.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from topic_lexicon import APPEAR, ACTIVITY, OBJECT
from topic_lexicon import categorize as _color
from topic_lexicon import distinctive_terms as _distinct
from topic_lexicon import topic_profiles as _profiles

_FOCUS = {APPEAR: ("weather \\& sky", "2166AC"), ACTIVITY: ("maneuvers \\& actions", "C2451E"),
          OBJECT: ("objects \\& vehicles", "1B7837")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clustering-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="LaTeX table (omit to just print)")
    ap.add_argument("--topn", type=int, default=8)
    args = ap.parse_args(argv)

    cols = [("k50_cosmos", "Cosmos-Embed1"), ("k50_caption", "Caption (Qwen3-Embedding-8B)"),
            ("k50_visual", "Visual (Florence-2/SigLIP)")]
    runs = {e: json.loads((args.clustering_dir / r / "cluster_topics.json").read_text())["topics"]
            for r, e in [(r, e) for r, e in [(c[0], c[1]) for c in cols]]}
    prof = _profiles(runs)
    embs = [name for _, name in cols]
    terms = {e: _distinct(prof, e, embs, args.topn) for e in embs}
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

    if args.out:
        # One row per embedding: name + coloured focus label + its distinctive terms,
        # so each line ties explicitly to one embedding (no ambiguous columns).
        rows = [r"\begin{tabular}{@{}l l p{0.52\linewidth}@{}}", r"\toprule"]
        for _, n in cols:
            rows.append(r"\textbf{%s} & \textcolor[HTML]{%s}{\textbf{%s}} & %s \\[1pt]"
                        % (n, focus[n][1], focus[n][0], ", ".join(terms[n])))
        rows += [r"\bottomrule", r"\end{tabular}"]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(rows) + "\n")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
