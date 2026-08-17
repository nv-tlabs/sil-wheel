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

"""What do the leading principal components of the caption embedding encode?

Mean-centers the monolithic caption embeddings, takes the top PCs, and for each
one lists the caption terms most over-represented at its high- vs low-projection
extremes (top/bottom decile of clips). Shows that the dominant directions track
broad scene type (lighting / road type / weather) rather than the target
scenarios, which is the structure ``caption_pc_ablation.remove_top_pcs`` strips.
Produces the paper's PC-topics table data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

# generic caption boilerplate to drop so scene terms surface
BOILERPLATE = {
    "ego", "vehicle", "vehicles", "drives", "driving", "drive", "road", "roads",
    "camera", "view", "views", "captures", "capturing", "dashcam", "perspective",
    "scene", "forward", "speed", "steady", "low", "moves", "moving", "travels",
    "traveling", "ahead", "lane", "lanes", "right", "left", "straight", "maintains",
    "continues", "continuing", "video", "car", "cars", "person", "clip", "first",
    "second", "sky", "visible", "slightly", "slow", "gentle", "approaches",
}


def pole_terms(X, vocab, keep, mask, topn=8):
    """Terms most over-represented in `mask` clips vs the rest (freq difference)."""
    p_in = np.asarray(X[mask].sum(0)).ravel() / max(int(mask.sum()), 1)
    p_out = np.asarray(X[~mask].sum(0)).ravel() / max(int((~mask).sum()), 1)
    score = np.where(keep, p_in - p_out, -9.0)
    return vocab[np.argsort(score)[::-1][:topn]].tolist()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caption-npz", type=Path, required=True)
    ap.add_argument("--captions", type=Path, required=True,
                    help="parquet/csv with clip_id + summary/caption for term extraction.")
    ap.add_argument("--out", type=Path, default=Path("pc_topics.json"))
    ap.add_argument("--n-pcs", type=int, default=5)
    ap.add_argument("--sample", type=int, default=12000)
    ap.add_argument("--min-df", type=int, default=150)
    args = ap.parse_args(argv)

    data = np.load(args.caption_npz, allow_pickle=True)
    ids = [str(x) for x in data["clip_ids"]]
    E = normalize(np.ascontiguousarray(data["embeddings"], dtype=np.float32))
    df = (pd.read_parquet(args.captions) if args.captions.suffix == ".parquet"
          else pd.read_csv(args.captions))
    cap_col = next(c for c in ("summary", "caption", "qwen35_caption") if c in df.columns)
    cap = df.set_index("clip_id")[cap_col]
    texts = [str(cap.get(c, "")) for c in ids]

    rng = np.random.default_rng(0)
    n = min(args.sample, len(E))
    samp = rng.choice(len(E), n, replace=False)
    Es, Ts = E[samp], [texts[i] for i in samp]
    mrl = float(np.linalg.norm(Es.mean(0)))  # shared-direction strength

    C = Es - Es.mean(0)
    _, S, Vt = np.linalg.svd(C, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()

    cv = CountVectorizer(stop_words="english", min_df=args.min_df, token_pattern=r"[a-zA-Z]{3,}")
    X = cv.fit_transform(Ts)
    vocab = np.array(cv.get_feature_names_out())
    keep = np.array([w not in BOILERPLATE for w in vocab])

    rows = []
    for k in range(args.n_pcs):
        proj = C @ Vt[k]
        hi = proj >= np.quantile(proj, 0.90)
        lo = proj <= np.quantile(proj, 0.10)
        rows.append({"pc": k + 1, "var": round(float(var[k]), 4),
                     "high": pole_terms(X, vocab, keep, hi),
                     "low": pole_terms(X, vocab, keep, lo)})
        print(f"PC{k + 1} (var {var[k]:.3f})\n  + {', '.join(rows[-1]['high'])}"
              f"\n  - {', '.join(rows[-1]['low'])}", flush=True)

    out = {"mean_resultant_length": round(mrl, 3),
           "top_pcs_var_fraction": round(float(var[:args.n_pcs].sum()), 3),
           "n_dims": int(E.shape[1]), "components": rows}
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\ntop-{args.n_pcs} PCs = {out['top_pcs_var_fraction']:.1%} of variance; "
          f"mean-resultant-length {mrl:.3f}. wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
