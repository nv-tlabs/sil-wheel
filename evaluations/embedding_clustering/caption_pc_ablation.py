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

"""Common-component removal on the caption embedding (clustering ablation).

Caption clusters look diffuse because a few leading principal components carry
broad, shared scene structure (lighting / road type / weather) that is largely
orthogonal to the scenarios users search for (see ``pc_topics.py``). Projecting
those components out before clustering sharpens the label-relevant structure a
little. This writes the ablated embedding as a normal ``caption_pc<r>.npz``
encoder so it can be clustered / scored like any other.

Fit the components on the FULL corpus (as done here): fitting on a labeled
slice removes discriminative variance and hurts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def remove_top_pcs(emb: np.ndarray, r: int, fit_on: np.ndarray | None = None) -> np.ndarray:
    """Mean-center and project out the top-``r`` principal components.

    Components are fit on ``fit_on`` (default: ``emb`` itself, i.e. the full
    corpus). Returns L2-normalized rows. ``r=0`` is plain centering.
    """
    emb = np.asarray(emb, dtype=np.float32)
    mu = emb.mean(0)
    src = emb if fit_on is None else np.asarray(fit_on, dtype=np.float32)
    if r > 0:
        _, _, Vt = np.linalg.svd(src - src.mean(0), full_matrices=False)
        P = Vt[:r]
        ec = emb - mu
        out = ec - (ec @ P.T) @ P
    else:
        out = emb - mu
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-12)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caption-npz", type=Path, required=True,
                    help="Monolithic caption <caption>.npz to ablate.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--remove-pcs", type=int, default=5)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.caption_npz, allow_pickle=True)
    ids = [str(x) for x in data["clip_ids"]]
    emb = np.ascontiguousarray(data["embeddings"], dtype=np.float32)
    pc = remove_top_pcs(emb, args.remove_pcs, fit_on=emb)  # fit on the full corpus
    out = args.out_dir / f"caption_pc{args.remove_pcs}.npz"
    np.savez(out, clip_ids=np.array(ids, dtype=object), embeddings=pc)
    print(f"[pc-ablation] wrote {out.name}: {pc.shape} "
          f"(removed top {args.remove_pcs} PCs)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
