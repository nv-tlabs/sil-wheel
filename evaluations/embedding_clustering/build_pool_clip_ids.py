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

"""Build clip-id pools for the embedding-clustering evaluation.

Each pool is a JSON list of clip_ids present in ALL THREE embedding indices
(cosmos / visual / caption), so the same clip set can be clustered on every
embedding and the per-cluster topic distributions are directly comparable:

* ``full``  = the entire cosmos∩visual∩caption intersection (deterministic).
* ``large`` = a random ``--n-large`` sample of that intersection.
* ``pai``   = an optional curated subset, intersected with the common set,
              identified by clip-id path files passed via ``--pai-path-files``
              (one S3/local path per line; the clip id is the basename before
              the first ``.``). Skipped if no path files are given.

It also writes a full-coverage caption ``clip_id -> faiss row`` map next to the
pools, so the caption clustering run can resolve every clip even when the
caption index only ships a legacy ``index_to_clip`` map.

    python build_pool_clip_ids.py \
        --wheel-data-dir /path/to/wheel-data --out ./emb_pools
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def _uuids_from_path_file(path: Path) -> set[str]:
    out: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(line.split("/")[-1].split(".")[0])
    return out


def _load_cosmos_keys(cti_pkl: Path) -> set[str]:
    with open(cti_pkl, "rb") as f:
        return set(pickle.load(f).keys())


def _load_visual_keys(ids_npy: Path) -> set[str]:
    return {str(x) for x in np.load(ids_npy, allow_pickle=True)}


def _load_caption_itc(itc_pkl: Path) -> dict:
    """Legacy index->clip map ({row_index: clip_id}) for the full caption index."""
    with open(itc_pkl, "rb") as f:
        return pickle.load(f)


def _build_caption_cti(itc: dict, keep: set[str], out_dir: Path,
                       caption_index: Path, caption_tag: str) -> int:
    """Invert the legacy index->clip map into a clip_id->row map restricted to
    ``keep`` and persist it where ``embed_io.load_clip_to_index`` will find it
    (a writable caption dir), so the run never tries to cache into a read-only
    data dir. First (smallest) row wins per clip, matching wheel.
    """
    cti: dict[str, int] = {}
    for row in sorted(itc):
        cid = str(itc[row])
        if cid in keep and cid not in cti:
            cti[cid] = int(row)
    cap_dir = out_dir / "caption_embeddings"
    cap_dir.mkdir(parents=True, exist_ok=True)
    cti_path = cap_dir / f"caption_clip_to_index_{caption_tag}.pkl"
    with open(cti_path, "wb") as f:
        pickle.dump(cti, f)
    # make the .index reachable from the writable dir too (symlink, idempotent)
    link = cap_dir / f"caption_embeddings_{caption_tag}.index"
    if not link.exists() and caption_index.exists():
        link.symlink_to(caption_index)
    print(f"  wrote caption cti {len(cti):,} entries -> {cti_path}", flush=True)
    return len(cti)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wheel-data-dir", type=Path, required=True,
                    help="dir holding the cosmos/visual/caption indices + id maps")
    ap.add_argument("--out", type=Path, required=True, help="output dir for the pools")
    ap.add_argument("--cosmos-tag", default="ivf4096_pq96x8")
    ap.add_argument("--visual-tag", default="ivf4096_pq64x8")
    ap.add_argument("--caption-tag", default="ivf4096_pq256x8")
    ap.add_argument("--pai-path-files", type=Path, nargs="*", default=None,
                    help="optional clip-id path files defining a curated 'pai' subset")
    ap.add_argument("--n-large", type=int, default=2_500_000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    wd = args.wheel_data_dir
    cosmos_cti = wd / f"cosmos_clip_to_index_{args.cosmos_tag}.pkl"
    visual_ids = wd / "visual_embeddings" / f"visual_clip_ids_{args.visual_tag}.npy"
    caption_itc = wd / "caption_embeddings" / f"caption_index_to_clip_{args.caption_tag}.pkl"
    caption_index = wd / "caption_embeddings" / f"caption_embeddings_{args.caption_tag}.index"

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print("loading visual keys...", flush=True)
    common = _load_visual_keys(visual_ids)
    print(f"  visual: {len(common):,}", flush=True)
    print("loading cosmos keys + intersecting...", flush=True)
    common &= _load_cosmos_keys(cosmos_cti)
    print(f"  cosmos: {len(common):,}", flush=True)
    print("loading caption itc + intersecting...", flush=True)
    cap_itc = _load_caption_itc(caption_itc)
    common &= {str(v) for v in cap_itc.values()}
    print(f"  caption (full common set): {len(common):,}", flush=True)

    # --- full pool: the ENTIRE common set, deterministic (sorted, no sampling) ---
    full = sorted(common)
    (out / "full_clip_ids.json").write_text(json.dumps(full))
    print(f"full: {len(full):,} clips (entire common set)", flush=True)

    # full-coverage caption clip->row map for the caption clustering run
    n_cti = _build_caption_cti(cap_itc, common, out, caption_index, args.caption_tag)
    del cap_itc

    # --- optional pai pool: curated uuids restricted to the common set ---
    n_pai = 0
    if args.pai_path_files:
        pai: set[str] = set()
        for pf in args.pai_path_files:
            pai |= _uuids_from_path_file(pf)
        pai_common = sorted(pai & common)
        (out / "pai_clip_ids.json").write_text(json.dumps(pai_common))
        n_pai = len(pai_common)
        print(f"pai: {len(pai):,} uuids -> {n_pai:,} in common set", flush=True)

    # --- large pool: random sample of the common set ---
    rng = np.random.default_rng(args.seed)
    common_arr = np.array(sorted(common), dtype=object)
    n_large = min(args.n_large, len(common_arr))
    sel = rng.choice(len(common_arr), n_large, replace=False)
    large = sorted(common_arr[sel].tolist())
    (out / "large_clip_ids.json").write_text(json.dumps(large))
    print(f"large: sampled {len(large):,} of {len(common_arr):,}", flush=True)

    summary = {
        "common_set": len(common),
        "full": len(full),
        "caption_cti": n_cti,
        "pai": n_pai,
        "large": len(large),
        "n_large_requested": args.n_large,
        "seed": args.seed,
    }
    (out / "pool_summary.json").write_text(json.dumps(summary, indent=2))
    print("summary:", summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
