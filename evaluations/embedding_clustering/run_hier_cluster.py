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

"""Drive the recursive (hierarchical) k-means wrapper on real embeddings.

Reconstructs a pool's vectors from a wheel FAISS index and runs
:func:`sil_wheel.cluster_hierarchy.build_hierarchical_clustering` with topic
extraction at EVERY level, producing a progressively finer cluster-topic
taxonomy (``hier_assignments.parquet`` + ``hier_topics.json``).

Put the repo root and its ``scripts/`` on PYTHONPATH so ``sil_wheel`` and
``embed_io`` import:

    PYTHONPATH=$REPO:$REPO/scripts python run_hier_cluster.py \
        --wheel-data-dir /path/to/wheel-data --captions-db /path/to/captions.db \
        --pools-dir ./emb_pools --pool full --embed cosmos \
        --branching 10 --max-depth 2 --out ./hier/full_cosmos
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
from embed_io import load_clip_to_index

from sil_wheel.cluster_hierarchy import build_hierarchical_clustering


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wheel-data-dir", type=Path, required=True)
    ap.add_argument("--captions-db", type=Path, required=True)
    ap.add_argument("--pools-dir", type=Path, required=True,
                    help="dir with <pool>_clip_ids.json (from build_pool_clip_ids.py)")
    ap.add_argument("--pool", required=True, help="pool name = <pool>_clip_ids.json")
    ap.add_argument("--embed", choices=["cosmos", "visual", "caption"], default="cosmos")
    ap.add_argument("--branching", type=int, default=10)
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--min-cluster-size", type=int, default=200)
    ap.add_argument("--no-spherical", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    wd = args.wheel_data_dir
    embed_dirs = {
        "cosmos": (wd, "ivf4096_pq96x8"),
        "visual": (wd / "visual_embeddings", "ivf4096_pq64x8"),
        # caption uses the writable pools dir holding the full-coverage cti map
        "caption": (args.pools_dir / "caption_embeddings", "ivf4096_pq256x8"),
    }
    embed_dir, tag = embed_dirs[args.embed]
    idx_path = embed_dir / f"{args.embed}_embeddings_{tag}.index"
    print(f"[hier] loading {args.embed} index {idx_path} (mmap)...", flush=True)
    index = faiss.read_index(str(idx_path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    index.make_direct_map()
    clip_to_index = load_clip_to_index(str(embed_dir), args.embed, tag)
    print(f"[hier] index ntotal={index.ntotal:,}", flush=True)

    pool_clip_ids = json.loads((args.pools_dir / f"{args.pool}_clip_ids.json").read_text())
    rows, clip_ids = [], []
    for c in (str(x) for x in pool_clip_ids):
        r = clip_to_index.get(c)
        if r is not None:
            rows.append(int(r))
            clip_ids.append(c)
    print(f"[hier] pool {args.pool}: {len(clip_ids):,} clips present in index", flush=True)

    t0 = time.perf_counter()
    embeddings = index.reconstruct_batch(np.asarray(rows, dtype="int64"))
    print(f"[hier] reconstructed {embeddings.shape} in {time.perf_counter()-t0:.1f}s", flush=True)

    root = build_hierarchical_clustering(
        embeddings,
        clip_ids,
        branching=args.branching,
        max_depth=args.max_depth,
        min_cluster_size=args.min_cluster_size,
        captions_db_path=str(args.captions_db),
        spherical=not args.no_spherical,
        output_dir=args.out,
    )

    print(f"\n[hier] taxonomy written to {args.out}", flush=True)
    print(f"[hier] level-1 topics ({len(root.children)} clusters):", flush=True)
    for path, child in list(root.children.items())[:12]:
        kws = ", ".join(child.keywords[:6])
        print(f"  {path:>4}  n={child.size:>7}  {kws}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
