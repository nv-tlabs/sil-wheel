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

"""Recursive (hierarchical) k-means taxonomy from raw (exact) vectors in an npz.

Companion to ``run_hier_cluster.py`` for the case where vectors are on hand as
an npz (from ``ingest_raw_embeddings.py``) rather than a served FAISS index.
Runs ``sil_wheel.cluster_hierarchy.build_hierarchical_clustering`` with topic
extraction at every level, writing ``hier_assignments.parquet`` +
``hier_topics.json`` (rendered by ``make_hier_viz.py``).

    PYTHONPATH=$REPO:$REPO/scripts python hier_raw.py \
        --npz cosmos.npz --captions-db captions.db \
        --branching 10 --max-depth 2 --out ./hier/pai_cosmos
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sil_wheel.cluster_hierarchy import build_hierarchical_clustering


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=Path, required=True, help="npz with clip_ids + embeddings")
    ap.add_argument("--captions-db", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=None,
                    help="optional JSON list of clip_ids to restrict to")
    ap.add_argument("--branching", type=int, default=10)
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--min-cluster-size", type=int, default=200)
    ap.add_argument("--no-spherical", action="store_true")
    ap.add_argument("--center", action="store_true",
                    help="mean-center + renormalize before clustering (anisotropic encoders, e.g. visual)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    d = np.load(args.npz, allow_pickle=True)
    clip_ids = [str(x) for x in d["clip_ids"]]
    emb = np.ascontiguousarray(d["embeddings"], dtype=np.float32)
    if args.pool is not None:
        keep = {str(x) for x in json.loads(args.pool.read_text())}
        idx = np.array([i for i, c in enumerate(clip_ids) if c in keep], dtype=np.int64)
        emb = np.ascontiguousarray(emb[idx])
        clip_ids = [clip_ids[i] for i in idx.tolist()]
    if args.center:
        emb = emb - emb.mean(axis=0, keepdims=True)
        nrm = np.linalg.norm(emb, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        emb = np.ascontiguousarray(emb / nrm, dtype=np.float32)
        print("[hier] mean-centered + renormalized", flush=True)

    print(f"[hier] {emb.shape[0]:,} clips × {emb.shape[1]} "
          f"(branching={args.branching}, max_depth={args.max_depth})", flush=True)
    t0 = time.perf_counter()
    root = build_hierarchical_clustering(
        emb,
        clip_ids,
        branching=args.branching,
        max_depth=args.max_depth,
        min_cluster_size=args.min_cluster_size,
        captions_db_path=str(args.captions_db),
        spherical=not args.no_spherical,
        output_dir=args.out,
    )
    print(f"[hier] taxonomy written to {args.out} ({time.perf_counter()-t0:.1f}s)", flush=True)
    print(f"[hier] level-1 topics ({len(root.children)} clusters):", flush=True)
    for path, child in list(root.children.items())[:12]:
        print(f"  {path:>4}  n={child.size:>7}  {', '.join(child.keywords[:6])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
