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

"""Cluster raw (pre-index, exact) embedding vectors from an npz.

Companion to ``ingest_raw_embeddings.py`` for the case where the embeddings are
on hand as exact vectors rather than a served FAISS index. Clusters the EXACT
vectors directly — no PQ reconstruction — via
``sil_wheel.cluster_build.build_clustering_run``, so the run directory it writes
(cluster_assignments / centroids / umap.json / cluster_topics.json / metadata)
is byte-identical in shape to what ``cluster_clips_and_select.py`` produces and
is consumed unchanged by the figure scripts.

Optionally restricts to a clip-id pool (e.g. the cross-encoder intersection from
``ingest_raw_embeddings.py``) so every embedding clusters the same clips.

    python cluster_raw.py --npz cosmos.npz --embed cosmos --k 1000 --spherical \
        --pool pai_clip_ids.json --captions-db captions.db \
        --clustering-dir ./clustering --runs-tsv ./runs.tsv --pool-name pai
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from sil_wheel.cluster_build import build_clustering_run, generate_run_id


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=Path, required=True, help="npz with clip_ids + embeddings")
    ap.add_argument("--embed", choices=["cosmos", "caption", "visual"], required=True,
                    help="embed_type label written into metadata.json")
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--spherical", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-points-per-centroid", type=int, default=256)
    ap.add_argument("--pool", type=Path, default=None,
                    help="optional JSON list of clip_ids to restrict to")
    ap.add_argument("--captions-db", type=Path, default=None)
    ap.add_argument("--caption-model", default=None)
    ap.add_argument("--clustering-dir", type=Path, required=True,
                    help="parent dir; a <run_id> subdir is created")
    ap.add_argument("--runs-tsv", type=Path, default=None,
                    help="append a run-tracking row (matches run_full_cluster.sh)")
    ap.add_argument("--pool-name", default="pai", help="pool label for runs.tsv")
    ap.add_argument("--center", action="store_true",
                    help="mean-center + renormalize before clustering; fixes anisotropic "
                         "encoders (e.g. visual/SigLIP) whose vectors occupy a narrow cone")
    ap.add_argument("--run-id", default=None, help="force this run id (default: random)")
    args = ap.parse_args(argv)

    d = np.load(args.npz, allow_pickle=True)
    clip_ids = [str(x) for x in d["clip_ids"]]
    emb = np.ascontiguousarray(d["embeddings"], dtype=np.float32)

    if args.pool is not None:
        import json
        keep = {str(x) for x in json.loads(args.pool.read_text())}
        idx = np.array([i for i, c in enumerate(clip_ids) if c in keep], dtype=np.int64)
        emb = np.ascontiguousarray(emb[idx])
        clip_ids = [clip_ids[i] for i in idx.tolist()]

    if args.center:
        mu = emb.mean(axis=0, keepdims=True)
        emb = emb - mu
        emb = np.ascontiguousarray(emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8),
                                   dtype=np.float32)
        print(f"[{args.embed}] mean-centered + renormalized (anisotropy fix)", flush=True)

    print(f"[{args.embed}] clustering {emb.shape[0]:,} clips × {emb.shape[1]} "
          f"(k={args.k}, spherical={args.spherical}, center={args.center})", flush=True)

    run_id = args.run_id or generate_run_id()
    t0 = time.perf_counter()
    run_dir = build_clustering_run(
        args.clustering_dir,
        emb,
        clip_ids,
        n_clusters=args.k,
        embed_type=args.embed,
        spherical=args.spherical,
        max_points_per_centroid=args.max_points_per_centroid,
        seed=args.seed,
        captions_db_path=str(args.captions_db) if args.captions_db else None,
        caption_model=args.caption_model,
        run_id=run_id,
    )
    secs = time.perf_counter() - t0
    print(f"[{args.embed}] run {run_id} -> {run_dir} ({secs:.1f}s)", flush=True)

    if args.runs_tsv is not None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        line = f"{ts}\t{args.pool_name}\t{args.embed}\t{run_id}\t{run_dir}\tDONE_rc=0\n"
        with open(args.runs_tsv, "a") as f:
            f.write(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
