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
vectors directly (no PQ reconstruction); optionally restricts to a clip-id pool
so every embedding clusters the same clips, and ``--center`` mean-centers +
renormalizes for anisotropic encoders (e.g. visual/SigLIP).

Two modes:

* flat (default) -- spherical k-means via ``sil_wheel.cluster_build``; writes a
  run dir (cluster_assignments / centroids / umap.json / cluster_topics.json /
  metadata) byte-identical in shape to the served-index path, consumed unchanged
  by the figure scripts.
* ``--hierarchical`` -- recursive k-means taxonomy with topic extraction at
  every level; an iterative frontier of (path, depth, row-indices) nodes split
  with ``faiss.Kmeans`` (indices-into-X scheme as in sklearn's
  BisectingKMeans); writes ``hier_assignments.parquet`` + ``hier_topics.json``.

    # flat
    python cluster_raw.py --npz cosmos.npz --embed cosmos --k 1000 --spherical \
        --pool pai_clip_ids.json --captions-db captions.db \
        --clustering-dir ./clustering --run-id k50_cosmos
    # hierarchical
    python cluster_raw.py --npz cosmos.npz --embed cosmos --hierarchical \
        --captions-db captions.db --branching 10 --max-depth 2 --out ./hier/pai_cosmos
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from sil_wheel.cluster_build import (
    build_clustering_run,
    generate_run_id,
    write_cluster_assignments,
)
from sil_wheel.stores.cluster_topics import extract_topics_for_run, read_topics


def _load(npz, pool, center, label):
    """Load clip_ids + embeddings, optionally restrict to a pool and mean-center."""
    d = np.load(npz, allow_pickle=True)
    clip_ids = [str(x) for x in d["clip_ids"]]
    emb = np.ascontiguousarray(d["embeddings"], dtype=np.float32)
    if pool is not None:
        keep = {str(x) for x in json.loads(Path(pool).read_text())}
        idx = np.array([i for i, c in enumerate(clip_ids) if c in keep], dtype=np.int64)
        emb = np.ascontiguousarray(emb[idx])
        clip_ids = [clip_ids[i] for i in idx.tolist()]
    if center:
        emb = emb - emb.mean(axis=0, keepdims=True)
        nrm = np.linalg.norm(emb, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        emb = np.ascontiguousarray(emb / nrm, dtype=np.float32)
        print(f"[{label}] mean-centered + renormalized (anisotropy fix)", flush=True)
    return emb, clip_ids


def _run_flat(args, emb, clip_ids):
    print(
        f"[{args.embed}] clustering {emb.shape[0]:,} clips × {emb.shape[1]} "
        f"(k={args.k}, spherical={args.spherical}, center={args.center})",
        flush=True,
    )
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
    print(
        f"[{args.embed}] run {run_id} -> {run_dir} ({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )
    if args.runs_tsv is not None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with open(args.runs_tsv, "a") as f:
            f.write(
                f"{ts}\t{args.pool_name}\t{args.embed}\t{run_id}\t{run_dir}\tDONE_rc=0\n"
            )
    return 0


def _split_kmeans(X, k, spherical, seed):
    """k-means one node's members with faiss; returns per-row labels."""
    km = faiss.Kmeans(
        X.shape[1],
        k,
        niter=25,
        spherical=spherical,
        seed=seed,
        gpu=False,
        max_points_per_centroid=256,
        verbose=False,
    )
    km.train(X)
    _, labels = km.index.search(X, 1)
    return labels.ravel()


def _node_topics(clip_ids, labels, k, args):
    """Keywords/description per child cluster via a temp run dir."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        write_cluster_assignments(
            run_dir, labels, np.zeros(len(labels), dtype=np.float32), list(clip_ids), k
        )
        extract_topics_for_run(
            run_dir,
            str(args.captions_db),
            model_name=args.caption_model,
            samples_per_cluster=50,
        )
        return read_topics(run_dir).get("topics", {})


def _run_hier(args, emb, clip_ids):
    print(
        f"[hier] {emb.shape[0]:,} clips × {emb.shape[1]} "
        f"(branching={args.branching}, max_depth={args.max_depth})",
        flush=True,
    )
    t0 = time.perf_counter()
    clip_ids = np.asarray(clip_ids, dtype=object)
    k, spherical = args.branching, not args.no_spherical
    topics = {}  # dotted path -> {keywords, description, size, depth}
    leaves = []  # (path, depth, row indices into emb)
    frontier = [("", 0, np.arange(len(clip_ids)))]
    while frontier:
        path, depth, idx = frontier.pop()
        if depth >= args.max_depth or len(idx) < max(args.min_cluster_size, 2 * k):
            leaves.append((path, depth, idx))
            continue
        labels = _split_kmeans(np.ascontiguousarray(emb[idx]), k, spherical, args.seed)
        node_topics = _node_topics(clip_ids[idx], labels, k, args)
        for cid in range(k):
            child_idx = idx[labels == cid]
            if len(child_idx) == 0:
                continue
            child_path = f"{path}.{cid}" if path else str(cid)
            t = node_topics.get(str(cid), node_topics.get(cid, {}))
            topics[child_path] = {
                "keywords": list(t.get("keywords", [])),
                "description": t.get("description", ""),
                "size": int(len(child_idx)),
                "depth": depth + 1,
            }
            frontier.append((child_path, depth + 1, child_idx))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "hier_topics.json").write_text(json.dumps(topics, indent=2))
    sizes = [len(idx) for _, _, idx in leaves]
    pd.DataFrame(
        {
            "clip_id": np.concatenate([clip_ids[idx] for _, _, idx in leaves]),
            "path": np.repeat([p for p, _, _ in leaves], sizes),
            "depth": np.repeat([d for _, d, _ in leaves], sizes),
        }
    ).to_parquet(args.out / "hier_assignments.parquet", index=False)
    print(
        f"[hier] taxonomy written to {args.out} ({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )
    level1 = [(p, t) for p, t in topics.items() if t["depth"] == 1]
    print(f"[hier] level-1 topics ({len(level1)} clusters):", flush=True)
    for p, t in level1[:12]:
        print(f"  {p:>4}  n={t['size']:>7}  {', '.join(t['keywords'][:6])}", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # shared
    ap.add_argument(
        "--npz", type=Path, required=True, help="npz with clip_ids + embeddings"
    )
    ap.add_argument(
        "--embed",
        choices=["cosmos", "caption", "visual"],
        default=None,
        help="embed_type label (required for flat; written into metadata.json)",
    )
    ap.add_argument(
        "--pool",
        type=Path,
        default=None,
        help="optional JSON list of clip_ids to restrict to",
    )
    ap.add_argument(
        "--captions-db",
        type=Path,
        default=None,
        help="SQLite captions DB for topic extraction (required for --hierarchical)",
    )
    ap.add_argument(
        "--center",
        action="store_true",
        help="mean-center + renormalize before clustering; fixes anisotropic "
        "encoders (e.g. visual/SigLIP) whose vectors occupy a narrow cone",
    )
    ap.add_argument(
        "--hierarchical",
        action="store_true",
        help="recursive k-means taxonomy instead of a single flat clustering",
    )
    # flat
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--spherical", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-points-per-centroid", type=int, default=256)
    ap.add_argument("--caption-model", default=None)
    ap.add_argument(
        "--clustering-dir",
        type=Path,
        default=None,
        help="[flat] parent dir; a <run_id> subdir is created",
    )
    ap.add_argument(
        "--runs-tsv",
        type=Path,
        default=None,
        help="[flat] append a run-tracking row (matches run_embedding_clustering.py)",
    )
    ap.add_argument("--pool-name", default="pai", help="[flat] pool label for runs.tsv")
    ap.add_argument(
        "--run-id", default=None, help="[flat] force this run id (default: random)"
    )
    # hierarchical
    ap.add_argument("--branching", type=int, default=10)
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--min-cluster-size", type=int, default=200)
    ap.add_argument(
        "--no-spherical", action="store_true", help="[hierarchical] disable spherical"
    )
    ap.add_argument(
        "--out", type=Path, default=None, help="[hierarchical] taxonomy output dir"
    )
    args = ap.parse_args(argv)

    if args.hierarchical:
        if args.captions_db is None or args.out is None:
            ap.error("--hierarchical requires --captions-db and --out")
    elif args.embed is None or args.clustering_dir is None:
        ap.error("flat clustering requires --embed and --clustering-dir")

    emb, clip_ids = _load(
        args.npz, args.pool, args.center, "hier" if args.hierarchical else args.embed
    )
    return (
        _run_hier(args, emb, clip_ids)
        if args.hierarchical
        else _run_flat(args, emb, clip_ids)
    )


if __name__ == "__main__":
    raise SystemExit(main())
