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

"""Pre-index (exact) vs after-index (PQ-reconstructed) clustering comparison.

Clusters the SAME clips twice with identical settings: once on exact vectors and
once on the product-quantized (PQ) vectors served at scale. Reports how much PQ
perturbs the partition (ARI/NMI agreement), the storage it saves
(bytes/vec + compression), the wall-clock reconstruct/cluster time, and
intrinsic cluster quality on each side (PQ-OPTIMISTIC -- see notes below).

Two modes:
  * ``--wheel-data-dir`` (flat-index mode): exact from a Flat index, PQ from the
    production PQ index, for an encoder that ships both (e.g. cosmos).
  * ``--raw-npz`` (proxy mode): exact from a raw ``.npz`` of vectors, PQ by
    self-quantizing to ``--index-spec`` -- isolates the pure quantization effect
    where no production Flat index exists.

    python preindex_compare.py --wheel-data-dir /path/to/wheel-data --embed cosmos
    python preindex_compare.py --raw-npz enc.npz --index-spec IVF4096,PQ256x8 --embed caption
"""

import argparse
import json
import pickle
import re
import time
from pathlib import Path

import faiss
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def _load_map(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)


def _reconstruct(index_path, row_ids):
    ix = faiss.read_index(str(index_path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    # IVF/PQ indexes need a direct map to reconstruct by id; Flat indexes
    # reconstruct directly and have no make_direct_map.
    if hasattr(ix, "make_direct_map"):
        try:
            ix.make_direct_map()
        except (AttributeError, RuntimeError):
            pass
    return np.ascontiguousarray(ix.reconstruct_batch(row_ids), dtype=np.float32)


def _quantize_reconstruct(X, index_spec, seed):
    """Build a FAISS index of the given spec from raw vectors, then reconstruct.

    Isolates the *pure* product-quantization loss: same vectors, only the
    codebook differs. Returned rows are in input order (aligned with X)."""
    d = X.shape[1]
    ix = faiss.index_factory(d, index_spec, faiss.METRIC_INNER_PRODUCT)
    n_train = min(len(X), 400_000)
    rng = np.random.default_rng(seed)
    train = X[rng.choice(len(X), n_train, replace=False)] if n_train < len(X) else X
    ix.train(np.ascontiguousarray(train))
    ix.add(np.ascontiguousarray(X))
    if hasattr(ix, "make_direct_map"):
        ix.make_direct_map()
    return np.ascontiguousarray(ix.reconstruct_n(0, len(X)), dtype=np.float32)


def _cluster(X, k, seed):
    """Spherical (cosine) k-means via faiss, matching wheel's FaissKMeans
    (spherical=True, max_points_per_centroid=256).

    Returns ``(labels, centroids, train_secs, assign_secs)``.
    """
    d = X.shape[1]
    km = faiss.Kmeans(
        d,
        k,
        niter=25,
        nredo=1,
        spherical=True,
        seed=seed,
        gpu=False,
        max_points_per_centroid=256,
        verbose=False,
    )
    t0 = time.perf_counter()
    km.train(np.ascontiguousarray(X, dtype=np.float32))
    train_secs = time.perf_counter() - t0
    t1 = time.perf_counter()
    _, labels = km.index.search(np.ascontiguousarray(X, dtype=np.float32), 1)
    assign_secs = time.perf_counter() - t1
    return (
        labels.ravel().astype(np.int64),
        km.centroids.reshape(k, d),
        train_secs,
        assign_secs,
    )


def _pq_bytes_per_vec(index_spec):
    """Bytes stored per vector for a PQ index_spec like 'IVF4096,PQ96x8'
    (96 sub-quantizers x 8 bits = 96 bytes). None if not a PQ spec."""
    m = re.search(r"PQ(\d+)x(\d+)", index_spec or "")
    if not m:
        return None
    n_sub, nbits = int(m.group(1)), int(m.group(2))
    return (n_sub * nbits) // 8


def _centroid_gap(X, labels, centroids, sample, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    Xs = X[idx]
    Xs = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-8)
    C = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    sims = Xs @ C.T  # (S, k) cosine to every centroid
    own = sims[np.arange(len(idx)), labels[idx]]
    between = (sims.sum(1) - own) / (C.shape[0] - 1)
    return float(own.mean() - between.mean())


def _silhouette(X, labels, sample, seed):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return float(
        silhouette_score(
            Xn,
            labels,
            metric="cosine",
            sample_size=min(sample, len(X)),
            random_state=seed,
        )
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--embed",
        default="cosmos",
        help="encoder name; used to build flat/PQ index filenames and label the run",
    )
    ap.add_argument(
        "--wheel-data-dir",
        type=Path,
        default=None,
        help="flat-index mode: dir with <embed>_embeddings_flat.index + the PQ index",
    )
    ap.add_argument(
        "--pq-tag",
        default="ivf4096_pq96x8",
        help="flat-index mode: tag for the production PQ index filename",
    )
    ap.add_argument(
        "--raw-npz",
        default=None,
        help="proxy mode: path to <encoder>.npz with 'clip_ids' + 'embeddings'",
    )
    ap.add_argument(
        "--index-spec",
        default="IVF4096,PQ96x8",
        help="PQ index_factory spec (self-quantization in proxy mode; "
        "also used to report the compression ratio)",
    )
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument(
        "--center",
        action="store_true",
        help="mean-center then re-normalize before clustering. Fixes the "
        "anisotropy 'cosine collapse' (e.g. Florence/SigLIP): removes the "
        "shared dominant direction so cosine becomes discriminative. "
        "Both exact and PQ are centered by the exact mean (fair).",
    )
    ap.add_argument("--n", type=int, default=0, help="cap clips (0 = all overlap)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sil-sample", type=int, default=10000)
    ap.add_argument("--gap-sample", type=int, default=50000)
    ap.add_argument("--out", type=Path, default=Path("preindex_compare.json"))
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    index_spec = args.index_spec
    recon_exact_secs = recon_pq_secs = None
    if args.raw_npz:
        # proxy mode: exact = raw vectors; PQ = self-quantized via index_spec.
        print(f"[{args.embed}] loading raw npz {args.raw_npz} ...", flush=True)
        data = np.load(args.raw_npz, allow_pickle=True)
        X_exact = np.ascontiguousarray(data["embeddings"], dtype=np.float32)
        if args.n and args.n < len(X_exact):
            rng = np.random.default_rng(args.seed)
            X_exact = X_exact[sorted(rng.choice(len(X_exact), args.n, replace=False))]
        print(
            f"  raw {X_exact.shape}; self-quantizing with '{args.index_spec}' ...",
            flush=True,
        )
        tq = time.perf_counter()
        X_pq = _quantize_reconstruct(X_exact, args.index_spec, args.seed)
        recon_pq_secs = time.perf_counter() - tq
    else:
        # flat-index mode: exact from a Flat index, PQ from the production PQ index.
        if args.wheel_data_dir is None:
            ap.error("flat-index mode needs --wheel-data-dir (or use --raw-npz)")
        wd = args.wheel_data_dir
        flat_idx = wd / f"{args.embed}_embeddings_flat.index"
        flat_map = wd / f"{args.embed}_clip_to_index_flat.pkl"
        pq_idx = wd / f"{args.embed}_embeddings_{args.pq_tag}.index"
        pq_map = wd / f"{args.embed}_clip_to_index_{args.pq_tag}.pkl"
        print(f"[{args.embed}] loading clip maps...", flush=True)
        fmap = {str(k): v for k, v in _load_map(flat_map).items()}
        pmap = {str(k): v for k, v in _load_map(pq_map).items()}
        common = sorted(set(fmap) & set(pmap))
        print(
            f"  exact={len(fmap):,}  pq={len(pmap):,}  overlap={len(common):,}",
            flush=True,
        )
        if args.n and args.n < len(common):
            rng = np.random.default_rng(args.seed)
            common = [
                common[i]
                for i in sorted(rng.choice(len(common), args.n, replace=False))
            ]
        flat_rows = np.array([int(fmap[c]) for c in common], dtype="int64")
        pq_rows = np.array([int(pmap[c]) for c in common], dtype="int64")
        print(
            f"[{args.embed}] reconstructing {len(common):,} exact + PQ vectors...",
            flush=True,
        )
        te = time.perf_counter()
        X_exact = _reconstruct(flat_idx, flat_rows)
        recon_exact_secs = time.perf_counter() - te
        tp = time.perf_counter()
        X_pq = _reconstruct(pq_idx, pq_rows)
        recon_pq_secs = time.perf_counter() - tp
    n_clips = len(X_exact)
    d = int(X_exact.shape[1])
    if args.center:
        # Remove the shared dominant direction (anisotropy) so cosine separates;
        # center BOTH sides by the exact mean so only PQ remains the variable.
        mu = X_exact.mean(axis=0, keepdims=True)

        def _cn(A):
            A = A - mu
            return np.ascontiguousarray(
                A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8), dtype=np.float32
            )

        X_exact, X_pq = _cn(X_exact), _cn(X_pq)
        print("  applied mean-centering + renormalization (--center)", flush=True)
    print(
        f"  prepared {X_exact.shape}/{X_pq.shape} in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    print(
        f"[{args.embed}] clustering k={args.k} spherical (exact, then PQ)...",
        flush=True,
    )
    lab_exact, cen_exact, train_e, assign_e = _cluster(X_exact, args.k, args.seed)
    lab_pq, cen_pq, train_p, assign_p = _cluster(X_pq, args.k, args.seed)

    ari = float(adjusted_rand_score(lab_exact, lab_pq))
    nmi = float(normalized_mutual_info_score(lab_exact, lab_pq))

    # --- footprint / compression: the dominant "much faster at scale" lever ---
    # We always reconstruct to float32 before k-means, so RAM during clustering
    # is identical; the real win is the *stored* index (what gets loaded and
    # streamed). Exact = d x 4 bytes/vec; PQ = (n_sub x nbits)/8 bytes/vec.
    bytes_exact = d * 4
    bytes_pq = _pq_bytes_per_vec(index_spec)
    compression = (bytes_exact / bytes_pq) if bytes_pq else None

    res = {
        "embed": args.embed,
        "k": args.k,
        "n_clips": n_clips,
        "dim": d,
        "mode": "raw-npz" if args.raw_npz else "flat-vs-pq",
        "index_spec": index_spec,
        "centered": bool(args.center),
        # --- agreement: does PQ change the partition? (the "lose little" axis) ---
        "ari_exact_vs_pq": ari,
        "nmi_exact_vs_pq": nmi,
        # --- footprint: stored bytes/vec + compression (the "much smaller" axis) ---
        "stored_bytes_per_vec_exact": bytes_exact,
        "stored_bytes_per_vec_pq": bytes_pq,
        "compression_ratio": round(compression, 1) if compression else None,
        # --- timing (wall-clock, this host): reconstruct + cluster, exact vs PQ.
        #     NOTE: in raw-npz mode recon_secs_pq includes one-time PQ *training*,
        #     so it is not a per-use reconstruct time -- cite flat-vs-pq instead. ---
        "recon_secs_exact": round(recon_exact_secs, 2)
        if recon_exact_secs is not None
        else None,
        "recon_secs_pq": round(recon_pq_secs, 2) if recon_pq_secs is not None else None,
        "cluster_secs_exact": round(train_e + assign_e, 2),
        "cluster_secs_pq": round(train_p + assign_p, 2),
        # --- intrinsic (PQ-OPTIMISTIC: codebook discretization inflates these;
        #     not a fair exact-vs-PQ quality comparison -- agreement is) ---
        "delta_gap_exact": _centroid_gap(
            X_exact, lab_exact, cen_exact, args.gap_sample, args.seed
        ),
        "delta_gap_pq": _centroid_gap(X_pq, lab_pq, cen_pq, args.gap_sample, args.seed),
        "silhouette_exact": _silhouette(X_exact, lab_exact, args.sil_sample, args.seed),
        "silhouette_pq": _silhouette(X_pq, lab_pq, args.sil_sample, args.seed),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    allres = json.loads(out.read_text()) if out.exists() else {}
    allres[args.embed] = res
    out.write_text(json.dumps(allres, indent=2))

    print("\n=== RESULT ===", flush=True)
    for kk, vv in res.items():
        print(f"  {kk}: {vv}", flush=True)
    if compression:
        full_exact_gb = bytes_exact * n_clips / 1e9
        full_pq_gb = bytes_pq * n_clips / 1e9
        print(
            f"\n[message] PQ keeps the coarse partition (NMI={nmi:.2f}) while "
            f"storing {compression:.0f}x fewer bytes/vec "
            f"({bytes_exact}B->{bytes_pq}B): the {n_clips:,}-clip index is "
            f"{full_exact_gb:.1f} GB exact vs {full_pq_gb:.2f} GB PQ. "
            f"Fine assignments shift (ARI={ari:.2f}); intrinsic scores are "
            f"PQ-optimistic and not a fair cross-comparison.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
