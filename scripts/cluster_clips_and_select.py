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

import argparse
import json
import random
import time
from pathlib import Path

import faiss
import numpy as np

from embed_io import load_clip_to_index
from sil_wheel.cluster_build import (
    FaissKMeans,
    fit_and_write_umap,
    write_centroids,
    write_cluster_assignments,
    write_metadata,
)
from sil_wheel.stores.cluster_topics import extract_topics_for_run


def faiss_index_to_gpu(cpu_index: faiss.Index, gpu_id: int = 0) -> faiss.Index:
    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    co.useFloat16 = False
    co.usePrecomputed = False
    return faiss.index_cpu_to_gpu(res, gpu_id, cpu_index, co)


def get_features_and_predict(features_index, ids_to_fetch, kmeans_model, batch_size=2_000_000):
    n_total = len(ids_to_fetch)
    all_labels = []
    all_distances = []

    t0 = time.perf_counter()
    for i in range(0, n_total, batch_size):
        batch_ids = ids_to_fetch[i : i + batch_size]

        # 1. Reconstruct only a small chunk (e.g., 100k points = ~300MB)
        batch_features = features_index.reconstruct_batch(batch_ids)
        # 2. Predict on this chunk immediately
        labels, dists = kmeans_model.predict(batch_features)

        all_labels.append(labels.squeeze())
        all_distances.append(dists.squeeze())

    print(f"Time for reconstruct + assign ({n_total} clips): {time.perf_counter() - t0:.3f}s")
    return np.concatenate(all_labels), np.concatenate(all_distances)


def spec_to_tag(spec):
    tag = spec.strip().lower().replace(",", "_").replace(" ", "")
    return tag.replace("/", "_").replace("-", "_")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Script for clustering clips and selecting subsets"
    )
    parser.add_argument("output_dir", help="Path to output dir")
    parser.add_argument("path_to_embeddings", help="Path to embeddings")
    parser.add_argument("n_clusters", type=int, help="Number of clusters")
    parser.add_argument(
        "--path_to_clip_ids",
        default=None,
        help=(
            "Optional JSON file containing clip_ids to cluster. "
            "Can be either a list [id1, id2, ...] or {'clip_ids': [...]}. "
            "If not provided, a random 500k sample is drawn from all available clips."
        ),
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=25,
        help="Number of KMeans iterations (passed to FAISS KMeans)",
    )
    parser.add_argument(
        "--n_redo",
        type=int,
        default=1,
        help="Number of KMeans restarts (FAISS nredo parameter)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed used by FAISS KMeans initialization",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging during KMeans training",
    )
    parser.add_argument(
        "--spherical_kmeans",
        action="store_true",
        help="Use spherical KMeans (cosine-style clustering) instead of L2",
    )
    parser.add_argument(
        "--max_points_per_centroid",
        type=int,
        default=256,
        help=(
            "FAISS KMeans max_points_per_centroid: training is capped at "
            "n_clusters × this value. Default (256) matches the FAISS "
            "built-in default. Raise this to use more data per cluster "
            "during training; all clips are still assigned afterwards."
        ),
    )
    parser.add_argument(
        "--embed_type",
        choices=["cosmos", "caption", "visual"],
        default="cosmos",
        help="Which embedding type to cluster on.",
    )
    parser.add_argument(
        "--index_tag",
        default="ivf4096_pq96x8",
        help="Pre-computed tag string for the FAISS index filename (e.g. ivf4096_pq96x8).",
    )
    parser.add_argument(
        "--captions_db",
        default=None,
        help=(
            "Optional path to the SQLite captions DB. If provided, per-cluster "
            "topic keywords are extracted as the final clustering step and "
            "written to cluster_topics.json in the output directory."
        ),
    )
    parser.add_argument(
        "--caption_model",
        default=None,
        help="Caption model name (exact match against captions.model_name) "
             "to use for topic extraction. If omitted, the topic step "
             "auto-selects the model with the most coverage of this run's "
             "clips.",
    )
    parser.add_argument(
        "--topic_threads",
        type=int,
        default=8,
        help="Threads used to fetch captions during topic extraction.",
    )
    args = parser.parse_args(argv)

    n_clusters = min(args.n_clusters, 20000)

    path_to_output = Path(args.output_dir)
    path_to_output.mkdir(parents=True, exist_ok=True)

    tag = args.index_tag
    path_to_faiss_index = (
        Path(args.path_to_embeddings) / f"{args.embed_type}_embeddings_{tag}.index"
    )

    if not path_to_faiss_index.exists():
        raise NotImplementedError("You need to specify a path to a FAISS index")

    # Memory-map the FAISS index so the on-disk file (which can run to
    # tens of GB for visual indices) doesn't have to be resident.
    features_index = faiss.read_index(
        str(path_to_faiss_index),
        faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,
    )
    clip_to_index = load_clip_to_index(
        args.path_to_embeddings, args.embed_type, tag,
    )
    print(f"[{tag}] Loaded {args.embed_type} index from {path_to_faiss_index}")
    print("ntotal:", features_index.ntotal)
    print("FAISS OMP threads:", faiss.omp_get_max_threads())
    features_index.make_direct_map()

    if args.path_to_clip_ids is None:
        clip_ids = random.sample(list(clip_to_index.keys()), 5000000)
    else:
        with open(args.path_to_clip_ids, "r") as f:
            clip_ids = json.load(f)

        if isinstance(clip_ids, list):
            clip_ids = [str(x) for x in clip_ids]
        elif isinstance(clip_ids, dict) and "clip_ids" in clip_ids:
            clip_ids = [str(x) for x in clip_ids["clip_ids"]]
        else:
            raise ValueError(
                f"Unsupported clip_ids JSON format in {args.path_to_clip_ids}"
            )

    kept_row_ids = []
    kept_clip_ids = []
    for c in clip_ids:
        if c not in clip_to_index:
            continue
        kept_row_ids.append(int(clip_to_index[c]))
        kept_clip_ids.append(c)
    del clip_ids, clip_to_index

    ids_to_fetch = np.asarray(kept_row_ids, dtype="int64")
    del kept_row_ids
    n_total = len(ids_to_fetch)

    if n_total > 5_000_000:
        rng = np.random.default_rng(args.seed)

        # Subsample IDs for assignment up-front — before any reconstruction —
        # so that n_total is bounded and the training sample is drawn from a
        # representative but much smaller pool of IDs (not vectors).
        MAX_ASSIGN = 5_000_000
        if n_total > MAX_ASSIGN:
            subset_idx = rng.choice(n_total, MAX_ASSIGN, replace=False)
            ids_to_fetch = ids_to_fetch[subset_idx]
            kept_clip_ids = [kept_clip_ids[i] for i in subset_idx.tolist()]
            n_total = MAX_ASSIGN
            print(f"Subsampled to {MAX_ASSIGN:,} clips for assignment.")

        # Reconstruct only the training cap — avoids allocating the full
        # n_total matrix before subsampling (e.g. 56 GB for 15 M clips).
        n_train = min(n_clusters * args.max_points_per_centroid, n_total)
        train_ids = ids_to_fetch[rng.choice(n_total, n_train, replace=False)]

        BATCH = 2_000_000
        print(
            f"Reconstructing {n_train:,} training vectors "
            f"(sampled from {n_total:,})..."
        )
        t0_reconstruct = time.perf_counter()
        train_features = np.empty(
            (n_train, features_index.d), dtype=np.float32
        )
        for i in range(0, n_train, BATCH):
            batch = features_index.reconstruct_batch(
                train_ids[i : i + BATCH]
            )
            train_features[i : i + len(batch)] = batch
        del train_ids
        print(
            f"Reconstructed training features: {train_features.shape} "
            f"({time.perf_counter() - t0_reconstruct:.3f}s)"
        )

        kmeans = FaissKMeans(
            feature_dim=train_features.shape[1],
            n_clusters=n_clusters,
            niter=args.n_iter,
            nredo=args.n_redo,
            verbose=bool(args.verbose),
            seed=args.seed,
            spherical_kmeans=bool(args.spherical_kmeans),
            max_points_per_centroid=args.max_points_per_centroid,
        )
        kmeans.fit(train_features)
        del train_features

        cluster_assignments, distances = get_features_and_predict(
            features_index, ids_to_fetch, kmeans
        )
        npfeatures = None
    else:
        print(f"Reconstructing {n_total} vectors in batch...")
        t0_reconstruct = time.perf_counter()
        npfeatures = features_index.reconstruct_batch(ids_to_fetch)
        print(
            f"Reconstructed features: {npfeatures.shape} "
            f"({time.perf_counter() - t0_reconstruct:.3f}s)"
        )
        kmeans = FaissKMeans(
            feature_dim=npfeatures.shape[1],
            n_clusters=n_clusters,
            niter=args.n_iter,
            nredo=args.n_redo,
            verbose=bool(args.verbose),
            seed=args.seed,
            spherical_kmeans=bool(args.spherical_kmeans),
            max_points_per_centroid=args.max_points_per_centroid,
        )
        kmeans.fit(npfeatures)
        cluster_assignments, distances = kmeans.predict(npfeatures)

    write_cluster_assignments(
        path_to_output, cluster_assignments, distances, kept_clip_ids, n_clusters,
    )
    write_centroids(path_to_output, kmeans.centroids)

    MAX_UMAP_CLIPS = 50000
    n_sub = min(MAX_UMAP_CLIPS, n_total)
    sub_idx = np.random.default_rng(args.seed).choice(n_total, n_sub, replace=False)
    if npfeatures is not None:
        sub_embeddings = npfeatures[sub_idx]
        del npfeatures
    else:
        sub_embeddings = features_index.reconstruct_batch(ids_to_fetch[sub_idx])
    sub_cluster_assignments = cluster_assignments[sub_idx]
    sub_distances = distances[sub_idx]
    sub_clip_ids = [kept_clip_ids[i] for i in sub_idx.tolist()]

    # Topics first so the server's done-detection (which keys on
    # umap.json existing) flips after topics are also persisted.
    if args.captions_db:
        try:
            t0_topics = time.perf_counter()
            extract_topics_for_run(
                path_to_output,
                args.captions_db,
                model_name=args.caption_model,
                n_threads=args.topic_threads,
            )
            print(
                f"Time for topic extraction: "
                f"{time.perf_counter() - t0_topics:.2f}s"
            )
        except Exception as e:
            print(f"[topics] extraction failed: {e}")

    fit_and_write_umap(
        path_to_output,
        n_clusters=n_clusters,
        centroids=kmeans.centroids,
        sub_embeddings=sub_embeddings,
        sub_cluster_assignments=sub_cluster_assignments,
        sub_clip_ids=sub_clip_ids,
        sub_distances=sub_distances,
    )
    write_metadata(
        path_to_output,
        run_id=path_to_output.name,
        n_clusters=n_clusters,
        n_input_clips=n_total,
        embed_type=args.embed_type,
        spherical_kmeans=bool(args.spherical_kmeans),
        max_points_per_centroid=args.max_points_per_centroid,
    )


if __name__ == "__main__":
    main()
