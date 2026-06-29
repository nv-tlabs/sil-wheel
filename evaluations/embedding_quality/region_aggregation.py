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

"""Region-embedding aggregators over the Florence-2 / SigCLIP detection archive.

The visual archive is *per-detection*: each clip contributes one
``__full_frame__`` row plus one row per detected object crop (SigLIP2 image
embedding). The other encoders are one vector per clip, so to put a "Region"
row in the same table we have to aggregate the detection set per clip. Two
aggregators are implemented here:

  - **Region BoF** — a TF-IDF bag-of-features histogram over a visual
    vocabulary fit on the detections (a per-clip vector; rides the normal
    vector metrics path).
  - **Region Chamfer** — no per-clip vector at all. A clip is its *set* of
    detections and the only geometry is the symmetric Chamfer similarity S;
    clustering / kNN / few-shot are computed natively on S (see
    ``chamfer_cluster``).

Both read the grounded-balanced archive produced by
``scripts/extract_florence2_sigclip_embeddings.py`` (grounding-phrase
detection + label-balanced selection). Benchmark-scale only: the scan holds
all wanted clips' detections in memory and S is dense N×N.
"""
from __future__ import annotations

import glob
import os
import pickle
import time

import numpy as np

from chamfer_cluster import (
    build_chamfer_similarity,
    cluster_all,
    few_shot_binary_knn_precomputed,
    knn_purity_precomputed,
    metrics_from_clusters,
)

# Default archive glob. Override with the env var or the CLI flag in
# run_embedding_quality.py. Points at the grounded-balanced extraction output.
DEFAULT_REGION_ARCHIVE = os.environ.get(
    "REGION_ARCHIVE_GLOB",
    "/lustre/fs12/portfolios/nvr/projects/nvr_torontoai_3dgenfoundation/"
    "wheel_data/visual_embeddings_grounding_balanced_benchmark/"
    "physical_ai/**/*.pkl",
)


def scan_region_sets(
    archive_glob: str, wanted: set[str],
) -> tuple[dict[str, list[tuple[str, np.ndarray]]], dict[str, np.ndarray]]:
    """Scan pkl shards and return detection rows + full-frame fallbacks.

    Returns:
        detections:  {clip_id: [(label, emb), ...]}  — detection rows only
        full_frames: {clip_id: emb}                  — __full_frame__ fallback

    Clips are dropped from `remaining` after the shard they appear in is
    processed (assumes all rows for a clip live in one shard, matching the
    archive's per-shard grouping). Memory scales with `wanted`.
    """
    files = sorted(glob.glob(archive_glob, recursive=True))
    if not files:
        raise FileNotFoundError(archive_glob)

    detections: dict[str, list[tuple[str, np.ndarray]]] = {c: [] for c in wanted}
    full_frames: dict[str, np.ndarray] = {}
    remaining = set(wanted)

    t0 = time.time()
    for fpath in files:
        if not remaining:
            break
        with open(fpath, "rb") as f:
            d = pickle.load(f)
        emb = d["embeddings"]
        items = d["items"]

        shard_hits: set[str] = set()
        for i, it in enumerate(items):
            cid = it.get("clip_id")
            if cid not in remaining:
                continue
            # np.array (not np.asarray) copies the view so the shard buffer
            # stays garbage-collectable across the multi-shard scan.
            vec = np.array(emb[i], dtype=np.float32)
            label = it.get("label", "")
            if label == "__full_frame__":
                full_frames[cid] = vec
            else:
                detections[cid].append((label, vec))
            shard_hits.add(cid)
        remaining -= shard_hits

    n_present = sum(
        1 for c in wanted if detections[c] or c in full_frames
    )
    print(
        f"[region] scanned {len(files)} shards: {n_present} / {len(wanted)} "
        f"clips have rows ({time.time() - t0:.1f}s)",
        flush=True,
    )
    return detections, full_frames


def region_clip_ids(
    detections: dict[str, list[tuple[str, np.ndarray]]],
    full_frames: dict[str, np.ndarray],
    wanted: set[str],
) -> list[str]:
    """Sorted clip_ids with at least one row (detection or full-frame)."""
    return sorted(
        c for c in wanted if detections.get(c) or c in full_frames
    )


def build_bof(
    detections: dict[str, list[tuple[str, np.ndarray]]],
    full_frames: dict[str, np.ndarray],
    wanted: set[str],
    n_vocab: int = 128,
) -> tuple[list[str], np.ndarray]:
    """TF-IDF bag-of-features histogram per clip over a visual vocabulary.

    1. Fit MiniBatchKMeans(V) on all detection embeddings from wanted clips.
    2. Hard-assign each detection to its nearest vocab word.
    3. Per-clip h[k] = count(assignments == k).
    4. TF-IDF weight: h_idf = h * idf, idf[k] = log(N / df[k]).
    5. L2-normalise.
    Clips with no detection rows fall back to the __full_frame__ embedding
    treated as a single detection, so every matched clip gets a V-dim vector.
    """
    from sklearn.cluster import MiniBatchKMeans

    t0 = time.time()
    clip_ids_flat: list[str] = []
    embs_list: list[np.ndarray] = []
    for cid, rows in detections.items():
        for _, emb in rows:
            clip_ids_flat.append(cid)
            embs_list.append(emb)

    if not embs_list:
        ids = [c for c in sorted(wanted) if c in full_frames]
        X = np.stack([full_frames[c] for c in ids]) if ids else np.zeros(
            (0, 0), dtype=np.float32
        )
        return ids, X

    all_embs = np.stack(embs_list)            # (total_detections, dim)
    actual_vocab = min(n_vocab, len(all_embs))
    kmeans = MiniBatchKMeans(
        n_clusters=actual_vocab, n_init=3, random_state=0, batch_size=4096,
    )
    assignments = kmeans.fit_predict(all_embs)
    print(
        f"[region] BoF vocab fitted ({actual_vocab} centers, "
        f"{len(all_embs)} detections)",
        flush=True,
    )

    clip_to_idxs: dict[str, list[int]] = {}
    for idx, cid in enumerate(clip_ids_flat):
        clip_to_idxs.setdefault(cid, []).append(idx)

    N = len(wanted)
    word_doc_count = np.zeros(actual_vocab, dtype=np.int32)
    for idxs in clip_to_idxs.values():
        word_doc_count[np.unique(assignments[idxs])] += 1
    idf = np.log(N / np.maximum(word_doc_count, 1)).astype(np.float32)

    found_ids: list[str] = []
    found_vecs: list[np.ndarray] = []
    for cid in sorted(wanted):
        idxs = clip_to_idxs.get(cid)
        if idxs:
            hist = np.bincount(
                assignments[idxs], minlength=actual_vocab,
            ).astype(np.float32) * idf
        elif cid in full_frames:
            k0 = int(kmeans.predict(full_frames[cid].reshape(1, -1))[0])
            hist = np.zeros(actual_vocab, dtype=np.float32)
            hist[k0] = idf[k0]
        else:
            continue
        norm = float(np.linalg.norm(hist))
        found_ids.append(cid)
        found_vecs.append(hist / norm if norm > 0 else hist)

    X = np.stack(found_vecs) if found_vecs else np.zeros((0, 0), dtype=np.float32)
    print(
        f"[region] BoF: {len(found_ids)} / {N} clips, {X.shape} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )
    return found_ids, X


def build_detection_matrix(
    clip_ids: list[str],
    detections: dict[str, list[tuple[str, np.ndarray]]],
    full_frames: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Stack the given clips' detection rows into a contiguous matrix.

    Args:
        clip_ids: clips to include, in the desired (sorted) row-block order.

    Returns:
        present_ids: subset of clip_ids that had at least one row, same order
        offsets:     (n_clips+1,) int64; clip i owns D[offsets[i]:offsets[i+1]]
        D:           (N_total, dim) float32, clip-contiguous, L2-normalized
                     (the archive vectors are already unit norm).
    """
    clip_row_lists: list[list[np.ndarray]] = []
    present_ids: list[str] = []
    for cid in clip_ids:
        rows = [emb for _, emb in detections.get(cid, [])]
        if not rows and cid in full_frames:
            rows = [full_frames[cid]]
        if rows:
            clip_row_lists.append(rows)
            present_ids.append(cid)

    n_clips = len(present_ids)
    offsets = np.zeros(n_clips + 1, dtype=np.int64)
    for i, rows in enumerate(clip_row_lists):
        offsets[i + 1] = offsets[i] + len(rows)
    n_total = int(offsets[-1])
    dim = clip_row_lists[0][0].shape[0]
    D = np.empty((n_total, dim), dtype=np.float32)
    for i, rows in enumerate(clip_row_lists):
        D[offsets[i]:offsets[i + 1]] = np.stack(rows)
    return present_ids, offsets, D


def _label_indep_keys(cluster_ks: list[int]) -> set[str]:
    """Chamfer metric keys that don't depend on GT labels (clustering is
    unsupervised), pulled to the top level in multi-label mode."""
    keys: set[str] = set()
    for k in cluster_ks:
        keys.update({f"intra_sim_k{k}", f"inter_sim_k{k}",
                     f"separation_k{k}", f"n_empty_clusters_k{k}"})
    return keys


def chamfer_metrics(
    detections: dict[str, list[tuple[str, np.ndarray]]],
    full_frames: dict[str, np.ndarray],
    inter_order: list[str],
    *,
    multi_label: bool,
    labels_for_pos: dict[str, list[str]],
    label_names: list[str],
    inter_labels: np.ndarray,
    ks: list[int],
    cluster_ks: list[int],
    few_shot_n: list[int],
    few_shot_trials: int,
    few_shot_seed: int,
    seed: int = 0,
) -> dict:
    """Set-native Chamfer K-medoids metrics over the run's intersection.

    Builds S over exactly `inter_order` (so the Region row is scored on the
    same clip set as every other encoder), clusters with K-medoids once
    (label-independent), then derives per-label metrics. The returned dict has
    the same shape as the vector encoders' metrics, so build_table renders it
    unchanged; intra/inter are Chamfer *set* similarities, not cosines.
    """
    present_ids, offsets, D = build_detection_matrix(
        inter_order, detections, full_frames,
    )
    assert present_ids == inter_order, (
        "every intersection clip must have at least one region row; "
        f"{len(present_ids)} present vs {len(inter_order)} expected"
    )
    t0 = time.time()
    S = build_chamfer_similarity(D, offsets)
    print(
        f"[region] Chamfer S {S.shape} built in {time.time() - t0:.1f}s",
        flush=True,
    )
    clusters = cluster_all(S, algo="kmedoids", k_values=cluster_ks, seed=seed)
    li_keys = _label_indep_keys(cluster_ks)

    def _per_label(lbl_vec: np.ndarray) -> dict:
        cm = metrics_from_clusters(S, lbl_vec, clusters)
        knn_m = knn_purity_precomputed(S, lbl_vec, k_values=ks)
        cm.update({k: v for k, v in knn_m.items()
                   if k not in ("n_clips", "n_pos", "n_neg")})
        return cm

    if multi_label:
        per_label: dict[str, dict] = {}
        label_indep: dict = {}
        for lbl in label_names:
            lbl_vec = np.array(
                [1 if lbl in labels_for_pos.get(c, []) else 0
                 for c in inter_order], dtype=np.int8)
            n_pos_lbl = int(lbl_vec.sum())
            n_neg_lbl = len(lbl_vec) - n_pos_lbl
            if n_pos_lbl < 2 or n_neg_lbl < 2:
                print(
                    f"[region] chamfer/{lbl}: too few pos/neg "
                    f"({n_pos_lbl}/{n_neg_lbl}); skipping",
                    flush=True,
                )
                continue
            cm = _per_label(lbl_vec)
            if not label_indep:
                label_indep = {k: cm[k] for k in li_keys if k in cm}
            per_dep = {k: v for k, v in cm.items()
                       if k not in li_keys
                       and k not in ("n_clips", "n_pos", "n_neg")}
            per_dep["n_pos"] = n_pos_lbl
            per_dep["n_neg"] = n_neg_lbl
            if few_shot_n:
                per_dep.update(few_shot_binary_knn_precomputed(
                    S, lbl_vec, n_values=list(few_shot_n),
                    n_trials=few_shot_trials, seed=few_shot_seed))
            per_label[lbl] = per_dep
        return {
            "n_dims": int(len(inter_order)),
            "n_clips": int(len(inter_order)),
            **label_indep,
            "per_label": per_label,
        }

    metrics = _per_label(inter_labels)
    metrics = {k: v for k, v in metrics.items()
               if k not in ("n_clips",)}
    if few_shot_n:
        metrics.update(few_shot_binary_knn_precomputed(
            S, inter_labels, n_values=list(few_shot_n),
            n_trials=few_shot_trials, seed=few_shot_seed))
    metrics["n_dims"] = int(len(inter_order))
    metrics["n_clips"] = int(len(inter_order))
    return metrics
