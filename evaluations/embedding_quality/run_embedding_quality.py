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

"""Run supervised embedding-quality probes from public ``.npz`` embeddings."""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from embeddings_io import load_embeddings
from metrics import cluster_metrics, few_shot_binary_knn, knn_purity
from region_aggregation import (
    DEFAULT_REGION_ARCHIVE,
    build_bof,
    chamfer_metrics,
    region_clip_ids,
    scan_region_sets,
)


# Region encoders are aggregated from the per-detection Florence/SigCLIP
# archive instead of a per-clip <name>.npz. BoF is a per-clip vector and rides
# the normal vector metrics path; Chamfer is set-native and has its own branch.
REGION_BOF_KEY = "florence2_sigclip_grounding_balanced_bof"
REGION_CHAMFER_KEY = "florence2_sigclip_grounding_balanced_chamfer_kmedoids"
REGION_KEYS = {REGION_BOF_KEY, REGION_CHAMFER_KEY}

DEFAULT_EMBEDDINGS = [
    "cosmos",
    "qwen3_vl_8b",
    "pe_core_g14",
    "caption",
    REGION_BOF_KEY,
    REGION_CHAMFER_KEY,
    "trajectory",
    "random",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    pos_group = parser.add_mutually_exclusive_group(required=True)
    pos_group.add_argument(
        "--positive-csv",
        type=Path,
        help="CSV of positive clip_ids for single-class mode.",
    )
    pos_group.add_argument(
        "--labels-csv",
        type=Path,
        help="CSV of (clip_id, label) rows for multi-label mode.",
    )
    parser.add_argument(
        "--negative-csv",
        required=True,
        type=Path,
        help="CSV of negative clip_ids.",
    )
    parser.add_argument(
        "--embeddings-dir",
        required=True,
        type=Path,
        help="Directory containing one <name>.npz per encoder.",
    )
    parser.add_argument(
        "--embeddings",
        nargs="+",
        default=DEFAULT_EMBEDDINGS,
        help="Encoder names to evaluate. Each must have <name>.npz in "
        "--embeddings-dir unless it should be skipped.",
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[1, 10],
        help="k values for kNN consistency.",
    )
    parser.add_argument(
        "--cluster-ks",
        nargs="+",
        type=int,
        default=[4, 8, 16, 32, 64],
        help="k values for k-means purity and NMI.",
    )
    parser.add_argument(
        "--few-shot-n",
        nargs="*",
        type=int,
        default=[],
        help="Seeds-per-class values for few-shot binary kNN.",
    )
    parser.add_argument(
        "--few-shot-trials",
        type=int,
        default=20,
        help="Trials for few-shot evaluation.",
    )
    parser.add_argument(
        "--few-shot-seed",
        type=int,
        default=0,
        help="Base RNG seed for few-shot evaluation.",
    )
    parser.add_argument(
        "--region-archive",
        default=DEFAULT_REGION_ARCHIVE,
        help="Glob for the grounded-balanced Florence/SigCLIP detection pkl "
        "shards, scanned by the Region BoF and Region Chamfer encoders.",
    )
    parser.add_argument(
        "--region-bof-vocab",
        type=int,
        default=128,
        help="Visual vocabulary size for the Region BoF histogram.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Deterministically subsample positives/negatives before loading.",
    )
    parser.add_argument("--debug-n-pos", type=int, default=20)
    parser.add_argument("--debug-n-neg", type=int, default=80)
    parser.add_argument("--debug-seed", type=int, default=0)
    parser.add_argument(
        "--no-spherical-kmeans",
        action="store_true",
        help="Use sklearn KMeans on normalized features instead of FAISS "
        "spherical k-means. Trajectory embeddings always use non-spherical.",
    )
    return parser.parse_args(argv)


def read_clip_ids(csv_path):
    """Return the clip_id column from a single-class label CSV."""
    with open(csv_path, newline="") as f:
        return [
            row["clip_id"]
            for row in csv.DictReader(f)
            if row.get("clip_id")
        ]


def read_labels_csv(csv_path):
    """Parse a multi-label CSV into clip_id -> labels and ordered labels."""
    labels_for = {}
    label_order = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            clip_id = row.get("clip_id")
            label = row.get("label")
            if not clip_id or not label:
                continue
            if label not in label_order:
                label_order.append(label)
            labels_for.setdefault(clip_id, [])
            if label not in labels_for[clip_id]:
                labels_for[clip_id].append(label)
    return labels_for, label_order


def label_indep_keys(cluster_ks):
    """Keys from cluster_metrics that do not depend on GT labels."""
    keys = set()
    for k in cluster_ks:
        keys.update({
            f"intra_sim_k{k}",
            f"inter_sim_k{k}",
            f"separation_k{k}",
            f"n_empty_clusters_k{k}",
        })
    return keys


def compute_one_vs_rest(X, binary_labels, ks, cluster_ks, spherical=True):
    """Run kNN consistency + cluster metrics for one binary label."""
    knn_m = knn_purity(X, binary_labels, k_values=ks)
    cluster_m = cluster_metrics(
        X,
        binary_labels,
        k_values=cluster_ks,
        spherical=spherical,
    )
    for dup in ("n_clips", "n_pos", "n_neg"):
        cluster_m.pop(dup, None)
    knn_m.update(cluster_m)
    return knn_m


def macro_average(per_label):
    """Unweighted mean of numeric per-label metrics, skipping NaNs."""
    if not per_label:
        return {}
    skip_keys = {"n_pos", "n_neg", "fewshot_skipped_ns", "fewshot_n_trials"}
    all_keys = set()
    for metrics in per_label.values():
        all_keys.update(metrics.keys())
    all_keys -= skip_keys

    out = {}
    for key in all_keys:
        vals = []
        for metrics in per_label.values():
            value = metrics.get(key)
            if not isinstance(value, (int, float)):
                continue
            value_f = float(value)
            if np.isnan(value_f):
                continue
            vals.append(value_f)
        if vals:
            out[key] = float(np.mean(vals))
    return out


def _read_labels(args):
    multi_label = args.labels_csv is not None
    if multi_label:
        labels_for_pos, label_names = read_labels_csv(args.labels_csv)
        pos_ids = list(labels_for_pos.keys())
        neg_ids = read_clip_ids(args.negative_csv)
        overlap = set(pos_ids) & set(neg_ids)
        if overlap:
            print(
                f"[run] WARNING: {len(overlap)} clip_ids appear in BOTH "
                "labels and negatives csv; dropping from negatives",
                flush=True,
            )
            neg_ids = [clip_id for clip_id in neg_ids if clip_id not in overlap]
        per_label_n = {
            label: sum(
                1 for clip_id in pos_ids if label in labels_for_pos[clip_id]
            )
            for label in label_names
        }
        print(
            f"[run] multi-label: {len(label_names)} labels = {label_names}",
            flush=True,
        )
        print(f"[run] per-label positives: {per_label_n}", flush=True)
        print(
            f"[run] {len(pos_ids)} distinct positives, {len(neg_ids)} negatives",
            flush=True,
        )
    else:
        labels_for_pos, label_names = {}, []
        pos_ids = read_clip_ids(args.positive_csv)
        neg_ids = read_clip_ids(args.negative_csv)
        overlap = set(pos_ids) & set(neg_ids)
        if overlap:
            print(
                f"[run] WARNING: {len(overlap)} clip_ids appear in BOTH csvs; "
                "dropping from negatives",
                flush=True,
            )
            neg_ids = [clip_id for clip_id in neg_ids if clip_id not in overlap]
        print(f"[run] {len(pos_ids)} positives, {len(neg_ids)} negatives", flush=True)
    return multi_label, labels_for_pos, label_names, pos_ids, neg_ids


def _debug_subsample(args, multi_label, labels_for_pos, label_names, pos_ids, neg_ids):
    rng = random.Random(args.debug_seed)
    if multi_label:
        keep_pos = set()
        for label in label_names:
            label_clips = [
                clip_id
                for clip_id in pos_ids
                if label in labels_for_pos[clip_id]
            ]
            n_keep = min(args.debug_n_pos, len(label_clips))
            keep_pos.update(rng.sample(label_clips, n_keep))
        pos_ids = [clip_id for clip_id in pos_ids if clip_id in keep_pos]
        labels_for_pos = {clip_id: labels_for_pos[clip_id] for clip_id in pos_ids}
    else:
        n_pos = min(args.debug_n_pos, len(pos_ids))
        pos_ids = rng.sample(pos_ids, n_pos)
    n_neg = min(args.debug_n_neg, len(neg_ids))
    neg_ids = rng.sample(neg_ids, n_neg)
    print(
        f"[run] --debug: subsampled to {len(pos_ids)} pos + "
        f"{len(neg_ids)} neg (seed={args.debug_seed})",
        flush=True,
    )
    return labels_for_pos, pos_ids, neg_ids


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (
        multi_label,
        labels_for_pos,
        label_names,
        pos_ids,
        neg_ids,
    ) = _read_labels(args)

    if args.debug:
        labels_for_pos, pos_ids, neg_ids = _debug_subsample(
            args,
            multi_label,
            labels_for_pos,
            label_names,
            pos_ids,
            neg_ids,
        )

    wanted = set(pos_ids) | set(neg_ids)
    label_for = {clip_id: 1 for clip_id in pos_ids}
    label_for.update({clip_id: 0 for clip_id in neg_ids})
    assert set(pos_ids).isdisjoint(neg_ids), (
        "pos/neg sets must be disjoint after overlap removal"
    )
    assert wanted.issubset(label_for), "every wanted clip must have a label"

    # Region encoders share one scan of the per-detection archive.
    region_scan = None
    if any(name in REGION_KEYS for name in args.embeddings):
        try:
            region_scan = scan_region_sets(args.region_archive, wanted)
        except FileNotFoundError as exc:
            print(
                f"[run] region archive missing ({exc}); skipping region "
                "encoders",
                flush=True,
            )

    loaded = {}
    for name in args.embeddings:
        print(f"\n=== Loading {name} ===", flush=True)
        if name in REGION_KEYS:
            if region_scan is None:
                print(f"[run] {name}: no region archive; skipping", flush=True)
                continue
            detections, full_frames = region_scan
            if name == REGION_BOF_KEY:
                ids, X = build_bof(
                    detections, full_frames, wanted,
                    n_vocab=args.region_bof_vocab,
                )
            else:  # REGION_CHAMFER_KEY: set-native, no per-clip vector
                ids, X = region_clip_ids(detections, full_frames, wanted), None
            if not ids:
                print(f"[run] {name}: 0 clips matched; skipping", flush=True)
                continue
            assert set(ids).issubset(wanted), (
                f"{name}: region scan returned clip_ids outside wanted"
            )
            loaded[name] = (ids, X)
            continue
        try:
            ids, X = load_embeddings(args.embeddings_dir, name, wanted)
        except FileNotFoundError as exc:
            print(f"[run] {name}: missing embeddings ({exc}); skipping", flush=True)
            continue
        if not ids:
            print(f"[run] {name}: 0 clips matched; skipping", flush=True)
            continue
        assert len(ids) == X.shape[0], (
            f"{name}: loader returned mismatched ids ({len(ids)}) and "
            f"X rows ({X.shape[0]})"
        )
        assert set(ids).issubset(wanted), (
            f"{name}: loader returned clip_ids outside the wanted set"
        )
        loaded[name] = (ids, X)

    if not loaded:
        print("[run] no embeddings loaded; exiting", file=sys.stderr)
        return 1

    print("\n=== Per-embedding coverage ===", flush=True)
    coverage = {}
    for name, (ids, _) in loaded.items():
        n_pos = sum(1 for clip_id in ids if label_for[clip_id] == 1)
        n_neg = sum(1 for clip_id in ids if label_for[clip_id] == 0)
        coverage[name] = {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_total": len(ids),
        }
        print(
            f"  {name}: {n_pos} pos + {n_neg} neg = {len(ids)} "
            f"({100 * len(ids) / len(wanted):.1f}% of wanted)",
            flush=True,
        )

    inter = set.intersection(*(set(ids) for ids, _ in loaded.values()))
    inter_pos = sum(1 for clip_id in inter if label_for[clip_id] == 1)
    inter_neg = sum(1 for clip_id in inter if label_for[clip_id] == 0)
    print(
        f"\n[run] intersection across {len(loaded)} embeddings: "
        f"{len(inter)} clips ({inter_pos} pos + {inter_neg} neg)",
        flush=True,
    )
    if inter_pos < 2 or inter_neg < 2 or len(inter) < max(args.ks) + 1:
        print("[run] intersection too small for kNN; exiting", file=sys.stderr)
        return 1

    inter_order = sorted(inter)
    inter_labels = np.array(
        [label_for[clip_id] for clip_id in inter_order],
        dtype=np.int8,
    )
    assert len(inter_order) == len(inter_labels) == len(inter)
    assert int(inter_labels.sum()) == inter_pos, (
        f"inter_labels positives ({int(inter_labels.sum())}) "
        f"don't match inter_pos ({inter_pos})"
    )

    summary = {
        "mode": "multi_label" if multi_label else "single_class",
        "positive_csv": None if multi_label else str(args.positive_csv),
        "labels_csv": str(args.labels_csv) if multi_label else None,
        "negative_csv": str(args.negative_csv),
        "embeddings_dir": str(args.embeddings_dir),
        "label_names": label_names if multi_label else None,
        "n_positives_input": len(pos_ids),
        "n_negatives_input": len(neg_ids),
        "intersection_n_total": len(inter),
        "intersection_n_pos": inter_pos,
        "intersection_n_neg": inter_neg,
        "ks": args.ks,
        "cluster_ks": args.cluster_ks,
        "few_shot_n": list(args.few_shot_n),
        "few_shot_trials": args.few_shot_trials,
        "few_shot_seed": args.few_shot_seed,
        "coverage_per_embedding": coverage,
        "metrics_per_embedding": {},
    }
    li_keys = label_indep_keys(args.cluster_ks)

    for name, (ids, X) in loaded.items():
        print(f"\n=== metrics: {name} ===", flush=True)

        if name == REGION_CHAMFER_KEY:
            t0 = time.time()
            detections, full_frames = region_scan
            metrics = chamfer_metrics(
                detections,
                full_frames,
                inter_order,
                multi_label=multi_label,
                labels_for_pos=labels_for_pos,
                label_names=label_names,
                inter_labels=inter_labels,
                ks=args.ks,
                cluster_ks=args.cluster_ks,
                few_shot_n=list(args.few_shot_n),
                few_shot_trials=args.few_shot_trials,
                few_shot_seed=args.few_shot_seed,
            )
            if multi_label:
                metrics["macro_avg"] = macro_average(metrics["per_label"])
            metrics["embedding"] = name
            metrics["runtime_s"] = round(time.time() - t0, 2)
            emb_dir = args.output_dir / name
            emb_dir.mkdir(parents=True, exist_ok=True)
            (emb_dir / "numbers.json").write_text(json.dumps(metrics, indent=2))
            scalar = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            print(json.dumps(scalar, indent=2), flush=True)
            summary["metrics_per_embedding"][name] = metrics
            continue

        id_to_row = {clip_id: i for i, clip_id in enumerate(ids)}
        Xi = X[[id_to_row[clip_id] for clip_id in inter_order]]
        assert Xi.shape[0] == len(inter_order), (
            f"{name}: re-indexed Xi has {Xi.shape[0]} rows, "
            f"expected {len(inter_order)}"
        )

        use_spherical = not args.no_spherical_kmeans and name != "trajectory"
        t0 = time.time()
        if multi_label:
            per_label = {}
            label_indep = {}
            for label in label_names:
                label_vec = np.array(
                    [
                        1 if (
                            clip_id in labels_for_pos
                            and label in labels_for_pos[clip_id]
                        )
                        else 0
                        for clip_id in inter_order
                    ],
                    dtype=np.int8,
                )
                n_pos_label = int(label_vec.sum())
                n_neg_label = int(len(label_vec) - n_pos_label)
                if n_pos_label < 2 or n_neg_label < 2:
                    print(
                        f"[run] {name}/{label}: too few pos/neg "
                        f"({n_pos_label}/{n_neg_label}); skipping",
                        flush=True,
                    )
                    continue

                per = compute_one_vs_rest(
                    Xi,
                    label_vec,
                    ks=args.ks,
                    cluster_ks=args.cluster_ks,
                    spherical=use_spherical,
                )
                if not label_indep:
                    label_indep = {k: per[k] for k in li_keys if k in per}
                per_dep = {
                    k: v
                    for k, v in per.items()
                    if k not in li_keys and k not in ("n_clips", "n_pos", "n_neg")
                }
                per_dep["n_pos"] = n_pos_label
                per_dep["n_neg"] = n_neg_label
                if args.few_shot_n:
                    fs = few_shot_binary_knn(
                        Xi,
                        label_vec,
                        n_values=list(args.few_shot_n),
                        S=args.few_shot_trials,
                        seed=args.few_shot_seed,
                    )
                    per_dep.update(fs)
                per_label[label] = per_dep

            metrics = {
                "embedding": name,
                "n_dims": int(Xi.shape[1]),
                "n_clips": int(Xi.shape[0]),
                **label_indep,
                "per_label": per_label,
                "macro_avg": macro_average(per_label),
            }
        else:
            metrics = knn_purity(Xi, inter_labels, k_values=args.ks)
            cluster = cluster_metrics(
                Xi,
                inter_labels,
                k_values=args.cluster_ks,
                spherical=use_spherical,
            )
            for dup in ("n_clips", "n_pos", "n_neg"):
                cluster.pop(dup, None)
            metrics.update(cluster)
            if args.few_shot_n:
                fs = few_shot_binary_knn(
                    Xi,
                    inter_labels,
                    n_values=list(args.few_shot_n),
                    S=args.few_shot_trials,
                    seed=args.few_shot_seed,
                )
                metrics.update(fs)
            metrics["embedding"] = name
            metrics["n_dims"] = int(Xi.shape[1])

        metrics["runtime_s"] = round(time.time() - t0, 2)
        emb_dir = args.output_dir / name
        emb_dir.mkdir(parents=True, exist_ok=True)
        (emb_dir / "numbers.json").write_text(json.dumps(metrics, indent=2))
        scalar = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        print(json.dumps(scalar, indent=2), flush=True)
        summary["metrics_per_embedding"][name] = metrics

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[run] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
