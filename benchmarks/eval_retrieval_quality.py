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

"""
Evaluate retrieval quality of search modules against Alpamayo manual annotations.

For each label and module, computes Average Precision (AP), R-Precision, P@K,
R@K, and full P-R curves. Combinations of modules are fused via Reciprocal
Rank Fusion (RRF).

Primary metrics:
  AP          — area under the P-R curve; threshold-free summary of ranking quality
  R-Precision — P@K where K=|GT|; size-normalised, directly comparable across labels

Ground truth: clips with label_type='manual' under the Alpamayo project.
Trajectory search: shape-based using the first ground-truth clip found in the
trajectory index as the reference clip.

Usage (run from repo root):
    python -m benchmarks.eval_retrieval_quality \\
        config/wheel_launch_dev_server_config.yaml results.parquet plots/ results.md \\
        [--min-count 50] [--ks 100 500 1000]
"""
import argparse
import fnmatch
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks import write_markdown
from sil_wheel.stores.caption_embeddings_store import CaptionEmbeddingsStore
from sil_wheel.stores.cosmos_embeddings_store import CosmosEmbeddingsStore
from sil_wheel.stores.sqlite_caption_store import FTSCaptionStore
from sil_wheel.stores.trajectory_store import TrajectoryStore
from sil_wheel.stores.visual_embeddings_store import CLIPEmbeddingStore

PROJECT = "Alpamayo"
LABEL_TYPE = "manual"
RRF_K = 60
DENYLIST = ["vlm_distill_*", "reason_*", "distill_*", "scenario_*"]
MAX_K = 100_000


def load_ground_truth(db_path, project, min_count, denylist=None):
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        "SELECT key, clip_id FROM annotations WHERE project=? AND label_type=?",
        (project, LABEL_TYPE),
    )
    gt = defaultdict(set)
    for row in cur:
        gt[row[0]].add(row[1])
    conn.close()
    result = {k: v for k, v in gt.items() if len(v) >= min_count}
    if denylist:
        result = {
            k: v for k, v in result.items()
            if not any(fnmatch.fnmatch(k, pat) for pat in denylist)
        }
    return result



def rrf(ranked_lists, k=RRF_K):
    """Reciprocal Rank Fusion over a list of ranked [(clip_id, score)] lists."""
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (clip_id, _) in enumerate(ranked, start=1):
            scores[clip_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def average_precision(ranked, ground_truth):
    """AP = (1/|GT|) * sum of precision at each rank where a relevant item appears."""
    if not ground_truth:
        return 0.0
    n = len(ground_truth)
    tp = 0
    ap = 0.0
    for i, (clip_id, _) in enumerate(ranked):
        if clip_id in ground_truth:
            tp += 1
            ap += tp / (i + 1)
    return ap / n


def r_precision(ranked, ground_truth):
    """Precision at rank K=|GT|. Size-normalised and directly comparable across labels."""
    if not ground_truth:
        return 0.0
    k = len(ground_truth)
    top_k = {cid for cid, _ in ranked[:k]}
    return len(top_k & ground_truth) / k


def precision_at_k(ranked, ground_truth, k):
    top_k = {cid for cid, _ in ranked[:k]}
    return len(top_k & ground_truth) / k if k > 0 else 0.0


def recall_at_k(ranked, ground_truth, k):
    if not ground_truth:
        return 0.0
    top_k = {cid for cid, _ in ranked[:k]}
    return len(top_k & ground_truth) / len(ground_truth)


def pr_curve(ranked, ground_truth):
    """Returns (recalls, precisions) as numpy arrays, starting at (0, 1)."""
    recalls, precisions = [0.0], [1.0]
    tp = 0
    n = len(ground_truth)
    for i, (clip_id, _) in enumerate(ranked):
        if clip_id in ground_truth:
            tp += 1
        precisions.append(tp / (i + 1))
        recalls.append(tp / n)
    return np.array(recalls), np.array(precisions)


def _run_module(search_fn, gt):
    """Rank all labels with search_fn; return {label: ranked_list}."""
    ranked = {}
    for label, gt_clips in gt.items():
        t0 = time.time()
        ranked[label] = search_fn(label, gt_clips)
        print(f"    [{label}] {len(ranked[label])} results ({time.time() - t0:.1f}s)", flush=True)
    return ranked


def run_random(cfg, gt, corpus):
    def search(_, __):
        shuffled = corpus.copy()
        np.random.shuffle(shuffled)
        return [(cid, 1.0) for cid in shuffled]

    return _run_module(search, gt)


def run_semantic(cfg, gt, corpus):
    ds = cfg["datastores"]
    store = CosmosEmbeddingsStore(
        ds["cosmos_embed_store"]["embeddings_dir"],
        index_spec=ds["cosmos_embed_store"]["index_spec"],
    )
    params = store._make_selector_params(corpus)
    return _run_module(
        lambda q, _: sorted(store.search_with_text(q, k=MAX_K, params=params), key=lambda x: -x[1]),
        gt,
    )


def run_caption_embed(cfg, gt, corpus):
    ds = cfg["datastores"]
    store = CaptionEmbeddingsStore(
        ds["caption_embed_store"]["embeddings_dir"],
        index_spec=ds["caption_embed_store"].get("index_spec", "IVF4096,PQ128x8"),
    )
    params = store._make_selector_params(set(corpus))
    return _run_module(
        lambda q, _: sorted(store.search_with_text(q, k=MAX_K, params=params), key=lambda x: -x[1]),
        gt,
    )


def run_clip(cfg, gt, corpus):
    ds = cfg["datastores"]
    store = CLIPEmbeddingStore(ds["clip_embed_store"]["embeddings_dir"])
    params = store._make_selector_params(set(corpus))
    return _run_module(
        lambda q, _: sorted(store.search_with_text(q, k=MAX_K, params=params), key=lambda x: -x[1]),
        gt,
    )


def run_caption_fts(cfg, gt, corpus):
    ds = cfg["datastores"]
    corpus_set = set(corpus)
    fts = FTSCaptionStore(ds["captions_db"])

    def search(query, _):
        try:
            clip_ids = fts._inner_search(query)
        except sqlite3.OperationalError:
            clip_ids = []
        clip_ids = [cid for cid in clip_ids if cid in corpus_set]
        return [(cid, 1.0 / (i + 1)) for i, cid in enumerate(clip_ids)]

    return _run_module(search, gt)


def run_trajectory(cfg, gt, corpus):
    ds = cfg["datastores"]
    traj = TrajectoryStore(ds["trajectory_store"]["trajectory_dir"], debug=False)
    _, traj_index, _ = traj.get_feature_params_index("full")
    params = traj._make_selector_params(corpus, tag="full")

    # For each label, find the first annotated clip (by DB insertion order) that
    # has trajectory data — this is the reference used for shape-based search.
    conn = sqlite3.connect(ds["annotations_db"])
    label_ref = {}
    for label in gt:
        cur = conn.execute(
            "SELECT clip_id FROM annotations "
            "WHERE project=? AND label_type=? AND key=? ORDER BY rowid",
            (PROJECT, LABEL_TYPE, label),
        )
        for (clip_id,) in cur:
            if clip_id in traj_index:
                label_ref[label] = clip_id
                break
    conn.close()

    def search(label, _):
        ref = label_ref.get(label)
        if ref is None:
            return []
        results = traj.search_with_video_clip(ref, None, None, params=params)
        return sorted(
            ((cid, -dist) for cid, dist in results if cid != ref),
            key=lambda x: -x[1],
        )

    return _run_module(search, gt)


BASE_MODULES = {
    "random": run_random,
    "semantic": run_semantic,
    "caption_embed": run_caption_embed,
    "clip": run_clip,
    "caption_fts": run_caption_fts,
    "trajectory": run_trajectory,
}


def score_ranked(ranked_per_label, gt, ks):
    """Compute metrics from pre-ranked lists. Returns {label: entry}."""
    results = {}
    for label, gt_clips in gt.items():
        ranked = ranked_per_label.get(label, [])
        ap = average_precision(ranked, gt_clips)
        entry = {
            "gt_size": len(gt_clips),
            "ap": ap,
            "r_precision": r_precision(ranked, gt_clips),
            "pr_curve": pr_curve(ranked, gt_clips),
        }
        for k in ks:
            entry[f"p@{k}"] = precision_at_k(ranked, gt_clips, k)
            entry[f"r@{k}"] = recall_at_k(ranked, gt_clips, k)
        results[label] = entry
    mAP = np.mean([e["ap"] for e in results.values()])
    print(f"    mAP={mAP:.4f}", flush=True)
    return results


def _fmt_table(results, modules, metric):
    """Return (headers, rows) for a wide table: rows=labels, cols=modules."""
    headers = ["label", "gt_size"] + list(modules)
    rows = []
    for label, mod_results in sorted(results.items()):
        gt_size = mod_results[list(mod_results)[0]]["gt_size"]
        row = [label, str(gt_size)] + [
            f"{mod_results[mod][metric]:.3f}" for mod in modules
        ]
        rows.append(tuple(row))
    return headers, rows


def print_results(results, modules, ks):
    for metric in ["ap", "r_precision"] + [f"p@{k}" for k in ks] + [f"r@{k}" for k in ks]:
        headers, rows = _fmt_table(results, modules, metric)
        col_w = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

        def fmt(cells):
            return " | ".join(c.ljust(w) for c, w in zip(cells, col_w))

        print(f"\n=== {metric.upper()} ===")
        print(fmt(headers))
        print("-" * (sum(col_w) + 3 * (len(col_w) - 1)))
        for row in rows:
            print(fmt(row))


def save_parquet(results, modules, ks, output_path):
    rows = []
    for label, mod_results in sorted(results.items()):
        for mod in modules:
            entry = mod_results[mod]
            row = {
                "label": label,
                "module": mod,
                "gt_size": entry["gt_size"],
                "ap": entry["ap"],
                "r_precision": entry["r_precision"],
            }
            for k in ks:
                row[f"p@{k}"] = entry[f"p@{k}"]
                row[f"r@{k}"] = entry[f"r@{k}"]
            rows.append(row)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    print(f"Results written to {output_path}")


def save_markdown(results, modules, ks, output_path, project, min_count):
    first = True
    for metric in ["ap", "r_precision"] + [f"p@{k}" for k in ks] + [f"r@{k}" for k in ks]:
        headers, rows = _fmt_table(results, modules, metric)
        write_markdown(
            output_path,
            f"Retrieval Quality — {metric.upper()}",
            headers,
            rows,
            metadata={"project": project, "min_count": str(min_count), "metric": metric},
            append=not first,
        )
        first = False


def save_plots(results, modules, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, mod_results in results.items():
        safe = label.replace(" ", "_").replace("/", "-")
        for mod in modules:
            recalls, precisions = mod_results[mod]["pr_curve"]
            np.savez(
                output_dir / f"{safe}_{mod}.npz",
                recalls=recalls,
                precisions=precisions,
            )
        fig, ax = plt.subplots(figsize=(8, 6))
        for mod in modules:
            recalls, precisions = mod_results[mod]["pr_curve"]
            ax.plot(recalls, precisions, label=mod, linewidth=1.2)
        gt_size = mod_results[list(mod_results)[0]]["gt_size"]
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{label}  (gt={gt_size})")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.savefig(output_dir / f"{safe}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {len(results)} P-R plots and numpy curves to {output_dir}")


def save_barplots(results, modules, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = set(BASE_MODULES.keys())
    colors = ["#4878d0" if m in base else "#ee854a" for m in modules]

    # Summary: mean AP per module, sorted
    mean_ap = {m: np.mean([results[label][m]["ap"] for label in results]) for m in modules}
    order = sorted(modules, key=lambda m: mean_ap[m])
    fig, ax = plt.subplots(figsize=(8, max(4, len(modules) * 0.45)))
    bars = ax.barh(order, [mean_ap[m] for m in order],
                   color=["#4878d0" if m in base else "#ee854a" for m in order])
    for bar, m in zip(bars, order):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{mean_ap[m]:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Mean AP")
    ax.set_xlim(0, 1)
    ax.set_title("Mean Average Precision per module")
    fig.tight_layout()
    fig.savefig(output_dir / "summary_map.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Per-label: horizontal barplot of AP per module
    for label, mod_results in results.items():
        safe = label.replace(" ", "_").replace("/", "-")
        gt_size = mod_results[list(mod_results)[0]]["gt_size"]
        aps = [mod_results[m]["ap"] for m in modules]
        fig, ax = plt.subplots(figsize=(8, max(4, len(modules) * 0.45)))
        ax.barh(modules, aps, color=colors)
        ax.set_xlabel("AP")
        ax.set_xlim(0, 1)
        ax.set_title(f"{label}  (gt={gt_size})")
        fig.tight_layout()
        fig.savefig(output_dir / f"{safe}_ap_bar.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved barplots to {output_dir}")


def main():
    parser = argparse.ArgumentParser("Evaluate retrieval APIs and their combination")
    parser.add_argument("config", help="YAML server config (e.g. config/wheel_launch_dev_server_config.yaml)")
    parser.add_argument("output", help="Write Parquet scores to this file")
    parser.add_argument("output_dir", help="Directory for P-R curve plots and numpy files")
    parser.add_argument("output_md", help="Write markdown tables to this file")
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument(
        "--min-count", type=int, default=50,
        help="Skip labels with fewer than this many annotated clips (default: 50)",
    )
    parser.add_argument(
        "--denylist", nargs="*", default=DENYLIST,
        help="Glob patterns for labels to exclude (default: vlm_distill_* reason_* distill_* scenario_*)",
    )
    parser.add_argument(
        "--ks", type=int, nargs="+", default=[10, 50, 100],
        help="Evaluation cutoffs for P@K / R@K (default: 10 50 100)",
    )
    parser.add_argument(
        "--no-combos", action="store_true",
        help="Skip RRF combination modules, run base modules only",
    )
    parser.add_argument(
        "--noise", type=int, default=2000,
        help="Number of random non-GT clips to add as distractors (default: 2000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for distractor sampling (default: 42)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("Loading ground truth...")
    gt = load_ground_truth(
        cfg["datastores"]["annotations_db"], args.project, args.min_count, args.denylist
    )
    print(f"  {len(gt)} labels with >= {args.min_count} clips (after denylist)")

    gt_clips = {cid for clips in gt.values() for cid in clips}
    db_path = cfg["datastores"]["annotations_db"]
    conn = sqlite3.connect(db_path)
    all_clip_ids = [r[0] for r in conn.execute("SELECT DISTINCT clip_id FROM annotations")]
    conn.close()
    non_gt = [c for c in all_clip_ids if c not in gt_clips]
    rng = np.random.default_rng(args.seed)
    noise = rng.choice(non_gt, size=min(args.noise, len(non_gt)), replace=False).tolist()
    corpus = list(gt_clips) + noise
    print(f"  Corpus: {len(gt_clips):,} GT clips + {len(noise):,} distractors = {len(corpus):,} total")

    all_results = {label: {} for label in gt}
    base_ranked = {}

    for name, run_fn in BASE_MODULES.items():
        print(f"\nRunning {name}...")
        ranked = run_fn(cfg, gt, corpus)
        base_ranked[name] = ranked
        for label, entry in score_ranked(ranked, gt, args.ks).items():
            all_results[label][name] = entry

    COMBOS = [
        ("semantic", "caption_fts"),
        ("semantic", "caption_embed"),
        ("semantic", "clip"),
        ("semantic", "trajectory"),
        ("caption_fts", "trajectory"),
        ("semantic", "caption_embed", "caption_fts"),
        ("semantic", "clip", "caption_fts"),
        ("semantic", "caption_embed", "clip", "caption_fts", "trajectory"),
    ]

    if not args.no_combos:
        for combo in COMBOS:
            name = "+".join(combo)
            print(f"\nRunning {name}...")
            ranked = {
                label: rrf([base_ranked[p][label] for p in combo])
                for label in gt
            }
            for label, entry in score_ranked(ranked, gt, args.ks).items():
                all_results[label][name] = entry

    all_modules = list(next(iter(all_results.values())).keys())
    print_results(all_results, all_modules, args.ks)
    save_parquet(all_results, all_modules, args.ks, args.output)
    save_markdown(all_results, all_modules, args.ks, args.output_md, args.project, args.min_count)
    save_plots(all_results, all_modules, args.output_dir)
    save_barplots(all_results, all_modules, args.output_dir)


if __name__ == "__main__":
    main()
