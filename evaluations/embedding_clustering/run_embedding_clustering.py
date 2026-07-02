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

"""One runner for the embedding-space clustering analysis.

Orchestrates the clustering experiments end to end and writes their result
files plus a single ``summary.json``, the clustering-side counterpart to
``embedding_quality/run_embedding_quality.py``. Each stage is the existing
stage script called in-process (``main(argv)``); this runner only sequences
them, resolves paths, and records what ran. Supersedes ``run_full_cluster.sh``.

Stages (select with ``--stages``; default all that have their inputs):
  flat        per-encoder spherical k-means + TF-IDF topics   (cluster_raw)
  hierarchical recursive taxonomy + per-level topics           (cluster_raw --hierarchical)
  preindex    exact vs PQ-reconstructed agreement table        (preindex_compare)
  pc-ablation caption common-component removal + PC topics      (caption_pc_ablation, pc_topics)
  figures     rendered tables / UMAP overviews                 (make_figures)

Inputs are resolved from flags or the matching env vars used by the old shell
driver (WHEEL_DATA_DIR / CAPTIONS_DB / POOLS_DIR / CLUSTER_OUT). A stage whose
inputs are missing is skipped with a message rather than failing the run.

  python run_embedding_clustering.py \
      --npz-dir ./embeddings --pool ./pools/pai_clip_ids.json --pool-name pai \
      --captions-db ./captions.db --output-dir ./out --k 1000
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Encoders that need mean-centering (anisotropic; e.g. SigLIP visual crops).
CENTER_EMBEDS = {"visual"}
ALL_STAGES = ("flat", "hierarchical", "preindex", "pc-ablation", "figures")


def _env_path(name: str) -> str | None:
    v = os.environ.get(name)
    return v or None


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz-dir", type=Path, default=_env_path("WHEEL_DATA_DIR"),
                    help="dir with <encoder>.npz (cosmos/caption/visual).")
    ap.add_argument("--embeds", nargs="+", default=["cosmos", "caption", "visual"])
    ap.add_argument("--pool", type=Path, default=None,
                    help="JSON list of clip_ids to restrict every encoder to.")
    ap.add_argument("--pool-name", default="pai")
    ap.add_argument("--captions-db", type=Path, default=_env_path("CAPTIONS_DB"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path(_env_path("CLUSTER_OUT") or "./cluster_out"))
    ap.add_argument("--k", type=int, default=1000, help="flat cluster count.")
    ap.add_argument("--branching", type=int, default=10)
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--remove-pcs", type=int, default=5)
    ap.add_argument("--index-specs", nargs="*", default=[],
                    help="[preindex] one FAISS index_factory spec per encoder to "
                    "compare exact vs PQ; omit to skip the preindex stage.")
    ap.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=list(ALL_STAGES))
    return ap.parse_args(argv)


def _npz(npz_dir: Path, embed: str) -> Path:
    return npz_dir / f"{embed}.npz"


def _run_stage(fn, name: str, summary: dict):
    """Run one stage, timing it and recording status; never abort the whole run."""
    t0 = time.perf_counter()
    try:
        result = fn()
        status = "ok" if result is not False else "skipped"
    except SystemExit as e:  # a stage main() that arg-errored
        status = f"skipped ({e})"
        result = None
    except Exception as e:  # pragma: no cover - stage-specific runtime failure
        status = f"error ({type(e).__name__}: {e})"
        result = None
    dt = round(time.perf_counter() - t0, 1)
    summary["stages"][name] = {"status": status, "seconds": dt}
    print(f"[run] stage {name}: {status} ({dt}s)", flush=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"pool_name": args.pool_name, "k": args.k,
                     "embeds": args.embeds, "stages": {}}

    have_npz = args.npz_dir is not None
    pool_args = (["--pool", str(args.pool)] if args.pool else [])
    db_args = (["--captions-db", str(args.captions_db)] if args.captions_db else [])

    # --- flat clustering + topics ---
    if "flat" in args.stages:
        def flat():
            if not (have_npz and args.captions_db):
                print("[run] flat: need --npz-dir and --captions-db; skipping", flush=True)
                return False
            import cluster_raw
            for e in args.embeds:
                npz = _npz(args.npz_dir, e)
                if not npz.exists():
                    print(f"[run] flat/{e}: {npz} missing; skipping", flush=True)
                    continue
                argv_e = ["--npz", str(npz), "--embed", e, "--k", str(args.k),
                          "--spherical", "--seed", str(args.seed),
                          "--clustering-dir", str(out / "clustering"),
                          "--run-id", f"k{args.k}_{e}", "--pool-name", args.pool_name,
                          "--runs-tsv", str(out / "runs.tsv"), *pool_args, *db_args]
                if e in CENTER_EMBEDS:
                    argv_e.append("--center")
                cluster_raw.main(argv_e)
        _run_stage(flat, "flat", summary)

    # --- hierarchical taxonomy ---
    if "hierarchical" in args.stages:
        def hier():
            if not (have_npz and args.captions_db):
                print("[run] hierarchical: need --npz-dir and --captions-db; skipping", flush=True)
                return False
            import cluster_raw
            for e in args.embeds:
                npz = _npz(args.npz_dir, e)
                if not npz.exists():
                    continue
                argv_e = ["--npz", str(npz), "--embed", e, "--hierarchical",
                          "--branching", str(args.branching),
                          "--max-depth", str(args.max_depth),
                          "--out", str(out / "hier" / f"{args.pool_name}_{e}"), *db_args]
                if e in CENTER_EMBEDS:
                    argv_e.append("--center")
                cluster_raw.main(argv_e)
        _run_stage(hier, "hierarchical", summary)

    # --- pre-index vs after-index comparison ---
    if "preindex" in args.stages:
        def preindex():
            if not (have_npz and args.index_specs):
                print("[run] preindex: need --npz-dir and --index-specs; skipping", flush=True)
                return False
            import preindex_compare
            for e, spec in zip(args.embeds, args.index_specs):
                npz = _npz(args.npz_dir, e)
                if not npz.exists():
                    continue
                # proxy path: self-quantize the raw vectors with the given spec
                argv_e = ["--raw-npz", str(npz), "--index-spec", spec, "--embed", e,
                          "--k", str(args.k), "--seed", str(args.seed),
                          "--out", str(out / f"preindex_{e}.json")]
                if e in CENTER_EMBEDS:
                    argv_e.append("--center")
                preindex_compare.main(argv_e)
        _run_stage(preindex, "preindex", summary)

    # --- caption common-component ablation + PC topics ---
    if "pc-ablation" in args.stages:
        def pc_ablation():
            cap = _npz(args.npz_dir, "caption") if have_npz else None
            if not (cap and cap.exists()):
                print("[run] pc-ablation: need caption.npz; skipping", flush=True)
                return False
            import caption_pc_ablation
            caption_pc_ablation.main(["--caption-npz", str(cap),
                                      "--out-dir", str(args.npz_dir),
                                      "--remove-pcs", str(args.remove_pcs)])
        _run_stage(pc_ablation, "pc-ablation", summary)

    # --- figures / tables ---
    if "figures" in args.stages:
        def figures():
            if not (out / "clustering").exists():
                print("[run] figures: no clustering runs yet; skipping", flush=True)
                return False
            import make_figures
            make_figures.main(["tables", "--clustering-dir", str(out / "clustering"),
                               "--what", "both",
                               "--topics-out", str(out / "figures" / "topics.tex"),
                               "--distinctive-out", str(out / "figures" / "distinctive.tex")])
        _run_stage(figures, "figures", summary)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[run] wrote {out / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
