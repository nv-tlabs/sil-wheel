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

"""Input-prep utilities for the embedding-space analysis. Subcommands:

* ``pool``      -- clip-id pools present in all three embedding indices.
* ``fig-runs``  -- fig_runs.json (input to make_figures umap-overview) from runs.tsv.
* ``synthetic`` -- a tiny synthetic wheel-data set to smoke-test the workflow.

    python prep.py pool --wheel-data-dir /path/to/wheel-data --out ./emb_pools
    python prep.py fig-runs --runs ./emb_pools/runs.tsv --clustering-dir ./clustering --out fig_runs.json
    python prep.py synthetic --out ./synth_starter
"""

import argparse
import json
import pickle
import sqlite3
from pathlib import Path

import faiss
import numpy as np


# pool -- clip-id pools in the cosmos∩visual∩caption intersection


def _uuids_from_path_file(path):
    out = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(line.split("/")[-1].split(".")[0])
    return out


def _load_cosmos_keys(cti_pkl):
    with open(cti_pkl, "rb") as f:
        return set(pickle.load(f).keys())


def _load_visual_keys(ids_npy):
    return {str(x) for x in np.load(ids_npy, allow_pickle=True)}


def _load_caption_itc(itc_pkl):
    """Legacy index->clip map ({row_index: clip_id}) for the full caption index."""
    with open(itc_pkl, "rb") as f:
        return pickle.load(f)


def _build_caption_cti(itc, keep, out_dir, caption_index, caption_tag):
    """Invert the legacy index->clip map into a clip_id->row map restricted to
    ``keep`` and persist it where ``embed_io.load_clip_to_index`` will find it
    (a writable caption dir). First (smallest) row wins per clip, matching wheel."""
    cti = {}
    for row in sorted(itc):
        cid = str(itc[row])
        if cid in keep and cid not in cti:
            cti[cid] = int(row)
    cap_dir = out_dir / "caption_embeddings"
    cap_dir.mkdir(parents=True, exist_ok=True)
    cti_path = cap_dir / f"caption_clip_to_index_{caption_tag}.pkl"
    with open(cti_path, "wb") as f:
        pickle.dump(cti, f)
    link = cap_dir / f"caption_embeddings_{caption_tag}.index"
    if not link.exists() and caption_index.exists():
        link.symlink_to(caption_index)
    print(f"  wrote caption cti {len(cti):,} entries -> {cti_path}", flush=True)
    return len(cti)


def cmd_pool(args):
    wd = args.wheel_data_dir
    cosmos_cti = wd / f"cosmos_clip_to_index_{args.cosmos_tag}.pkl"
    visual_ids = wd / "visual_embeddings" / f"visual_clip_ids_{args.visual_tag}.npy"
    caption_itc = (
        wd / "caption_embeddings" / f"caption_index_to_clip_{args.caption_tag}.pkl"
    )
    caption_index = (
        wd / "caption_embeddings" / f"caption_embeddings_{args.caption_tag}.index"
    )

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print("loading visual keys...", flush=True)
    common = _load_visual_keys(visual_ids)
    print(f"  visual: {len(common):,}", flush=True)
    print("loading cosmos keys + intersecting...", flush=True)
    common &= _load_cosmos_keys(cosmos_cti)
    print(f"  cosmos: {len(common):,}", flush=True)
    print("loading caption itc + intersecting...", flush=True)
    cap_itc = _load_caption_itc(caption_itc)
    common &= {str(v) for v in cap_itc.values()}
    print(f"  caption (full common set): {len(common):,}", flush=True)

    full = sorted(common)
    (out / "full_clip_ids.json").write_text(json.dumps(full))
    print(f"full: {len(full):,} clips (entire common set)", flush=True)

    n_cti = _build_caption_cti(cap_itc, common, out, caption_index, args.caption_tag)
    del cap_itc

    n_pai = 0
    if args.pai_path_files:
        pai = set()
        for pf in args.pai_path_files:
            pai |= _uuids_from_path_file(pf)
        pai_common = sorted(pai & common)
        (out / "pai_clip_ids.json").write_text(json.dumps(pai_common))
        n_pai = len(pai_common)
        print(f"pai: {len(pai):,} uuids -> {n_pai:,} in common set", flush=True)

    rng = np.random.default_rng(args.seed)
    common_arr = np.array(sorted(common), dtype=object)
    n_large = min(args.n_large, len(common_arr))
    sel = rng.choice(len(common_arr), n_large, replace=False)
    large = sorted(common_arr[sel].tolist())
    (out / "large_clip_ids.json").write_text(json.dumps(large))
    print(f"large: sampled {len(large):,} of {len(common_arr):,}", flush=True)

    summary = {
        "common_set": len(common),
        "full": len(full),
        "caption_cti": n_cti,
        "pai": n_pai,
        "large": len(large),
        "n_large_requested": args.n_large,
        "seed": args.seed,
    }
    (out / "pool_summary.json").write_text(json.dumps(summary, indent=2))
    print("summary:", summary, flush=True)
    return 0


# fig-runs -- fig_runs.json from runs.tsv

EMBEDS = [
    {"key": "cosmos", "label": "Cosmos-Embed1", "color": "#4C78A8"},
    {"key": "caption", "label": "Caption (Qwen3-Emb-8B)", "color": "#F58518"},
    {"key": "visual", "label": "Florence-2/SigLIP", "color": "#54A24B"},
]


def cmd_fig_runs(args):
    latest = {}
    pool_order = []
    for line in args.runs.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) != 6 or not parts[5].startswith("DONE_rc=0"):
            continue
        _, pool, emb, rid, _out, _status = parts
        latest[(pool, emb)] = rid
        if pool not in pool_order:
            pool_order.append(pool)

    pools_wanted = args.pools or pool_order
    emb_keys = [e["key"] for e in EMBEDS]

    pools = []
    for pool in pools_wanted:
        runs = {e: latest[(pool, e)] for e in emb_keys if (pool, e) in latest}
        if len(runs) < len(emb_keys):
            missing = [e for e in emb_keys if e not in runs]
            print(
                f"[warn] pool {pool!r} missing successful runs for {missing}; skipping",
                flush=True,
            )
            continue
        n = 0
        meta = args.clustering_dir / next(iter(runs.values())) / "metadata.json"
        if meta.exists():
            n = int(json.loads(meta.read_text()).get("n_input_clips", 0))
        pools.append({"label": pool.capitalize(), "n": n, "runs": runs})

    if not pools:
        raise SystemExit(
            "no complete pools found in runs.tsv (need cosmos+caption+visual each)"
        )

    args.out.write_text(json.dumps({"embeds": EMBEDS, "pools": pools}, indent=2))
    print(
        f"wrote {args.out} with pools: "
        + ", ".join(f"{p['label']}({p['n']})" for p in pools)
    )
    return 0


# tiny wheel-data set to smoke-test the workflow

# name -> (dim, filename tag, faiss build spec); small nlist for a tiny set.
EMB = {
    "cosmos": (384, "ivf4096_pq96x8", "IVF16,PQ96x8"),
    "caption": (512, "ivf4096_pq256x8", "IVF16,PQ256x8"),
    "visual": (384, "ivf4096_pq64x8", "IVF16,PQ64x8"),
}
# cosmos cleanest, visual noisiest -> three visibly different geometries.
NOISE = {"cosmos": 0.5, "caption": 0.8, "visual": 1.2}
THEME_WORDS = {
    "highway merge": ["highway", "merge", "ramp", "lane", "accelerate", "overtaking"],
    "pedestrian crossing": [
        "pedestrian",
        "crosswalk",
        "yield",
        "walking",
        "sidewalk",
        "halt",
    ],
    "roundabout": ["roundabout", "circular", "yield", "exit", "navigate", "island"],
    "parking": ["parking", "reverse", "stall", "maneuver", "garage", "space"],
    "traffic light": ["intersection", "red", "green", "signal", "queue", "wait"],
    "rain night": ["rain", "night", "wet", "headlights", "reflection", "dark"],
    "construction": ["cones", "construction", "closed", "worker", "detour", "barrier"],
    "cyclist": ["cyclist", "bike", "overtake", "share", "shoulder", "helmet"],
}


def _normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def cmd_synthetic(args):
    rng = np.random.default_rng(args.seed)
    themes = list(THEME_WORDS)[: max(2, min(args.themes, len(THEME_WORDS)))]
    clip_ids = [f"clip_{i:06d}" for i in range(args.n)]
    assign = rng.integers(0, len(themes), size=args.n)

    base_d = 64
    centers = rng.normal(size=(len(themes), base_d)) * 3.0
    latent = centers[assign] + rng.normal(size=(args.n, base_d))

    out = args.out
    wd = out / "wheel-data"
    (wd / "visual_embeddings").mkdir(parents=True, exist_ok=True)
    (wd / "caption_embeddings").mkdir(parents=True, exist_ok=True)

    for name, (d, tag, spec) in EMB.items():
        proj = rng.normal(size=(base_d, d))
        vecs = _normalize(
            latent @ proj + rng.normal(size=(args.n, d)) * NOISE[name]
        ).astype("float32")
        ix = faiss.index_factory(d, spec, faiss.METRIC_INNER_PRODUCT)
        ix.train(np.ascontiguousarray(vecs))
        ix.add(np.ascontiguousarray(vecs))
        if name == "cosmos":
            faiss.write_index(ix, str(wd / f"cosmos_embeddings_{tag}.index"))
            with open(wd / f"cosmos_clip_to_index_{tag}.pkl", "wb") as f:
                pickle.dump({c: i for i, c in enumerate(clip_ids)}, f)
            np.savez(out / "cosmos.npz", clip_ids=np.array(clip_ids), embeddings=vecs)
        elif name == "visual":
            vdir = wd / "visual_embeddings"
            faiss.write_index(ix, str(vdir / f"visual_embeddings_{tag}.index"))
            np.save(vdir / f"visual_clip_ids_{tag}.npy", np.array(clip_ids))
            np.save(vdir / f"visual_position_of_row_{tag}.npy", np.arange(args.n))
        else:  # caption: ships a legacy index->clip map
            cdir = wd / "caption_embeddings"
            faiss.write_index(ix, str(cdir / f"caption_embeddings_{tag}.index"))
            with open(cdir / f"caption_index_to_clip_{tag}.pkl", "wb") as f:
                pickle.dump({i: c for i, c in enumerate(clip_ids)}, f)
        print(f"  {name}: {args.n}x{d} -> {spec} index + id map", flush=True)

    db = out / "captions.db"
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE captions(uid INTEGER PRIMARY KEY, clip_id TEXT NOT NULL, "
        "model_name TEXT NOT NULL, caption TEXT NOT NULL, data_source TEXT, "
        "start_time REAL, end_time REAL)"
    )
    rows = []
    for i, c in enumerate(clip_ids):
        words = rng.choice(THEME_WORDS[themes[assign[i]]], size=5, replace=True)
        rows.append(
            (c, "synthetic_vlm", "the ego vehicle " + " ".join(words), "synthetic")
        )
    con.executemany(
        "INSERT INTO captions(clip_id, model_name, caption, data_source) VALUES (?,?,?,?)",
        rows,
    )
    con.commit()
    con.close()

    print(f"\nwrote synthetic wheel-data + captions to {out}/")
    print("next:")
    print(f"  export WD={wd.resolve()} DB={db.resolve()}")
    print(f"  python prep.py pool --wheel-data-dir $WD --out {out}/emb_pools")
    print(
        f"  WHEEL_DATA_DIR=$WD CAPTIONS_DB=$DB POOLS_DIR={out}/emb_pools CLUSTER_OUT={out}/clustering \\"
    )
    print("      python run_embedding_clustering.py --k 20 (see its README section)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "pool", help="clip-id pools in the cosmos∩visual∩caption intersection"
    )
    p.add_argument(
        "--wheel-data-dir",
        type=Path,
        required=True,
        help="dir holding the cosmos/visual/caption indices + id maps",
    )
    p.add_argument("--out", type=Path, required=True, help="output dir for the pools")
    p.add_argument("--cosmos-tag", default="ivf4096_pq96x8")
    p.add_argument("--visual-tag", default="ivf4096_pq64x8")
    p.add_argument("--caption-tag", default="ivf4096_pq256x8")
    p.add_argument(
        "--pai-path-files",
        type=Path,
        nargs="*",
        default=None,
        help="optional clip-id path files defining a curated 'pai' subset",
    )
    p.add_argument("--n-large", type=int, default=2_500_000)
    p.add_argument("--seed", type=int, default=1234)
    p.set_defaults(func=cmd_pool)

    p = sub.add_parser("fig-runs", help="fig_runs.json from runs.tsv")
    p.add_argument(
        "--runs", type=Path, required=True, help="runs.tsv from run_embedding_clustering.py"
    )
    p.add_argument("--clustering-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("fig_runs.json"))
    p.add_argument(
        "--pools",
        nargs="*",
        default=None,
        help="restrict to these pool names (default: all seen, in order)",
    )
    p.set_defaults(func=cmd_fig_runs)

    p = sub.add_parser(
        "synthetic", help="tiny synthetic wheel-data set for smoke tests"
    )
    p.add_argument("--out", type=Path, default=Path("synth_starter"))
    p.add_argument("--n", type=int, default=2000, help="number of synthetic clips")
    p.add_argument("--themes", type=int, default=8, help="number of blobs (<= 8)")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synthetic)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
