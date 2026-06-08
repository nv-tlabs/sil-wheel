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

"""Generate a tiny synthetic 'wheel-data' set so the embedding-clustering
pipeline runs end-to-end in seconds -- no multi-GB indices, GPU, or downloads.

It plants Gaussian blobs (driving "themes") and emits three correlated-but-
different embeddings (cosmos / caption / visual) for ``--n`` synthetic clips,
each written as a wheel FAISS index + id map in the layout the scripts expect.
It also writes a toy captions DB whose words track each clip's theme (so the
recovered clusters get meaningful topics) and a ``cosmos.npz`` for the
pre/post-index comparison.

This is SYNTHETIC data for verifying the workflow / smoke-testing the scripts;
for real results, point the scripts at actual embedding indices -- see the
README's nuScenes path.

    python evaluations/embedding_clustering/make_synthetic_starter.py --out ./synth_starter
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3
from pathlib import Path

import numpy as np

# name -> (dim, filename tag, faiss build spec). The tag is only a filename
# label (matches what wheel ships); the build spec uses a small nlist so it
# trains on a tiny synthetic set.
EMB = {
    "cosmos":  (384, "ivf4096_pq96x8",  "IVF16,PQ96x8"),
    "caption": (512, "ivf4096_pq256x8", "IVF16,PQ256x8"),
    "visual":  (384, "ivf4096_pq64x8",  "IVF16,PQ64x8"),
}
# Per-embedding noise: cosmos cleanest (strong global structure), visual noisiest
# -- so the three induce visibly different geometries on the same clips.
NOISE = {"cosmos": 0.5, "caption": 0.8, "visual": 1.2}

THEME_WORDS = {
    "highway merge":       ["highway", "merge", "ramp", "lane", "accelerate", "overtaking"],
    "pedestrian crossing": ["pedestrian", "crosswalk", "yield", "walking", "sidewalk", "halt"],
    "roundabout":          ["roundabout", "circular", "yield", "exit", "navigate", "island"],
    "parking":             ["parking", "reverse", "stall", "maneuver", "garage", "space"],
    "traffic light":       ["intersection", "red", "green", "signal", "queue", "wait"],
    "rain night":          ["rain", "night", "wet", "headlights", "reflection", "dark"],
    "construction":        ["cones", "construction", "closed", "worker", "detour", "barrier"],
    "cyclist":             ["cyclist", "bike", "overtake", "share", "shoulder", "helmet"],
}


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("synth_starter"))
    ap.add_argument("--n", type=int, default=2000, help="number of synthetic clips")
    ap.add_argument("--themes", type=int, default=8, help="number of blobs (<= 8)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import faiss

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
        vecs = _normalize(latent @ proj + rng.normal(size=(args.n, d)) * NOISE[name]).astype("float32")
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

    # toy captions DB (schema matches sil_wheel.stores.sqlite_caption_store)
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
        rows.append((c, "synthetic_vlm", "the ego vehicle " + " ".join(words), "synthetic"))
    con.executemany(
        "INSERT INTO captions(clip_id, model_name, caption, data_source) VALUES (?,?,?,?)", rows
    )
    con.commit()
    con.close()

    print(f"\nwrote synthetic wheel-data + captions to {out}/")
    print("next:")
    print(f"  export WD={wd.resolve()} DB={db.resolve()}")
    print(f"  python evaluations/embedding_clustering/build_pool_clip_ids.py --wheel-data-dir $WD --out {out}/emb_pools")
    print(f"  WHEEL_DATA_DIR=$WD CAPTIONS_DB=$DB POOLS_DIR={out}/emb_pools CLUSTER_OUT={out}/clustering \\")
    print("      POOL=full K=20 bash evaluations/embedding_clustering/run_full_cluster.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
