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

"""Ingest raw (pre-index) embedding dumps into one deduped npz per encoder.

Takes the per-encoder embedding dumps physical-AI ships (parquet for the
clip-level encoders, pickled region detections for the visual encoder) and
emits, per encoder, a single ``<name>.npz`` holding ``clip_ids`` (str) +
``embeddings`` (float32, one vector per clip). These npz feed the raw-vector
clustering driver (``cluster_raw.py``) and ``preindex_compare.py --raw-npz``
directly — no FAISS index round-trip.

Layout assumed (configurable via --root / per-encoder globs):

    <root>/<encoder-dir>/physical_ai/{avfoundation,alpamayo}/<shards>

* parquet encoders (cosmos / caption / qwen3_vl / pe_core): columns
  ``clip_id`` + ``embeddings`` (or ``embedding``); one row per clip.
* visual (Florence-2/SigLIP): ``.pkl`` dicts ``{embeddings: (N,d) f32,
  items: [{clip_id, camera, frame_index, bbox_xyxy, label}, ...]}``. We keep
  the ``__full_frame__`` row at ``frame_index == 0`` per clip — the scene-level
  vector — to match what the served visual index clusters on.

Also writes ``<pool-name>_clip_ids.json`` = the intersection of the requested
encoders' clip sets (so every embedding clusters the same clips and their
per-cluster topic distributions are comparable), plus ``pool_summary.json``.

    python ingest_raw_embeddings.py \
        --root /media/.../pai_embeddings_complete --out /media/.../pai_npz \
        --encoders cosmos caption visual --pool-name pai
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# (encoder key -> (subdir under <root>, file glob, format)).
# The encoder key is also the npz basename and the embed_type label used
# downstream (cosmos/caption/visual are the three the paper figures use).
ENCODERS = {
    "cosmos":   ("cosmos_embeddings",                   "cosmos_embed1_448p_*.parquet", "parquet"),
    "caption":  ("qwen3-8b-embed-qwen3.5-27b-caption",  "qwen3_embed_8b_*.parquet",     "parquet"),
    "visual":   ("visual_embeddings",                   "florence2_sigclip_*.pkl",      "pkl"),
    "qwen3_vl": ("qwen3_vl_embeddings",                 "qwen3_vl_embed_8b_*.parquet",  "parquet"),
    "pe_core":  ("pe_core_embeddings",                  "pe_core_g14_448p_*.parquet",   "parquet"),
}
SPLITS = ("avfoundation", "alpamayo")
_EMB_COLS = ("embeddings", "embedding")


def _shards(root: Path, subdir: str, glob: str) -> list[Path]:
    out: list[Path] = []
    for split in SPLITS:
        out += sorted((root / subdir / "physical_ai" / split).glob(glob))
    return out


def _read_parquet(files: list[Path]) -> tuple[list[str], np.ndarray]:
    """One vector per clip from list<double> parquet shards (first wins on dup)."""
    seen: dict[str, int] = {}
    ids: list[str] = []
    chunks: list[np.ndarray] = []
    d = None
    for f in files:
        sc = pq.ParquetFile(f).schema_arrow
        emb_col = next((c for c in _EMB_COLS if c in sc.names), None)
        if emb_col is None:
            raise ValueError(f"{f}: no embedding column among {_EMB_COLS} (cols={sc.names})")
        t = pq.read_table(f, columns=["clip_id", emb_col])
        cids = [str(x) for x in t.column("clip_id").to_pylist()]
        la = t.column(emb_col).combine_chunks()
        if d is None:
            d = len(la[0])
        vals = la.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        mat = vals.reshape(len(cids), d)
        keep_rows = []
        for i, c in enumerate(cids):
            if c not in seen:
                seen[c] = 1
                ids.append(c)
                keep_rows.append(i)
        if keep_rows:
            chunks.append(mat[keep_rows])
    emb = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, d or 0), np.float32)
    return ids, np.ascontiguousarray(emb, dtype=np.float32)


def _read_visual_pkl(files: list[Path]) -> tuple[list[str], np.ndarray]:
    """Scene-level vector per clip = the __full_frame__ row at frame_index 0."""
    seen: set[str] = set()
    ids: list[str] = []
    chunks: list[np.ndarray] = []
    for f in files:
        with open(f, "rb") as fh:
            o = pickle.load(fh)
        emb = np.asarray(o["embeddings"], dtype=np.float32)
        rows = []
        cids = []
        for i, it in enumerate(o["items"]):
            if it.get("label") != "__full_frame__" or int(it.get("frame_index", 0)) != 0:
                continue
            c = str(it["clip_id"])
            if c in seen:
                continue
            seen.add(c)
            cids.append(c)
            rows.append(i)
        if rows:
            ids.extend(cids)
            chunks.append(emb[rows])
        del o, emb
    out = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), np.float32)
    return ids, np.ascontiguousarray(out, dtype=np.float32)


def ingest_one(name: str, root: Path, out_dir: Path) -> list[str]:
    subdir, glob, fmt = ENCODERS[name]
    files = _shards(root, subdir, glob)
    if not files:
        raise FileNotFoundError(f"{name}: no shards under {root/subdir} matching {glob}")
    t0 = time.perf_counter()
    print(f"[{name}] {len(files)} shards ({fmt}) ...", flush=True)
    if fmt == "parquet":
        ids, emb = _read_parquet(files)
    else:
        ids, emb = _read_visual_pkl(files)
    npz = out_dir / f"{name}.npz"
    np.savez(npz, clip_ids=np.array(ids, dtype=object), embeddings=emb)
    print(f"[{name}] {emb.shape[0]:,} clips × {emb.shape[1]} dim -> {npz} "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)
    return ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="dir holding the per-encoder dump dirs")
    ap.add_argument("--out", type=Path, required=True, help="output dir for the npz + pool files")
    ap.add_argument("--encoders", nargs="+", default=["cosmos", "caption", "visual"],
                    choices=list(ENCODERS))
    ap.add_argument("--pool-name", default="pai", help="basename for the intersection clip-id pool")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    per_enc: dict[str, list[str]] = {}
    for name in args.encoders:
        per_enc[name] = ingest_one(name, args.root, args.out)

    common = set(per_enc[args.encoders[0]])
    for name in args.encoders[1:]:
        common &= set(per_enc[name])
    pool = sorted(common)
    (args.out / f"{args.pool_name}_clip_ids.json").write_text(json.dumps(pool))

    summary = {
        "encoders": {n: len(ids) for n, ids in per_enc.items()},
        f"{args.pool_name}_intersection": len(pool),
    }
    (args.out / "pool_summary.json").write_text(json.dumps(summary, indent=2))
    print("summary:", summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
