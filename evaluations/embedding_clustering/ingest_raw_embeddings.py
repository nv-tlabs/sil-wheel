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

"""Ingest per-encoder embedding dumps into one deduped npz per encoder.

Emits, per encoder, a single ``<name>.npz`` holding ``clip_ids`` (str) +
``embeddings`` (float32, one vector per clip). These npz feed the raw-vector
clustering driver (``cluster_raw.py``), ``preindex_compare.py --raw-npz``, and
the embedding-quality evaluation (``../embedding_quality``) directly — no FAISS
index round-trip and no internal data paths.

The default ``--layout hf`` reads the wheel-data directory produced by the
public Physical AI getting-started example
(``examples/getting-started-physical-ai-autonomous-vehicles/setup_physical_ai.py``),
whose extract steps leave one shard set per encoder under ``<workdir>``:

    <workdir>/cosmos_embeddings/cosmos_embed1_448p_group_*.parquet   (clip_id, embeddings)
    <workdir>/caption_embeddings/group_*.parquet                     (clip_id, embedding)
    <workdir>/visual_embeddings/florence2_sigclip_group_*.pkl        (Florence-2/SigLIP)

``--layout internal`` reads the per-split research dumps instead
(``<root>/<encoder-dir>/physical_ai/{avfoundation,alpamayo}/<shards>``) and also
exposes the qwen3_vl / pe_core encoders used for the quality table.

Shard formats:

* parquet encoders (cosmos / caption / qwen3_vl / pe_core): columns
  ``clip_id`` + ``embeddings`` (or ``embedding``); one row per clip.
* visual (Florence-2/SigLIP): ``.pkl`` dicts ``{embeddings: (N,d) f32,
  items: [{clip_id, camera, frame_index, bbox_xyxy, label}, ...]}``. We keep,
  per clip, the earliest ``__full_frame__`` row (smallest ``frame_index``) — the
  scene-level vector the served visual index clusters on.

Also writes ``<pool-name>_clip_ids.json`` = the intersection of the requested
encoders' clip sets (so every embedding clusters the same clips and their
per-cluster topic distributions are comparable), plus ``pool_summary.json``.

    # public: ingest the getting-started wheel-data dir
    python ingest_raw_embeddings.py \
        --root ./wheel-data-physical-ai --out ./npz \
        --encoders cosmos caption visual --pool-name pai

    # internal research dumps
    python ingest_raw_embeddings.py --layout internal \
        --root /path/to/raw_dumps --out ./npz --encoders cosmos caption visual
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Per-layout encoder source map: encoder key -> (subdir, file glob, format).
# The encoder key is also the npz basename and the embed_type label used
# downstream (cosmos/caption/visual are the three the paper figures use).
# ``splits`` lists the per-split subdirs nested under <root>/<subdir>/physical_ai/;
# an empty tuple means the shards sit directly under <root>/<subdir>/ (HF layout).
LAYOUTS = {
    # Matches setup_physical_ai.py's wheel-data workdir (flat group_* shards).
    "hf": {
        "splits": (),
        "encoders": {
            "cosmos": ("cosmos_embeddings", "cosmos_embed1_448p_group_*.parquet", "parquet"),
            "caption": ("caption_embeddings", "group_*.parquet", "parquet"),
            "visual": ("visual_embeddings", "florence2_sigclip_group_*.pkl", "pkl"),
        },
    },
    # Per-split research dumps on internal storage.
    "internal": {
        "splits": ("avfoundation", "alpamayo"),
        "encoders": {
            "cosmos": ("cosmos_embeddings", "cosmos_embed1_448p_*.parquet", "parquet"),
            "caption": (
                "qwen3-8b-embed-qwen3.5-27b-caption",
                "qwen3_embed_8b_*.parquet",
                "parquet",
            ),
            "visual": ("visual_embeddings", "florence2_sigclip_*.pkl", "pkl"),
            "qwen3_vl": ("qwen3_vl_embeddings", "qwen3_vl_embed_8b_*.parquet", "parquet"),
            "pe_core": ("pe_core_embeddings", "pe_core_g14_448p_*.parquet", "parquet"),
        },
    },
}
_EMB_COLS = ("embeddings", "embedding")


def _shards(root, subdir, glob, splits):
    if not splits:  # flat layout: shards directly under <root>/<subdir>/
        return sorted((root / subdir).glob(glob))
    out = []
    for split in splits:
        out += sorted((root / subdir / "physical_ai" / split).glob(glob))
    return out


def _read_parquet(files):
    """One vector per clip from list<double> parquet shards (first wins on dup)."""
    seen = {}
    ids = []
    chunks = []
    d = None
    for f in files:
        sc = pq.ParquetFile(f).schema_arrow
        emb_col = next((c for c in _EMB_COLS if c in sc.names), None)
        if emb_col is None:
            raise ValueError(
                f"{f}: no embedding column among {_EMB_COLS} (cols={sc.names})"
            )
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
    emb = (
        np.concatenate(chunks, axis=0) if chunks else np.zeros((0, d or 0), np.float32)
    )
    return ids, np.ascontiguousarray(emb, dtype=np.float32)


def _read_visual_pkl(files):
    """Scene-level vector per clip = the earliest ``__full_frame__`` row.

    The Florence-2/SigLIP extractor writes one ``__full_frame__`` row per sampled
    frame, so a clip has several; we keep the one with the smallest
    ``frame_index`` (the first sampled full frame). Absolute frame indices depend
    on the sampler and need not start at 0, so we select by minimum rather than
    by ``frame_index == 0``.
    """
    best = {}  # clip_id -> (frame_index, vector)
    order = []  # first-seen clip order, to keep output deterministic
    for f in files:
        with open(f, "rb") as fh:
            o = pickle.load(fh)
        emb = np.asarray(o["embeddings"], dtype=np.float32)
        for i, it in enumerate(o["items"]):
            if it.get("label") != "__full_frame__":
                continue
            c = str(it["clip_id"])
            fi = int(it.get("frame_index", 0))
            if c not in best:
                order.append(c)
                best[c] = (fi, np.array(emb[i], dtype=np.float32))
            elif fi < best[c][0]:
                best[c] = (fi, np.array(emb[i], dtype=np.float32))
        del o, emb
    if not order:
        return [], np.zeros((0, 0), np.float32)
    out = np.stack([best[c][1] for c in order], axis=0)
    return order, np.ascontiguousarray(out, dtype=np.float32)


def ingest_one(name, root, out_dir, layout):
    subdir, glob, fmt = LAYOUTS[layout]["encoders"][name]
    splits = LAYOUTS[layout]["splits"]
    files = _shards(root, subdir, glob, splits)
    if not files:
        loc = root / subdir
        if splits:
            loc = loc / "physical_ai" / f"{{{','.join(splits)}}}"
        raise FileNotFoundError(
            f"{name}: no shards under {loc} matching {glob} (layout={layout})"
        )
    t0 = time.perf_counter()
    print(f"[{name}] {len(files)} shards ({fmt}) ...", flush=True)
    if fmt == "parquet":
        ids, emb = _read_parquet(files)
    else:
        ids, emb = _read_visual_pkl(files)
    npz = out_dir / f"{name}.npz"
    np.savez(npz, clip_ids=np.array(ids, dtype=object), embeddings=emb)
    print(
        f"[{name}] {emb.shape[0]:,} clips × {emb.shape[1]} dim -> {npz} "
        f"({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )
    return ids


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="hf layout: the setup_physical_ai.py wheel-data workdir; "
        "internal layout: the dir holding the per-encoder dump dirs",
    )
    ap.add_argument(
        "--out", type=Path, required=True, help="output dir for the npz + pool files"
    )
    ap.add_argument(
        "--layout",
        choices=list(LAYOUTS),
        default="hf",
        help="source directory layout (default: hf = public getting-started output)",
    )
    ap.add_argument(
        "--encoders",
        nargs="+",
        default=["cosmos", "caption", "visual"],
        help="encoders to ingest; must exist in the chosen --layout",
    )
    ap.add_argument(
        "--pool-name", default="pai", help="basename for the intersection clip-id pool"
    )
    args = ap.parse_args(argv)

    known = LAYOUTS[args.layout]["encoders"]
    bad = [e for e in args.encoders if e not in known]
    if bad:
        ap.error(
            f"encoders {bad} not available in layout {args.layout!r} "
            f"(choices: {sorted(known)})"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    per_enc = {}
    for name in args.encoders:
        per_enc[name] = ingest_one(name, args.root, args.out, args.layout)

    common = set(per_enc[args.encoders[0]])
    for name in args.encoders[1:]:
        common &= set(per_enc[name])
    pool = sorted(common)
    (args.out / f"{args.pool_name}_clip_ids.json").write_text(json.dumps(pool))

    summary = {
        "layout": args.layout,
        "encoders": {n: len(ids) for n, ids in per_enc.items()},
        f"{args.pool_name}_intersection": len(pool),
    }
    (args.out / "pool_summary.json").write_text(json.dumps(summary, indent=2))
    print("summary:", summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
