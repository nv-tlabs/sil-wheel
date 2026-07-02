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

"""Caption faceting: split the dense caption into four aligned views, two steps.

The dense caption is decomposed into four facets aligned with the caption-quality
axes (scene / road entities / action / temporal) and each facet is embedded
separately, producing ordinary ``caption_facet_{scene,road_entities,action,
temporal}.npz`` encoders read by ``embeddings_io.load_embeddings`` and scored on
the standard vector path.

Subcommands (``FACETS`` is the single source of truth for the twelve fields):

  segment  text-only decomposition of each caption into the 12 structured fields
           with gpt-oss-20b via the NVIDIA inference API (no video, no labels;
           multi-key, resumable). Writes a JSONL of clip_id + fields.
  embed    embed each of the four facet texts with Qwen3-Embedding-8B and write
           the per-facet npz encoders (needs a CUDA GPU).

The related common-component-removal ablation lives in
``evaluations/embedding_clustering/caption_pc_ablation.py``.

  set -a; . .env; set +a   # NV_INFERENCE_API_KEYS=key1,key2,...
  python caption_faceting.py segment --captions captions.parquet --out facets.jsonl
  python caption_faceting.py embed   --facets facets.jsonl --out-dir ./embeddings
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

# Paper's caption-quality axes -> the 12 structured fields. Single source of
# truth: the segmentation schema and the facet grouping both derive from this.
FACETS: dict[str, list[str]] = {
    "scene": ["weather", "illumination", "road_type", "road_curvature"],
    "road_entities": ["key_objects", "road_information",
                      "vehicle_density", "pedestrian_density"],
    "action": ["ego_meta_action", "ego_speed", "rule_following_or_violation"],
    "temporal": ["key_events"],
}
EXPECTED = {k for keys in FACETS.values() for k in keys}

_SYS = (
    "You extract structured attributes from a written autonomous-driving scene "
    "caption. Use ONLY information stated in the caption. State every field as a "
    "positive observation; never use negations (no/none/not/without). Omit what "
    "is not mentioned. Output ONLY a JSON object."
)
_SCHEMA = (
    '{"key_objects":["<object>: <desc>"],"key_events":["<event, chronological order>"],'
    '"vehicle_density":"none/low/medium/high + brief reason","pedestrian_density":"none/low/medium/high + brief reason",'
    '"weather":"clear/rain/snow/fog/etc. + brief reason","illumination":"day/night + lighting + brief reason",'
    '"ego_speed":"standing/low/city/highway + brief reason","road_curvature":"low/medium/high",'
    '"road_type":"rural/residential/urban/highway/etc.","road_information":"lane counts, intersections, etc.",'
    '"ego_meta_action":"ego maneuvers in chronological order","rule_following_or_violation":"compliant/violation + brief reason"}'
)


def facet_text(structured: dict, keys: list[str]) -> str:
    """Join a facet's structured fields into one text (lists flattened)."""
    parts: list[str] = []
    for k in keys:
        v = structured.get(k)
        parts.append(" ".join(map(str, v)) if isinstance(v, list) else str(v))
    return " ".join(parts)


# ── segment: caption -> 12 structured fields via gpt-oss-20b ────────────────
def _keys_from_env() -> list[str]:
    pool = os.environ.get("NV_INFERENCE_API_KEYS", "")
    keys = [k.strip() for k in pool.split(",") if k.strip()]
    if not keys:
        single = os.environ.get("NV_INFERENCE_API_KEY", "").strip()
        keys = [single] if single else []
    if not keys:
        raise SystemExit("set NV_INFERENCE_API_KEYS (comma-sep) or NV_INFERENCE_API_KEY")
    return keys


def _load_captions(path: Path) -> list[tuple[str, str]]:
    import pandas as pd
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    col = next((c for c in ("summary", "caption", "qwen35_caption") if c in df.columns), None)
    if col is None:
        raise SystemExit(f"{path}: need a summary/caption/qwen35_caption column")
    return list(df[["clip_id", col]].itertuples(index=False, name=None))


def segment(args) -> int:
    from sil_wheel.llm.llm_client import LLMClient

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.out.exists():
        for line in args.out.open():
            try:
                done.add(json.loads(line)["clip_id"])
            except Exception:
                pass
    todo = [(c, s) for c, s in _load_captions(args.captions) if c not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[segment] {len(done)} done; {len(todo)} to process with {args.model}", flush=True)
    if not todo:
        return 0

    keys = _keys_from_env()
    clients = [LLMClient(api_key=k, model=args.model, temperature=0.2,
                         max_tokens=args.max_tokens) for k in keys]
    lock = threading.Lock()
    fh = args.out.open("a")
    counter = {"ok": 0, "err": 0, "t0": time.time()}

    def work(idx_item):
        i, (clip_id, caption) = idx_item
        client = clients[i % len(clients)]
        user = f"Caption:\n{caption}\n\nProduce the JSON with exactly these fields:\n{_SCHEMA}"
        for attempt in range(args.retries + 1):
            try:
                txt = client.generate(prompt=user, system_prompt=_SYS,
                                      response_format={"type": "json_object"})
                if not txt or not txt.strip():
                    raise ValueError("empty response")
                d = client.parse_json(txt)
                if not EXPECTED.issubset(d.keys()):
                    raise ValueError(f"missing fields: {EXPECTED - set(d.keys())}")
                rec = {"clip_id": clip_id, **{k: d.get(k) for k in EXPECTED}}
                with lock:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    counter["ok"] += 1
                    n = counter["ok"] + counter["err"]
                    if n % 100 == 0:
                        rate = n / (time.time() - counter["t0"])
                        print(f"[segment] {n} done ({counter['err']} err) {rate:.1f}/s", flush=True)
                return
            except Exception as e:
                if attempt < args.retries:
                    time.sleep(min(2 ** attempt, 20))
                    continue
                with lock:
                    counter["err"] += 1
                print(f"[segment] FAILED {clip_id}: {e}", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(keys) * args.workers_per_key) as ex:
        list(ex.map(work, enumerate(todo)))
    fh.close()
    print(f"[segment] DONE ok={counter['ok']} err={counter['err']} "
          f"in {(time.time() - counter['t0']) / 60:.1f}min", flush=True)
    return 0


# ── embed: facet texts -> per-facet npz encoders ────────────────────────────
def embed(args) -> int:
    from sentence_transformers import SentenceTransformer

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in args.facets.open()]
    seen: set[str] = set()
    recs = [r for r in recs if not (r["clip_id"] in seen or seen.add(r["clip_id"]))]
    clip_ids = [r["clip_id"] for r in recs]
    print(f"[embed] {len(clip_ids)} clips with facets", flush=True)

    model = SentenceTransformer(args.model, model_kwargs={"torch_dtype": "float16"})
    for facet, keys in FACETS.items():
        texts = [facet_text(r, keys) for r in recs]
        emb = model.encode(texts, normalize_embeddings=True,
                           batch_size=args.batch_size, show_progress_bar=True)
        out = args.out_dir / f"caption_facet_{facet}.npz"
        np.savez(out, clip_ids=np.array(clip_ids, dtype=object),
                 embeddings=emb.astype(np.float32))
        print(f"[embed] wrote {out.name}: {emb.shape}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("segment", help="caption -> 12 structured fields (gpt-oss-20b)")
    s.add_argument("--captions", type=Path, required=True,
                   help="parquet/csv with clip_id + summary/caption column.")
    s.add_argument("--out", type=Path, required=True, help="output JSONL.")
    s.add_argument("--model", default="nvidia/openai/gpt-oss-20b")
    s.add_argument("--workers-per-key", type=int, default=8)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--max-tokens", type=int, default=2500)
    s.add_argument("--retries", type=int, default=4)
    s.set_defaults(fn=segment)

    e = sub.add_parser("embed", help="facet texts -> per-facet npz encoders")
    e.add_argument("--facets", type=Path, required=True, help="JSONL from `segment`.")
    e.add_argument("--out-dir", type=Path, required=True)
    e.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    e.add_argument("--batch-size", type=int, default=64)
    e.set_defaults(fn=embed)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
