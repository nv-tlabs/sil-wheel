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

"""Facet-segment dense captions into the 12-field structured schema (text-only).

A single text-to-JSON pass over each dense caption with gpt-oss-20b (served via
the NVIDIA inference API through ``sil_wheel.llm.llm_client``). No video, no
labels: it only reorganizes the text already in the caption. Multi-key,
multi-worker, resumable (skips clip_ids already in ``--out``). The twelve
fields are grouped into four facets by ``caption_faceting.FACETS``.

  set -a; . .env; set +a   # NV_INFERENCE_API_KEYS=key1,key2,...
  python facet_segment.py --captions captions.parquet --out facets.jsonl \
      --workers-per-key 12
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from sil_wheel.llm.llm_client import LLMClient

SYS = (
    "You extract structured attributes from a written autonomous-driving scene "
    "caption. Use ONLY information stated in the caption. State every field as a "
    "positive observation; never use negations (no/none/not/without). Omit what "
    "is not mentioned. Output ONLY a JSON object."
)
SCHEMA = (
    '{"key_objects":["<object>: <desc>"],"key_events":["<event, chronological order>"],'
    '"vehicle_density":"none/low/medium/high + brief reason","pedestrian_density":"none/low/medium/high + brief reason",'
    '"weather":"clear/rain/snow/fog/etc. + brief reason","illumination":"day/night + lighting + brief reason",'
    '"ego_speed":"standing/low/city/highway + brief reason","road_curvature":"low/medium/high",'
    '"road_type":"rural/residential/urban/highway/etc.","road_information":"lane counts, intersections, etc.",'
    '"ego_meta_action":"ego maneuvers in chronological order","rule_following_or_violation":"compliant/violation + brief reason"}'
)
EXPECTED = set(json.loads(SCHEMA).keys())


def keys_from_env() -> list[str]:
    pool = os.environ.get("NV_INFERENCE_API_KEYS", "")
    keys = [k.strip() for k in pool.split(",") if k.strip()]
    if not keys:
        single = os.environ.get("NV_INFERENCE_API_KEY", "").strip()
        keys = [single] if single else []
    if not keys:
        raise SystemExit("set NV_INFERENCE_API_KEYS (comma-sep) or NV_INFERENCE_API_KEY")
    return keys


def load_captions(path: Path) -> list[tuple[str, str]]:
    """Return [(clip_id, caption), ...] from a parquet/csv with clip_id + a
    caption column (``summary``, ``caption``, or ``qwen35_caption``)."""
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    cap_col = next((c for c in ("summary", "caption", "qwen35_caption") if c in df.columns), None)
    if cap_col is None:
        raise SystemExit(f"{path}: need a summary/caption/qwen35_caption column")
    return list(df[["clip_id", cap_col]].itertuples(index=False, name=None))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="nvidia/openai/gpt-oss-20b")
    ap.add_argument("--workers-per-key", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if args.out.exists():
        for line in args.out.open():
            try:
                done.add(json.loads(line)["clip_id"])
            except Exception:
                pass
    todo = [(c, s) for c, s in load_captions(args.captions) if c not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[facet] {len(done)} done; {len(todo)} to process with {args.model}", flush=True)
    if not todo:
        return 0

    keys = keys_from_env()
    clients = [LLMClient(api_key=k, model=args.model, temperature=0.2,
                         max_tokens=args.max_tokens) for k in keys]
    lock = threading.Lock()
    fh = args.out.open("a")
    counter = {"ok": 0, "err": 0, "t0": time.time()}

    def work(idx_item):
        i, (clip_id, caption) = idx_item
        client = clients[i % len(clients)]
        user = f"Caption:\n{caption}\n\nProduce the JSON with exactly these fields:\n{SCHEMA}"
        for attempt in range(args.retries + 1):
            try:
                txt = client.generate(prompt=user, system_prompt=SYS,
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
                        print(f"[facet] {n} done ({counter['err']} err) {rate:.1f}/s", flush=True)
                return
            except Exception as e:
                if attempt < args.retries:
                    time.sleep(min(2 ** attempt, 20))
                    continue
                with lock:
                    counter["err"] += 1
                print(f"[facet] FAILED {clip_id}: {e}", flush=True)

    with ThreadPoolExecutor(max_workers=len(keys) * args.workers_per_key) as ex:
        list(ex.map(work, enumerate(todo)))
    fh.close()
    el = time.time() - counter["t0"]
    print(f"[facet] DONE ok={counter['ok']} err={counter['err']} in {el / 60:.1f}min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
