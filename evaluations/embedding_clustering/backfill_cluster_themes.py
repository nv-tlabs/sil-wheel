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

"""Offline LLM theme summaries for clustering runs.

The LLM summarization normally runs during clustering; when a run was clustered
without an LLM (e.g. locally), this backfills it: for each cluster it sends the
TF-IDF keywords to an OpenAI-compatible chat endpoint and stores the one-phrase
result as the cluster's ``description`` in ``cluster_topics.json`` (re-upload the
run afterwards so the server/UI shows the themes).

Endpoint via ``--base-url`` (or ``LLM_BASE_URL``) + ``--model`` (or ``LLM_MODEL``)
+ ``NV_INFERENCE_API_KEY``/``OPENAI_API_KEY``. Low concurrency + backoff because
the gateway rate-limits.

    NV_INFERENCE_API_KEY=... python backfill_cluster_themes.py \
        --clustering-dir /media/.../clustering_pai_complete \
        --runs x9a2gvcsz3 t2isxvlj4i iifi76mehh \
        --base-url https://<gateway>/v1 --model gcp/google/gemini-3.1-flash-lite-preview
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError

SYSTEM = (
    "You are given TF-IDF keyword terms from a cluster of driving-scene video "
    "captions. Reply with ONE short noun phrase (4-9 words) naming the common "
    "SCENE or SCENARIO (road type, setting, weather, lighting, time of day, or "
    "traffic situation). Do NOT mention the camera or viewpoint: never use "
    "'first-person', 'first person view', 'dashcam', 'POV', 'point of view', "
    "'ego vehicle', or 'driving perspective'. Output only the phrase: no quotes, "
    "no leading 'A '/'The ', no trailing punctuation."
)


def _summarize(keywords, base, model, key, retries=6):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": "Keywords: " + ", ".join(keywords)}],
        "max_tokens": 1024, "temperature": 0.2,
    }).encode()
    url = base.rstrip("/") + "/chat/completions"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                method="POST")
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            content = d["choices"][0]["message"].get("content")
            if content and content.strip():
                return content.strip().strip('"').strip("'").rstrip(".")
            if attempt < retries - 1:        # null content (reasoning-model truncation) -> retry
                time.sleep(1.0 + attempt)
                continue
            return ""
        except HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except (URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clustering-dir", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="run-id subdirs to backfill")
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL"))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    ap.add_argument("--workers", type=int, default=2, help="concurrency (gateway rate-limits)")
    ap.add_argument("--top-k", type=int, default=10, help="keywords sent per cluster")
    ap.add_argument("--topics-name", default="cluster_topics.json",
                    help="topics file per run dir (use hier_topics.json for hierarchical runs)")
    args = ap.parse_args(argv)

    key = os.environ.get("NV_INFERENCE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not (key and args.base_url and args.model):
        raise SystemExit("need a key (NV_INFERENCE_API_KEY) + --base-url + --model")

    for rid in args.runs:
        p = args.clustering_dir / rid / args.topics_name
        tj = json.loads(p.read_text())
        flat = "topics" not in tj            # hier_topics.json is a flat {node: {...}} dict
        topics = tj if flat else tj["topics"]
        cids = [c for c, v in topics.items() if isinstance(v, dict) and v.get("keywords")]

        def work(c):
            try:
                return c, _summarize(topics[c]["keywords"][:args.top_k], args.base_url, args.model, key)
            except Exception as e:
                print(f"  {rid} {c} failed: {type(e).__name__} {str(e)[:80]}", flush=True)
                return c, ""

        n_ok = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for c, desc in ex.map(work, cids):
                if desc:
                    topics[c]["description"] = desc
                    n_ok += 1
        if not flat:
            tj["topics"] = topics
        p.write_text(json.dumps(tj))
        print(f"{rid}: {n_ok}/{len(cids)} nodes themed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
