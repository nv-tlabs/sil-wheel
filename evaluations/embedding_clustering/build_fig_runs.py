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

"""Build fig_runs.json (input to make_umap_overview / make_topic_focus) from the
runs.tsv that run_full_cluster.sh appends, so you don't hand-edit run ids.

For each (pool, embedding) it takes the most recent successful (``DONE_rc=0``)
run, reads the pool size from that run's metadata.json, and writes the
pools/embeddings spec the figure scripts expect.

    python build_fig_runs.py --runs ./emb_pools/runs.tsv \
        --clustering-dir ./clustering --out ./fig_runs.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Display defaults for the known embeddings (column order + labels + colors).
EMBEDS = [
    {"key": "cosmos",  "label": "Cosmos-Embed1",          "color": "#4C78A8"},
    {"key": "caption", "label": "Caption (Qwen3-Emb-8B)", "color": "#F58518"},
    {"key": "visual",  "label": "Florence-2/SigLIP",      "color": "#54A24B"},
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, required=True, help="runs.tsv from run_full_cluster.sh")
    ap.add_argument("--clustering-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("fig_runs.json"))
    ap.add_argument("--pools", nargs="*", default=None,
                    help="restrict to these pool names (default: all seen, in first-seen order)")
    args = ap.parse_args(argv)

    # latest successful run id per (pool, embed)
    latest: dict[tuple[str, str], str] = {}
    pool_order: list[str] = []
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
            print(f"[warn] pool {pool!r} missing successful runs for {missing}; skipping", flush=True)
            continue
        n = 0
        meta = args.clustering_dir / next(iter(runs.values())) / "metadata.json"
        if meta.exists():
            n = int(json.loads(meta.read_text()).get("n_input_clips", 0))
        pools.append({"label": pool.capitalize(), "n": n, "runs": runs})

    if not pools:
        raise SystemExit("no complete pools found in runs.tsv (need cosmos+caption+visual each)")

    args.out.write_text(json.dumps({"embeds": EMBEDS, "pools": pools}, indent=2))
    print(f"wrote {args.out} with pools: " + ", ".join(f"{p['label']}({p['n']})" for p in pools))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
