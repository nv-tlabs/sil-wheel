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

"""Topic-focus figure: contrastive distinctiveness bars + LLM phrase.

All embeddings draw topics from the SAME caption corpus, so the most-common
per-cluster keywords look nearly identical across embeddings. Instead, for each
embedding we surface the caption terms that *concentrate* in its clusters but
spread out across the other embeddings' clusters (a weighted log-odds). Each
panel shows those distinctive terms as horizontal bars (length = distinctiveness)
titled by an optional LLM one-phrase summary of what the embedding emphasizes.

Pools/embeddings/labels/colors and run ids come from a ``fig_runs.json``
(see fig_runs.example.json). Set ``NV_INFERENCE_API_KEY`` for the per-panel LLM
titles; without it the panels still render (bars only).

    NV_INFERENCE_API_KEY=... python make_topic_focus.py \
        --clustering-dir /path/to/clustering --fig-runs fig_runs.json \
        --out topic_focus.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ARTICLES = ("the ", "a ", "an ")

# Generic AV scene words + caption-template fragments: shared (non-distinctive)
# or caption-model boilerplate. Dropped BEFORE the contrastive scoring so a
# caption artifact can't masquerade as "distinctive to one embedding". Tune for
# your corpus.
_DROP = {
    "parked", "parked cars", "street", "intersection", "highway", "residential",
    "nighttime", "urban", "houses", "buildings", "overcast", "wet", "green",
    "light", "streetlights", "headlights", "nighttime drive",
    "seconds", "reason", "seconds action", "action", "action the", "captures",
    "captures nighttime", "drive multi", "highway under", "highway viewed",
    "drive", "ego", "ego vehicle", "multi", "partly", "sky overcast",
    "stop", "covered",
    "suggesting", "likely", "appears", "perspective", "perspective from",
    "viewed", "showing", "featuring", "various", "captured", "seen", "shows",
    "indicating", "possibly", "image", "scene", "view",
}

# Token-level noise: any keyword whose tokens include one of these is dropped
# (catches glued bigrams like "reason the", "key driving", "with scattered"
# that whole-term stopwording misses). Function words + caption-template
# scaffolding + vague descriptors.
_FILLER_TOKENS = {
    "the", "a", "an", "with", "of", "and", "to", "in", "on", "at", "is", "are",
    "this", "that", "by", "as", "its",
    "reason", "reasons", "seconds", "second", "action", "actions", "key",
    "depicts", "depicting", "suggesting", "suggests", "likely", "appears",
    "appear", "viewed", "showing", "shows", "featuring", "various", "captured",
    "captures", "capturing", "seen", "indicating", "indicates", "possibly",
    "image", "scene", "view", "multi", "partly", "under", "casting",
    "childhood", "driving", "depict", "showcasing", "showcase",
}


def _clean_kw(kw: str) -> str:
    kw = kw.strip().lower()
    for art in _ARTICLES:
        if kw.startswith(art):
            return kw[len(art):].strip()
    return kw


def _is_noise(term: str) -> bool:
    return term in _DROP or any(t in _FILLER_TOKENS for t in term.split())


def _profile(run_dir: Path, top_k: int = 8) -> dict:
    """word -> fraction of clusters whose top-`top_k` keywords contain it."""
    tj = json.loads((run_dir / "cluster_topics.json").read_text())
    topics = tj.get("topics", {})
    c: Counter = Counter()
    n = 0
    for v in topics.values():
        kws = {_clean_kw(k) for k in v.get("keywords", [])[:top_k]}
        kws = {k for k in kws if k and not _is_noise(k)}
        if not kws:
            continue
        n += 1
        for k in kws:
            c[k] += 1
    return {w: cnt / max(n, 1) for w, cnt in c.items()}


def _distinctive(profiles: dict, pi: int, emb_key: str, emb_keys: list[str],
                 topn: int, floor: float = 0.01) -> list[tuple[str, float]]:
    """Terms frequent in (pool pi, emb_key) but rare in the other embeddings of
    the same pool, as (term, score) pairs. Weighted log-odds:
    p_e * log((p_e+k)/(p_bg+k)), p_bg = mean over the other embeddings."""
    p_e = profiles[(pi, emb_key)]
    others = [profiles[(pi, e)] for e in emb_keys if e != emb_key]
    eps = 1e-3
    scored = []
    for w, pe in p_e.items():
        if pe < floor:
            continue
        p_bg = sum(o.get(w, 0.0) for o in others) / max(len(others), 1)
        scored.append((w, pe * math.log((pe + eps) / (p_bg + eps))))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:topn]


_SYSTEM = (
    "You are given the keyword terms that most DISTINGUISH one video-embedding "
    "model's automatic clusters from other embedding models clustered on the "
    "SAME video corpus. In one short noun phrase (5-12 words), say what kind of "
    "content, scene structure, or detail this embedding's clusters emphasize "
    "relative to the others. Output ONLY the phrase: no quotes, no leading "
    "'A '/'The ', no trailing punctuation, no explanation."
)


def _summarize(terms_by_panel: dict) -> dict:
    """One LLM call per panel -> short phrase. Returns {} if no API key."""
    if not os.environ.get("NV_INFERENCE_API_KEY"):
        print("[topics] NV_INFERENCE_API_KEY not set; rendering bars only", flush=True)
        return {}
    from sil_wheel.llm.llm_client import LLMClient

    client = LLMClient()
    if not client.config.api_key:
        print("[topics] LLM client has no api_key; rendering bars only", flush=True)
        return {}
    out = {}
    for key, terms in terms_by_panel.items():
        if not terms:
            continue
        try:
            txt = client.generate(
                prompt="Distinctive terms: " + ", ".join(terms),
                system_prompt=_SYSTEM, temperature=0.2, max_tokens=40,
            )
            out[key] = (txt or "").strip().strip('"').strip("'").rstrip(".")
        except Exception as e:
            print(f"[topics] summary failed for {key}: {e}", flush=True)
            out[key] = ""
        print(f"  {key}: {out.get(key, '')!r}", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clustering-dir", type=Path, required=True)
    ap.add_argument("--fig-runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("topic_focus.png"))
    ap.add_argument("--top-k", type=int, default=8, help="keywords/cluster into the profile")
    ap.add_argument("--topn", type=int, default=8, help="distinctive terms (bars) per panel")
    args = ap.parse_args(argv)

    spec = json.loads(args.fig_runs.read_text())
    embeds = spec["embeds"]
    pools = spec["pools"]
    emb_keys = [e["key"] for e in embeds]

    profiles = {(pi, e["key"]): _profile(args.clustering_dir / pool["runs"][e["key"]], args.top_k)
                for pi, pool in enumerate(pools) for e in embeds}
    terms = {(pi, e["key"]): _distinctive(profiles, pi, e["key"], emb_keys, args.topn)
             for pi, pool in enumerate(pools) for e in embeds}
    phrases = _summarize({k: [t for t, _ in v] for k, v in terms.items()})

    fig, axes = plt.subplots(len(pools), len(embeds), figsize=(15, 8), squeeze=False)
    for i, pool in enumerate(pools):
        for j, emb in enumerate(embeds):
            ax = axes[i][j]
            color = emb.get("color", "#4C78A8")
            items = terms[(i, emb["key"])][::-1]   # most distinctive at top
            labels = [t for t, _ in items]
            vals = [s for _, s in items]
            ax.barh(range(len(labels)), vals, color=color, alpha=0.85, height=0.72)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xticks([])
            ax.tick_params(length=0)
            for s in ("top", "right", "bottom"):
                ax.spines[s].set_visible(False)
            ax.spines["left"].set_color("#cccccc")
            ax.margins(x=0.02)
            phrase = phrases.get((i, emb["key"]), "")
            if phrase:
                ax.set_title("\n".join(textwrap.wrap(f"“{phrase}”", 34)),
                             fontsize=10.5, color="#222222", style="italic", pad=6)
    fig.tight_layout(rect=(0.05, 0, 1, 0.90))
    for j, emb in enumerate(embeds):
        p = axes[0][j].get_position()
        fig.text((p.x0 + p.x1) / 2, 0.935, emb["label"], ha="center",
                 fontsize=13, fontweight="bold", color=emb.get("color", "#333333"))
    for i, pool in enumerate(pools):
        p = axes[i][0].get_position()
        fig.text(0.012, (p.y0 + p.y1) / 2, f"{pool['label']}\n({pool['n']:,} clips)",
                 rotation=90, va="center", ha="center", fontsize=11, fontweight="bold")
    fig.suptitle("What each embedding's clusters emphasize: caption terms that concentrate "
                 "here but disperse across the other embeddings\n"
                 "(bar = distinctiveness; panel title = LLM one-phrase summary)",
                 fontsize=12.5, y=0.995)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
