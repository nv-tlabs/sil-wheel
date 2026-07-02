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

"""Text-video retrieval benchmark for MSR-VTT, PVD-Bench, and OpenDV."""
import argparse
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import orjson

from embeddings_utils import (
    Split,
    load_florence_sigclip_embeddings,
    load_subclip_caption_embeddings,
    load_video_embeddings,
    read_csv,
    read_jsonl,
    read_parquet,
    score_per_video,
)
from fusion import rrf_term, zscore_term
from metrics import ranks_for_paired, recall_at_k, t2v_and_v2t
from text_encoders import encode_text


# Each baseline mapped to its model family. Fusion skips combinations
# with two members of the same family (e.g. two Cosmos-Embed1 resolutions
# share an encoder, so fusing them is redundant).
BASELINES = {
    "cosmos_embed1_224p": "cosmos_embed1",
    "cosmos_embed1_336p": "cosmos_embed1",
    "cosmos_embed1_448p": "cosmos_embed1",
    "qwen3_vl_embed_2b": "qwen3_vl_embed",
    "qwen3_vl_embed_8b": "qwen3_vl_embed",
    "pe_core_b16_224p": "pe_core",
    "pe_core_l14_336p": "pe_core",
    "pe_core_g14_448p": "pe_core",
    "florence_sigclip2": "florence_sigclip2",
    "caption_embedding": "caption_embedding",
}

FUSER_NAMES = ["RRF", "zscore"]


def load_msrvtt_split(args):
    """MSR-VTT 1K-A: the JSFusion test CSV (``video_id`` / ``sentence``)."""
    if not args.gt_path.exists():
        raise FileNotFoundError(
            f"{args.gt_path} not found. Download MSRVTT_JSFUSION_test.csv from "
            "huggingface.co/datasets/friedrichor/MSR-VTT (raw_data/) and pass "
            "it as --gt-path."
        )
    rows = read_csv(args.gt_path)
    assert len(rows) == 1000, (
        f"expected 1000 JSFusion 1K-A rows, got {len(rows)}"
    )
    return Split(
        video_ids=[r["video_id"] for r in rows],
        sentences=[r["sentence"] for r in rows],
    )


def load_pvdbench_split(args):
    """PVD-Bench: clips with a human caption from the metadata parquet."""
    rows = read_parquet(args.gt_path)
    assert rows and {"clip_id", "human_caption"} <= rows[0].keys(), (
        f"{args.gt_path} needs clip_id + human_caption columns; "
        "expected output of extract_pe_video_metadata.py"
    )
    rows = [r for r in rows if isinstance(r["human_caption"], str)]
    return Split(
        video_ids=[str(r["clip_id"]) for r in rows],
        sentences=[r["human_caption"] for r in rows],
    )


def load_opendv_split(args):
    """OpenDV: captions JSONL with short/medium/long variants per clip.

    The ground-truth ``clip_id`` is the clip-path basename ``<vid>_<seg>`` that
    matches the embeddings, so it is used directly.
    """
    rows = read_jsonl(args.gt_path)
    return Split(
        video_ids=[r["clip_id"] for r in rows],
        sentences=[r[args.caption_length] for r in rows],
    )


def compute_sim(name, split, args):
    """Build the ``(n_text, n_video)`` similarity matrix for one baseline."""
    if name == "florence_sigclip2":
        text_emb = encode_text(name, split.sentences, args.cache_dir)
        rows, owners = load_florence_sigclip_embeddings(
            Path(args.embeddings_dir) / "florence_sigclip2", split.video_ids
        )
        return score_per_video(text_emb, rows, owners, split.video_ids)
    if name == "caption_embedding":
        text_emb = encode_text("caption_embedding", split.sentences, args.cache_dir)
        rows, owners = load_subclip_caption_embeddings(
            Path(args.embeddings_dir) / "caption_embeddings_group_0_1.parquet",
            split.video_ids,
        )
        return score_per_video(text_emb, rows, owners, split.video_ids)
    text_emb = encode_text(name, split.sentences, args.cache_dir)
    video_emb = load_video_embeddings(
        Path(args.embeddings_dir) / f"{name}_group_0_1.parquet",
        split.video_ids,
    )
    return score_per_video(text_emb, video_emb)


def score(sim):
    t, v = t2v_and_v2t(sim)
    return {"t2v": t.as_dict(), "v2t": v.as_dict()}


def collect_failures(sim, split, rank_threshold):
    """T2V queries whose GT video ranks worse than ``rank_threshold``, each
    paired with the wrongly-retrieved top-1 caption for inspection.
    """
    n = sim.shape[0]
    assert sim.shape == (n, n), f"expected square sim, got {sim.shape}"
    assert len(split.video_ids) == n and len(split.sentences) == n
    ranks = ranks_for_paired(sim, np.arange(n))
    top1 = sim.argmax(axis=1)
    fails = []
    for i in range(n):
        if ranks[i] <= rank_threshold:
            continue
        fails.append({
            "clip_id": split.video_ids[i],
            "gt_caption": split.sentences[i],
            "rank": int(ranks[i]),
            "top1_clip_id": split.video_ids[int(top1[i])],
            "top1_caption": split.sentences[int(top1[i])],
        })
    fails.sort(key=lambda r: -r["rank"])
    return fails


def render_table(title, rows):
    """Markdown table sorted by T2V R@1 descending."""
    header = (
        "| Combination | Fuser "
        "| T2V R@1 | T2V R@5 | T2V R@10 | T2V MedR "
        "| V2T R@1 | V2T R@5 | V2T R@10 | V2T MedR |"
    )
    sep = (
        "| --- | --- "
        "| ---: | ---: | ---: | ---: "
        "| ---: | ---: | ---: | ---: |"
    )
    lines = [f"## {title}", "", header, sep]
    items = sorted(rows.items(), key=lambda kv: -kv[1]["t2v"]["R@1"])
    for tag, row in items:
        if " (" in tag:
            combo, fuser = tag.rsplit(" (", 1)
            fuser = fuser.rstrip(")")
        else:
            combo, fuser = tag, "—"
        t = row["t2v"]
        v = row["v2t"]
        cells = [
            combo, fuser,
            f"{t['R@1']*100:.1f}", f"{t['R@5']*100:.1f}",
            f"{t['R@10']*100:.1f}", f"{t['MedR']:.1f}",
            f"{v['R@1']*100:.1f}", f"{v['R@5']*100:.1f}",
            f"{v['R@10']*100:.1f}", f"{v['MedR']:.1f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cross_family(combo):
    families = [BASELINES[m] for m in combo]
    return len(set(families)) == len(families)


def eval_combos(size, rrf_terms, zscore_terms, available, label):
    """Score every cross-family ``size``-combination under both fusers."""
    combos = [c for c in combinations(available, size) if cross_family(c)]
    print(
        f"\n[{label}] {len(combos)} cross-family combos "
        f"x {len(FUSER_NAMES)} fusers", flush=True,
    )
    t0 = time.time()
    out = {}
    for combo in combos:
        rrf_fused = sum(rrf_terms[m] for m in combo)
        zscore_fused = sum(zscore_terms[m] for m in combo)
        out[" + ".join(combo) + " (RRF)"] = score(rrf_fused)
        out[" + ".join(combo) + " (zscore)"] = score(zscore_fused)
    print(f"[{label}] done in {time.time()-t0:.1f}s", flush=True)
    return out


def write_results(sections, args):
    """Print every section's table; also save the Markdown leaderboard if a
    path is given."""
    for title, section in sections:
        print()
        print(render_table(title, section), flush=True)
    if args.results_md:
        body = [f"# {args.dataset} retrieval benchmark", ""]
        for title, section in sections:
            body.append(render_table(title, section))
            body.append("")
        args.results_md.write_text("\n".join(body))
        print(f"Wrote {args.results_md}")


def main():
    supported_datasets = {
        "msrvtt": load_msrvtt_split,
        "pvdbench": load_pvdbench_split,
        "opendv": load_opendv_split,
    }
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval baselines on a dataset."
    )
    parser.add_argument(
        "dataset", choices=sorted(supported_datasets),
        help="Which dataset to benchmark.",
    )
    parser.add_argument(
        "embeddings_dir", type=Path,
        help="Directory of precomputed per-baseline embeddings "
             "(see the README for the expected layout).",
    )
    parser.add_argument(
        "cache_dir", type=Path,
        help="Directory for cached query-text encodings, one .npy per "
             "encoder keyed by the sentence hash.",
    )
    parser.add_argument(
        "gt_path", type=Path,
        help="Ground-truth file for the dataset: the MSR-VTT JSFusion CSV, "
             "the PVD-Bench metadata parquet, or the OpenDV captions JSONL.",
    )
    parser.add_argument(
        "--caption_length", choices=["short", "medium", "long"],
        default="long",
        help="Which OpenDV caption variant to use as the query text "
             "(default: long). Only used by the opendv dataset.",
    )
    parser.add_argument(
        "--results_md", type=Path,
        help="Path to write the results as a Markdown leaderboard.",
    )
    parser.add_argument(
        "--failures_json", type=Path, default=None,
        help="Path to dump per-baseline T2V failures (queries whose GT video "
             "ranks worse than --failures_rank_threshold). Each failure "
             "records clip_id, gt_caption, rank, top1_clip_id, top1_caption.",
    )
    parser.add_argument(
        "--failures_rank_threshold", type=int, default=10,
        help="A query is a failure when its GT rank exceeds this threshold "
             "(default: 10, i.e. R@10 misses).",
    )
    parser.add_argument(
        "--failures_fusion_combo", type=str, default=None,
        help="Fusion combo whose T2V failures should also be dumped, "
             "formatted as baseline1+baseline2+...:RRF|zscore. The key in "
             "--failures_json is the combo string verbatim.",
    )
    args = parser.parse_args()

    split = supported_datasets[args.dataset](args)

    standalone = {}
    rrf_terms = {}
    zscore_terms = {}
    failures = {}
    available_baselines = []

    print(f"[standalone] computing {len(BASELINES)} baselines", flush=True)
    for i, name in enumerate(BASELINES, 1):
        t0 = time.time()
        try:
            sim = compute_sim(name, split, args)
        except FileNotFoundError as e:
            print(f"  [{i}/{len(BASELINES)}] {name} skipped: {e}", flush=True)
            continue
        available_baselines.append(name)
        standalone[name] = score(sim)
        if args.failures_json:
            failures[name] = collect_failures(
                sim, split, args.failures_rank_threshold
            )
        rrf_terms[name] = rrf_term(sim)
        zscore_terms[name] = zscore_term(sim)
        del sim
        print(
            f"  [{i}/{len(BASELINES)}] {name} "
            f"(R@1 t2v={standalone[name]['t2v']['R@1']*100:.1f}, "
            f"{time.time()-t0:.1f}s)",
            flush=True,
        )

    if args.failures_fusion_combo:
        combo_str, _, fuser = args.failures_fusion_combo.rpartition(":")
        combo = combo_str.split("+")
        assert fuser in ("RRF", "zscore"), f"bad fuser: {fuser}"
        assert all(m in available_baselines for m in combo), (
            f"combo {combo} not in available baselines {available_baselines}"
        )
        terms = rrf_terms if fuser == "RRF" else zscore_terms
        fused_sim = sum(terms[m] for m in combo)
        failures[args.failures_fusion_combo] = collect_failures(
            fused_sim, split, args.failures_rank_threshold
        )

    if args.failures_json:
        args.failures_json.write_bytes(
            orjson.dumps(failures, option=orjson.OPT_INDENT_2)
        )
        print(f"Wrote {args.failures_json}")

    pairs = eval_combos(2, rrf_terms, zscore_terms,
                        available_baselines, "pairs")
    triplets = eval_combos(3, rrf_terms, zscore_terms,
                           available_baselines, "triplets")

    write_results([
        (f"Standalone ({len(standalone)} baselines)", standalone),
        (f"Pairs ({len(pairs)} rows)", pairs),
        (f"Triplets ({len(triplets)} rows)", triplets),
    ], args)


if __name__ == "__main__":
    main()
