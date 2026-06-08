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

"""Evaluate caption quality against human annotations and (optionally) source video.

Reference-based: ``nlg``, ``bertscore``, ``lingojudge``, ``llm_judge``.
Reference-free:  ``vlm_judge``, ``evqa`` (gated behind ``[evqa]`` extra).

Usage::

    python evaluations/caption_quality/caption_quality.py \\
        config/wheel_launch_dev_server_config.yaml \\
        caption_quality_results.md \\
        --reference-model human \\
        --prediction-model "Qwen2.5-VL-7B-Instruct" \\
        --metrics nlg,bertscore,llm_judge --num-samples 200
"""
import argparse
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from scorers import (
    REF_FREE_METRICS,
    available_metrics,
    build_scorer,
)
from reporting import write_markdown
from sil_wheel.stores.sqlite_caption_store import FTSCaptionStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

HUMAN_REFERENCE = "human"

# Auto-emitted annotation keys (not human-curated). Mirrors eval_retrieval_quality.
DEFAULT_KEY_DENYLIST = ["vlm_distill_%", "reason_%", "distill_%", "scenario_%"]

DEFAULT_NUM_SCENARIOS = 20
DEFAULT_MIN_PRED_CLIPS = 5
DEFAULT_SCENARIO_POOL = 100


def select_scenarios(
    annotations_db: str,
    captions_db: str,
    prediction_model: str,
    project: Optional[str] = None,
    n: int = DEFAULT_NUM_SCENARIOS,
    key_denylist: Optional[List[str]] = None,
    min_pred_clips: int = DEFAULT_MIN_PRED_CLIPS,
    pool_size: int = DEFAULT_SCENARIO_POOL,
) -> List[str]:
    """Pick ~n scenarios prioritised by caption-version coverage, spread across clip-count quantiles.

    Two-step lookup (~4s on Alpamayo): Q1 pulls top ``pool_size`` candidate
    keys from the annotations DB; Q2 counts ``prediction_model`` clips and
    distinct caption versions per candidate. Ranks by (n_versions,
    n_pred_clips) then stratified-samples by ``n_clips`` for complexity diversity.
    """
    import sqlite3

    # ----- Q1: candidate scenarios + clip counts -----
    sql_anno = """
        SELECT key, COUNT(DISTINCT clip_id) AS n_clips FROM annotations
        WHERE label_type='manual' AND key IS NOT NULL AND key != ''
    """
    p_anno: List[Any] = []
    if project:
        sql_anno += " AND project = ?"
        p_anno.append(project)
    for pat in (key_denylist or DEFAULT_KEY_DENYLIST):
        sql_anno += " AND key NOT LIKE ?"
        p_anno.append(pat)
    sql_anno += " GROUP BY key HAVING n_clips >= ? ORDER BY n_clips DESC LIMIT ?"
    p_anno.extend([min_pred_clips, pool_size])

    anno = sqlite3.connect(annotations_db)
    candidates = list(anno.execute(sql_anno, p_anno))

    # ----- Q2: per-candidate caption-version coverage -----
    cap = sqlite3.connect(captions_db)
    scored: List[Dict[str, Any]] = []
    for key, n_clips in candidates:
        sql_clips = "SELECT DISTINCT clip_id FROM annotations WHERE label_type='manual' AND key=?"
        p_clips: List[Any] = [key]
        if project:
            sql_clips += " AND project=?"
            p_clips.append(project)
        clip_ids = [r[0] for r in anno.execute(sql_clips, p_clips)]
        if not clip_ids:
            continue
        ph = ",".join("?" * len(clip_ids))
        n_pred = cap.execute(
            f"SELECT COUNT(DISTINCT clip_id) FROM captions "
            f"WHERE model_name=? AND clip_id IN ({ph})",
            (prediction_model, *clip_ids),
        ).fetchone()[0]
        if n_pred < min_pred_clips:
            continue
        n_ver = cap.execute(
            f"SELECT COUNT(DISTINCT model_name) FROM captions WHERE clip_id IN ({ph})",
            clip_ids,
        ).fetchone()[0]
        scored.append({"key": key, "n_clips": n_clips, "n_pred": n_pred, "n_ver": n_ver})
    anno.close()
    cap.close()

    if not scored:
        return []
    if len(scored) <= n:
        scored.sort(key=lambda s: (-s["n_ver"], -s["n_pred"]))
        return [s["key"] for s in scored]

    # ----- stratified sampling: top 2N by versions, then spread by clip count -----
    scored.sort(key=lambda s: (-s["n_ver"], -s["n_pred"]))
    pool = scored[:max(2 * n, n + 10)]
    pool.sort(key=lambda s: s["n_clips"])
    step = len(pool) / n
    selected: List[str] = []
    seen: set = set()
    for i in range(n):
        s = pool[min(int(i * step), len(pool) - 1)]
        if s["key"] not in seen:
            seen.add(s["key"])
            selected.append(s["key"])
    return selected


def load_pairs(
    captions_db: str,
    reference_model: str,
    prediction_model: str,
    annotations_db: Optional[str] = None,
    annotation_project: Optional[str] = None,
    annotation_key_denylist: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None,
    num_scenarios: int = DEFAULT_NUM_SCENARIOS,
    data_source: Optional[str] = None,
    num_samples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load (clip, reference, prediction) rows.

    ``reference_model='human'`` emits one row per ``(clip, scenario)`` from
    the annotations DB (auto-selecting scenarios if not given). Any other
    value joins captions vs captions on ``clip_id``.
    """
    if reference_model == HUMAN_REFERENCE:
        if not annotations_db:
            raise ValueError(
                "--reference-model human requires an annotations DB; pass "
                "annotations_db= or set datastores.annotations_db in the config."
            )
        if not scenarios:
            scenarios = select_scenarios(
                annotations_db=annotations_db,
                captions_db=captions_db,
                prediction_model=prediction_model,
                project=annotation_project,
                n=num_scenarios,
                key_denylist=annotation_key_denylist if annotation_key_denylist is not None else DEFAULT_KEY_DENYLIST,
            )
            logger.info("Auto-selected %d scenarios: %s", len(scenarios), scenarios)
        return _load_human_pairs(
            captions_db=captions_db,
            annotations_db=annotations_db,
            prediction_model=prediction_model,
            scenarios=scenarios,
            project=annotation_project,
            data_source=data_source,
            num_samples=num_samples,
            seed=seed,
        )
    return _load_caption_pairs(
        captions_db=captions_db,
        reference_model=reference_model,
        prediction_model=prediction_model,
        data_source=data_source,
        num_samples=num_samples,
        seed=seed,
    )


def _load_caption_pairs(
    captions_db, reference_model, prediction_model, data_source, num_samples, seed,
):
    store = FTSCaptionStore(captions_db)
    # Carry a per-item question when the captions table has one (e.g. a QA
    # dataset like LingoQA), so question-aware metrics (lingojudge) use it
    # instead of a generic framing. Optional -- absent on plain caption corpora.
    has_question = any(
        c[1] == "question" for c in store.conn.execute("PRAGMA table_info(captions)")
    )
    q_select = ", r.question AS question" if has_question else ""
    sql = f"""
        SELECT r.clip_id, r.caption AS reference, p.caption AS prediction,
               r.data_source AS data_source{q_select}
        FROM captions r
        JOIN captions p
          ON p.clip_id = r.clip_id
         AND p.model_name = ?
        WHERE r.model_name = ?
    """
    params: List[Any] = [prediction_model, reference_model]
    if data_source:
        sql += " AND r.data_source = ?"
        params.append(data_source)
    sql += " GROUP BY r.clip_id ORDER BY r.clip_id"

    with store.lock:
        rows = store.conn.execute(sql, params).fetchall()

    pairs = [
        {
            "clip_id": row["clip_id"],
            "reference": row["reference"],
            "prediction": row["prediction"],
            "data_source": row["data_source"] or "unknown",
            **({"question": row["question"]} if has_question else {}),
        }
        for row in rows
        if row["reference"] and row["prediction"]
    ]
    if num_samples and len(pairs) > num_samples:
        rng = random.Random(seed)
        pairs = rng.sample(pairs, num_samples)
    logger.info(
        "Loaded %d caption-vs-caption pairs (reference=%s, prediction=%s)",
        len(pairs), reference_model, prediction_model,
    )
    return pairs


def _load_human_pairs(
    captions_db, annotations_db, prediction_model, scenarios, project,
    data_source, num_samples, seed,
):
    """One ``(clip, scenario)`` pair per matching annotation, balanced across scenarios.

    Each row has ``reference = scenario`` (single tag) and a ``scenario``
    field for grouped aggregation. With ``num_samples`` set, each scenario
    gets a quota of ``num_samples // len(scenarios)``.
    """
    import sqlite3

    if not scenarios:
        return []
    per_scenario_cap = (num_samples // len(scenarios)) if num_samples else None
    rng = random.Random(seed)

    anno = sqlite3.connect(annotations_db)
    cap = sqlite3.connect(captions_db)
    cap.row_factory = sqlite3.Row

    pairs: List[Dict[str, Any]] = []
    for scenario in scenarios:
        sql_clips = "SELECT DISTINCT clip_id FROM annotations WHERE label_type='manual' AND key=?"
        p_clips: List[Any] = [scenario]
        if project:
            sql_clips += " AND project=?"
            p_clips.append(project)
        clip_ids = [r[0] for r in anno.execute(sql_clips, p_clips)]
        if not clip_ids:
            continue
        rng.shuffle(clip_ids)

        scenario_pairs: List[Dict[str, Any]] = []
        for start in range(0, len(clip_ids), 500):
            batch = clip_ids[start:start + 500]
            placeholders = ",".join("?" * len(batch))
            sql_cap = (
                f"SELECT clip_id, caption, data_source FROM captions "
                f"WHERE model_name=? AND clip_id IN ({placeholders})"
            )
            p_cap: List[Any] = [prediction_model, *batch]
            if data_source:
                sql_cap += " AND data_source=?"
                p_cap.append(data_source)
            seen: set = set()
            for row in cap.execute(sql_cap, p_cap):
                cid = row["clip_id"]
                if cid in seen or not row["caption"]:
                    continue
                seen.add(cid)
                scenario_pairs.append({
                    "clip_id": cid,
                    "reference": scenario,
                    "prediction": row["caption"],
                    "data_source": row["data_source"] or "unknown",
                    "scenario": scenario,
                })
                if per_scenario_cap and len(scenario_pairs) >= per_scenario_cap:
                    break
            if per_scenario_cap and len(scenario_pairs) >= per_scenario_cap:
                break
        pairs.extend(scenario_pairs)
    anno.close()
    cap.close()

    logger.info(
        "Loaded %d human-vs-prediction pairs across %d scenarios (project=%s, prediction=%s)",
        len(pairs), len(scenarios), project or "*", prediction_model,
    )
    return pairs


def attach_video_paths(
    pairs: List[Dict[str, Any]], annotations_db: str
) -> List[Dict[str, Any]]:
    """Attach a local ``video_path`` per pair from the annotations DB; drop unresolvable ones."""
    import sqlite3

    conn = sqlite3.connect(annotations_db)
    conn.row_factory = sqlite3.Row
    clip_ids = [p["clip_id"] for p in pairs]
    placeholders = ",".join("?" * len(clip_ids))
    rows = conn.execute(
        f"SELECT clip_id, path FROM video_paths WHERE clip_id IN ({placeholders})",
        clip_ids,
    ).fetchall()
    conn.close()

    paths = {row["clip_id"]: row["path"] for row in rows}
    missing = 0
    out: List[Dict[str, Any]] = []
    for p in pairs:
        path = paths.get(p["clip_id"])
        if path and os.path.exists(path):
            out.append({**p, "video_path": path})
        else:
            missing += 1
    if missing:
        logger.warning("Dropped %d/%d pairs without resolvable video", missing, len(pairs))
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_scores(
    per_clip: List[Dict[str, float]], group_keys: List[str],
) -> Dict[str, Dict[str, float]]:
    """Group per-clip rows by ``group_keys``; return {group: {metric: mean}}. Non-numerics are dropped."""
    by_group: Dict[Tuple, List[Dict]] = defaultdict(list)
    for row in per_clip:
        key = tuple(row.get(k, "unknown") for k in group_keys)
        by_group[key].append(row)
    by_group[("__overall__",) * len(group_keys)] = list(per_clip)

    out: Dict[str, Dict[str, float]] = {}
    for key, rows in by_group.items():
        label = "all" if key[0] == "__overall__" else " / ".join(map(str, key))
        agg: Dict[str, float] = {"n": float(len(rows))}
        metrics = {k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and k not in group_keys}
        for m in metrics:
            vals = [r[m] for r in rows if isinstance(r.get(m), (int, float))]
            if vals:
                agg[m] = float(np.mean(vals))
        out[label] = agg
    return out


def render_metric_table(
    metric_name: str, aggregated: Dict[str, Dict[str, float]],
) -> Tuple[List[str], List[Tuple[str, ...]]]:
    """Format aggregated scores as (headers, rows) for write_markdown."""
    metric_cols: List[str] = []
    for grp in aggregated.values():
        for k in grp.keys():
            if k != "n" and k not in metric_cols:
                metric_cols.append(k)
    headers = ["group", "n"] + metric_cols
    rows: List[Tuple[str, ...]] = []
    for grp_label, agg in sorted(aggregated.items(), key=lambda x: (x[0] != "all", x[0])):
        row = [grp_label, str(int(agg.get("n", 0)))]
        for col in metric_cols:
            val = agg.get(col)
            row.append(f"{val:.3f}" if isinstance(val, float) else "—")
        rows.append(tuple(row))
    return headers, rows


# ---------------------------------------------------------------------------
# Metric dispatch
# ---------------------------------------------------------------------------

def _scorer_kwargs(name: str, args) -> Dict[str, Any]:
    """argparse Namespace → per-metric scorer kwargs."""
    if name == "nlg":
        return {}
    if name == "bertscore":
        return {"model_type": args.bertscore_model}
    if name == "lingojudge":
        return {"device": args.lingojudge_device}
    if name == "llm_judge":
        return {
            "provider": args.llm_provider,
            "model": args.llm_model,
            "max_workers": args.llm_workers,
        }
    if name == "vlm_judge":
        return {
            "model": args.vlm_model,
            "max_workers": args.vlm_workers,
            "max_frames": args.vlm_max_frames,
            "fps": args.vlm_fps,
        }
    if name == "evqa":
        return {
            "backend": args.evqa_backend,
            "yolo_path": args.evqa_yolo_path,
            "frame_interval": args.evqa_frame_interval,
            "keyword_provider": args.llm_provider,
            "keyword_model": args.llm_model,
            "keyword_workers": args.llm_workers,
            "device": args.evqa_device,
            "cache_dir": args.evqa_cache_dir,
        }
    raise ValueError(f"No kwargs mapping registered for metric: {name!r}")


def run_metric(name: str, pairs: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    """Build a scorer via the factory, run it on ``pairs``, release resources."""
    t0 = time.time()
    scorer = build_scorer(name, **_scorer_kwargs(name, args))
    try:
        scores = scorer.score_batch(pairs)
    finally:
        if hasattr(scorer, "close"):
            scorer.close()
    logger.info(
        "Metric %s: scored %d/%d clips in %.1fs",
        name, len(scores), len(pairs), time.time() - t0,
    )
    return scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate caption quality against human references and source video.",
    )
    parser.add_argument("config", help="YAML server config")
    parser.add_argument("output_md", help="Markdown results file")
    parser.add_argument(
        "--reference-model", required=True,
        help="captions-DB model_name to use as reference, or 'human' for annotations DB",
    )
    parser.add_argument("--prediction-model", required=True, help="captions-DB model_name to score")
    parser.add_argument(
        "--annotation-project", default="Alpamayo",
        help="(human mode) project to scope to; '' merges all. Default: Alpamayo",
    )
    parser.add_argument(
        "--scenarios", action="append", default=None,
        help="(human mode) explicit scenario key, repeatable. If omitted, auto-selected.",
    )
    parser.add_argument(
        "--num-scenarios", type=int, default=DEFAULT_NUM_SCENARIOS,
        help=f"(human mode) auto-select count when --scenarios is omitted (default {DEFAULT_NUM_SCENARIOS})",
    )
    parser.add_argument("--data-source", default=None, help="restrict to one data_source token")
    parser.add_argument(
        "--metrics", default="nlg,bertscore,llm_judge",
        help=f"comma-separated subset of: {','.join(available_metrics())}",
    )
    parser.add_argument("--num-samples", type=int, default=None, help="cap on pairs to score")
    parser.add_argument("--seed", type=int, default=42)

    # LLM (llm_judge + evqa keyword extraction)
    parser.add_argument("--llm-provider", default="auto")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-workers", type=int, default=20, help="thread workers for LLM API calls")

    # BERTScore
    parser.add_argument("--bertscore-model", default="microsoft/deberta-xlarge-mnli")

    # LingoJudge
    parser.add_argument("--lingojudge-device", default="cuda")

    # VLM judge
    parser.add_argument("--vlm-model", default="gcp/google/gemini-3-flash-preview")
    parser.add_argument("--vlm-workers", type=int, default=20)
    parser.add_argument("--vlm-max-frames", type=int, default=8)
    parser.add_argument("--vlm-fps", type=float, default=1.0)

    # EVQAScore
    parser.add_argument("--evqa-backend", default="siglip", choices=["siglip", "clip"])
    parser.add_argument("--evqa-yolo-path", default="yolo11x-seg.pt")
    parser.add_argument("--evqa-frame-interval", type=int, default=30)
    parser.add_argument("--evqa-device", default="cuda")
    parser.add_argument(
        "--evqa-cache-dir", default=None,
        help="per-video visual-feature sidecar dir; warm runs skip SigLIP+YOLO",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    known = available_metrics()
    unknown = [m for m in metrics if m not in known]
    if unknown:
        raise SystemExit(f"Unknown metrics: {unknown}. Choose from {known}.")

    if not Path(args.config).exists():
        raise SystemExit(f"Config not found: {args.config}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pairs = load_pairs(
        captions_db=cfg["datastores"]["captions_db"],
        reference_model=args.reference_model,
        prediction_model=args.prediction_model,
        annotations_db=cfg["datastores"].get("annotations_db"),
        annotation_project=args.annotation_project,
        scenarios=args.scenarios,
        num_scenarios=args.num_scenarios,
        data_source=args.data_source,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    if not pairs:
        raise SystemExit("No reference/prediction pairs found.")

    if any(m in REF_FREE_METRICS for m in metrics):
        pairs = attach_video_paths(pairs, cfg["datastores"]["annotations_db"])
        if not pairs:
            raise SystemExit("Reference-free metrics requested but no video paths resolved.")

    # Group by scenario in human-reference mode; else by data_source.
    group_keys = ["scenario"] if any("scenario" in p for p in pairs) else ["data_source"]

    metadata = {
        "reference_model": args.reference_model,
        "prediction_model": args.prediction_model,
        "n_pairs": str(len(pairs)),
    }
    if args.data_source:
        metadata["data_source"] = args.data_source
    if args.reference_model == HUMAN_REFERENCE and args.annotation_project:
        metadata["annotation_project"] = args.annotation_project
    if group_keys == ["scenario"]:
        scenarios_seen = sorted({p["scenario"] for p in pairs if "scenario" in p})
        metadata["n_scenarios"] = str(len(scenarios_seen))

    first = True
    for metric_name in metrics:
        try:
            per_clip = run_metric(metric_name, pairs, args)
        except Exception as exc:
            # One failing metric must not kill the rest of the run.
            logger.exception("Metric %s failed: %s", metric_name, exc)
            continue
        if not per_clip:
            logger.warning("Metric %s returned no rows", metric_name)
            continue
        aggregated = aggregate_scores(per_clip, group_keys=group_keys)
        headers, rows = render_metric_table(metric_name, aggregated)
        write_markdown(
            args.output_md,
            f"Caption Quality — {metric_name}",
            headers,
            rows,
            metadata=metadata,
            append=not first,
        )
        first = False


if __name__ == "__main__":
    main()
