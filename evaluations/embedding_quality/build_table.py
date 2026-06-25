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

"""Render embedding-quality ``summary.json`` files as CSV and LaTeX tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


EMBEDDING_DISPLAY = {
    "cosmos": "Cosmos-Embed1 (768-d)",
    "qwen3_vl_8b": "Qwen3-VL-8B (4096-d)",
    "pe_core_g14": "PE-Core-G14 (1280-d)",
    "caption": "Captions: Qwen3-Embed-8B / Qwen3.5-27B-VL (4096-d)",
    "florence2_sigclip": "Florence-2 / SigCLIP (768-d)",
    "trajectory": "Trajectory shape (1210-d)",
    "random": "Random Gaussian (768-d)",
}

PAPER_DISPLAY = {
    "cosmos": "Cosmos-Embed1",
    "qwen3_vl_8b": "Qwen3-VL-8B",
    "pe_core_g14": "PE-Core-G14",
    "caption": "Caption embedding",
    "florence2_sigclip": "Region embeddings",
    "trajectory": "Trajectory shape",
    "random": "Random Gaussian",
}

PAPER_ORDER = [
    "cosmos",
    "qwen3_vl_8b",
    "pe_core_g14",
    "caption",
    "florence2_sigclip",
    "trajectory",
    "random",
]

LABEL_MAP = {
    "Person with Hand Gestures": "Hand Gestures",
    "Person Holding Traffic Sign": "Traffic Sign",
}

COMMENT_LABELS = {"dog", "Traffic Sign"}

CAPTION = r"""  \caption{%
    \textbf{Embeddings quality on PAI subset}.
    Six labels cover scene-level,
    object, ego-trajectory, and long-tail scenarios.
    \emph{Caption embedding} = Qwen3-Embedding-8B on \wheelname captions;
    \emph{Region embeddings} = SigLIP2 on
    Florence2-detected crops.
    No single embedding dominates.
    Visual models (Cosmos-Embed1, Qwen3-VL-8B, PE-Core-G14) are
    strong on scene-level (Barrier Gate, Fog).
    Caption embeddings are competitive for specific objects
    (Animal Crossing, Hand Gestures) and in few shot setting.
    Only \emph{Trajectory shape} reliably captures
    ego-trajectory (U-turn).
  }
  \label{tab:embedding_quality_full}"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to run_embedding_quality.py summary.json.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        required=True,
        help="Output path without extension. Writes <stem>.csv and "
        "<stem>_paper.tex.",
    )
    parser.add_argument(
        "--intrinsic-k",
        type=int,
        default=64,
        help="Cluster k used for the single-class Intra/Inter column.",
    )
    parser.add_argument(
        "--purity-ks",
        type=int,
        nargs="+",
        default=None,
        help="Cluster k values to surface for purity and NMI. Defaults to "
        "[32, 64].",
    )
    return parser.parse_args(argv)


def _fmt(value: float | int | None) -> str:
    if not isinstance(value, (int, float)):
        return "---"
    if isinstance(value, float) and math.isnan(value):
        return "---"
    return f"{float(value):.3f}"


def _csv_value(value: float | int | None) -> float | int | str:
    if not isinstance(value, (int, float)):
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _ordered_embedding_items(summary: dict) -> list[tuple[str, dict]]:
    metrics_by_embedding = summary["metrics_per_embedding"]
    out: list[tuple[str, dict]] = []
    for key in PAPER_ORDER:
        if key in metrics_by_embedding:
            out.append((key, metrics_by_embedding[key]))
    for key, metrics in metrics_by_embedding.items():
        if key not in PAPER_DISPLAY:
            continue
        if key not in PAPER_ORDER:
            out.append((key, metrics))
    return out


def _metric_columns(purity_ks: list[int]) -> list[str]:
    return (
        ["nn_purity_k1", "nn_purity_k10"]
        + [f"cluster_purity_k{k}" for k in purity_ks]
        + [f"nmi_k{k}" for k in purity_ks]
        + ["few_shot_n5", "few_shot_n20"]
    )


def _row_from_metrics(
    embedding_key: str,
    label: str,
    metrics: dict,
    purity_ks: list[int],
    *,
    n_pos: int | None = None,
    commented: bool = False,
) -> dict:
    row = {
        "embedding_key": embedding_key,
        "embedding_display": EMBEDDING_DISPLAY[embedding_key],
        "paper_display": PAPER_DISPLAY[embedding_key],
        "label": label,
        "n_pos": n_pos,
        "commented": commented,
        "nn_purity_k1": metrics.get("nn_purity_k1"),
        "nn_purity_k10": metrics.get("nn_purity_k10"),
    }
    for k in purity_ks:
        row[f"cluster_purity_k{k}"] = metrics.get(f"cluster_purity_k{k}")
    for k in purity_ks:
        row[f"nmi_k{k}"] = metrics.get(f"nmi_k{k}")
    row["few_shot_n5"] = metrics.get("fewshot_acc_n5_mean")
    row["few_shot_n20"] = metrics.get("fewshot_acc_n20_mean")
    return row


def _paper_rows(summary: dict, purity_ks: list[int]) -> list[dict]:
    """Build long-form rows for the paper table, including Avg rows."""
    labels_raw = summary.get("label_names") or []
    labels = [(raw, LABEL_MAP.get(raw, raw)) for raw in labels_raw]
    emb_items = _ordered_embedding_items(summary)
    rows: list[dict] = []

    for raw_label, display_label in labels:
        commented = display_label in COMMENT_LABELS
        for embedding_key, metrics in emb_items:
            per_label = metrics.get("per_label", {})
            if raw_label not in per_label:
                continue
            label_metrics = per_label[raw_label]
            rows.append(_row_from_metrics(
                embedding_key,
                display_label,
                label_metrics,
                purity_ks,
                n_pos=label_metrics.get("n_pos"),
                commented=commented,
            ))

    rows.extend(_avg_rows(rows, purity_ks))
    return rows


def _avg_rows(rows: list[dict], purity_ks: list[int]) -> list[dict]:
    metric_cols = _metric_columns(purity_ks)
    present_keys = [
        key
        for key in PAPER_ORDER
        if any(row["embedding_key"] == key for row in rows)
    ]
    values: dict[str, dict[str, list[float]]] = {
        key: {metric: [] for metric in metric_cols}
        for key in present_keys
    }
    for row in rows:
        if row["commented"]:
            continue
        key = row["embedding_key"]
        if key not in values:
            continue
        for metric in metric_cols:
            value = row.get(metric)
            if not isinstance(value, (int, float)):
                continue
            value_f = float(value)
            if math.isnan(value_f):
                continue
            values[key][metric].append(value_f)

    avg_rows: list[dict] = []
    for key in present_keys:
        avg_metrics = {}
        for metric in metric_cols:
            vals = values[key][metric]
            avg_metrics[metric] = sum(vals) / len(vals) if vals else None
        avg_metrics["fewshot_acc_n5_mean"] = avg_metrics["few_shot_n5"]
        avg_metrics["fewshot_acc_n20_mean"] = avg_metrics["few_shot_n20"]
        avg_rows.append(_row_from_metrics(
            key,
            "Avg",
            avg_metrics,
            purity_ks,
            n_pos=None,
            commented=False,
        ))
    return avg_rows


def _write_csv(rows: list[dict], path: Path, purity_ks: list[int]) -> None:
    fieldnames = [
        "embedding_key",
        "embedding_display",
        "paper_display",
        "label",
        "n_pos",
        "commented",
    ] + _metric_columns(purity_ks)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: _csv_value(row.get(key))
                if key in _metric_columns(purity_ks)
                else row.get(key)
                for key in fieldnames
            })


def _col_max(rows: list[dict], metric_cols: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in metric_cols:
        vals = [
            float(row[col])
            for row in rows
            if isinstance(row.get(col), (int, float))
            and not math.isnan(float(row[col]))
        ]
        if vals:
            out[col] = max(vals)
    return out


def _table_cell(row: dict, col: str, col_max: dict[str, float]) -> str:
    value = row.get(col)
    cell = _fmt(value)
    if (
        isinstance(value, (int, float))
        and not math.isnan(float(value))
        and col in col_max
        and abs(float(value) - col_max[col]) < 1e-12
    ):
        return f"\\textbf{{{cell}}}"
    return cell


def _paper_row(row: dict, purity_ks: list[int], col_max: dict[str, float]) -> str:
    pur_cells = " ".join(
        f"& {_table_cell(row, f'cluster_purity_k{k}', col_max)}"
        for k in purity_ks
    )
    nmi_cells = " ".join(
        f"& {_table_cell(row, f'nmi_k{k}', col_max)}"
        for k in purity_ks
    )
    return (
        f"{_latex_escape(row['paper_display'])} "
        f"& {_table_cell(row, 'nn_purity_k1', col_max)} "
        f"& {_table_cell(row, 'nn_purity_k10', col_max)} "
        f"{pur_cells} {nmi_cells} "
        f"& {_table_cell(row, 'few_shot_n5', col_max)} "
        f"& {_table_cell(row, 'few_shot_n20', col_max)} \\\\"
    )


def _group_rows(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    groups: list[tuple[str, list[dict]]] = []
    for row in rows:
        if groups and groups[-1][0] == row["label"]:
            groups[-1][1].append(row)
        else:
            groups.append((row["label"], [row]))
    return groups


def _render_paper_latex(rows: list[dict], purity_ks: list[int]) -> str:
    n_pur = len(purity_ks)
    col_spec = "l l cc " + ("c" * n_pur) + " " + ("c" * n_pur) + " cc"
    knn_first = 3
    knn_last = 4
    pur_first = 5
    pur_last = 4 + n_pur
    nmi_first = pur_last + 1
    nmi_last = nmi_first + n_pur - 1
    few_first = nmi_last + 1
    few_last = few_first + 1
    pur_header = " & ".join(f"$k\\!=\\!{k}$" for k in purity_ks)

    metric_cols = _metric_columns(purity_ks)
    body: list[str] = []
    groups = _group_rows(rows)
    for group_i, (label, group) in enumerate(groups):
        maxima = _col_max(group, metric_cols)
        n_rows = len(group)
        group_commented = all(row["commented"] for row in group)
        for row_i, row in enumerate(group):
            prefix = "%" if row["commented"] else ""
            label_cell = (
                f"\\multirow{{{n_rows}}}{{*}}{{{_latex_escape(label)}}}"
                if row_i == 0 else ""
            )
            body.append(
                f"    {prefix}{label_cell} & "
                f"{_paper_row(row, purity_ks, maxima)}"
            )
        if group_i < len(groups) - 1:
            body.append("    %\\midrule" if group_commented else "    \\midrule")

    return (
        "\\begin{table}[!ht]\n"
        "\\centering \\small\n"
        "  \\setlength{\\tabcolsep}{4pt}\n"
        f"  \\begin{{tabular}}{{{col_spec}}}\n"
        "    \\toprule\n"
        "              &       & \\multicolumn{2}{c}{$k$-NN cons.}"
        f" & \\multicolumn{{{n_pur}}}{{c}}{{Purity}}"
        f" & \\multicolumn{{{n_pur}}}{{c}}{{NMI}}"
        " & \\multicolumn{2}{c}{Few-shot kNN} \\\\\n"
        f"    \\cmidrule(lr){{{knn_first}-{knn_last}}}"
        f" \\cmidrule(lr){{{pur_first}-{pur_last}}}"
        f" \\cmidrule(lr){{{nmi_first}-{nmi_last}}}"
        f" \\cmidrule(lr){{{few_first}-{few_last}}}\n"
        "    Label & Embedding & $k\\!=\\!1$ & $k\\!=\\!10$"
        f" & {pur_header} & {pur_header}"
        " & $n\\!=\\!5$ & $n\\!=\\!20$ \\\\\n"
        "    \\midrule\n"
        + "\n".join(body)
        + "\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        f"{CAPTION}\n"
        "\\end{table}\n"
    )


def _flat_for_avg(metrics: dict) -> dict:
    if "per_label" not in metrics:
        return metrics
    out = {
        key: value
        for key, value in metrics.items()
        if key not in ("per_label", "macro_avg")
    }
    out.update(metrics.get("macro_avg", {}))
    return out


def _single_class_rows(
    summary: dict,
    purity_ks: list[int],
    intrinsic_k: int,
) -> list[dict]:
    rows = []
    for key, metrics in _ordered_embedding_items(summary):
        flat = _flat_for_avg(metrics)
        row = {
            "embedding_key": key,
            "embedding_display": EMBEDDING_DISPLAY[key],
            "paper_display": PAPER_DISPLAY[key],
            "intra_inter": (
                f"{flat[f'intra_sim_k{intrinsic_k}']:.3f} / "
                f"{flat[f'inter_sim_k{intrinsic_k}']:.3f}"
            ),
            "nn_purity_k1": flat.get("nn_purity_k1"),
            "nn_purity_k10": flat.get("nn_purity_k10"),
            "few_shot_n5": flat.get("fewshot_acc_n5_mean"),
            "few_shot_n20": flat.get("fewshot_acc_n20_mean"),
        }
        for k in purity_ks:
            row[f"cluster_purity_k{k}"] = flat.get(f"cluster_purity_k{k}")
        rows.append(row)
    return rows


def _render_single_class_latex(rows: list[dict], purity_ks: list[int]) -> str:
    n_pur = len(purity_ks)
    col_spec = "l c cc " + ("c" * n_pur) + " cc"
    pur_header = " & ".join(f"$k\\!=\\!{k}$" for k in purity_ks)
    body = []
    for row in rows:
        pur_cells = " ".join(
            f"& {_fmt(row[f'cluster_purity_k{k}'])}" for k in purity_ks
        )
        body.append(
            f"    {_latex_escape(row['paper_display'])} "
            f"& {row['intra_inter']} "
            f"& {_fmt(row['nn_purity_k1'])} & {_fmt(row['nn_purity_k10'])} "
            f"{pur_cells} "
            f"& {_fmt(row['few_shot_n5'])} & {_fmt(row['few_shot_n20'])} \\\\"
        )
    return (
        "\\begin{table}[!ht]\n"
        "\\centering \\small\n"
        f"  \\begin{{tabular}}{{{col_spec}}}\n"
        "    \\toprule\n"
        "    Embedding & Intra/Inter & $k\\!=\\!1$ & $k\\!=\\!10$"
        f" & {pur_header} & $n\\!=\\!5$ & $n\\!=\\!20$ \\\\\n"
        "    \\midrule\n"
        + "\n".join(body)
        + "\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text())
    purity_ks = args.purity_ks if args.purity_ks else [32, 64]
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)

    if summary.get("mode") == "multi_label":
        rows = _paper_rows(summary, purity_ks)
        if not rows:
            print(f"[build_table] no paper rows in {args.summary}; exiting")
            return 1
        csv_path = args.output_stem.with_suffix(".csv")
        tex_path = args.output_stem.with_name(args.output_stem.name + "_paper").with_suffix(".tex")
        _write_csv(rows, csv_path, purity_ks)
        tex_path.write_text(_render_paper_latex(rows, purity_ks))
    else:
        rows = _single_class_rows(summary, purity_ks, args.intrinsic_k)
        if not rows:
            print(f"[build_table] no embeddings in {args.summary}; exiting")
            return 1
        csv_path = args.output_stem.with_suffix(".csv")
        tex_path = args.output_stem.with_name(args.output_stem.name + "_paper").with_suffix(".tex")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        tex_path.write_text(_render_single_class_latex(rows, purity_ks))

    print(f"[build_table] wrote {csv_path}")
    print(f"[build_table] wrote {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
