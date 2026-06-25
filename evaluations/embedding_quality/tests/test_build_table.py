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

import csv
import json

import build_table


def _label_metrics(base, n_pos=10):
    return {
        "n_pos": n_pos,
        "n_neg": 100,
        "nn_purity_k1": base,
        "nn_purity_k10": base + 0.01,
        "cluster_purity_k4": base + 0.02,
        "cluster_purity_k8": base + 0.03,
        "nmi_k4": base / 10,
        "nmi_k8": base / 10 + 0.01,
        "fewshot_acc_n5_mean": base + 0.04,
        "fewshot_acc_n20_mean": base + 0.05,
    }


def _embedding(per_label_base):
    return {
        "embedding": "unused",
        "n_dims": 4,
        "n_clips": 120,
        "per_label": {
            "Animal Crossing": _label_metrics(per_label_base),
            "dog": _label_metrics(0.95),
            "Person Holding Traffic Sign": _label_metrics(0.90),
        },
        "macro_avg": {},
    }


def test_paper_table_drops_unlisted_and_recomputes_avg(tmp_path):
    summary = {
        "mode": "multi_label",
        "label_names": [
            "Animal Crossing",
            "dog",
            "Person Holding Traffic Sign",
        ],
        "metrics_per_embedding": {
            "random": _embedding(0.20),
            "qwen3_vl_8b": _embedding(0.80),
            "cosmos": _embedding(0.60),
            "florence2_sigclip_idf": _embedding(0.99),
        },
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))

    rc = build_table.main([
        "--summary",
        str(summary_path),
        "--output-stem",
        str(tmp_path / "table"),
        "--purity-ks",
        "4",
        "8",
    ])

    assert rc == 0
    tex = (tmp_path / "table_paper.tex").read_text()
    assert "Cosmos-Embed1" in tex
    assert "Qwen3-VL-8B" in tex
    assert "Random Gaussian" in tex
    assert "florence2_sigclip_idf" not in tex
    assert "Traffic Sign" in tex
    assert "%\\multirow{3}{*}{dog}" in tex
    assert "%\\multirow{3}{*}{Traffic Sign}" in tex
    assert tex.find("Cosmos-Embed1") < tex.find("Qwen3-VL-8B")
    assert tex.find("Qwen3-VL-8B") < tex.find("Random Gaussian")

    with open(tmp_path / "table.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    avg_cosmos = [
        row
        for row in rows
        if row["embedding_key"] == "cosmos" and row["label"] == "Avg"
    ][0]
    assert float(avg_cosmos["nn_purity_k1"]) == 0.60
    assert float(avg_cosmos["few_shot_n5"]) == 0.64
    assert avg_cosmos["commented"] == "False"
