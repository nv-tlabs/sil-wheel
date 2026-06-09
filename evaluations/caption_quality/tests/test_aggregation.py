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

import pytest

from run_caption_eval import aggregate_scores, render_metric_table


def test_aggregate_groups_and_overall():
    per_clip = [
        {"clip_id": "a", "data_source": "x", "bleu4": 0.2},
        {"clip_id": "b", "data_source": "x", "bleu4": 0.4},
        {"clip_id": "c", "data_source": "y", "bleu4": 1.0},
    ]
    agg = aggregate_scores(per_clip, group_keys=["data_source"])

    # One row per group plus a synthetic "all" row over every clip.
    assert set(agg) == {"x", "y", "all"}
    assert agg["x"]["n"] == 2
    assert agg["x"]["bleu4"] == pytest.approx(0.3)
    assert agg["y"]["bleu4"] == pytest.approx(1.0)
    assert agg["all"]["n"] == 3
    assert agg["all"]["bleu4"] == pytest.approx(0.5333333, rel=1e-5)


def test_aggregate_drops_non_numeric_and_group_keys():
    per_clip = [
        {"clip_id": "a", "data_source": "x", "bleu4": 0.5, "llm_motivation": "looks good"},
        {"clip_id": "b", "data_source": "x", "bleu4": 0.5, "llm_motivation": "fine"},
    ]
    agg = aggregate_scores(per_clip, group_keys=["data_source"])
    # Strings (llm_motivation) and the group key itself are not averaged.
    assert "llm_motivation" not in agg["x"]
    assert "data_source" not in agg["x"]
    assert agg["x"]["bleu4"] == pytest.approx(0.5)


def test_aggregate_skips_missing_metric_in_some_rows():
    # bertscore present on one clip only; mean is taken over present values.
    per_clip = [
        {"clip_id": "a", "data_source": "x", "bert_f1": 0.8},
        {"clip_id": "b", "data_source": "x"},
    ]
    agg = aggregate_scores(per_clip, group_keys=["data_source"])
    assert agg["x"]["n"] == 2
    assert agg["x"]["bert_f1"] == pytest.approx(0.8)


def test_render_metric_table_orders_all_first_and_marks_missing():
    aggregated = {
        "y": {"n": 1.0, "bleu4": 0.1},
        "x": {"n": 2.0, "bleu4": 0.5, "meteor": 0.3},
        "all": {"n": 3.0, "bleu4": 0.3, "meteor": 0.3},
    }
    headers, rows = render_metric_table("nlg", aggregated)

    assert headers[:2] == ["group", "n"]
    assert set(headers[2:]) == {"bleu4", "meteor"}
    # "all" sorts first; remaining groups alphabetical.
    assert [r[0] for r in rows] == ["all", "x", "y"]
    # "y" has no meteor -> em dash placeholder, not a number.
    y_row = next(r for r in rows if r[0] == "y")
    assert "—" in y_row
