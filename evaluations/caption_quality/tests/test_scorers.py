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

import scorers


def test_base_metrics_registered():
    available = scorers.available_metrics()
    for name in ("nlg", "bertscore", "lingojudge", "llm_judge", "vlm_judge"):
        assert name in available


def test_ref_free_metrics_constant():
    assert scorers.REF_FREE_METRICS == frozenset({"vlm_judge", "evqa"})


def test_build_scorer_unknown_raises():
    with pytest.raises(ValueError, match="Unknown metric"):
        scorers.build_scorer("does_not_exist")


def test_evqa_only_listed_when_available():
    # evqa is gated on ultralytics being importable.
    listed = "evqa" in scorers.available_metrics()
    assert listed == scorers._evqa_available()


def test_nlg_scores_identical_captions_high():
    # Skip cleanly when the [caption-quality] extra is not installed.
    pytest.importorskip("pycocoevalcap")
    pytest.importorskip("rouge_score")
    pytest.importorskip("nltk")
    scorer = scorers.build_scorer("nlg")
    pairs = [
        {"clip_id": "a", "data_source": "s", "reference": "a car turns left at the junction",
         "prediction": "a car turns left at the junction"},
    ]
    out = scorer.score_batch(pairs)
    assert len(out) == 1
    # Identical strings: ROUGE-L F and METEOR should be near 1.
    assert out[0]["rougeL_f"] == pytest.approx(1.0, abs=1e-6)
    assert out[0]["meteor"] > 0.9
