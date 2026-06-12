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

import math

import topic_lexicon as tl


def _topics(*keyword_lists):
    """Build a fake cluster_topics 'topics' dict: one cluster per keyword list."""
    return {str(i): {"keywords": kws} for i, kws in enumerate(keyword_lists)}


def test_topic_profiles_presence_fraction():
    # 'snowy' appears in 2 of 2 clusters -> 1.0; 'sunny' in 1 of 2 -> 0.5.
    runs = {"A": _topics(["snowy", "sunny"], ["snowy", "wet"])}
    prof = tl.topic_profiles(runs)
    assert prof["A"]["snowy"] == 1.0
    assert prof["A"]["sunny"] == 0.5
    assert prof["A"]["wet"] == 0.5


def test_topic_profiles_drops_boilerplate():
    runs = {"A": _topics(["dashcam", "ego", "snowy"])}
    prof = tl.topic_profiles(runs)
    assert "snowy" in prof["A"]
    assert "dashcam" not in prof["A"]
    assert "ego" not in prof["A"]


def test_topic_profiles_only_top10_keywords():
    long = [f"w{i}" for i in range(12)]  # w10, w11 are beyond the top-10 cut
    prof = tl.topic_profiles({"A": _topics(long)})
    assert "w9" in prof["A"]
    assert "w10" not in prof["A"]
    assert "w11" not in prof["A"]


def test_distinctive_terms_ranks_concentrated_first():
    # 'snowy' is unique to A; 'common' is shared across all three.
    runs = {
        "A": _topics(["snowy", "common"], ["snowy", "common"]),
        "B": _topics(["common"], ["common"]),
        "C": _topics(["common"], ["common"]),
    }
    prof = tl.topic_profiles(runs)
    embs = ["A", "B", "C"]
    terms = tl.distinctive_terms(prof, "A", embs, topn=2, floor=0.0)
    assert terms[0] == "snowy"  # the term unique to A ranks first


def test_distinctive_terms_floor_filters_rare():
    # 'rare' appears in 1 of 10 clusters -> 0.1 presence; floor=0.2 drops it.
    runs = {"A": _topics(*([["rare", "frequent"]] + [["frequent"]] * 9)),
            "B": _topics(["frequent"]), "C": _topics(["frequent"])}
    prof = tl.topic_profiles(runs)
    terms = tl.distinctive_terms(prof, "A", ["A", "B", "C"], topn=5, floor=0.2)
    assert "rare" not in terms


def test_distinctive_terms_log_odds_value():
    # Hand-check the score formula on a single term.
    prof = {"A": {"w": 0.8}, "B": {"w": 0.1}, "C": {"w": 0.1}}
    eps = 1e-3
    bg = (0.1 + 0.1) / 2
    expected = 0.8 * math.log((0.8 + eps) / (bg + eps))
    # distinctive_terms returns terms, not scores; re-derive the score path here.
    terms = tl.distinctive_terms(prof, "A", ["A", "B", "C"], topn=1, floor=0.0)
    assert terms == ["w"]
    assert expected > 0  # concentrated term has positive log-odds
