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

import numpy as np
import pandas as pd

import figlib


# ---------------------------------------------------------------- lexicon


def test_categorize_known_signature_words():
    assert figlib.categorize("snowy") == figlib.APPEAR
    assert figlib.categorize("overcast") == figlib.APPEAR
    assert figlib.categorize("roundabout") == figlib.ACTIVITY
    assert figlib.categorize("pedestrians") == figlib.ACTIVITY
    assert figlib.categorize("truck") == figlib.OBJECT
    assert figlib.categorize("parked") == figlib.OBJECT


def test_categorize_shared_words_are_neutral():
    # Words common to all three embeddings must not be coloured.
    for w in ("intersection", "highway", "street", "residential", "night"):
        assert figlib.categorize(w) == figlib.NEUTRAL


def test_categorize_strips_case_and_punctuation():
    assert figlib.categorize("Snowy,") == figlib.APPEAR
    assert figlib.categorize("  TRUCK.") == figlib.OBJECT
    assert figlib.categorize("tree-lined") == figlib.OBJECT  # hyphen preserved


def test_categorize_unknown_is_neutral():
    assert figlib.categorize("zphlbx") == figlib.NEUTRAL
    assert figlib.categorize("") == figlib.NEUTRAL


def test_latex_escape():
    assert figlib.latex_escape("a_b & c% #d") == r"a\_b \& c\% \#d"
    assert figlib.latex_escape("plain") == "plain"


def test_categories_are_disjoint():
    sets = list(figlib.CATEGORY_WORDS.values())
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not (sets[i] & sets[j]), "a word may belong to only one category"


# ----------------------------------------------------- distinctive terms


def _topics(*keyword_lists):
    """One cluster per keyword list, as a cluster_topics 'topics' dict."""
    return {str(i): {"keywords": kws} for i, kws in enumerate(keyword_lists)}


def test_topic_profiles_presence_fraction():
    runs = {"A": _topics(["snowy", "sunny"], ["snowy", "wet"])}
    prof = figlib.topic_profiles(runs)
    assert prof["A"]["snowy"] == 1.0
    assert prof["A"]["sunny"] == 0.5
    assert prof["A"]["wet"] == 0.5


def test_topic_profiles_drops_boilerplate():
    prof = figlib.topic_profiles({"A": _topics(["dashcam", "ego", "snowy"])})
    assert "snowy" in prof["A"]
    assert "dashcam" not in prof["A"]
    assert "ego" not in prof["A"]


def test_topic_profiles_only_top10_keywords():
    long = [f"w{i}" for i in range(12)]
    prof = figlib.topic_profiles({"A": _topics(long)})
    assert "w9" in prof["A"]
    assert "w10" not in prof["A"]


def test_distinctive_terms_ranks_concentrated_first():
    runs = {
        "A": _topics(["snowy", "common"], ["snowy", "common"]),
        "B": _topics(["common"], ["common"]),
        "C": _topics(["common"], ["common"]),
    }
    prof = figlib.topic_profiles(runs)
    terms = figlib.distinctive_terms(prof, "A", ["A", "B", "C"], topn=2, floor=0.0)
    assert terms[0] == "snowy"


def test_distinctive_terms_floor_filters_rare():
    runs = {
        "A": _topics(*([["rare", "frequent"]] + [["frequent"]] * 9)),
        "B": _topics(["frequent"]),
        "C": _topics(["frequent"]),
    }
    prof = figlib.topic_profiles(runs)
    terms = figlib.distinctive_terms(prof, "A", ["A", "B", "C"], topn=5, floor=0.2)
    assert "rare" not in terms


def test_distinctive_terms_log_odds_positive_for_concentrated():
    prof = {"A": {"w": 0.8}, "B": {"w": 0.1}, "C": {"w": 0.1}}
    bg = (0.1 + 0.1) / 2
    assert 0.8 * math.log((0.8 + 1e-3) / (bg + 1e-3)) > 0
    assert figlib.distinctive_terms(prof, "A", ["A", "B", "C"], topn=1, floor=0.0) == [
        "w"
    ]


# --------------------------------------------------- cluster selection


def test_farthest_first_picks_opposite_then_orthogonal():
    X = np.array([[1, 0], [0.999, 0.001], [-1, 0], [0, 1]], float)
    chosen = figlib.farthest_first(X, seed=0, k=3)
    assert chosen[0] == 0
    assert chosen[1] == 2  # -x is the single farthest point
    assert 1 not in chosen[:3]  # the +x near-duplicate is never preferred


def test_farthest_first_caps_at_n_points():
    assert len(figlib.farthest_first(np.eye(3), seed=0, k=10)) == 3


def test_farthest_first_no_repeats():
    X = np.random.default_rng(0).standard_normal((20, 8))
    chosen = figlib.farthest_first(X, seed=3, k=6)
    assert len(chosen) == len(set(chosen)) == 6


def test_distinct_clusters_uses_centroids_and_size_floor(tmp_path):
    cents = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], float)
    np.save(tmp_path / "centroids.npy", cents)
    sizes = pd.Series({0: 100, 1: 90, 2: 80, 3: 1})  # cluster 3 << mean
    out = figlib.distinct_clusters(tmp_path, sizes, k=3, min_frac=0.5)
    assert 3 not in out
    assert out[0] == 0  # seeded with the largest cluster
    assert set(out) <= {0, 1, 2} and len(out) == 3


def test_distinct_clusters_falls_back_when_all_filtered(tmp_path):
    np.save(tmp_path / "centroids.npy", np.array([[1, 0], [-1, 0]], float))
    sizes = pd.Series({0: 1, 1: 1})
    assert sorted(figlib.distinct_clusters(tmp_path, sizes, k=2, min_frac=10.0)) == [
        0,
        1,
    ]


def test_dense_xy_finds_the_cluster_not_the_outlier():
    x, y = figlib.dense_xy([0.0] * 100 + [50.0], [0.0] * 100 + [50.0])
    assert abs(x) < 1.0 and abs(y) < 1.0


def test_dense_xy_small_cloud_uses_median():
    x, y = figlib.dense_xy([1, 2, 3], [10, 20, 30])
    assert x == 2.0 and y == 20.0
