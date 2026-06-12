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

import topic_lexicon as tl


def test_categorize_known_signature_words():
    assert tl.categorize("snowy") == tl.APPEAR
    assert tl.categorize("overcast") == tl.APPEAR
    assert tl.categorize("roundabout") == tl.ACTIVITY
    assert tl.categorize("pedestrians") == tl.ACTIVITY
    assert tl.categorize("truck") == tl.OBJECT
    assert tl.categorize("parked") == tl.OBJECT


def test_categorize_shared_words_are_neutral():
    # Words common to all three embeddings must not be coloured, or the columns
    # stop looking distinct.
    for w in ("intersection", "highway", "street", "residential", "night"):
        assert tl.categorize(w) == tl.NEUTRAL


def test_categorize_strips_case_and_punctuation():
    assert tl.categorize("Snowy,") == tl.APPEAR
    assert tl.categorize("  TRUCK.") == tl.OBJECT
    # hyphen is preserved (tree-lined is a signature OBJECT word)
    assert tl.categorize("tree-lined") == tl.OBJECT


def test_categorize_unknown_is_neutral():
    assert tl.categorize("zphlbx") == tl.NEUTRAL
    assert tl.categorize("") == tl.NEUTRAL


def test_latex_escape():
    assert tl.latex_escape("a_b & c% #d") == r"a\_b \& c\% \#d"
    assert tl.latex_escape("plain") == "plain"


def test_categories_are_disjoint():
    sets = list(tl.CATEGORY_WORDS.values())
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not (sets[i] & sets[j]), "a word may belong to only one category"
