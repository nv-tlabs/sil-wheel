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

"""Shared topic lexicon and distinctive-terms scoring for the §4.4 figures.

This is the single source of truth for two things the cluster-topic figures share:

1. The category lexicon used to colour keywords in cluster themes
   (``categorize`` -> one of four hex colours: appearance / activity / objects /
   neutral). Only each embedding's *signature* vocabulary is coloured; words that
   show up across all three embeddings (night, intersection, highway, ...) stay
   neutral so the distinctive focus of each column actually stands out.

2. The weighted log-odds that score how much a term concentrates in one
   embedding's clusters relative to the other two (``topic_profiles`` +
   ``distinctive_terms``), the discriminator behind ``emb_distinctive_terms.tex``.

``make_taxonomy_compare.py`` keeps its own (now superseded) copy of the lexicon;
this module is the version the active figure generators import.
"""
from __future__ import annotations

import math
import re
from collections import Counter

# --- Category colours (HTML hex). Used both for keyword colouring and as the
# --- stable category keys for distinctive-terms focus labels.
APPEAR = "#2166AC"    # appearance / lighting / weather
ACTIVITY = "#C2451E"  # activity / signals / agents
OBJECT = "#1B7837"    # objects / scenery / vehicles
NEUTRAL = "#7a7a7a"   # shared road/place vocabulary (uncoloured)

# Only each embedding's SIGNATURE vocabulary is coloured; words shared across all
# three (night, illuminated, intersection, street, highway, residential, ...) stay
# neutral so the distinctive focus of each column actually stands out.
CATEGORY_WORDS = {
    APPEAR: {"sunny", "sunlit", "daytime", "overcast", "cloudy", "cloud", "clouds",
             "wet", "rainy", "rain", "snow", "snowy", "dusk", "sunset", "twilight",
             "gray", "grey", "gloomy", "clear", "sky", "shadows", "shadow", "blue"},
    ACTIVITY: {"traffic", "light", "lights", "signal", "signals", "red", "roundabout",
               "roundabouts", "stop", "stopping", "stopped", "waiting", "yield",
               "yielding", "accelerating", "turning", "flowing", "changing", "crossing",
               "pedestrians", "cyclists"},
    OBJECT: {"parked", "cars", "trees", "tree-lined", "bare", "forest", "fields",
             "field", "concrete", "buildings", "building", "houses", "house", "village",
             "tunnel", "tunnels", "construction", "sedan", "suv", "suvs", "trucks",
             "truck", "semi", "vehicles", "pole", "poles"},
}
_WORD2COL = {w: c for c, ws in CATEGORY_WORDS.items() for w in ws}

# Generic / cross-cutting terms that win log-odds but are not on-theme: dropping
# them keeps the distinctive-terms columns crisp. Covers caption boilerplate
# (dashcam, ego, forward, ...) and the road/place words common to all embeddings.
DROP = {"the", "a", "an", "of", "with", "and", "to", "in", "on", "at", "is", "are", "under",
        "through", "lined", "quiet", "day", "road", "street", "drive", "driving", "scene",
        "video", "view", "dashcam", "first", "person", "perspective", "captures", "vehicle",
        "ego", "forward", "camera", "shows", "captured", "travels",
        "intersection", "overcast", "multi", "multi lane", "lane", "blue", "red", "modern",
        "sign", "signs", "car", "houses", "city", "urban", "highway", "two", "rural",
        "traffic", "lights", "light", "headlights", "taillights", "slick", "night", "nighttime"}


def categorize(word: str) -> str:
    """Map a keyword to its category colour, or ``NEUTRAL`` if it is not a
    signature word. Punctuation and case are stripped before lookup."""
    return _WORD2COL.get(re.sub(r"[^a-z-]", "", word.lower()), NEUTRAL)


def latex_escape(s: str) -> str:
    """Escape the LaTeX specials that occur in cluster themes."""
    return s.replace("&", r"\&").replace("_", r"\_").replace("#", r"\#").replace("%", r"\%")


def topic_profiles(runs: dict, drop: set | None = None) -> dict:
    """Per-embedding term-presence profiles.

    ``runs`` maps embedding name -> ``cluster_topics.json`` ``topics`` dict. For
    each embedding, ``p_e(w)`` is the fraction of that embedding's clusters whose
    top-10 keywords contain term ``w`` (after dropping boilerplate). Returns
    ``{embedding: {term: p_e(term)}}``."""
    drop = DROP if drop is None else drop
    prof = {}
    for e, t in runs.items():
        c, n = Counter(), 0
        for v in t.values():
            kws = {k.lower().strip() for k in v.get("keywords", [])[:10]}
            kws = {w for w in kws if w and w not in drop
                   and not all(tok in drop for tok in w.split())}
            if kws:
                n += 1
            for w in kws:
                c[w] += 1
        prof[e] = {w: cnt / max(n, 1) for w, cnt in c.items()}
    return prof


def distinctive_terms(prof: dict, e: str, embs: list, topn: int, floor: float = 0.06) -> list:
    """Top ``topn`` terms for embedding ``e`` by weighted log-odds against the
    other embeddings: ``s_e(w) = p_e(w) * log((p_e(w)+eps) / (p_bar_{!e}(w)+eps))``
    with ``eps = 1e-3``, where ``p_bar_{!e}`` is the mean presence across the other
    embeddings. Terms below ``floor`` presence in ``e`` are skipped."""
    others = [o for o in embs if o != e]
    eps = 1e-3
    sc = []
    for w, p in prof[e].items():
        if p < floor:
            continue
        bg = sum(prof[o].get(w, 0.0) for o in others) / max(len(others), 1)
        sc.append((w, p * math.log((p + eps) / (bg + eps))))
    return [w for w, _ in sorted(sc, key=lambda x: -x[1])[:topn]]
