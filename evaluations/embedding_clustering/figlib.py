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

"""Shared helpers for the embedding-clustering figures.

Topic lexicon: the category colours + signature word sets used to colour
cluster-topic keywords (``categorize``), LaTeX escaping (``latex_escape``), and
the weighted log-odds that score how much a term concentrates in one embedding's
clusters relative to the others (``topic_profiles`` / ``distinctive_terms``).

Cluster selection: the k most mutually-distinct clusters of a run
(``distinct_clusters`` via ``farthest_first``) and a label anchor over a 2D point
cloud (``dense_xy``). Plus ``use_nvidia_style`` for matplotlib font/palette.
"""

import math
import os
import re
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import font_manager

# Category colours (HTML hex), also the stable keys for distinctive-terms focus.
APPEAR = "#2166AC"  # appearance / lighting / weather
ACTIVITY = "#C2451E"  # activity / signals / agents
OBJECT = "#1B7837"  # objects / scenery / vehicles
NEUTRAL = "#7a7a7a"  # shared road/place vocabulary (uncoloured)

# Only each embedding's SIGNATURE vocabulary is coloured; words shared across all
# three (night, illuminated, intersection, street, ...) stay neutral so the
# distinctive focus of each column actually stands out.
CATEGORY_WORDS = {
    APPEAR: {
        "sunny",
        "sunlit",
        "daytime",
        "overcast",
        "cloudy",
        "cloud",
        "clouds",
        "wet",
        "rainy",
        "rain",
        "snow",
        "snowy",
        "dusk",
        "sunset",
        "twilight",
        "gray",
        "grey",
        "gloomy",
        "clear",
        "sky",
        "shadows",
        "shadow",
        "blue",
    },
    ACTIVITY: {
        "traffic",
        "light",
        "lights",
        "signal",
        "signals",
        "red",
        "roundabout",
        "roundabouts",
        "stop",
        "stopping",
        "stopped",
        "waiting",
        "yield",
        "yielding",
        "accelerating",
        "turning",
        "flowing",
        "changing",
        "crossing",
        "pedestrians",
        "cyclists",
    },
    OBJECT: {
        "parked",
        "cars",
        "trees",
        "tree-lined",
        "bare",
        "forest",
        "fields",
        "field",
        "concrete",
        "buildings",
        "building",
        "houses",
        "house",
        "village",
        "tunnel",
        "tunnels",
        "construction",
        "sedan",
        "suv",
        "suvs",
        "trucks",
        "truck",
        "semi",
        "vehicles",
        "pole",
        "poles",
    },
}
_WORD2COL = {w: c for c, ws in CATEGORY_WORDS.items() for w in ws}

# Generic / cross-cutting terms that win log-odds but are not on-theme: dropping
# them keeps the distinctive-terms columns crisp. Covers caption boilerplate
# (dashcam, ego, forward, ...) and the road/place words common to all embeddings.
DROP = {
    "the",
    "a",
    "an",
    "of",
    "with",
    "and",
    "to",
    "in",
    "on",
    "at",
    "is",
    "are",
    "under",
    "through",
    "lined",
    "quiet",
    "day",
    "road",
    "street",
    "drive",
    "driving",
    "scene",
    "video",
    "view",
    "dashcam",
    "first",
    "person",
    "perspective",
    "captures",
    "vehicle",
    "ego",
    "forward",
    "camera",
    "shows",
    "captured",
    "travels",
    "intersection",
    "overcast",
    "multi",
    "multi lane",
    "lane",
    "blue",
    "red",
    "modern",
    "sign",
    "signs",
    "car",
    "houses",
    "city",
    "urban",
    "highway",
    "two",
    "rural",
    "traffic",
    "lights",
    "light",
    "headlights",
    "taillights",
    "slick",
    "night",
    "nighttime",
}

# Defaults to a repo-relative ``fonts/`` dir (override with NVIDIA_SANS_DIR);
# falls back to DejaVu Sans when the dir is absent.
NVIDIA_SANS_DIR = Path(
    os.environ.get("NVIDIA_SANS_DIR", str(Path(__file__).resolve().parent / "fonts"))
)
FONT_FAMILY = "DejaVu Sans"


def categorize(word):
    """Map a keyword to its category colour, or NEUTRAL if it is not a signature
    word. Punctuation and case are stripped before lookup."""
    return _WORD2COL.get(re.sub(r"[^a-z-]", "", word.lower()), NEUTRAL)


def latex_escape(s):
    """Escape the LaTeX specials that occur in cluster themes."""
    return (
        s.replace("&", r"\&")
        .replace("_", r"\_")
        .replace("#", r"\#")
        .replace("%", r"\%")
    )


def topic_profiles(runs, drop=None):
    """Per-embedding term-presence profiles.

    ``runs`` maps embedding name -> ``cluster_topics.json`` ``topics`` dict. For
    each embedding, ``p_e(w)`` is the fraction of that embedding's clusters whose
    top-10 keywords contain term ``w`` (after dropping boilerplate). Returns
    ``{embedding: {term: p_e(term)}}``.
    """
    drop = DROP if drop is None else drop
    prof = {}
    for e, t in runs.items():
        c, n = Counter(), 0
        for v in t.values():
            kws = {k.lower().strip() for k in v.get("keywords", [])[:10]}
            kws = {
                w
                for w in kws
                if w and w not in drop and not all(tok in drop for tok in w.split())
            }
            if kws:
                n += 1
            for w in kws:
                c[w] += 1
        prof[e] = {w: cnt / max(n, 1) for w, cnt in c.items()}
    return prof


def distinctive_terms(prof, e, embs, topn, floor=0.06):
    """Top ``topn`` terms for embedding ``e`` by weighted log-odds against the
    other embeddings: ``s_e(w) = p_e(w) * log((p_e(w)+eps)/(p_bar(w)+eps))`` with
    ``eps = 1e-3``, where ``p_bar`` is the mean presence across the other
    embeddings. Terms below ``floor`` presence in ``e`` are skipped.
    """
    others = [o for o in embs if o != e]
    eps = 1e-3
    sc = []
    for w, p in prof[e].items():
        if p < floor:
            continue
        bg = sum(prof[o].get(w, 0.0) for o in others) / max(len(others), 1)
        sc.append((w, p * math.log((p + eps) / (bg + eps))))
    return [w for w, _ in sorted(sc, key=lambda x: -x[1])[:topn]]


def farthest_first(X, seed, k):
    """Row indices of ``k`` mutually-distant vectors of ``X`` by cosine distance,
    seeded with row ``seed`` and greedily adding the row whose minimum cosine
    distance to the chosen set is largest. ``X`` is L2-normalised internally.
    """
    X = np.asarray(X, dtype=float)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    chosen = [int(seed)]
    while len(chosen) < min(k, len(X)):
        mind = (1.0 - X @ X[chosen].T).min(axis=1)
        for i in chosen:
            mind[i] = -1.0
        chosen.append(int(np.argmax(mind)))
    return chosen


def distinct_clusters(run_dir, sizes, k, min_frac):
    """The ``k`` most mutually-distinct clusters of a run: farthest-first (cosine)
    over the run's L2-normalised centroids, restricted to clusters at least
    ``min_frac`` of the mean size so tiny outliers don't dominate, seeded with the
    largest. ``sizes`` is a pandas Series indexed by cluster id; returns ids.
    """
    cents = np.load(Path(run_dir) / "centroids.npy")
    ids = sorted(int(c) for c in sizes.index)
    mean_sz = float(sizes.mean())
    cand = [c for c in ids if sizes[c] >= min_frac * mean_sz] or ids
    seed = max(range(len(cand)), key=lambda i: sizes[cand[i]])
    chosen = farthest_first(cents[cand].astype(float), seed, k)
    return [cand[i] for i in chosen]


def dense_xy(xs, ys, bins=20):
    """Anchor (x, y) for a label over a 2D point cloud: the mean of the points in
    the densest 2D-histogram bin (median fallback for tiny clouds)."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if len(xs) < 5:
        return float(np.median(xs)), float(np.median(ys))
    H, xe, ye = np.histogram2d(xs, ys, bins=bins)
    bx, by = np.unravel_index(int(np.argmax(H)), H.shape)
    m = (xs >= xe[bx]) & (xs <= xe[bx + 1]) & (ys >= ye[by]) & (ys <= ye[by + 1])
    if m.any():
        return float(xs[m].mean()), float(ys[m].mean())
    return float(np.median(xs)), float(np.median(ys))


def use_nvidia_style():
    """Register every NVIDIA Sans TTF and set it as the matplotlib default font.

    Call once at the top of a figure routine. Returns the resolved family name
    (e.g. for plotly ``font=dict(family=...)``). Overridable via NVIDIA_SANS_DIR.
    """
    global FONT_FAMILY
    if not NVIDIA_SANS_DIR.is_dir():
        return FONT_FAMILY
    names = set()
    for ttf in sorted(NVIDIA_SANS_DIR.glob("*.ttf")):
        try:
            font_manager.fontManager.addfont(str(ttf))
            names.add(font_manager.FontProperties(fname=str(ttf)).get_name())
        except Exception:
            pass
    if names:
        FONT_FAMILY = "NVIDIA Sans" if "NVIDIA Sans" in names else sorted(names)[0]
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [FONT_FAMILY, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    return FONT_FAMILY
