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

"""Extract per-cluster TF-IDF keywords from captions.

Topics are computed once at clustering time (by ``cluster_clips_and_select.py``
when ``--captions_db`` is provided) and persisted to ``cluster_topics.json``
inside the run directory. The server only ever reads that file.

Each cluster entry has shape::

    {"keywords": ["...", ...],   # top TF-IDF terms, ordered by score
     "description": "..."}       # one-phrase LLM summary

The ``description`` is produced by an LLM call against the keyword list and is
omitted for a cluster if the call fails or no API key is configured.
"""

import argparse
import json
import math
import os
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_CAPTION_MODEL = "Qwen2.5-7B (yotta)"

# Generic AV-domain filler that adds noise without distinguishing clusters.
# Applied as a post-TF-IDF filter on the top-N terms.
_AV_STOPWORDS = {
    "student", "students", "shot", "shots", "speed", "position",
    "positioned", "trajectory", "direction", "side", "sides", "current",
    "visible", "clearly", "clear", "video", "frame", "frames", "initial",
    "throughout", "during", "while", "moving", "movement", "move", "moves", "close", "closer", "far",
    "further", "section", "area", "areas", "moment", "time", "times",
    "point", "points", "way", "ways", "around", "through", "along",
    "across", "past", "towards", "away", "observation", "observations",
    "behavior", "ensuring", "ensure", "ensures",
    "control", "controlled", "controls", "scenario", "situation",
    "conditions", "condition", "task", "structure", "context",
    "reasoning", "none", "type",
    # Vehicle makes / brand emblems: caption-verbosity noise, not scene content.
    "mercedes", "benz", "bmw", "audi", "toyota", "honda", "ford", "chevrolet",
    "chevy", "nissan", "hyundai", "kia", "tesla", "jeep", "volkswagen", "vw",
    "subaru", "lexus", "mazda", "dodge", "ram", "gmc", "buick", "cadillac",
    "volvo", "porsche", "ferrari", "lamborghini", "bentley", "jaguar", "acura",
    "infiniti", "chrysler", "mitsubishi", "peugeot", "renault", "fiat",
    "logo", "emblem", "badge", "indicated", "brand",
    # caption-verb scaffolding that forms noise n-grams
    "shows", "show", "depicts", "depict", "showing", "depicting",
    "showcasing", "showcase", "featuring", "captures", "capturing", "captured",
    # Cosmos-Reason caption-template hedging / boilerplate
    "seconds", "second", "action", "actions", "suggesting", "suggests",
    "suggest", "appears", "appear", "indicating", "indicates", "reason",
    "reasons", "depicting", "ego",
    # qwen3.5 caption viewpoint / narration boilerplate
    "perspective", "dashcam", "person", "first", "footage", "viewpoint",
    "travels", "forward", "scene", "video", "captures",
}

# Bare number words are noise on their own ("two") but meaningful inside an
# n-gram ("two lane"), so they are dropped only as a WHOLE term, not per-token.
_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "several", "multiple",
}


def _preprocess_caption(text):
    text = text.lower()
    text = re.sub(r"[#*\-\d]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


# Light suffix stemmer for keyword dedup. Collapses common English plural /
# verb-inflection variants ("pedestrian"/"pedestrians", "merge"/"merging")
# without pulling in NLTK. Order matters: try longest suffixes first so
# "ing"/"ies" win over "s".
_INFLECTION_SUFFIXES = (
    ("ies", "y"),
    ("ing", ""),
    ("ed", ""),
    ("es", ""),
    ("s", ""),
)


def _stem_token(tok):
    for suf, repl in _INFLECTION_SUFFIXES:
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            return tok[: -len(suf)] + repl
    return tok


def _canonical_key(term):
    """Order-independent canonical form for an n-gram. Stems each token and
    sorts so "lane changing"/"changing lane" and "pedestrian"/"pedestrians"
    collapse to the same key."""
    return tuple(sorted(_stem_token(t) for t in term.split()))


# SQLite's compiled-in default for SQLITE_MAX_VARIABLE_NUMBER is 999 on older
# builds and 32766 on newer ones. Stay well under the conservative limit.
_BATCH_IN = 800


def pick_highest_coverage_captions(db_path, clip_ids):
    """Return ``(model_name, distinct_clip_count)`` for the caption model
    that covers the most clips in ``clip_ids``. Returns ``(None, 0)`` if no
    captions exist for any clip.

    Counts distinct clip_ids per model_name across SQL batches. Batches are
    disjoint slices of the deduplicated clip_ids list, so per-batch distinct
    counts add up to the total distinct count per model.
    """
    if not clip_ids:
        return None, 0
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        per_model = {}
        for i in range(0, len(clip_ids), _BATCH_IN):
            ids = clip_ids[i : i + _BATCH_IN]
            placeholders = ",".join("?" * len(ids))
            rows = conn.execute(
                "SELECT model_name, COUNT(DISTINCT clip_id) "
                f"FROM captions WHERE clip_id IN ({placeholders}) "
                "GROUP BY model_name",
                ids,
            ).fetchall()
            for model_name, n in rows:
                per_model[model_name] = per_model.get(model_name, 0) + int(n)
    finally:
        conn.close()
    if not per_model:
        return None, 0
    return max(per_model.items(), key=lambda kv: kv[1])


def _fetch_chunk(db_path, clip_ids, model_name):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    out = {}
    try:
        for i in range(0, len(clip_ids), _BATCH_IN):
            ids = clip_ids[i : i + _BATCH_IN]
            placeholders = ",".join("?" * len(ids))
            sql = (
                "SELECT clip_id, caption FROM captions "
                f"WHERE clip_id IN ({placeholders}) AND model_name = ?"
            )
            rows = conn.execute(sql, (*ids, model_name)).fetchall()
            for r in rows:
                cid = r["clip_id"]
                cap = r["caption"]
                if not cap:
                    continue
                if cid in out:
                    out[cid] = out[cid] + " " + cap
                else:
                    out[cid] = cap
    finally:
        conn.close()
    return out


def _fetch_captions(db_path, clip_ids, model_name, n_threads):
    """Return dict clip_id -> concatenated caption text. Threaded over chunks
    of clip_ids; each thread uses its own SQLite connection (WAL mode allows
    concurrent readers)."""
    clip_ids = list(set(str(c) for c in clip_ids))
    if not clip_ids:
        return {}
    if n_threads <= 1 or len(clip_ids) < 2 * _BATCH_IN:
        return _fetch_chunk(db_path, clip_ids, model_name)

    chunk_size = max(_BATCH_IN, math.ceil(len(clip_ids) / n_threads))
    chunks = [clip_ids[i : i + chunk_size] for i in range(0, len(clip_ids), chunk_size)]
    result = {}
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        for partial in pool.map(
            lambda c: _fetch_chunk(db_path, c, model_name), chunks
        ):
            result.update(partial)
    return result


# --------------------------------------------------------------------------
# TF-IDF
# --------------------------------------------------------------------------

def _english_stopwords():
    try:
        import nltk
        nltk.data.find("corpora/stopwords")
        from nltk.corpus import stopwords as nltk_sw
        return set(nltk_sw.words("english"))
    except Exception:
        return set()


def _build_topics(cluster_clip_ids, clip_to_caption, keywords_top_k=15):
    """Run per-clip TF-IDF and aggregate per cluster. Returns dict
    ``cid -> {"keywords": [...]}``.

    Each clip caption is its own document. After vectorization we average
    the per-clip TF-IDF rows belonging to each cluster. This dampens the
    influence of generic words that appear once in every caption and lets
    less-frequent but still-distinctive terms surface.
    """
    import numpy as np

    base_stopwords = _english_stopwords()

    clip_docs = []
    clip_to_cid = []
    for cid in sorted(cluster_clip_ids.keys(), key=int):
        for clip_id in cluster_clip_ids[cid]:
            cap = clip_to_caption.get(clip_id)
            if not isinstance(cap, str):
                continue
            cleaned = _preprocess_caption(cap)
            if not cleaned:
                continue
            clip_docs.append(cleaned)
            clip_to_cid.append(cid)

    if len(clip_docs) < 2:
        return {}

    # New default params: stricter min_df/max_df make sense now that the
    # corpus is per-clip (50 × n_clusters docs, not n_clusters). The
    # small-run fallback below keeps tiny debug runs from tripping the
    # "max_df < min_df" sklearn error.
    n_docs = len(clip_docs)
    min_df = 3
    max_df = 0.6
    max_features = 20_000
    if n_docs < 10:
        # Way too few documents — relax everything so sklearn doesn't fail.
        min_df = 1
        max_df = 1.0
    else:
        # Make sure max_df_count ≥ min_df + 1, otherwise sklearn will raise.
        max_df = max(max_df, (min_df + 1) / n_docs)

    # `sublinear_tf=True` replaces raw TF with 1+log(TF), which damps
    # repetition within a single caption. The AV stopword filter is
    # applied later on the top-N terms.
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=list(base_stopwords) if base_stopwords else None,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),
        token_pattern=r"\b[a-zA-Z]{3,}\b",
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(clip_docs)
    feature_names = vectorizer.get_feature_names_out()

    # Aggregate per cluster: mean of the rows belonging to each cluster.
    # `dict.fromkeys` preserves insertion order so results are deterministic.
    clip_to_cid_arr = np.asarray(clip_to_cid)
    cluster_ids_ordered = list(dict.fromkeys(clip_to_cid))

    topics = {}
    for cid in cluster_ids_ordered:
        mask = clip_to_cid_arr == cid
        rows = tfidf_matrix[mask]
        if rows.shape[0] == 0:
            continue
        # `rows.mean(axis=0)` returns a 1×V numpy.matrix for sparse input;
        # convert to a flat ndarray for argsort.
        score = np.asarray(rows.mean(axis=0)).ravel()

        idx = score.argsort()[::-1]
        # Walk top terms in score order, dropping AV-jargon noise and
        # near-duplicate inflections, until we have ``keywords_top_k``
        # survivors. Highest-scoring variant wins because we walk in score
        # order and skip later canon-matches.
        keywords = []
        seen_canon = set()
        for j in idx:
            if score[j] <= 0:
                break
            term = feature_names[j]
            if term in _NUMBER_WORDS:
                continue
            if any(part in _AV_STOPWORDS for part in term.split()):
                continue
            canon = _canonical_key(term)
            if canon in seen_canon:
                continue
            seen_canon.add(canon)
            keywords.append(term)
            if len(keywords) >= keywords_top_k:
                break

        if keywords:
            topics[cid] = {"keywords": keywords}

    return topics


# --------------------------------------------------------------------------
# LLM summarization
# --------------------------------------------------------------------------

_SUMMARY_SYSTEM_PROMPT = (
    "You summarize topic keywords extracted from a cluster of video captions "
    "into one short noun phrase (4-10 words) describing the common scenario. "
    "Emphasize actions, interactions, or other interesting details. "
    "Output ONLY the phrase: no quotes, no leading 'A '/'The ', no trailing "
    "punctuation, no explanation."
)


def _summarize_keywords(keywords, client):
    """One LLM call: keywords -> short phrase. Returns ``""`` on failure.

    Robust to ``None`` content (some providers return that for filtered or
    empty completions) and any post-processing error, so one bad cluster
    can't drop the whole batch."""
    try:
        text = client.generate(
            prompt="Keywords: " + ", ".join(keywords),
            system_prompt=_SUMMARY_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=40,
        )
        if not text:
            return ""
        return text.strip().strip('"').strip("'").rstrip(".")
    except Exception as e:
        print(f"[topics] summary failed: {e}")
        return ""


def _summarize_topics(topics, n_threads=20):
    """Add a ``"description"`` field to each cluster in ``topics`` using one
    LLM call per cluster, in parallel. No-op if no API key is configured."""
    if not topics:
        return
    from sil_wheel.llm.llm_client import LLMClient
    client = LLMClient()
    if not client.config.api_key:
        print("[topics] no LLM API key set; skipping topic summaries")
        return
    cids = list(topics.keys())
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        descs = pool.map(lambda c: _summarize_keywords(topics[c]["keywords"], client), cids)
        for cid, desc in zip(cids, descs):
            if desc:
                topics[cid]["description"] = desc


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------

def read_topics(run_dir):
    """Return the full cluster_topics.json payload, or ``{}`` if missing.

    Shape::

        {
            "topics":          {cid: {"keywords": [...],
                                       "description": "..."}, ...},
            "caption_model":   "Qwen2.5-7B (yotta)",
            "captions_found":  35200,   # clips with >=1 caption row
            "captions_total":  36743,   # clips that went into the lookup
        }
    """
    p = Path(run_dir) / "cluster_topics.json"
    if not p.exists():
        return {}
    try:
        with open(p, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def extract_topics_for_run(run_dir, captions_db_path,
                           model_name=None, n_threads=8,
                           samples_per_cluster=50, sample_seed=42):
    """Compute and persist topics for a clustering run.

    Reads ``cluster_assignments.parquet`` for the run, fetches captions from
    the SQLite DB (in parallel), runs topic extraction and writes
    the result to ``cluster_topics.json``.

    ``model_name=None`` (default) auto-selects the caption model with the
    most coverage of the run's clip_ids. Pass an explicit string to pin a
    specific model. The chosen model and its coverage are persisted in the
    output JSON so the UI can label the run.

    ``samples_per_cluster`` controls how many clips per cluster feed into
    TF-IDF. Clips are drawn at **random** (with a fixed ``sample_seed`` for
    reproducibility). Set ``samples_per_cluster=0`` to use
    every clip.
    """
    run_dir = Path(run_dir)
    parquet_path = run_dir / "cluster_assignments.parquet"
    clusters_path = run_dir / "representative_by_cluster.json"
    if not parquet_path.exists() or not clusters_path.exists():
        return {}

    import pandas as pd

    df = pd.read_parquet(
        parquet_path, columns=["clip_id", "cluster_id", "distance"]
    )
    if samples_per_cluster and samples_per_cluster > 0:
        # Random sample up to N clips per cluster (smaller clusters: take
        # everything). `random_state` is fixed so re-running on the same
        # parquet yields the same topics. We collect sampled indices in a
        # plain loop instead of `.apply(lambda g: g.sample(...))` because
        # pandas 2.2+ excludes the grouping column from the apply result
        # by default.
        sampled_idx = []
        for _, group in df.groupby("cluster_id", sort=False):
            n = min(len(group), samples_per_cluster)
            sampled_idx.extend(
                group.sample(n=n, random_state=sample_seed).index.tolist()
            )
        df = df.loc[sampled_idx]

    cluster_clip_ids = {}
    for cid_int, group in df.groupby("cluster_id"):
        cluster_clip_ids[str(cid_int)] = group["clip_id"].astype(str).tolist()
    all_clip_ids = df["clip_id"].astype(str).tolist()
    del df

    if not cluster_clip_ids:
        return {}

    # Dedupe up front so the coverage count and the fetch see the same set.
    unique_clip_ids = list(set(all_clip_ids))

    if model_name is None:
        model_name, _ = pick_highest_coverage_captions(
            captions_db_path, unique_clip_ids
        )
        if model_name is None:
            # No captions for any clip under any model. Persist an empty
            # marker so the server stops polling.
            try:
                with open(run_dir / "cluster_topics.json", "w") as f:
                    json.dump({
                        "topics": {},
                        "caption_model": None,
                        "captions_found": 0,
                        "captions_total": len(unique_clip_ids),
                    }, f)
            except OSError:
                pass
            return {}
        print(f"[topics] auto-selected caption model: {model_name!r}")

    clip_to_caption = _fetch_captions(
        captions_db_path, unique_clip_ids, model_name, n_threads
    )

    payload = {
        "topics": {},
        "caption_model": model_name,
        "captions_found": len(clip_to_caption),
        "captions_total": len(unique_clip_ids),
    }
    if clip_to_caption:
        payload["topics"] = _build_topics(cluster_clip_ids, clip_to_caption)
        # Summarization is best-effort: never block the keyword payload on
        # it. Per-cluster failures are caught inside _summarize_keywords;
        # this outer guard catches orchestration breakage (thread pool,
        # import error, etc.).
        try:
            _summarize_topics(payload["topics"])
        except Exception as e:
            print(f"[topics] summarization batch failed: {e}")

    try:
        with open(run_dir / "cluster_topics.json", "w") as f:
            json.dump(payload, f)
    except OSError:
        pass

    return payload["topics"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("captions_db")
    parser.add_argument(
        "--model_name", default=None,
        help="Caption model name (exact match against captions.model_name). "
             "If omitted, auto-selects the model with the most coverage of "
             "the run's clips.",
    )
    parser.add_argument("--n_threads", type=int,
                        default=min(8, (os.cpu_count() or 4)))
    parser.add_argument(
        "--samples_per_cluster", type=int, default=50,
        help="Random-sample N clips per cluster for topic extraction. "
             "0 disables sampling (use every clip).",
    )
    parser.add_argument(
        "--sample_seed", type=int, default=42,
        help="Random seed for per-cluster clip sampling.",
    )
    args = parser.parse_args(argv)
    extract_topics_for_run(
        args.run_dir,
        args.captions_db,
        model_name=args.model_name,
        n_threads=args.n_threads,
        samples_per_cluster=args.samples_per_cluster,
        sample_seed=args.sample_seed,
    )


if __name__ == "__main__":
    main()
