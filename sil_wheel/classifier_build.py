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
"""Build a classifier run from positive/negative samples."""
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import orjson
from sklearn.linear_model import LogisticRegression

from sil_wheel.cluster_build import generate_run_id


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x, dtype=np.float32))


def train_logistic_regression(X_pos, X_neg, C=100, max_iter=100000):
    """Fit a sklearn LogisticRegression on stacked positive/negative features.

    Returns the fitted model. Caller is responsible for persisting it.
    """
    X = np.vstack([
        np.asarray(X_pos, dtype=np.float32),
        np.asarray(X_neg, dtype=np.float32),
    ])
    y = np.concatenate([
        np.ones(len(X_pos), dtype=np.float32),
        np.zeros(len(X_neg), dtype=np.float32),
    ])
    model = LogisticRegression(n_jobs=-1, tol=1e-3, max_iter=max_iter, C=C)
    model.fit(X, y)
    return model


def score_corpus(model, items, features_index, chunk_size=50_000):
    """Score every clip in ``items`` (``[(clip_id, faiss_row), ...]``) in chunks.

    Returns ``(clip_ids: list[str], probs: np.ndarray[float32])`` aligned by
    index. Stays in float32 throughout to keep peak memory low on
    million-clip corpora.
    """
    coef = model.coef_.astype(np.float32).ravel()
    intercept = float(model.intercept_.astype(np.float32).ravel()[0])

    n = len(items)
    probs = np.empty(n, dtype=np.float32)
    clip_ids = [str(items[i][0]) for i in range(n)]
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        idxs = [idx for _, idx in items[i:end]]
        vecs = features_index.reconstruct_batch(idxs)
        Xc = np.asarray(vecs, dtype=np.float32, order="C")
        logits = Xc @ coef + intercept
        probs[i:end] = _sigmoid(logits)
    return clip_ids, probs


def select_top_scores(clip_ids, probs, save_threshold=0.3, max_clips=7_000_000):
    """Filter to ``probs > save_threshold`` and trim to ``max_clips`` by score.

    Returns ``{clip_id: float_score}`` ordered arbitrarily.
    """
    candidates = np.where(probs > save_threshold)[0]
    if max_clips > 0 and len(candidates) > max_clips:
        scores = probs[candidates]
        top = np.argpartition(scores, -max_clips)[-max_clips:]
        candidates = candidates[top]
    return {clip_ids[i]: float(probs[i]) for i in candidates}


def write_metadata(
    run_dir,
    run_id,
    embed_type,
    positive_labels,
    negative_labels,
    trained_by,
    n_positive_clips,
    n_negative_clips,
    use_autolabels=False,
    save_threshold=0.3,
    max_clips=7_000_000,
    C=100,
    search_params=None,
):
    """Write/merge metadata.json with status='done'.

    Mirrors ``cluster_build.write_metadata`` so a server pre-stamping the
    run dir (e.g. when launching from the UI) doesn't lose its fields.
    """
    run_dir = Path(run_dir)
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}
    metadata.setdefault("started_at", time.time())
    metadata.setdefault("run_id", run_id)
    metadata.setdefault("embed_type", embed_type)
    metadata.setdefault("positive_labels", list(positive_labels))
    metadata.setdefault("negative_labels", list(negative_labels))
    metadata.setdefault("trained_by", trained_by)
    metadata.setdefault("use_autolabels", bool(use_autolabels))
    metadata.setdefault("save_threshold", float(save_threshold))
    metadata.setdefault("max_clips", int(max_clips))
    metadata.setdefault("C", int(C))
    metadata.setdefault("search_params", search_params)
    metadata["n_positive_clips"] = int(n_positive_clips)
    metadata["n_negative_clips"] = int(n_negative_clips)
    metadata["status"] = "done"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)


def write_clip_lists(run_dir, positive_clips, negative_clips):
    run_dir = Path(run_dir)
    with open(run_dir / "positive_clips.json", "wb") as f:
        f.write(orjson.dumps(list(positive_clips)))
    with open(run_dir / "negative_clips.json", "wb") as f:
        f.write(orjson.dumps(list(negative_clips)))


def write_predicted_scores(run_dir, scores):
    with open(Path(run_dir) / "predicted_scores.json", "wb") as f:
        f.write(orjson.dumps(scores))


def write_lr_weights(run_dir, model):
    with open(Path(run_dir) / "LR_weights.pkl", "wb") as f:
        pickle.dump(model, f)


def validate_run_dir(run_dir) -> None:
    """Verify ``run_dir`` is a valid classifier run.

    Raises ``ValueError`` with a specific message on the first failed check.
    Used by ``/upload_classifier`` to reject malformed uploads before they
    are renamed into the live classifier directory.
    """
    run_dir = Path(run_dir)

    weights_path = run_dir / "LR_weights.pkl"
    if not weights_path.exists():
        raise ValueError("missing LR_weights.pkl")
    try:
        with open(weights_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        raise ValueError(f"LR_weights.pkl is not a valid pickle: {e}") from e
    if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
        raise ValueError("LR_weights.pkl does not look like a sklearn LR model")

    scores_path = run_dir / "predicted_scores.json"
    if not scores_path.exists():
        raise ValueError("missing predicted_scores.json")
    try:
        with open(scores_path, "rb") as f:
            scores = orjson.loads(f.read())
    except Exception as e:
        raise ValueError(f"predicted_scores.json is not valid JSON: {e}") from e
    if not isinstance(scores, dict):
        raise ValueError("predicted_scores.json is not a JSON object")

    for name in ("positive_clips.json", "negative_clips.json"):
        path = run_dir / name
        if not path.exists():
            raise ValueError(f"missing {name}")
        try:
            with open(path, "rb") as f:
                payload = orjson.loads(f.read())
        except Exception as e:
            raise ValueError(f"{name} is not valid JSON: {e}") from e
        if not isinstance(payload, list):
            raise ValueError(f"{name} is not a JSON array")

    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError("missing metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    for key in ("embed_type", "positive_labels", "trained_by"):
        if key not in metadata:
            raise ValueError(f"metadata.json missing key: {key}")
    if not isinstance(metadata["positive_labels"], list) or not metadata["positive_labels"]:
        raise ValueError("metadata.positive_labels must be a non-empty list")
    if not isinstance(metadata.get("negative_labels", []), list):
        raise ValueError("metadata.negative_labels must be a list")
    if not isinstance(metadata["trained_by"], str) or not metadata["trained_by"]:
        raise ValueError("metadata.trained_by must be a non-empty string")


def build_classifier_run(
    output_dir,
    positive_clips,
    negative_clips,
    positive_features,
    negative_features,
    corpus_items,
    features_index,
    embed_type,
    positive_labels,
    negative_labels,
    trained_by,
    use_autolabels=False,
    save_threshold=0.3,
    max_clips=7_000_000,
    C=100,
    chunk_size=50_000,
    run_id: Optional[str] = None,
    search_params: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Train an LR classifier and write the 5-file run directory.

    Parameters
    ----------
    output_dir
        Either a parent dir (a ``run_id`` subdir is created) or a path
        whose ``.name`` already equals ``run_id`` (treated as the run dir
        directly — useful for tests).
    positive_clips, negative_clips
        Clip-id lists used as training labels. Persisted verbatim into
        ``positive_clips.json`` / ``negative_clips.json``.
    positive_features, negative_features
        ``(n, d)`` float32 arrays, aligned with ``positive_clips`` /
        ``negative_clips`` row-by-row.
    corpus_items
        Iterable of ``(clip_id, faiss_row)`` for *every* clip the trained
        classifier should be scored against.
    features_index
        FAISS index supporting ``reconstruct_batch(rows)``. Caller must
        have already set up direct row access (``make_direct_map()``).
    embed_type
        ``cosmos`` / ``caption`` / ``visual`` / ``other`` — recorded in
        metadata for UI display, not used for training.
    positive_labels, negative_labels
        Annotation keys used to gather the training samples. Persisted in
        metadata so the UI can list runs by ``(positive_labels, negative_labels)``.
    trained_by
        Username of the person training. Required — no sentinel default.
    run_id
        Auto-generated 10-char alphanumeric id when None.
    search_params
        Optional URL query that produced the corpus — purely cosmetic,
        persisted in metadata for the runs list.
    """
    log = logger or logging.getLogger(__name__)
    if run_id is None:
        run_id = generate_run_id()

    output_dir = Path(output_dir)
    run_dir = output_dir if output_dir.name == run_id else output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    positive_clips = list(positive_clips)
    negative_clips = list(negative_clips)

    log.info(
        "Training classifier run %s (%d positives, %d negatives, embed_type=%s)",
        run_id, len(positive_clips), len(negative_clips), embed_type,
    )
    model = train_logistic_regression(
        positive_features, negative_features, C=C,
    )
    write_lr_weights(run_dir, model)
    write_clip_lists(run_dir, positive_clips, negative_clips)

    items = list(corpus_items)
    log.info("Scoring %d clips in chunks of %d", len(items), chunk_size)
    clip_ids, probs = score_corpus(
        model, items, features_index, chunk_size=chunk_size,
    )
    scores = select_top_scores(
        clip_ids, probs, save_threshold=save_threshold, max_clips=max_clips,
    )
    write_predicted_scores(run_dir, scores)
    log.info(
        "Wrote %d predictions (threshold=%.3f, max_clips=%d)",
        len(scores), save_threshold, max_clips,
    )

    write_metadata(
        run_dir,
        run_id=run_id,
        embed_type=embed_type,
        positive_labels=positive_labels,
        negative_labels=negative_labels,
        trained_by=trained_by,
        n_positive_clips=len(positive_clips),
        n_negative_clips=len(negative_clips),
        use_autolabels=use_autolabels,
        save_threshold=save_threshold,
        max_clips=max_clips,
        C=C,
        search_params=search_params,
    )
    return run_dir.resolve()
