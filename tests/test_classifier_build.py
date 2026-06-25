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

"""End-to-end tests for `build_classifier_run` and `validate_run_dir`."""
import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from sil_wheel.classifier_build import (
    build_classifier_run,
    select_top_scores,
    validate_run_dir,
)


def _build_index(embeddings):
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


@pytest.fixture()
def fake_corpus():
    """100 clips on two separable Gaussians + a wired-up FAISS index.

    Returns dict with embeddings, clip_ids, faiss index, clip_to_index,
    and split positive/negative training subsets.
    """
    rng = np.random.default_rng(0)
    n_pos = 30
    n_neg = 30
    n_other = 40
    pos_x = rng.normal(loc=5.0, scale=0.2, size=(n_pos, 8))
    neg_x = rng.normal(loc=-5.0, scale=0.2, size=(n_neg, 8))
    other_x = rng.normal(loc=0.0, scale=1.0, size=(n_other, 8))
    embeddings = np.vstack([pos_x, neg_x, other_x]).astype(np.float32)
    clip_ids = (
        [f"pos-{i:03d}" for i in range(n_pos)]
        + [f"neg-{i:03d}" for i in range(n_neg)]
        + [f"other-{i:03d}" for i in range(n_other)]
    )
    index = _build_index(embeddings)
    clip_to_index = {c: i for i, c in enumerate(clip_ids)}

    return {
        "embeddings": embeddings,
        "clip_ids": clip_ids,
        "index": index,
        "clip_to_index": clip_to_index,
        "positive_clips": clip_ids[:n_pos],
        "negative_clips": clip_ids[n_pos:n_pos + n_neg],
        "X_pos": embeddings[:n_pos],
        "X_neg": embeddings[n_pos:n_pos + n_neg],
    }


def test_build_writes_expected_files(tmp_path, fake_corpus):
    run_dir = build_classifier_run(
        output_dir=tmp_path,
        positive_clips=fake_corpus["positive_clips"],
        negative_clips=fake_corpus["negative_clips"],
        positive_features=fake_corpus["X_pos"],
        negative_features=fake_corpus["X_neg"],
        corpus_items=list(fake_corpus["clip_to_index"].items()),
        features_index=fake_corpus["index"],
        embed_type="cosmos",
        positive_labels=["snow"],
        negative_labels=["sun"],
        trained_by="alice",
        save_threshold=0.0,
        max_clips=-1,
        run_id="run-test",
    )
    run_dir = Path(run_dir)
    assert run_dir.name == "run-test"
    for name in (
        "metadata.json",
        "LR_weights.npz",
        "predicted_scores.json",
        "positive_clips.json",
        "negative_clips.json",
    ):
        assert (run_dir / name).exists(), f"missing {name}"


def test_build_metadata_has_required_fields(tmp_path, fake_corpus):
    run_dir = build_classifier_run(
        output_dir=tmp_path,
        positive_clips=fake_corpus["positive_clips"],
        negative_clips=fake_corpus["negative_clips"],
        positive_features=fake_corpus["X_pos"],
        negative_features=fake_corpus["X_neg"],
        corpus_items=list(fake_corpus["clip_to_index"].items()),
        features_index=fake_corpus["index"],
        embed_type="visual",
        positive_labels=["a", "b"],
        negative_labels=[],
        trained_by="alice",
        run_id="run-meta",
    )
    metadata = json.loads((Path(run_dir) / "metadata.json").read_text())
    assert metadata["run_id"] == "run-meta"
    assert metadata["embed_type"] == "visual"
    assert metadata["positive_labels"] == ["a", "b"]
    assert metadata["negative_labels"] == []
    assert metadata["trained_by"] == "alice"
    assert metadata["status"] == "done"
    assert metadata["n_positive_clips"] == 30
    assert metadata["n_negative_clips"] == 30


def test_lr_weights_persisted_and_loadable(tmp_path, fake_corpus):
    run_dir = build_classifier_run(
        output_dir=tmp_path,
        positive_clips=fake_corpus["positive_clips"],
        negative_clips=fake_corpus["negative_clips"],
        positive_features=fake_corpus["X_pos"],
        negative_features=fake_corpus["X_neg"],
        corpus_items=list(fake_corpus["clip_to_index"].items()),
        features_index=fake_corpus["index"],
        embed_type="cosmos",
        positive_labels=["x"],
        negative_labels=[],
        trained_by="alice",
        run_id="run-w",
    )
    with np.load(Path(run_dir) / "LR_weights.npz") as weights:
        assert "coef" in weights.files
        assert "intercept" in weights.files


def test_predicted_scores_separate_classes(tmp_path, fake_corpus):
    run_dir = build_classifier_run(
        output_dir=tmp_path,
        positive_clips=fake_corpus["positive_clips"],
        negative_clips=fake_corpus["negative_clips"],
        positive_features=fake_corpus["X_pos"],
        negative_features=fake_corpus["X_neg"],
        corpus_items=list(fake_corpus["clip_to_index"].items()),
        features_index=fake_corpus["index"],
        embed_type="cosmos",
        positive_labels=["x"],
        negative_labels=[],
        trained_by="alice",
        save_threshold=0.0,
        max_clips=-1,
        run_id="run-s",
    )
    scores = json.loads((Path(run_dir) / "predicted_scores.json").read_text())
    pos_avg = np.mean([scores[c] for c in fake_corpus["positive_clips"]])
    neg_avg = np.mean([scores[c] for c in fake_corpus["negative_clips"]])
    assert pos_avg > 0.9
    assert neg_avg < 0.1


def test_validate_run_dir_accepts_built_run(tmp_path, fake_corpus):
    run_dir = build_classifier_run(
        output_dir=tmp_path,
        positive_clips=fake_corpus["positive_clips"],
        negative_clips=fake_corpus["negative_clips"],
        positive_features=fake_corpus["X_pos"],
        negative_features=fake_corpus["X_neg"],
        corpus_items=list(fake_corpus["clip_to_index"].items()),
        features_index=fake_corpus["index"],
        embed_type="cosmos",
        positive_labels=["x"],
        negative_labels=[],
        trained_by="alice",
        run_id="run-v",
    )
    validate_run_dir(run_dir)  # must not raise


@pytest.mark.parametrize("missing", [
    "LR_weights.npz",
    "predicted_scores.json",
    "positive_clips.json",
    "negative_clips.json",
    "metadata.json",
])
def test_validate_rejects_missing_file(tmp_path, fake_corpus, missing):
    run_dir = build_classifier_run(
        output_dir=tmp_path,
        positive_clips=fake_corpus["positive_clips"],
        negative_clips=fake_corpus["negative_clips"],
        positive_features=fake_corpus["X_pos"],
        negative_features=fake_corpus["X_neg"],
        corpus_items=list(fake_corpus["clip_to_index"].items()),
        features_index=fake_corpus["index"],
        embed_type="cosmos",
        positive_labels=["x"],
        negative_labels=[],
        trained_by="alice",
        run_id="run-m",
    )
    (Path(run_dir) / missing).unlink()
    with pytest.raises(ValueError, match=f"missing {missing}"):
        validate_run_dir(run_dir)


def test_validate_rejects_missing_metadata_keys(tmp_path, fake_corpus):
    run_dir = build_classifier_run(
        output_dir=tmp_path,
        positive_clips=fake_corpus["positive_clips"],
        negative_clips=fake_corpus["negative_clips"],
        positive_features=fake_corpus["X_pos"],
        negative_features=fake_corpus["X_neg"],
        corpus_items=list(fake_corpus["clip_to_index"].items()),
        features_index=fake_corpus["index"],
        embed_type="cosmos",
        positive_labels=["x"],
        negative_labels=[],
        trained_by="alice",
        run_id="run-mk",
    )
    metadata_path = Path(run_dir) / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata["trained_by"]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="trained_by"):
        validate_run_dir(run_dir)


def test_select_top_scores_thresholds_and_caps():
    clip_ids = [f"c-{i:02d}" for i in range(10)]
    probs = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
    out = select_top_scores(clip_ids, probs, save_threshold=0.5, max_clips=-1)
    assert all(v > 0.5 for v in out.values())
    # Strictly > 0.5 → 0.6, 0.7, 0.8, 0.9 → c-06..c-09
    assert set(out.keys()) == set(clip_ids[6:])

    capped = select_top_scores(clip_ids, probs, save_threshold=0.0, max_clips=3)
    assert len(capped) == 3
    # Top 3 by score should be the last 3 entries
    assert set(capped.keys()) == set(clip_ids[-3:])
