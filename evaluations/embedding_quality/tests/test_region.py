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

"""Region BoF + Chamfer aggregation, wired through the full run + table."""

import csv
import json
import pickle

import numpy as np

import build_table
import chamfer_cluster
import run_embedding_quality
from region_aggregation import build_chamfer_similarity, build_detection_matrix


def _unit(rng, n, dim):
    x = rng.standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_chamfer_recovers_separable_blobs():
    """Two well-separated detection clusters -> high K-medoids separation."""
    rng = np.random.default_rng(0)
    dim = 16
    # 12 clips: first 6 near direction A, last 6 near direction B.
    a = np.zeros(dim, dtype=np.float32); a[0] = 1.0
    b = np.zeros(dim, dtype=np.float32); b[1] = 1.0
    rows, offsets = [], [0]
    for i in range(12):
        center = a if i < 6 else b
        det = center + 0.05 * rng.standard_normal((4, dim)).astype(np.float32)
        det /= np.linalg.norm(det, axis=1, keepdims=True)
        rows.append(det)
        offsets.append(offsets[-1] + len(det))
    D = np.concatenate(rows).astype(np.float32)
    S = build_chamfer_similarity(D, np.array(offsets, dtype=np.int64))
    assert S.shape == (12, 12)
    assert np.allclose(np.diag(S), 1.0, atol=1e-3)

    gt = np.array([1] * 6 + [0] * 6, dtype=np.int8)
    clusters = chamfer_cluster.cluster_all(S, algo="kmedoids", k_values=[2], seed=0)
    m = chamfer_cluster.metrics_from_clusters(S, gt, clusters)
    assert m["cluster_purity_k2"] == 1.0
    assert m["separation_k2"] > 0.5


def _write_archive(path, clip_ids, dim, rng):
    """One pkl shard: each clip = 1 __full_frame__ row + 3 detection rows."""
    embeddings, items = [], []
    for cid in clip_ids:
        for label in ("__full_frame__", "obj", "obj", "rare"):
            embeddings.append(_unit(rng, 1, dim)[0])
            items.append({"clip_id": cid, "label": label})
    with open(path, "wb") as f:
        pickle.dump(
            {"embeddings": np.stack(embeddings).astype(np.float32), "items": items},
            f,
        )


def _write_npz(path, clip_ids, dim, rng):
    np.savez(
        path,
        clip_ids=np.array(clip_ids, dtype=object),
        embeddings=_unit(rng, len(clip_ids), dim),
    )


def test_region_encoders_land_in_one_summary_and_table(tmp_path):
    rng = np.random.default_rng(1)
    dim = 16
    clips = [f"clip{i:03d}" for i in range(20)]

    archive = tmp_path / "region_shard.pkl"
    _write_archive(archive, clips, dim, rng)

    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    _write_npz(emb_dir / "cosmos.npz", clips, dim, rng)
    _write_npz(emb_dir / "random.npz", clips, dim, rng)

    labels_csv = tmp_path / "labels.csv"
    with open(labels_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "label"])
        for cid in clips[:6]:
            w.writerow([cid, "Animal Crossing"])
        for cid in clips[6:10]:
            w.writerow([cid, "Stop sign"])
    neg_csv = tmp_path / "neg.csv"
    with open(neg_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id"])
        for cid in clips[10:]:
            w.writerow([cid])

    out_dir = tmp_path / "out"
    rc = run_embedding_quality.main([
        "--labels-csv", str(labels_csv),
        "--negative-csv", str(neg_csv),
        "--embeddings-dir", str(emb_dir),
        "--embeddings", "cosmos",
        run_embedding_quality.REGION_BOF_KEY,
        run_embedding_quality.REGION_CHAMFER_KEY, "random",
        "--region-archive", str(archive),
        "--region-bof-vocab", "8",
        "--ks", "1", "3",
        "--cluster-ks", "2", "4",
        "--few-shot-n", "2",
        "--few-shot-trials", "3",
        "--no-spherical-kmeans",
        "--output-dir", str(out_dir),
    ])
    assert rc == 0

    summary = json.loads((out_dir / "summary.json").read_text())
    mpe = summary["metrics_per_embedding"]
    # Both region encoders live in the SAME summary, alongside the others.
    for key in (
        "cosmos",
        run_embedding_quality.REGION_BOF_KEY,
        run_embedding_quality.REGION_CHAMFER_KEY,
        "random",
    ):
        assert key in mpe, f"{key} missing from summary"
        assert "Animal Crossing" in mpe[key]["per_label"]
    # Chamfer is set-native: its per-label metrics carry few-shot + purity.
    chamfer = mpe[run_embedding_quality.REGION_CHAMFER_KEY]["per_label"]["Stop sign"]
    assert "fewshot_acc_n2_mean" in chamfer
    assert 0.0 <= chamfer["nn_purity_k1"] <= 1.0

    # build_table renders Region BoF + Region Chamfer rows directly.
    rc = build_table.main([
        "--summary", str(out_dir / "summary.json"),
        "--output-stem", str(tmp_path / "table"),
        "--purity-ks", "2", "4",
    ])
    assert rc == 0
    tex = (tmp_path / "table_paper.tex").read_text()
    assert "Region BoF" in tex
    assert "Region Chamfer" in tex
    assert "Region embeddings" not in tex


def test_build_detection_matrix_falls_back_to_full_frame():
    """A clip with no detection rows still gets a row via its full frame."""
    dim = 4
    ff = np.ones(dim, dtype=np.float32)
    detections = {"a": [("obj", np.eye(dim, dtype=np.float32)[0])], "b": []}
    full_frames = {"b": ff}
    present, offsets, D = build_detection_matrix(["a", "b"], detections, full_frames)
    assert present == ["a", "b"]
    assert offsets.tolist() == [0, 1, 2]
    assert np.array_equal(D[1], ff)
