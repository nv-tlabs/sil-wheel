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

"""Ingest the public (hf) wheel-data layout into per-encoder npz."""

import json
import pickle

import numpy as np
import pandas as pd

import ingest_raw_embeddings as ing


def _write_parquet(path, clip_ids, dim, col, seed):
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    vecs = rng.normal(size=(len(clip_ids), dim)).astype("float32")
    df = pd.DataFrame({"clip_id": clip_ids, col: [v.tolist() for v in vecs]})
    df.to_parquet(path)
    return vecs


def _write_visual_pkl(path, clip_ids, dim, seed):
    """One pkl with, per clip, two __full_frame__ rows (frame_index 5 then 2)
    plus a detected-region row. The earliest full frame (index 2) is the target.
    """
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    embs = []
    items = []
    want = {}  # clip_id -> the index-2 full-frame vector we expect ingest to keep
    for c in clip_ids:
        v_late = rng.normal(size=dim).astype("float32")
        v_early = rng.normal(size=dim).astype("float32")
        v_region = rng.normal(size=dim).astype("float32")
        want[c] = v_early
        for fi, vec, label in (
            (5, v_late, "__full_frame__"),
            (2, v_early, "__full_frame__"),
            (2, v_region, "car"),
        ):
            embs.append(vec)
            items.append({"clip_id": c, "frame_index": fi, "label": label})
    with open(path, "wb") as f:
        pickle.dump({"embeddings": np.stack(embs), "items": items}, f)
    return want


def test_hf_ingest_npz_and_pool(tmp_path):
    root = tmp_path / "wheel-data-physical-ai"
    out = tmp_path / "npz"

    # cosmos/caption share clips a,b,c; visual drops c and adds d -> pool = {a,b}.
    cos = _write_parquet(
        root / "cosmos_embeddings" / "cosmos_embed1_448p_group_0_1.parquet",
        ["a", "b", "c"], dim=8, col="embeddings", seed=1,
    )
    _write_parquet(
        root / "caption_embeddings" / "group_0_1.parquet",
        ["a", "b", "c"], dim=16, col="embedding", seed=2,
    )
    want_visual = _write_visual_pkl(
        root / "visual_embeddings" / "florence2_sigclip_group_0_1.pkl",
        ["a", "b", "d"], dim=8, seed=3,
    )

    rc = ing.main(
        ["--root", str(root), "--out", str(out), "--encoders", "cosmos", "caption", "visual"]
    )
    assert rc == 0

    # Each encoder -> one npz, row-aligned clip_ids + 2D embeddings.
    cz = np.load(out / "cosmos.npz", allow_pickle=True)
    assert [str(x) for x in cz["clip_ids"]] == ["a", "b", "c"]
    assert cz["embeddings"].shape == (3, 8)
    np.testing.assert_allclose(cz["embeddings"], cos, rtol=1e-5)

    capz = np.load(out / "caption.npz", allow_pickle=True)
    assert capz["embeddings"].shape == (3, 16)  # caption's "embedding" column read

    # Visual: the earliest (min frame_index) __full_frame__ row per clip, regions dropped.
    vz = np.load(out / "visual.npz", allow_pickle=True)
    vids = [str(x) for x in vz["clip_ids"]]
    assert set(vids) == {"a", "b", "d"}
    assert vz["embeddings"].shape == (3, 8)
    for i, c in enumerate(vids):
        np.testing.assert_allclose(vz["embeddings"][i], want_visual[c], rtol=1e-5)

    # Pool = intersection across all three encoders.
    pool = json.loads((out / "pai_clip_ids.json").read_text())
    assert pool == ["a", "b"]
    summary = json.loads((out / "pool_summary.json").read_text())
    assert summary["layout"] == "hf"
    assert summary["pai_intersection"] == 2


def test_unknown_encoder_for_layout_errors(tmp_path):
    # qwen3_vl exists only in the internal layout; rejected under hf.
    with __import__("pytest").raises(SystemExit):
        ing.main(
            ["--root", str(tmp_path), "--out", str(tmp_path / "o"), "--encoders", "qwen3_vl"]
        )
