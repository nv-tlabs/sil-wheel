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

"""Tests for the HuggingFaceDataset reader."""
import io
import json
import tarfile

import pytest

from sil_wheel.datasets import base_dataset
from sil_wheel.datasets.base_dataset import HuggingFaceDataset


def _make_tar(tar_path, clip_ids):
    """Create a tar at ``tar_path`` containing one tiny mp4-named entry per id."""
    with tarfile.open(tar_path, "w") as tf:
        for clip_id in clip_ids:
            payload = f"fake-mp4-bytes-for-{clip_id}".encode()
            info = tarfile.TarInfo(name=f"{clip_id}.mp4")
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))


class _FakeHfApi:
    """Stand-in for ``huggingface_hub.HfApi`` that returns a fixed list."""

    def __init__(self, files):
        self._files = files

    def list_repo_files(self, repo_id, repo_type=None):
        return list(self._files)


@pytest.fixture
def synthetic_repo(tmp_path):
    """Build a fake repo on disk with two named tar shards."""
    repo_files = ["test/shard_a.tar", "test/shard_b.tar"]
    (tmp_path / "test").mkdir()
    _make_tar(tmp_path / "test" / "shard_a.tar", ["vid_A1", "vid_A2"])
    _make_tar(tmp_path / "test" / "shard_b.tar", ["vid_B1", "vid_B2"])
    return tmp_path, repo_files


def test_build_manifest_scans_given_shards(tmp_path):
    _make_tar(tmp_path / "shard_a.tar", ["video_A1", "video_A2"])
    _make_tar(tmp_path / "shard_b.tar", ["video_B1"])
    out_path = tmp_path / "manifest.json"

    HuggingFaceDataset._build_manifest(
        [tmp_path / "shard_a.tar", tmp_path / "shard_b.tar"], out_path
    )

    manifest = json.loads(out_path.read_text())
    assert set(manifest) == {"video_A1.mp4", "video_A2.mp4", "video_B1.mp4"}
    assert manifest["video_A1.mp4"].endswith("shard_a.tar")
    assert manifest["video_B1.mp4"].endswith("shard_b.tar")


def test_iteration_yields_video_bytes(synthetic_repo, monkeypatch):
    local_dir, repo_files = synthetic_repo
    monkeypatch.setattr(
        base_dataset, "HfApi", lambda: _FakeHfApi(repo_files)
    )
    monkeypatch.setattr(
        base_dataset, "snapshot_download", lambda **kw: str(local_dir)
    )

    dataset = HuggingFaceDataset(
        process_id=0,
        n_processes=1,
        repo_id="dummy/repo",
        allow_patterns=["test/*"],
    )

    rows = list(dataset)
    assert sorted(r[1] for r in rows) == [
        "vid_A1", "vid_A2", "vid_B1", "vid_B2",
    ]
    for buf, clip_id, camera in rows:
        assert camera is None
        assert buf.read() == f"fake-mp4-bytes-for-{clip_id}".encode()


def test_per_process_sharding_disjoint(synthetic_repo, monkeypatch):
    """Two ranks each get half the shards; their clip sets don't overlap."""
    local_dir, repo_files = synthetic_repo
    monkeypatch.setattr(
        base_dataset, "HfApi", lambda: _FakeHfApi(repo_files)
    )

    captured_allow = []

    def fake_snapshot(**kw):
        captured_allow.append(list(kw["allow_patterns"]))
        return str(local_dir)

    monkeypatch.setattr(base_dataset, "snapshot_download", fake_snapshot)

    ds0 = HuggingFaceDataset(
        process_id=0, n_processes=2,
        repo_id="dummy/repo", allow_patterns=["test/*"],
    )
    ds1 = HuggingFaceDataset(
        process_id=1, n_processes=2,
        repo_id="dummy/repo", allow_patterns=["test/*"],
    )

    # Each rank asked snapshot_download for exactly one shard, and
    # together they cover the full set.
    assert len(captured_allow[0]) == 1
    assert len(captured_allow[1]) == 1
    assert set(captured_allow[0]) | set(captured_allow[1]) == set(
        repo_files
    )

    clips0 = {r[1] for r in ds0}
    clips1 = {r[1] for r in ds1}
    assert clips0.isdisjoint(clips1)
    assert clips0 | clips1 == {"vid_A1", "vid_A2", "vid_B1", "vid_B2"}


def test_matching_manifest_is_not_rebuilt(synthetic_repo, monkeypatch):
    local_dir, repo_files = synthetic_repo
    # Pre-populate a manifest whose tar values match this run's
    # expected slice (both shards). The sentinel member name lets us
    # detect whether the scanner overwrote the file.
    real_tars = [
        str(local_dir / "test" / "shard_a.tar"),
        str(local_dir / "test" / "shard_b.tar"),
    ]
    sentinel = {
        "sentinel.mp4": real_tars[0],
        "another_sentinel.mp4": real_tars[1],
    }
    (local_dir / "hf_manifest_p0_n1.json").write_text(
        json.dumps(sentinel)
    )
    monkeypatch.setattr(
        base_dataset, "HfApi", lambda: _FakeHfApi(repo_files)
    )
    monkeypatch.setattr(
        base_dataset, "snapshot_download", lambda **kw: str(local_dir)
    )

    called = {"build": 0}
    real_build = HuggingFaceDataset._build_manifest

    def spy(tar_paths, out_path):
        called["build"] += 1
        return real_build(tar_paths, out_path)

    monkeypatch.setattr(
        HuggingFaceDataset, "_build_manifest", staticmethod(spy)
    )

    HuggingFaceDataset(
        process_id=0, n_processes=1,
        repo_id="dummy/repo", allow_patterns=["test/*"],
    )

    assert called["build"] == 0
    assert json.loads(
        (local_dir / "hf_manifest_p0_n1.json").read_text()
    ) == sentinel


def test_stale_manifest_is_rebuilt(synthetic_repo, monkeypatch):
    """A manifest from a previous --hf-allow-patterns invocation
    references different tars; the new run must regenerate it."""
    local_dir, repo_files = synthetic_repo
    stale = {"old_clip.mp4": "/no/such/old_shard.tar"}
    (local_dir / "hf_manifest_p0_n1.json").write_text(json.dumps(stale))
    monkeypatch.setattr(
        base_dataset, "HfApi", lambda: _FakeHfApi(repo_files)
    )
    monkeypatch.setattr(
        base_dataset, "snapshot_download", lambda **kw: str(local_dir)
    )

    HuggingFaceDataset(
        process_id=0, n_processes=1,
        repo_id="dummy/repo", allow_patterns=["test/*"],
    )

    rebuilt = json.loads(
        (local_dir / "hf_manifest_p0_n1.json").read_text()
    )
    assert "old_clip.mp4" not in rebuilt
    assert set(rebuilt) == {
        "vid_A1.mp4", "vid_A2.mp4", "vid_B1.mp4", "vid_B2.mp4",
    }
