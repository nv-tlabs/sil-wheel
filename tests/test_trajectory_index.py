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

"""Tests for trajectory_store module-level indexing/parsing functions.

All tests use artificial data and temporary directories.  The expensive
OPQ/IVF FAISS index calls are replaced with a simple FlatL2 index via
unittest.mock.patch so that the full code path can be exercised without
training data or a GPU.
"""
import json
import pickle
from unittest.mock import MagicMock, patch

import faiss
import numpy as np
import pytest
from safetensors.numpy import save_file

from sil_wheel.stores.trajectory_store import (
    parse_subtrajectory_data_from_dir,
    parse_trajectory_data_from_dir,
    parse_trajectory_mmap,
    update_index,
    update_subtrajectory_index,
)

# ---------------------------------------------------------------------------
# Constants mirroring trajectory_store internals
# ---------------------------------------------------------------------------

TRAJECTORY_DIM = 605
S_PER_SEC = max(1, TRAJECTORY_DIM // 20)  # = 30
D_FULL = 2 * TRAJECTORY_DIM               # = 1210

# File names used by parse_trajectory_data_from_dir (fast path)
FULL_IDX_NAME = "trajectory_data_OPQ121_IVF4096_PQ121x8_p1.index"
FULL_PKL_NAME = "trajectory_clip_to_index_OPQ121_IVF4096_PQ121x8_p1.pkl"

# File names used by update_index (p0 = input, p1 = output)
UPDATE_P0_IDX = "trajectory_data_OPQ121_IVF4096_PQ121x8_p0.index"
UPDATE_P0_PKL = "trajectory_clip_to_index_OPQ121_IVF4096_PQ121x8_p0.pkl"
UPDATE_P1_IDX = "trajectory_data_OPQ121_IVF4096_PQ121x8_p1.index"
UPDATE_P1_PKL = "trajectory_clip_to_index_OPQ121_IVF4096_PQ121x8_p1.pkl"

# Windows used by parse_subtrajectory_data_from_dir / update_subtrajectory_index
WINDOWS = {
    10: [(0, 10), (5, 15), (10, 20)],
    5:  [(0, 5), (5, 10), (10, 15), (15, 20)],
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_safetensors(tmp_dir, name, clips):
    """Write clips into tmp_dir/<name>/<name>.safetensors.

    This matches the */*.safetensors glob used by all four functions.
    All arrays are cast to float32 before saving.
    Returns the path to the created file.
    """
    sub = tmp_dir / name
    sub.mkdir(parents=True, exist_ok=True)
    path = sub / f"{name}.safetensors"
    save_file({k: v.astype(np.float32) for k, v in clips.items()}, str(path))
    return path


def _make_flat_index(d, vecs=None):
    """Return a FlatL2 FAISS index, optionally pre-populated with vecs."""
    idx = faiss.IndexFlatL2(d)
    if vecs is not None:
        idx.add(np.ascontiguousarray(vecs, dtype=np.float32))
    return idx


def _write_index_and_pkl(idx_path, pkl_path, idx, clip_to_index):
    """Persist a FAISS index and a clip_to_index mapping to disk."""
    faiss.write_index(idx, str(idx_path))
    with open(pkl_path, "wb") as f:
        pickle.dump(clip_to_index, f)


def _sub_d(sec):
    """Compute the FAISS vector dimension for a given sub-trajectory length."""
    sub_traj_dim = min(TRAJECTORY_DIM, sec * S_PER_SEC)
    return 2 * sub_traj_dim


def _sub_index_names(sec, M, version="p1"):
    """Return (index_filename, pkl_filename) for the given sec/M/version.

    The tag must match what production's ``spec_to_tag`` emits, i.e.,
    case preserved with commas replaced by underscores.
    """
    tag = f"OPQ{M}_IVF4096_PQ{M}x8"
    return (
        f"trajectory_data_{sec}s_{tag}_{version}.index",
        f"trajectory_clip_to_index_{sec}s_{tag}_{version}.pkl",
    )


# ---------------------------------------------------------------------------
# parse_trajectory_data_from_dir
# ---------------------------------------------------------------------------

class TestParseTrajectoryDataFromDir:

    def test_fast_path_loads_prebuilt_index(self, tmp_path):
        """If .index and .pkl already exist they are loaded; no safetensors read."""
        vecs = np.random.randn(5, D_FULL).astype(np.float32)
        idx = _make_flat_index(D_FULL, vecs)
        cti = {f"clip-{i}": i for i in range(5)}
        _write_index_and_pkl(
            tmp_path / FULL_IDX_NAME, tmp_path / FULL_PKL_NAME, idx, cti
        )

        loaded_idx, loaded_cti = parse_trajectory_data_from_dir(str(tmp_path))

        assert loaded_idx.ntotal == 5
        assert loaded_cti == cti

    def test_fast_path_ignores_safetensors(self, tmp_path):
        """Fast path takes priority even when extra safetensors are present."""
        vecs = np.random.randn(3, D_FULL).astype(np.float32)
        idx = _make_flat_index(D_FULL, vecs)
        cti = {"clip-0": 0, "clip-1": 1, "clip-2": 2}
        _write_index_and_pkl(
            tmp_path / FULL_IDX_NAME, tmp_path / FULL_PKL_NAME, idx, cti
        )
        # Safetensors with a clip not in the prebuilt index
        _make_safetensors(
            tmp_path, "sub", {"clip-extra": np.random.randn(700, 7).astype(np.float32)}
        )

        _, loaded_cti = parse_trajectory_data_from_dir(str(tmp_path))

        assert "clip-extra" not in loaded_cti

    @patch("sil_wheel.stores.trajectory_store.faiss.extract_index_ivf")
    @patch("sil_wheel.stores.trajectory_store.faiss.index_factory")
    def test_slow_path_indexes_all_clips(
        self, mock_factory, mock_extract_ivf, tmp_path
    ):
        """Without prebuilt files all clips from safetensors are indexed."""
        flat_idx = _make_flat_index(D_FULL)
        mock_factory.return_value = flat_idx
        mock_extract_ivf.return_value = MagicMock()

        clips = {
            "clip-A": np.random.randn(700, 7).astype(np.float32),
            "clip-B": np.random.randn(300, 7).astype(np.float32),
        }
        _make_safetensors(tmp_path, "sub1", clips)

        idx, cti = parse_trajectory_data_from_dir(str(tmp_path))

        assert idx.ntotal == 2
        assert set(cti.keys()) == {"clip-A", "clip-B"}
        assert cti["clip-A"] == 0
        assert cti["clip-B"] == 1

    @patch("sil_wheel.stores.trajectory_store.faiss.extract_index_ivf")
    @patch("sil_wheel.stores.trajectory_store.faiss.index_factory")
    def test_slow_path_long_trajectory_is_truncated(
        self, mock_factory, mock_extract_ivf, tmp_path
    ):
        """Trajectories longer than TRAJECTORY_DIM are truncated; vector dim is D_FULL."""
        flat_idx = _make_flat_index(D_FULL)
        mock_factory.return_value = flat_idx
        mock_extract_ivf.return_value = MagicMock()

        _make_safetensors(
            tmp_path, "sub1", {"clip-long": np.random.randn(1000, 7).astype(np.float32)}
        )

        idx, _ = parse_trajectory_data_from_dir(str(tmp_path))

        assert idx.ntotal == 1
        assert idx.reconstruct(0).shape == (D_FULL,)

    @patch("sil_wheel.stores.trajectory_store.faiss.extract_index_ivf")
    @patch("sil_wheel.stores.trajectory_store.faiss.index_factory")
    def test_slow_path_short_trajectory_is_zero_padded(
        self, mock_factory, mock_extract_ivf, tmp_path
    ):
        """Short trajectories are zero-padded; stored vector still has shape D_FULL."""
        flat_idx = _make_flat_index(D_FULL)
        mock_factory.return_value = flat_idx
        mock_extract_ivf.return_value = MagicMock()

        _make_safetensors(
            tmp_path, "sub1", {"clip-short": np.ones((50, 7), dtype=np.float32)}
        )

        idx, _ = parse_trajectory_data_from_dir(str(tmp_path))

        vec = idx.reconstruct(0)
        assert vec.shape == (D_FULL,)
        # Frames beyond the clip length should be zero-padded
        traj = vec.reshape(TRAJECTORY_DIM, 2)
        np.testing.assert_array_equal(traj[50:], 0.0)

    @patch("sil_wheel.stores.trajectory_store.faiss.extract_index_ivf")
    @patch("sil_wheel.stores.trajectory_store.faiss.index_factory")
    def test_slow_path_trajectory_is_centered(
        self, mock_factory, mock_extract_ivf, tmp_path
    ):
        """The embedding is centered: first XY position subtracted from all rows."""
        flat_idx = _make_flat_index(D_FULL)
        mock_factory.return_value = flat_idx
        mock_extract_ivf.return_value = MagicMock()

        # x increases linearly from a non-zero offset; y is constant
        data = np.zeros((700, 7), dtype=np.float32)
        data[:, 0] = np.arange(700, dtype=np.float32) * 0.5 + 10.0
        data[:, 1] = 3.0
        _make_safetensors(tmp_path, "sub1", {"clip-c": data})

        idx, _ = parse_trajectory_data_from_dir(str(tmp_path))

        traj = idx.reconstruct(0).reshape(TRAJECTORY_DIM, 2)
        # After centering the first point is (0, 0)
        np.testing.assert_allclose(traj[0], [0.0, 0.0], atol=1e-5)

    @patch("sil_wheel.stores.trajectory_store.faiss.extract_index_ivf")
    @patch("sil_wheel.stores.trajectory_store.faiss.index_factory")
    def test_slow_path_duplicate_clip_ids_indexed_once(
        self, mock_factory, mock_extract_ivf, tmp_path
    ):
        """A clip_id appearing in multiple safetensors files is indexed only once."""
        flat_idx = _make_flat_index(D_FULL)
        mock_factory.return_value = flat_idx
        mock_extract_ivf.return_value = MagicMock()

        dup = np.random.randn(700, 7).astype(np.float32)
        _make_safetensors(tmp_path, "sub1", {"clip-dup": dup})
        _make_safetensors(tmp_path, "sub2", {"clip-dup": dup, "clip-new": dup})

        idx, cti = parse_trajectory_data_from_dir(str(tmp_path))

        assert idx.ntotal == 2
        assert set(cti.keys()) == {"clip-dup", "clip-new"}

    @patch("sil_wheel.stores.trajectory_store.faiss.extract_index_ivf")
    @patch("sil_wheel.stores.trajectory_store.faiss.index_factory")
    def test_slow_path_persists_index_files(
        self, mock_factory, mock_extract_ivf, tmp_path
    ):
        """After building, the index and pkl files are written to disk."""
        flat_idx = _make_flat_index(D_FULL)
        mock_factory.return_value = flat_idx
        mock_extract_ivf.return_value = MagicMock()

        _make_safetensors(
            tmp_path, "sub1", {"clip-A": np.random.randn(700, 7).astype(np.float32)}
        )

        parse_trajectory_data_from_dir(str(tmp_path))

        assert (tmp_path / FULL_IDX_NAME).exists()
        assert (tmp_path / FULL_PKL_NAME).exists()


# ---------------------------------------------------------------------------
# update_index
# ---------------------------------------------------------------------------

class TestUpdateIndex:

    def _setup_p0(self, tmp_path, n_existing):
        """Write p0 index + pkl with n_existing dummy clips."""
        vecs = np.random.randn(n_existing, D_FULL).astype(np.float32) if n_existing else None
        idx = _make_flat_index(D_FULL, vecs)
        cti = {f"clip-old-{i}": i for i in range(n_existing)}
        _write_index_and_pkl(
            tmp_path / UPDATE_P0_IDX, tmp_path / UPDATE_P0_PKL, idx, cti
        )
        return idx, cti

    def test_new_clips_are_added(self, tmp_path):
        """Clips absent from the existing index are added and appear in p1."""
        old_idx, _ = self._setup_p0(tmp_path, n_existing=2)
        _make_safetensors(
            tmp_path, "sub1", {"clip-new": np.random.randn(700, 7).astype(np.float32)}
        )

        new_idx, new_cti = update_index(str(tmp_path))

        assert new_idx.ntotal == old_idx.ntotal + 1
        assert "clip-new" in new_cti

    def test_existing_clips_are_skipped(self, tmp_path):
        """Clips already present in clip_to_index are not re-added."""
        old_idx, _ = self._setup_p0(tmp_path, n_existing=1)
        clips = {
            "clip-old-0": np.random.randn(700, 7).astype(np.float32),
            "clip-new": np.random.randn(700, 7).astype(np.float32),
        }
        _make_safetensors(tmp_path, "sub1", clips)

        new_idx, new_cti = update_index(str(tmp_path))

        assert new_idx.ntotal == old_idx.ntotal + 1  # only clip-new added
        assert "clip-old-0" in new_cti
        assert "clip-new" in new_cti

    def test_nan_clips_are_skipped(self, tmp_path):
        """Clips containing any NaN value are silently skipped."""
        self._setup_p0(tmp_path, n_existing=0)
        nan_data = np.random.randn(700, 7).astype(np.float32)
        nan_data[5, 2] = float("nan")
        clips = {
            "clip-nan": nan_data,
            "clip-ok": np.random.randn(700, 7).astype(np.float32),
        }
        _make_safetensors(tmp_path, "sub1", clips)

        new_idx, new_cti = update_index(str(tmp_path))

        assert "clip-nan" not in new_cti
        assert "clip-ok" in new_cti
        assert new_idx.ntotal == 1

    def test_p1_files_are_written(self, tmp_path):
        """Output files are written with the p1 suffix."""
        self._setup_p0(tmp_path, n_existing=0)
        _make_safetensors(
            tmp_path, "sub1", {"clip-A": np.random.randn(700, 7).astype(np.float32)}
        )

        update_index(str(tmp_path))

        assert (tmp_path / UPDATE_P1_IDX).exists()
        assert (tmp_path / UPDATE_P1_PKL).exists()

    def test_offset_starts_from_existing_ntotal(self, tmp_path):
        """New clip indices are assigned starting from the current index ntotal."""
        self._setup_p0(tmp_path, n_existing=3)
        _make_safetensors(
            tmp_path, "sub1", {"clip-new": np.random.randn(700, 7).astype(np.float32)}
        )

        _, new_cti = update_index(str(tmp_path))

        assert new_cti["clip-new"] == 3  # existing ntotal was 3

    @patch("sil_wheel.stores.trajectory_store.faiss.read_index")
    def test_raises_on_untrained_index(self, mock_read_index, tmp_path):
        """An untrained index (is_trained=False) raises ValueError immediately."""
        mock_idx = MagicMock()
        mock_idx.is_trained = False
        mock_idx.ntotal = 0
        mock_read_index.return_value = mock_idx

        idx_path = tmp_path / UPDATE_P0_IDX
        pkl_path = tmp_path / UPDATE_P0_PKL
        idx_path.touch()
        with open(pkl_path, "wb") as f:
            pickle.dump({}, f)

        with pytest.raises(ValueError, match="must be trained"):
            update_index(str(tmp_path))


# ---------------------------------------------------------------------------
# parse_trajectory_mmap
# ---------------------------------------------------------------------------

class TestParseTrajectoryMmap:

    def test_creates_output_files(self, tmp_path):
        """Both trajectory_data.dat and clip_to_idx.json are created."""
        _make_safetensors(
            tmp_path, "sub1", {"clip-A": np.random.randn(50, 7).astype(np.float32)}
        )

        parse_trajectory_mmap(str(tmp_path))

        assert (tmp_path / "trajectory_data.dat").exists()
        assert (tmp_path / "clip_to_idx.json").exists()

    def test_memmap_shape(self, tmp_path):
        """The memmap has shape (total_rows, 7) across all clips."""
        clips = {
            "clip-A": np.random.randn(50, 7).astype(np.float32),
            "clip-B": np.random.randn(80, 7).astype(np.float32),
        }
        _make_safetensors(tmp_path, "sub1", clips)

        parse_trajectory_mmap(str(tmp_path))

        total_rows = 50 + 80
        mmap = np.memmap(
            str(tmp_path / "trajectory_data.dat"),
            dtype="float32", mode="r", shape=(total_rows, 7),
        )
        assert mmap.shape == (total_rows, 7)

    def test_clip_to_idx_ranges_cover_data_exactly(self, tmp_path):
        """Each (start, end) range matches the clip length; ranges are contiguous."""
        clips = {
            "clip-A": np.random.randn(30, 7).astype(np.float32),
            "clip-B": np.random.randn(50, 7).astype(np.float32),
        }
        _make_safetensors(tmp_path, "sub1", clips)

        parse_trajectory_mmap(str(tmp_path))

        with open(tmp_path / "clip_to_idx.json") as f:
            cti = json.load(f)

        for clip_id, (start, end) in cti.items():
            assert end - start == clips[clip_id].shape[0]

        # Ranges must be contiguous with no gaps or overlaps
        sorted_ranges = sorted(cti.values(), key=lambda x: x[0])
        prev_end = 0
        for start, end in sorted_ranges:
            assert start == prev_end, f"Gap/overlap at {start} (expected {prev_end})"
            prev_end = end

    def test_data_values_are_preserved(self, tmp_path):
        """Values read back from the memmap match the original array."""
        data_A = np.arange(21, dtype=np.float32).reshape(3, 7)
        _make_safetensors(tmp_path, "sub1", {"clip-A": data_A})

        parse_trajectory_mmap(str(tmp_path))

        with open(tmp_path / "clip_to_idx.json") as f:
            cti = json.load(f)

        start, end = cti["clip-A"]
        mmap = np.memmap(
            str(tmp_path / "trajectory_data.dat"),
            dtype="float32", mode="r", shape=(end, 7),
        )
        np.testing.assert_array_equal(mmap[start:end], data_A)

    def test_multiple_safetensors_no_overlap(self, tmp_path):
        """Clips from multiple safetensors files are all indexed without overlap."""
        clips1 = {"clip-X": np.random.randn(20, 7).astype(np.float32)}
        clips2 = {"clip-Y": np.random.randn(40, 7).astype(np.float32)}
        _make_safetensors(tmp_path, "sub1", clips1)
        _make_safetensors(tmp_path, "sub2", clips2)

        parse_trajectory_mmap(str(tmp_path))

        with open(tmp_path / "clip_to_idx.json") as f:
            cti = json.load(f)

        assert set(cti.keys()) == {"clip-X", "clip-Y"}
        sx, ex = cti["clip-X"]
        sy, ey = cti["clip-Y"]
        assert ex - sx == 20
        assert ey - sy == 40
        # Ranges must not overlap
        assert ex <= sy or ey <= sx


# ---------------------------------------------------------------------------
# parse_subtrajectory_data_from_dir
# ---------------------------------------------------------------------------

def _setup_sub_mocks(mock_factory, mock_extract_ivf, sec):
    """Wire up mock return values; return the FlatL2 that acts as the index."""
    d = _sub_d(sec)
    flat_idx = _make_flat_index(d)
    mock_factory.return_value = flat_idx
    mock_extract_ivf.return_value = MagicMock()
    return flat_idx


class TestParseSubtrajectoryDataFromDir:

    def test_fast_path_loads_existing_subindex(self, tmp_path):
        """Pre-built _p1 .index and .pkl for sec=10 are loaded without rebuilding."""
        sec, M = 10, 40
        d = _sub_d(sec)
        idx_name, pkl_name = _sub_index_names(sec, M)

        vecs = np.random.randn(4, d).astype(np.float32)
        idx = _make_flat_index(d, vecs)
        cti = {f"clip-{i}": [i] for i in range(4)}
        _write_index_and_pkl(tmp_path / idx_name, tmp_path / pkl_name, idx, cti)

        loaded_idx, loaded_cti = parse_subtrajectory_data_from_dir(
            str(tmp_path), sec, M
        )

        assert loaded_idx.ntotal == 4
        assert loaded_cti == cti

    @patch("sil_wheel.stores.trajectory_store.faiss.extract_index_ivf")
    @patch("sil_wheel.stores.trajectory_store.faiss.index_factory")
    def test_slow_path_window_count_sec10(
        self, mock_factory, mock_extract_ivf, tmp_path
    ):
        """For sec=10 each clip produces 3 index entries (one per window)."""
        _setup_sub_mocks(mock_factory, mock_extract_ivf, sec=10)
        clips = {
            "clip-A": np.random.randn(700, 7).astype(np.float32),
            "clip-B": np.random.randn(700, 7).astype(np.float32),
        }
        _make_safetensors(tmp_path, "sub1", clips)

        idx, cti = parse_subtrajectory_data_from_dir(str(tmp_path), sec=10, M=40)

        n_windows = len(WINDOWS[10])  # 3
        assert idx.ntotal == len(clips) * n_windows
        for clip_id in clips:
            assert clip_id in cti
            assert len(cti[clip_id]) == n_windows

    @patch("sil_wheel.stores.trajectory_store.faiss.extract_index_ivf")
    @patch("sil_wheel.stores.trajectory_store.faiss.index_factory")
    def test_slow_path_window_count_sec5(
        self, mock_factory, mock_extract_ivf, tmp_path
    ):
        """For sec=5 each clip produces 4 index entries (one per window)."""
        _setup_sub_mocks(mock_factory, mock_extract_ivf, sec=5)
        _make_safetensors(
            tmp_path, "sub1",
            {"clip-A": np.random.randn(700, 7).astype(np.float32)},
        )

        idx, cti = parse_subtrajectory_data_from_dir(str(tmp_path), sec=5, M=20)

        n_windows = len(WINDOWS[5])  # 4
        assert idx.ntotal == n_windows
        assert len(cti["clip-A"]) == n_windows

    @patch("sil_wheel.stores.trajectory_store.faiss.extract_index_ivf")
    @patch("sil_wheel.stores.trajectory_store.faiss.index_factory")
    def test_slow_path_subtrajectory_is_centered(
        self, mock_factory, mock_extract_ivf, tmp_path
    ):
        """Each window slice is centered: first row of the sub-trajectory is (0, 0)."""
        flat_idx = _setup_sub_mocks(mock_factory, mock_extract_ivf, sec=5)
        sec = 5
        sub_traj_dim = min(TRAJECTORY_DIM, sec * S_PER_SEC)  # 150

        # x increases linearly from a non-zero offset; y is zero
        data = np.zeros((700, 7), dtype=np.float32)
        data[:, 0] = np.arange(700, dtype=np.float32) + 100.0
        _make_safetensors(tmp_path, "sub1", {"clip-A": data})

        _, cti = parse_subtrajectory_data_from_dir(str(tmp_path), sec=sec, M=20)

        for offset in cti["clip-A"]:
            vec = flat_idx.reconstruct(int(offset))
            traj = vec.reshape(sub_traj_dim, 2)
            # After centering, the first XY is (0, 0)
            np.testing.assert_allclose(traj[0], [0.0, 0.0], atol=1e-5)

    @patch("sil_wheel.stores.trajectory_store.faiss.extract_index_ivf")
    @patch("sil_wheel.stores.trajectory_store.faiss.index_factory")
    def test_slow_path_short_clip_all_windows_still_indexed(
        self, mock_factory, mock_extract_ivf, tmp_path
    ):
        """A clip shorter than the first window boundary is still given all windows.

        Out-of-range windows produce all-zero sub-trajectories but the clip
        still appears in clip_to_index with the full window count.
        """
        _setup_sub_mocks(mock_factory, mock_extract_ivf, sec=5)
        # Clip has only 10 rows — shorter than even the first window (150 frames)
        _make_safetensors(
            tmp_path, "sub1",
            {"clip-short": np.ones((10, 7), dtype=np.float32)},
        )

        idx, cti = parse_subtrajectory_data_from_dir(str(tmp_path), sec=5, M=20)

        n_windows = len(WINDOWS[5])  # 4
        assert idx.ntotal == n_windows
        assert len(cti["clip-short"]) == n_windows

    @patch("sil_wheel.stores.trajectory_store.faiss.extract_index_ivf")
    @patch("sil_wheel.stores.trajectory_store.faiss.index_factory")
    def test_slow_path_duplicate_clips_indexed_once(
        self, mock_factory, mock_extract_ivf, tmp_path
    ):
        """A clip_id across multiple safetensors is indexed only once."""
        _setup_sub_mocks(mock_factory, mock_extract_ivf, sec=10)
        dup = np.random.randn(700, 7).astype(np.float32)
        _make_safetensors(tmp_path, "sub1", {"clip-dup": dup})
        _make_safetensors(tmp_path, "sub2", {"clip-dup": dup, "clip-new": dup})

        idx, cti = parse_subtrajectory_data_from_dir(str(tmp_path), sec=10, M=40)

        n_windows = len(WINDOWS[10])  # 3
        # 2 unique clips × 3 windows
        assert idx.ntotal == 2 * n_windows
        assert len(cti) == 2

    @patch("sil_wheel.stores.trajectory_store.faiss.extract_index_ivf")
    @patch("sil_wheel.stores.trajectory_store.faiss.index_factory")
    def test_slow_path_persists_index_files(
        self, mock_factory, mock_extract_ivf, tmp_path
    ):
        """Index and pkl are written to disk with the _p1 suffix."""
        _setup_sub_mocks(mock_factory, mock_extract_ivf, sec=10)
        _make_safetensors(
            tmp_path, "sub1",
            {"clip-A": np.random.randn(700, 7).astype(np.float32)},
        )

        parse_subtrajectory_data_from_dir(str(tmp_path), sec=10, M=40)

        idx_name, pkl_name = _sub_index_names(10, 40)
        assert (tmp_path / idx_name).exists()
        assert (tmp_path / pkl_name).exists()


# ---------------------------------------------------------------------------
# update_subtrajectory_index
# ---------------------------------------------------------------------------

class TestUpdateSubtrajectoryIndex:
    """Tests for update_subtrajectory_index, mirroring TestUpdateIndex."""

    def _setup_p0(self, tmp_path, sec, M, n_existing):
        """Write p0 index + pkl with n_existing dummy clips (n_windows offsets each)."""
        d = _sub_d(sec)
        n_windows = len(WINDOWS[sec])
        n_vecs = n_existing * n_windows
        vecs = np.random.randn(n_vecs, d).astype(np.float32) if n_vecs else None
        idx = _make_flat_index(d, vecs)
        cti = {
            f"clip-old-{i}": list(range(i * n_windows, (i + 1) * n_windows))
            for i in range(n_existing)
        }
        idx_name, pkl_name = _sub_index_names(sec, M, version="p0")
        _write_index_and_pkl(
            tmp_path / idx_name, tmp_path / pkl_name, idx, cti
        )
        return idx, cti

    def test_new_clips_are_added(self, tmp_path):
        """Clips absent from the existing index are added and appear in p1."""
        sec, M = 10, 40
        old_idx, _ = self._setup_p0(tmp_path, sec, M, n_existing=1)
        _make_safetensors(
            tmp_path, "sub1", {"clip-new": np.random.randn(700, 7).astype(np.float32)}
        )

        new_idx, new_cti = update_subtrajectory_index(str(tmp_path), sec, M)

        n_windows = len(WINDOWS[sec])
        assert new_idx.ntotal == old_idx.ntotal + n_windows
        assert "clip-new" in new_cti
        assert len(new_cti["clip-new"]) == n_windows

    def test_existing_clips_are_skipped(self, tmp_path):
        """Clips already present in clip_to_index are not re-added."""
        sec, M = 5, 20
        old_idx, _ = self._setup_p0(tmp_path, sec, M, n_existing=1)
        clips = {
            "clip-old-0": np.random.randn(700, 7).astype(np.float32),
            "clip-new": np.random.randn(700, 7).astype(np.float32),
        }
        _make_safetensors(tmp_path, "sub1", clips)

        new_idx, new_cti = update_subtrajectory_index(str(tmp_path), sec, M)

        n_windows = len(WINDOWS[sec])
        # Only clip-new was added
        assert new_idx.ntotal == old_idx.ntotal + n_windows
        assert "clip-old-0" in new_cti
        assert "clip-new" in new_cti

    def test_nan_clips_are_skipped(self, tmp_path):
        """Clips containing any NaN value are silently skipped."""
        sec, M = 10, 40
        self._setup_p0(tmp_path, sec, M, n_existing=0)
        nan_data = np.random.randn(700, 7).astype(np.float32)
        nan_data[5, 2] = float("nan")
        clips = {
            "clip-nan": nan_data,
            "clip-ok": np.random.randn(700, 7).astype(np.float32),
        }
        _make_safetensors(tmp_path, "sub1", clips)

        new_idx, new_cti = update_subtrajectory_index(str(tmp_path), sec, M)

        n_windows = len(WINDOWS[sec])
        assert "clip-nan" not in new_cti
        assert "clip-ok" in new_cti
        assert new_idx.ntotal == n_windows

    def test_p1_files_are_written(self, tmp_path):
        """Output files are written with the p1 suffix."""
        sec, M = 10, 40
        self._setup_p0(tmp_path, sec, M, n_existing=0)
        _make_safetensors(
            tmp_path, "sub1", {"clip-A": np.random.randn(700, 7).astype(np.float32)}
        )

        update_subtrajectory_index(str(tmp_path), sec, M)

        idx_name, pkl_name = _sub_index_names(sec, M, version="p1")
        assert (tmp_path / idx_name).exists()
        assert (tmp_path / pkl_name).exists()

    def test_offset_starts_from_existing_ntotal(self, tmp_path):
        """New clip window offsets start from the current index ntotal."""
        sec, M = 10, 40
        n_existing = 2
        self._setup_p0(tmp_path, sec, M, n_existing=n_existing)
        _make_safetensors(
            tmp_path, "sub1", {"clip-new": np.random.randn(700, 7).astype(np.float32)}
        )

        _, new_cti = update_subtrajectory_index(str(tmp_path), sec, M)

        n_windows = len(WINDOWS[sec])
        existing_ntotal = n_existing * n_windows
        expected_offsets = list(
            range(existing_ntotal, existing_ntotal + n_windows)
        )
        assert new_cti["clip-new"] == expected_offsets

    def test_multiple_windows_per_clip(self, tmp_path):
        """Each new clip gets exactly n_windows entries in clip_to_index."""
        for sec, M in [(5, 20), (10, 40)]:
            self._setup_p0(tmp_path, sec, M, n_existing=0)
            _make_safetensors(
                tmp_path, "sub1",
                {"clip-A": np.random.randn(700, 7).astype(np.float32)},
            )

            _, cti = update_subtrajectory_index(str(tmp_path), sec, M)

            n_windows = len(WINDOWS[sec])
            assert isinstance(cti["clip-A"], list)
            assert len(cti["clip-A"]) == n_windows

    @patch("sil_wheel.stores.trajectory_store.faiss.read_index")
    def test_raises_on_untrained_index(self, mock_read_index, tmp_path):
        """An untrained index (is_trained=False) raises ValueError immediately."""
        sec, M = 10, 40
        mock_idx = MagicMock()
        mock_idx.is_trained = False
        mock_idx.ntotal = 0
        mock_read_index.return_value = mock_idx

        idx_name, pkl_name = _sub_index_names(sec, M, version="p0")
        (tmp_path / idx_name).touch()
        with open(tmp_path / pkl_name, "wb") as f:
            pickle.dump({}, f)

        with pytest.raises(ValueError, match="must be trained"):
            update_subtrajectory_index(str(tmp_path), sec, M)
