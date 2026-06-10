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

"""Tests for OpenDV clip-window selection."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from sil_wheel.datasets.opendv.sampling import (
    clip_id_for, diverse_windows, find_video, sample, uniform_windows,
)


def test_find_video_handles_flat_and_nested_layouts(tmp_path):
    flat = tmp_path / "flat"
    (flat).mkdir()
    (flat / "vidA.mp4").touch()
    assert find_video(flat, "vidA") == flat / "vidA.mp4"

    # nested, e.g. the official OpenDV <youtuber>/<videoid> layout
    nested = tmp_path / "nested"
    (nested / "some_channel").mkdir(parents=True)
    target = nested / "some_channel" / "vidB.webm"
    target.touch()
    assert find_video(nested, "vidB") == target

    assert find_video(flat, "missing") is None


def test_uniform_windows_respects_interval_and_bounds():
    # end=65: window at 0 fits (0-20); next at 60 would be 60-80 > 65, dropped.
    assert uniform_windows(start=0.0, end=65.0, interval=60, clip_sec=20) == [(0.0, 20.0)]


def test_uniform_windows_multiple():
    # end=130: windows at 0 and 60 fit; 120-140 > 130, dropped.
    assert uniform_windows(start=0.0, end=130.0, interval=60, clip_sec=20) == \
        [(0.0, 20.0), (60.0, 80.0)]


def test_clip_id_for_matches_caption_benchmark_format():
    # <video_id>__<video_id>_<start>-<end>
    assert clip_id_for("--I-TdCe2_g", 410.0, 430.0) == "--I-TdCe2_g__--I-TdCe2_g_410-430"


def test_diverse_windows_prefers_rare_maneuvers_and_is_non_overlapping():
    cands = [
        {"a": 0.0,  "b": 20.0, "cap": "a road",     "cmd": "go straight"},
        {"a": 20.0, "b": 40.0, "cap": "a road",     "cmd": "go straight"},
        {"a": 40.0, "b": 60.0, "cap": "a road",     "cmd": "go straight"},
        {"a": 60.0, "b": 80.0, "cap": "a junction", "cmd": "turn left"},
    ]
    picks = diverse_windows(cands, k=2, lam=0.5)
    cmds = {p["cmd"] for p in picks}
    assert "turn left" in cmds          # rare maneuver selected first
    spans = sorted((p["a"], p["b"]) for p in picks)
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))


def test_diverse_windows_total_caps_count():
    cands = [{"a": float(i*20), "b": float(i*20+20), "cap": "x", "cmd": "go straight"}
             for i in range(10)]
    assert len(diverse_windows(cands, total=3)) == 3


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_sample_uniform_end_to_end(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    # 130 s black test video (libx264 so it runs without a GPU).
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=130",
                    "-c:v", "libx264", str(videos / "vidX.mp4")],
                   capture_output=True, check=True)
    records = [{"videoid": "vidX", "start_discard": 0, "end_discard": 0}]
    sample(records, videos, tmp_path / "out", method="uniform", interval=60, cut="libx264")
    lines = (tmp_path / "out" / "uniform" / "manifest.jsonl").read_text().splitlines()
    assert len(lines) == 2     # windows at 0 and 60 (120 doesn't fit in 130)
    row = json.loads(lines[0])
    assert row["clip_id"] == "vidX__vidX_0-20"
    assert Path(row["clip_path"]).exists()


from sil_wheel.datasets.opendv.sampling import _select_global


def test_select_global_total_round_robins_to_target_across_videos():
    # 3 videos x 5 ranked windows = 15 available; total=7 must return exactly 7,
    # round-robined so every video contributes before any video's 2nd pick.
    def wins():
        return [{"a": float(i * 20), "b": float(i * 20 + 20), "cap": "", "cmd": "go straight"}
                for i in range(5)]
    ranked = {("v1", "v1.mp4"): wins(), ("v2", "v2.mp4"): wins(), ("v3", "v3.mp4"): wins()}
    chosen = _select_global(ranked, method="diverse", select_k=None, total=7)
    assert len(chosen) == 7
    first_three_videos = {chosen[i][0][0] for i in range(3)}
    assert first_three_videos == {"v1", "v2", "v3"}   # round-robin, not video-by-video
