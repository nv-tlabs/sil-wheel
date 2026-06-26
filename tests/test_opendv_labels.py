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

"""Tests for OpenDV slim labels: dominant selection + start_discard alignment."""
from sil_wheel.datasets.opendv.labels import dominant, segments_from_entries


def _entries():
    # Two 4 s clips (40 frames @10 Hz), first/last frame as zero-padded jpg names.
    # clip A centered ~ frame 20 (t=2.0s): cmd 0 "go straight", blip "a road"
    # clip B centered ~ frame 220 (t=22s): cmd 2 "turn left",  blip "a junction"
    return [
        {"folder": "train_images/chan/vid1", "first_frame": "000000000.jpg",
         "last_frame": "000000039.jpg", "cmd": 0, "blip": "a road"},
        {"folder": "train_images/chan/vid1", "first_frame": "000000200.jpg",
         "last_frame": "000000239.jpg", "cmd": 2, "blip": "a junction"},
    ]


def test_segments_and_dominant_pick_max_overlap():
    segs = segments_from_entries(_entries(), bin_sec=5)["vid1"]
    cap, cmd = dominant(segs, 0.0, 4.0)
    assert cmd == "go straight"
    assert cap == "a road"
    cap2, cmd2 = dominant(segs, 20.0, 24.0)
    assert cmd2 == "turn left"
    assert cap2 == "a junction"


def test_dominant_returns_empty_outside_coverage():
    segs = segments_from_entries(_entries(), bin_sec=5)["vid1"]
    assert dominant(segs, 1000.0, 1004.0) == ("", "")
