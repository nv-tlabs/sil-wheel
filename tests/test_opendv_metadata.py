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

"""Tests for OpenDV metadata normalization."""
from sil_wheel.datasets.opendv.metadata import duration2length, parse_csv_text


def test_duration2length_handles_mm_ss_and_hh_mm_ss():
    assert duration2length("03:20") == 200
    assert duration2length("01:02:03") == 3723


def test_parse_csv_text_normalizes_columns_and_filters_subset():
    csv_text = (
        "Videoid,Link,Youtuber,Train / Val,Mini / Full Set,"
        "Discarded length at the begininning (second),"
        "Discarded length at the ending (second),Duration\r\n"
        "vid1,http://y/vid1,chan,train,Mini,5,2,03:20\r\n"
        "vid2,http://y/vid2,chan,val,Full,,,10:00\r\n"
    )
    records = parse_csv_text(csv_text)
    assert len(records) == 2
    r0 = records[0]
    assert r0["videoid"] == "vid1"
    assert r0["split"] == "train"
    assert r0["subset"] == "mini"          # lowercased
    assert r0["start_discard"] == 5 and r0["end_discard"] == 2
    assert r0["length"] == 200
    assert records[1]["start_discard"] == 0   # blank -> 0

    mini = [r for r in records if r["subset"] == "mini"]
    assert len(mini) == 1 and mini[0]["videoid"] == "vid1"
