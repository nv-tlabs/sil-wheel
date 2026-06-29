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

from types import SimpleNamespace

import orjson

from run_benchmark import load_opendv_split


def _write_jsonl(path, records):
    lines = b"\n".join(orjson.dumps(r) for r in records)
    path.write_bytes(lines + b"\n")


def test_opendv_uses_clip_id_directly(tmp_path):
    # The GT stores the clip-path basename "<vid>_<seg>" matching the
    # embeddings, so the clip_id is used as-is -- including ids whose YouTube
    # video id itself contains "__" (which a naive split would mangle).
    p = tmp_path / "gt.jsonl"
    p.write_bytes(
        orjson.dumps({"clip_id": "vidA_10-20", "long": "l1"}) + b"\n"
        + b"\n"  # blank lines are skipped
        + orjson.dumps({"clip_id": "e6BAP___NYQ_30-40", "long": "l2"}) + b"\n"
    )
    split = load_opendv_split(
        SimpleNamespace(gt_path=p, caption_length="long")
    )
    assert split.video_ids == ["vidA_10-20", "e6BAP___NYQ_30-40"]
    assert split.sentences == ["l1", "l2"]


def test_opendv_selects_caption_length(tmp_path):
    p = tmp_path / "gt.jsonl"
    _write_jsonl(p, [
        {"clip_id": "v_0-1", "short": "S", "medium": "M", "long": "L"},
    ])
    for length, expected in [("short", "S"), ("medium", "M"), ("long", "L")]:
        split = load_opendv_split(
            SimpleNamespace(gt_path=p, caption_length=length)
        )
        assert split.sentences == [expected]
