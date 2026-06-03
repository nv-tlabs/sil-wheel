#!/usr/bin/env python3
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

"""Clean-room end-to-end smoke test of the usage-skill, fully off-VPN.

Simulates what a fresh external user gets: only the public files, a server URL
they control (here a local mock), and the SDK. Runs the core usage workflows
the SKILL.md documents and asserts they behave. No NVIDIA network required.

    python tests/clean_room_smoke.py

Exit code 0 = every workflow worked against a from-scratch server.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                 # mock_wheel_server
sys.path.insert(0, str(HERE.parent))          # sil_wheel package

from mock_wheel_server import MockWheel  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}  {detail}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def main() -> int:
    with MockWheel() as wheel:
        # A fresh user points the agent at THEIR server via env - no hardcoding.
        os.environ["WHEEL_SERVER_URL"] = wheel.url
        os.environ["WHEEL_USERNAME"] = "demo"
        os.environ["WHEEL_PASSWORD"] = "demo"

        from sil_wheel_agent import WheelClient  # imported AFTER env is set

        print(f"Mock SIL Wheel at {wheel.url}\n")

        client = WheelClient()
        check("login", client.login(), "authenticated against fresh server")
        check("whoami", client.whoami().get("authenticated") is True)

        total, results = client.caption_search("rain")
        check("caption_search('rain')", total > 0 and len(results) > 0,
              f"-> {total} clips")
        if results:
            url = client.clip_url(results[0].clip_id)
            check("clip_url is server-relative", wheel.url in url, url)
            check("caption parsed", bool(results[0].caption_text),
                  results[0].caption_text[:48])

        ctotal, cresults = client.classifier_search("Snow", threshold=0.5)
        check("classifier_search('Snow')", ctotal > 0,
              f"-> {ctotal} clips, score={getattr(cresults[0], 'classifier_score', None) if cresults else None}")

        # Composed search (caption + data_source).
        t2, r2 = client.search(search="construction", data_source="MADS-1M")
        check("composed search", t2 > 0, f"-> {t2} clips in MADS-1M")

        # Discovery.
        sources = client.get_data_sources()
        check("get_data_sources", "MADS-1M" in sources, str(sources))

        # Export + set operations (the training-curation workflow).
        rain_ids = client.export_search_clip_ids(search="rain", data_source="MADS-1M")
        snow_ids = client.export_search_clip_ids(search="snow", data_source="MADS-1M")
        union = WheelClient.merge_clip_id_lists(rain_ids, snow_ids)
        check("export + union", len(union) >= len(rain_ids), f"union={len(union)}")

        with tempfile.TemporaryDirectory() as d:
            client.save_clip_ids(union, "clips.txt", output_dir=d)
            out = Path(d) / "clips.txt"
            check("save_clip_ids", out.exists() and out.read_text().strip() != "")

    print(f"\n{'=' * 48}\nclean-room smoke: {PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
