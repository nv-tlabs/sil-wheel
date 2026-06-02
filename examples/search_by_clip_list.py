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

"""Filter a wheel search by a precomputed list of clip_ids.

Workflow
--------
1. Prepare a JSON file containing the clip_ids you care about, either a
   bare array ``["clip_id_1", "clip_id_2", ...]`` or
   ``{"clip_ids": [...]}``.
2. Run this script. It POSTs the list to ``/upload_clip_list``, gets back
   a 16-hex content-addressed hash, and runs a search filtered to those
   clips. 

Usage
-----
::

    export WHEEL_PASSWORD=...
    python examples/search_by_clip_list.py \\
        --server-url http://wheel-host:8012 \\
        --username alice \\
        --clip-ids examples/sample_clip_list.json
"""
import argparse
import json
import os
from pathlib import Path

from sil_wheel.http_client import WheelHTTPClient


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server-url", required=True,
                        help="e.g. http://wheel-host:8012")
    parser.add_argument("--username")
    parser.add_argument("--clip-ids", required=True, type=Path,
                        help="Path to a JSON file with a list of clip_ids "
                             "(either a bare array or {clip_ids: [...]})")
    args = parser.parse_args()

    # Grab the user info from args or the environment (same convention as
    # examples/classifier_from_search.py and examples/cluster_from_search.py).
    username = args.username
    password = None
    if secrets := os.environ.get("WHEEL_SECRETS"):
        secrets = Path(secrets)
        if secrets.exists() and secrets.is_file():
            with secrets.open("r") as f:
                username, password = f.read().split(":")
    password = os.environ.get("WHEEL_PASSWORD", password)
    if not password:
        raise SystemExit("Set WHEEL_PASSWORD or WHEEL_SECRETS in the environment")
    if not username:
        raise SystemExit("Pass --username or set WHEEL_SECRETS")

    payload = json.loads(args.clip_ids.read_text())
    if isinstance(payload, dict):
        payload = payload.get("clip_ids", [])
    if not isinstance(payload, list) or not payload:
        raise SystemExit(f"{args.clip_ids} must contain a non-empty list of clip_ids")

    client = WheelHTTPClient(
        server_url=args.server_url, username=username, password=password,
    )
    info = client.upload_clip_list(payload)
    print(f"Uploaded: hash={info['hash']} count={info['count']} "
          f"created={info['created']}")

    result = client.search_clip_list(info["hash"])
    print(f"Search returned {len(result.clip_ids)} clips")


if __name__ == "__main__":
    main()
