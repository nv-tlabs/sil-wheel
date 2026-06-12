#!/usr/bin/env python
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

"""Upload clustering run dirs to the wheel server (login -> upload_clustering).

Server + creds from ``WHEEL_URL`` / ``WHEEL_USERNAME`` / ``WHEEL_PASSWORD`` (env),
overridable by flags. Always uploads with ``overwrite=True`` so re-uploads refresh
the server copy (e.g. after backfilling per-cluster themes).

    WHEEL_URL=http://sil-wheel.nvidia.com:8000 python upload_runs.py \
        --clustering-dir /media/.../clustering_pai_complete \
        --runs k50_cosmos k50_caption k50_visual
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from sil_wheel.http_client import WheelHTTPClient


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clustering-dir", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="run-id subdirs to upload")
    ap.add_argument("--url", default=os.environ.get("WHEEL_URL"))
    ap.add_argument("--username", default=os.environ.get("WHEEL_USERNAME"))
    ap.add_argument("--password", default=os.environ.get("WHEEL_PASSWORD"))
    args = ap.parse_args(argv)
    if not (args.url and args.username and args.password):
        raise SystemExit("need --url/--username/--password (or WHEEL_URL/_USERNAME/_PASSWORD)")

    client = WheelHTTPClient(args.url)
    client.login(args.username, args.password)
    for rid in args.runs:
        run_dir = args.clustering_dir / rid
        resp = client.upload_clustering_run(run_dir, run_id=rid, overwrite=True)
        print(f"uploaded {rid}: {resp}  ->  {args.url}/?clustering_run={rid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
