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

"""
Test for the VLM Judge in the Wheel server.

Usage:
  # Use clip IDs from your annotations DB (requires video_paths table)
  python tests/test_vlm_judge.py --annotations_db /path/to/annotations.db --server_url http://localhost:8000

  # Test a specific clip (e.g. from the UI)
  python tests/test_vlm_judge.py --server_url http://localhost:8010 --clip_id 54a0937e-5972-4a3d-a6f4-5cff2516776c
"""
import argparse
import json
import sqlite3
import sys


def get_clip_ids_from_db(annotations_db: str, limit: int = 3) -> list[str]:
    """Return up to `limit` clip_ids that have a video_path (so the judge can fetch the video)."""
    try:
        conn = sqlite3.connect(annotations_db)
        rows = conn.execute(
            "SELECT clip_id FROM video_paths LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            print(
                f"Error: {e}. Use the annotations DB (same as launch_server first arg), not the caption DB.",
                file=sys.stderr,
            )
        else:
            print(f"Error reading DB: {e}", file=sys.stderr)
        sys.exit(1)


def call_caption_score(server_url: str, clip_id: str, caption: str) -> dict:
    """Call launch_server /api/vlm_judge/caption_score and return JSON."""
    import urllib.request
    import urllib.error

    sample_caption = caption or "The vehicle drives forward on a clear road. The dashboard is visible at the bottom of the frame."

    from urllib.parse import quote

    base = server_url.rstrip("/")
    url = f"{base}/api/vlm_judge/caption_score?clip_id={quote(clip_id)}&caption={quote(sample_caption)}"
    req = urllib.request.Request(url, method="GET")

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        try:
            err = json.loads(body)
            return err
        except Exception:
            return {"error": f"HTTP {e.code}: {body or e.reason}"}
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser(
        description="Test launch_server in-process VLM Judge caption_score API"
    )
    ap.add_argument("--annotations_db", help="Annotations DB to read clip_ids from (must have video_paths)")
    ap.add_argument(
        "--server_url",
        default="http://localhost:8000",
        help="Wheel server base URL (served by scripts/launch_server.py)",
    )
    ap.add_argument("--clip_id", help="Specific clip_id to test (else use first from DB)")
    ap.add_argument("--caption", help="Caption text to score (default: sample caption)")
    ap.add_argument("-n", type=int, default=1, help="If using DB, how many clip_ids to try (default 1)")
    args = ap.parse_args()

    clip_ids = []
    if args.clip_id:
        clip_ids = [args.clip_id]
    elif args.annotations_db:
        clip_ids = get_clip_ids_from_db(args.annotations_db, limit=args.n)
        if not clip_ids:
            print("No clip_ids found in video_paths.", file=sys.stderr)
            sys.exit(1)
        print(f"Using clip_ids from DB: {clip_ids}")
    else:
        print("Provide --annotations_db or --clip_id.", file=sys.stderr)
        sys.exit(1)

    for i, cid in enumerate(clip_ids):
        print(f"\n--- Request {i+1}: clip_id={cid} ---")
        out = call_caption_score(args.server_url, cid, args.caption)
        print(json.dumps(out, indent=2))
        if out.get("error"):
            print("(Request failed.)")
        elif out.get("scores"):
            print("(Success: scores and reasoning above.)")


if __name__ == "__main__":
    main()
