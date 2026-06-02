#!/usr/bin/env python3
"""Minimal SIL Wheel Agent quickstart.

Requires a reachable SIL Wheel server. Set these first (or put them in .env):
    WHEEL_SERVER_URL=http://your-sil-wheel-host:8000
    WHEEL_USERNAME=...
    WHEEL_PASSWORD=...

    python examples/quickstart.py "construction zone in rain"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sil_wheel_agent import WheelClient


def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else "construction zone in rain"

    client = WheelClient()
    if not client.login():
        diag = client.check_connection()
        print(f"Login failed. Connection diagnosis: {diag.get('diagnosis')}")
        print("Check WHEEL_SERVER_URL / WHEEL_USERNAME / WHEEL_PASSWORD in .env.")
        return 1

    total, results = client.search(search=query, data_source="MADS-1M")
    print(f"'{query}' -> {total} clips\n")
    for r in results[:5]:
        print(f"  {r.clip_id}  {client.clip_url(r.clip_id)}")
        if r.caption_text:
            print(f"      {r.caption_text[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
