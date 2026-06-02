#!/usr/bin/env bash
# Prove the usage-skill works for a fresh, off-VPN user.
#
# Copies ONLY the public files into a throwaway temp dir (no .git, no NVIDIA
# env, no VPN), installs deps into a fresh venv, then runs:
#   1. offline unit smoke (pytest)        - the SDK parses/builds correctly
#   2. clean-room end-to-end (mock server) - the documented workflows run
#
# Usage:  bash tests/run_clean_room.sh
set -euo pipefail

AGENT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Pick a Python >= 3.10 (the SDK targets 3.10+). Override with PYTHON=...
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for cand in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
  done
fi
echo "Clean room: $TMP  (python: $("$PY" --version 2>&1))"

# Copy the public surface only - mimic a user who downloaded the release.
cp -R "$AGENT_DIR/sil_wheel_agent" "$AGENT_DIR/tests" "$AGENT_DIR/requirements.txt" "$TMP/"

cd "$TMP"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt

echo ""
echo "== 1. full pytest suite (offline smoke + full coverage + CLI, all vs mock) =="
python -m pytest tests/ -q

echo ""
echo "== 2. clean-room narrative end-to-end (mock SIL Wheel) =="
python tests/clean_room_smoke.py

echo ""
echo "Clean-room run complete."
