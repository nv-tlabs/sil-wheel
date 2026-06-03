#!/usr/bin/env bash
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

# Prove the usage-skill works for a fresh, off-VPN user.
#
# Makes a faithful copy of the whole agent/ folder into a throwaway temp dir
# (no .git, no NVIDIA env, no VPN), installs deps into a fresh venv, then runs:
#   1. the full pytest suite (offline smoke + full coverage + CLI + doc accuracy)
#   2. the clean-room narrative end-to-end (mock server)
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

# Faithful copy of the whole agent/ folder - mimic a user who downloaded the
# release (SKILL.md, knowledge/, examples/ included, not just code + tests).
DEST="$TMP/agent"
mkdir -p "$DEST"
cp -R "$AGENT_DIR/." "$DEST/"
find "$DEST" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf "$DEST/.pytest_cache" "$DEST/.venv" 2>/dev/null || true

cd "$DEST"
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
