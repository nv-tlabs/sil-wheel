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
#
# Large-scale clustering for a clip-id pool, on each of cosmos/visual/caption,
# with flat spherical k-means (k=1000) and always-on TF-IDF topic extraction.
# Runs sequentially (single GPU) and appends run state to runs.tsv.
#
# Configure via environment (no hardcoded paths):
#   WHEEL_DATA_DIR  dir with cosmos/visual/caption indices            (required)
#   CAPTIONS_DB     SQLite captions DB for topic extraction           (required)
#   POOLS_DIR       dir produced by build_pool_clip_ids.py            (required)
#   CLUSTER_OUT     dir to write run-id subdirs into                  (required)
#   PYTHON          python interpreter (default: python)
#   POOL            pool name = <POOL>_clip_ids.json (default: full)
#   K               number of clusters (default: 1000)
#
# NV_INFERENCE_API_KEY is intentionally unset here: the inline summarizer
# rate-limits under many threads. Extract keywords now; backfill one-phrase
# descriptions separately if desired.
set -u

: "${WHEEL_DATA_DIR:?set WHEEL_DATA_DIR}"
: "${CAPTIONS_DB:?set CAPTIONS_DB}"
: "${POOLS_DIR:?set POOLS_DIR}"
: "${CLUSTER_OUT:?set CLUSTER_OUT}"
PY="${PYTHON:-python}"
POOL="${POOL:-full}"
K="${K:-1000}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO}:${REPO}/scripts${PYTHONPATH:+:$PYTHONPATH}"
unset NV_INFERENCE_API_KEY

LOGS="${POOLS_DIR}/logs"; mkdir -p "$LOGS"
RUNS="${POOLS_DIR}/runs.tsv"
mkdir -p "$CLUSTER_OUT"

declare -A EPATH=( [cosmos]="$WHEEL_DATA_DIR"
                   [visual]="$WHEEL_DATA_DIR/visual_embeddings"
                   [caption]="$POOLS_DIR/caption_embeddings" )
declare -A ETAG=(  [cosmos]=ivf4096_pq96x8 [visual]=ivf4096_pq64x8 [caption]=ivf4096_pq256x8 )

run_one() {
  local emb=$1
  local rid; rid=$(tr -dc 'a-z0-9' </dev/urandom | head -c10)
  local out="$CLUSTER_OUT/$rid"
  local log="$LOGS/${POOL}_${emb}_${rid}.log"
  printf '%s\t%s\t%s\t%s\t%s\tSTARTED\n' "$(date -Is)" "$POOL" "$emb" "$rid" "$out" | tee -a "$RUNS"
  "$PY" "$REPO/scripts/cluster_clips_and_select.py" \
      "$out" "${EPATH[$emb]}" "$K" \
      --path_to_clip_ids "$POOLS_DIR/${POOL}_clip_ids.json" \
      --embed_type "$emb" --index_tag "${ETAG[$emb]}" \
      --spherical_kmeans \
      --captions_db "$CAPTIONS_DB" \
      > "$log" 2>&1
  local rc=$?
  printf '%s\t%s\t%s\t%s\t%s\tDONE_rc=%s\n' "$(date -Is)" "$POOL" "$emb" "$rid" "$out" "$rc" | tee -a "$RUNS"
}

echo "=== clustering pool='${POOL}' k=${K} on cosmos/visual/caption ==="
for emb in cosmos visual caption; do run_one "$emb"; done
echo "ALL RUNS DONE"
