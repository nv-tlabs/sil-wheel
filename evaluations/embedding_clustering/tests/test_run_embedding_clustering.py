# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner orchestration tests: arg parsing + graceful skips (no data/SDK)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_embedding_clustering as run


def test_parse_args_defaults_and_stage_selection():
    args = run.parse_args(["--output-dir", "/tmp/x"])
    assert args.stages == list(run.ALL_STAGES)
    assert args.k == 1000 and args.embeds == ["cosmos", "caption", "visual"]
    picked = run.parse_args(["--output-dir", "/tmp/x", "--stages", "flat", "preindex"])
    assert picked.stages == ["flat", "preindex"]


def test_main_skips_all_stages_without_inputs_and_writes_summary(tmp_path, monkeypatch):
    # no env-provided inputs -> every stage should skip, not crash
    for var in ("WHEEL_DATA_DIR", "CAPTIONS_DB", "POOLS_DIR", "CLUSTER_OUT"):
        monkeypatch.delenv(var, raising=False)
    rc = run.main(["--output-dir", str(tmp_path)])
    assert rc == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert set(summary["stages"]) == set(run.ALL_STAGES)
    assert all(v["status"].startswith("skipped") for v in summary["stages"].values())
