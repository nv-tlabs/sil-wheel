<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Knowledge Index

What each knowledge file covers and when to read it. These docs give an agent
the context to use the SIL Wheel SDK well. Start with the top of this list.

## Reading order

| # | File | What it covers | When to read |
|---|------|----------------|--------------|
| 1 | `overview.md` | Architecture, data coverage, search modes, API endpoints | Start here for the big picture |
| 2 | `search-modes.md` | Every search mode with parameters and examples | Before choosing a search strategy |
| 3 | `timing-reference.md` | Measured latencies per operation | Before running searches (fast vs slow choices) |
| 4 | `server-policy.md` | Read-only vs write rules, deep-link URL formats, safety | Before any write operations |
| 5 | `anti-patterns.md` | Recurring SDK footguns + fixes, and the `diagnose_zero_results()` / `WHEEL_STRICT=1` recovery paths | **First stop** when a search returns 0 or "wrong source" |
| 6 | `search-tool-calibration.md` | Honest precision/recall by tool and query type | Before reporting search results to a teammate or system |
| 7 | `feature-compat.md` | Which features work on which data sources (the silent-0 matrix) | Before composing classifier x source or trajectory x source filters |
| 8 | `support.md` | Where to get help | When stuck |

## Cross-reference

| Topic | Primary doc | Also in |
|-------|-------------|---------|
| Search modes & params | `search-modes.md` | `SKILL.md` |
| API timings (authoritative) | `timing-reference.md` | `search-modes.md`, `SKILL.md` |
| Server caps + silent failures | `timing-reference.md` | `anti-patterns.md` |
| SDK footguns + recovery | `anti-patterns.md` | `timing-reference.md`, `SKILL.md` |
| Strict mode (`WHEEL_STRICT=1`) | `anti-patterns.md` | `sil_wheel_agent/wheel_client.py` (`WheelZeroResultError`) |
| `diagnose_zero_results()` flow | `anti-patterns.md` | `sil_wheel_agent/wheel_client.py` |
| Precision/recall by tool | `search-tool-calibration.md` | `feature-compat.md` |
| Feature x data-source matrix | `feature-compat.md` | `anti-patterns.md` |
| Deep-link URL format | `server-policy.md` | `overview.md` |
| Read-only / write safety | `server-policy.md` | `SKILL.md` (`WHEEL_READONLY`) |
| Client methods & SearchResult | `SKILL.md` | `overview.md` |
| Clip ID format (MADS-1M) | `overview.md` | `anti-patterns.md` |

## Key files in this package

| File | Purpose |
|------|---------|
| `SKILL.md` | The usage skill an agent reads to set up + drive the API |
| `sil_wheel_agent/wheel_client.py` | The SDK (Python + CLI, all search modes) |
| `sil_wheel_agent/__init__.py` | `from sil_wheel_agent import WheelClient` |
| `tests/test_public_smoke.py` | Offline unit smoke (no network) |
| `tests/mock_wheel_server.py` + `tests/clean_room_smoke.py` | Off-server end-to-end usage check |
| `tests/run_clean_room.sh` | Fresh-venv clean-room runner |