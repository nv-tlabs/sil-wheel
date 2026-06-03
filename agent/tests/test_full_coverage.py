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

"""End-to-end coverage of the documented agent surface against a mock server.

Exercises every search mode, discovery call, curation workflow, URL builder,
and the safety/recovery edge cases the SKILL documents - all with no real
server and no network (in-process mock via the `client` fixture in conftest).

Not covered here (require a real server): VLM Judge scoring, server-side
clustering runs, 3D reconstruction, and live metrics - the mock advertises
those as unavailable so the graceful-degradation paths are what get tested.

    pytest tests/test_full_coverage.py -q
"""
import pytest

from sil_wheel_agent import WheelClient, SearchResult
from sil_wheel_agent.wheel_client import WheelZeroResultError


# ── Search modes (all return (total, results) unless noted) ──────────────

def test_caption_search(client):
    total, results = client.caption_search("rain")
    assert total > 0 and all(isinstance(r, SearchResult) for r in results)


def test_caption_search_any(client):
    total, results = client.caption_search_any(["rain", "snow"])
    assert total > 0


def test_caption_search_all_mode_long_query_warns(client, recwarn):
    # 4+ word mode='all' query should emit the pre-flight UserWarning.
    client.caption_search("rain night pedestrian crossing fog", mode="all")
    assert any(issubclass(w.category, UserWarning) for w in recwarn.list)


def test_semantic_search_by_text(client):
    total, results = client.semantic_search_by_text("rainy highway at night")
    assert total > 0


def test_semantic_search_by_clip_returns_list(client):
    results = client.semantic_search_by_clip("c0001")
    assert isinstance(results, list) and all(isinstance(r, SearchResult) for r in results)


def test_visual_search_by_text(client):
    total, results = client.visual_search_by_text("snowy road")
    assert total >= 0


def test_trajectory_search_by_clip(client):
    total, results = client.trajectory_search_by_clip("c0001")
    assert total >= 0


def test_classifier_search_scored(client):
    total, results = client.classifier_search("Snow", threshold=0.5)
    assert total > 0
    assert results[0].classifier_score is not None


def test_world_model_search(client):
    total, results = client.world_model_search("PEDESTRIAN_UNKNOWN", min_count=1)
    assert total >= 0


def test_country_search(client):
    total, results = client.country_search("US")
    assert total >= 0


def test_lookup_clip_exact(client):
    r = client.lookup_clip("c0001")
    assert r is not None and r.clip_id == "c0001"


def test_lookup_clip_missing_returns_none(client):
    assert client.lookup_clip("does-not-exist-uuid") is None


def test_composed_search(client):
    total, results = client.search(search="construction", data_source="MADS-1M", n=10)
    assert total > 0


# ── Discovery ────────────────────────────────────────────────────────────

def test_get_data_sources(client):
    assert "MADS-1M" in client.get_data_sources()


def test_get_classifiers(client):
    cl = client.get_classifiers()
    assert isinstance(cl, dict)


def test_list_classifiers(client):
    names = client.list_classifiers()
    assert "Snow" in names


def test_resolve_classifier_name(client):
    resolved = client.resolve_classifier_name("snow")
    assert resolved and "snow" in resolved.lower()


# ── Curation workflows ────────────────────────────────────────────────────

def test_find_clips_for_scenario(client):
    out = client.find_clips_for_scenario("construction zone in rain", data_source="MADS-1M")
    assert isinstance(out, dict) and out


def test_find_clips_for_scenario_ids(client):
    ids = client.find_clips_for_scenario_ids("construction zone in rain", data_source="MADS-1M")
    assert isinstance(ids, list) and len(ids) > 0


def test_find_similar_to_clip(client):
    sim = client.find_similar_to_clip("c0001", n=5)
    assert isinstance(sim, dict) and ("cosmos" in sim or "trajectory" in sim)


def test_expand_clip_set(client):
    expanded = client.expand_clip_set(["c0001", "c0002"], n_similar_per_clip=5, max_total=20)
    assert isinstance(expanded, list) and len(expanded) > 0


def test_export_search_clip_ids(client):
    ids = client.export_search_clip_ids(search="rain", data_source="MADS-1M")
    assert isinstance(ids, list) and len(ids) > 0


def test_multi_search_export(client):
    combined = client.multi_search_export([
        {"search": "rain", "data_source": "MADS-1M"},
        {"search": "snow", "data_source": "MADS-1M"},
    ])
    assert isinstance(combined, list) and len(combined) > 0


# ── URL builders + formatting (pure client-side) ──────────────────────────

def test_url_builders(client):
    cid = "c0001"
    assert client.clip_url(cid).endswith(cid) or cid in client.clip_url(cid)
    assert cid in client.video_url(cid)
    assert "search" in client.search_url(search="rain")


def test_format_results_with_urls(client):
    _, results = client.caption_search("rain")
    md = client.format_results_with_urls(results)
    assert isinstance(md, str) and results[0].clip_id in md


# ── Graceful degradation for server-dependent features ────────────────────

def test_vlm_judge_disabled_gracefully(client):
    status = client.vlm_judge_status()
    assert status.get("enabled") is False


def test_check_sdk_version_graceful(client):
    info = WheelClient.check_sdk_version(timeout=5)
    # The public skill URL may 404 / be unreachable - must not raise, must report.
    assert "local" in info


def test_check_connection_healthy(client):
    diag = client.check_connection(probe_vpn=False)
    assert diag.get("reachable") is True
    assert diag.get("diagnosis") == "healthy"


# ── Safety + recovery edge cases ──────────────────────────────────────────

def test_wheel_readonly_blocks_writes(client, monkeypatch):
    monkeypatch.setenv("WHEEL_READONLY", "1")
    assert client.is_production is True
    result = client.upload_labels(["c0001"], "TestLabel", project="TestProject")
    # Must be blocked (no successful upload) - returns an error sentinel.
    assert result is None or "read" in str(result).lower() or "error" in str(result).lower()


def test_writes_allowed_by_default(client):
    # No WHEEL_READONLY -> not "production" -> writes are permitted.
    assert client.is_production is False


def test_strict_mode_raises_on_composed_zero(client, monkeypatch):
    monkeypatch.setenv("WHEEL_STRICT", "1")
    with pytest.raises(WheelZeroResultError):
        client.search(search="rain", classifier_select="Snow",
                      data_source="NO_SUCH_SOURCE")


def test_diagnose_zero_results_never_raises(client):
    diag = client.diagnose_zero_results(query="zzz", data_source="NO_SUCH_SOURCE")
    assert isinstance(diag, dict) and "summary" in diag
