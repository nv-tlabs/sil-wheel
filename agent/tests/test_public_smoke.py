"""Offline unit smoke for the public SIL Wheel Agent (no network, no server).

Validates that the sanitized SDK parses server payloads, builds URLs, and runs
its pure-Python helpers. These never touch the network, so they pass anywhere -
the first signal that the public package is intact.

    pytest tests/test_public_smoke.py -q
"""
from sil_wheel_agent import WheelClient, SearchResult
from wheel_client import PROD_SERVER, DEV_SERVER


def test_no_internal_hosts_in_defaults():
    for const in (PROD_SERVER, DEV_SERVER, WheelClient._VPN_PROBE_URL):
        assert "nvidia.com" not in const, f"internal host leaked into default: {const}"
        assert "10.110." not in const, f"internal IP leaked into default: {const}"


def test_search_result_parses_video_dict():
    v = {
        "clip_id": "abc-123",
        "data_source": "MADS-1M",
        "captions": {"qwen2.5-7b": [{"caption": "a rainy highway"}]},
        "classifier_score": 0.77,
    }
    r = SearchResult.from_video_dict(v)
    assert r.clip_id == "abc-123"
    assert r.data_source == "MADS-1M"
    assert r.caption_text == "a rainy highway"
    assert r.classifier_score == 0.77


def test_clip_url_uses_configured_server():
    client = WheelClient(base_url="http://localhost:8000")
    url = client.clip_url("abc-123")
    assert url.startswith("http://localhost:8000")
    assert "abc-123" in url


def test_set_operations():
    a = ["c1", "c2", "c3"]
    b = ["c2", "c3", "c4"]
    assert set(WheelClient.merge_clip_id_lists(a, b)) == {"c1", "c2", "c3", "c4"}
    assert set(WheelClient.intersect_clip_id_lists(a, b)) == {"c2", "c3"}
    assert set(WheelClient.subtract_clip_id_lists(a, b)) == {"c1"}


def test_save_load_clip_ids_roundtrip(tmp_path):
    ids = ["c1", "c2", "c3"]
    client = WheelClient(base_url="http://localhost:8000")
    client.save_clip_ids(ids, "clips.txt", output_dir=str(tmp_path))
    out = tmp_path / "clips.txt"
    assert out.exists()
    assert WheelClient.load_clip_ids(str(out)) == ids
