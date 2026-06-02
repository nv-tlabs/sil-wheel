# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for /upload_clip_list, /clip_list, and the clip_id_list_hash filter."""
import json
import threading
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

import launch_server
from sil_wheel.search.search_pipeline import SearchPipeline
from sil_wheel.stores.clip_list_search import ClipListSearch


def _make_static_pages():
    root = Path(launch_server.__file__).resolve().parent.parent / "sil_wheel" / "app" / "static"
    return launch_server.StaticPages({
        "/login": ("text/html", root / "html/login.html"),
    })


def _passthrough():
    m = MagicMock()
    m.search = MagicMock(side_effect=lambda filters, results: results)
    return m


@pytest.fixture()
def clip_lists_dir(tmp_path):
    return tmp_path / "clip_lists"


@pytest.fixture()
def server(data_store, caption_db, users_db, clip_lists_dir):
    """HTTP server with a real ClipListSearch + datastore for end-to-end filtering."""
    embeddingsstore = _passthrough()
    clipembeddingsstore = _passthrough()
    captionembeddingsstore = _passthrough()
    trajectorystore = _passthrough()
    classifiersearch = _passthrough()
    clustersearch = _passthrough()
    metricstore = _passthrough()
    predictionsstore = MagicMock()
    autolabelsstore = MagicMock()
    bev_fetcher = None  # Pipeline skips when None; keeps chain clean.
    metricstore.has_metrics_index = MagicMock(return_value=False)
    wm_store = _passthrough()
    video_fetcher = MagicMock()

    cliplistsearch = ClipListSearch(clip_lists_dir)

    pipeline = SearchPipeline(
        datastore=data_store,
        captionstore=caption_db,
        captionembeddingsstore=captionembeddingsstore,
        embeddingsstore=embeddingsstore,
        clipembeddingsstore=clipembeddingsstore,
        classifiersearch=classifiersearch,
        clustersearch=clustersearch,
        cliplistsearch=cliplistsearch,
        trajectorystore=trajectorystore,
        metricstore=metricstore,
        bev_fetcher=bev_fetcher,
        wm_store=wm_store,
    )

    Handler = partial(
        launch_server.RequestHandler,
        datastore=data_store,
        captionstore=caption_db,
        trajectorystore=trajectorystore,
        embeddingsstore=embeddingsstore,
        clipembeddingsstore=clipembeddingsstore,
        captionembeddingsstore=captionembeddingsstore,
        classifiersearch=classifiersearch,
        wm_store=wm_store,
        classifier_dir="/tmp/classifiers",
        metricstore=metricstore,
        predictionsstore=predictionsstore,
        autolabelsstore=autolabelsstore,
        search_pipeline=pipeline,
        favicon=launch_server.load_favicon("images/car.png"),
        static_pages=_make_static_pages(),
        video_fetcher=video_fetcher,
        usersstore=users_db,
        classifier_jobs=MagicMock(),
        nurec_job=MagicMock(),
        clips_to_apis={},
        bev_fetcher=bev_fetcher,
        clustering_jobs=MagicMock(),
        clustering_dir="/tmp/clustering",
        clustersearch=clustersearch,
        cliplistsearch=cliplistsearch,
        slack_notifier=None,
        bug_report_spreadsheet_id=None,
        bug_report_credential_path=None,
        rewriter=None,
        arena_store=None,
        vlm_judge=None,
    )

    httpd = launch_server.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "url": f"http://127.0.0.1:{port}",
            "clip_lists_dir": clip_lists_dir,
            "cliplistsearch": cliplistsearch,
            "users_db": users_db,
        }
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _post(server, body, content_type="application/json"):
    return urlopen(
        Request(
            f"{server['url']}/upload_clip_list",
            data=body if isinstance(body, bytes) else body.encode(),
            headers={"Content-Type": content_type},
            method="POST",
        ),
        timeout=10,
    )


def _get_json(url, cookie=None):
    headers = {}
    if cookie is not None:
        headers["Cookie"] = cookie
    return json.loads(
        urlopen(Request(url, method="GET", headers=headers), timeout=10).read()
    )


def _login(server):
    users_db = server["users_db"]
    # Grant access to the two data sources used by the data_store
    # fixture so the search isn't filtered to empty.
    users_db.grant_datasource_to_all_users("src-A")
    users_db.grant_datasource_to_all_users("src-B")
    uid = users_db.create_user("alice", "pwd123")
    users_db.grant_datasource_to_all_users("src-A")
    users_db.grant_datasource_to_all_users("src-B")
    sid = users_db.create_session(uid)
    return f"{launch_server.SESSION_COOKIE}={sid}"


def test_upload_round_trip(server):
    resp = _post(server, json.dumps(["ds-A-clip-01", "ds-A-clip-02"]))
    assert resp.status == 201
    body = json.loads(resp.read())
    assert body["count"] == 2
    assert body["created"] is True
    assert len(body["hash"]) == 16
    # File appears on disk.
    assert (server["clip_lists_dir"] / f"{body['hash']}.json").exists()


def test_upload_idempotent(server):
    r1 = json.loads(_post(server, json.dumps(["a", "b"])).read())
    r2 = json.loads(_post(server, json.dumps(["b", "a"])).read())
    assert r1["hash"] == r2["hash"]
    assert r1["created"] is True
    assert r2["created"] is False


def test_upload_no_auth_required(server):
    # No cookies, no Authorization — endpoint accepts.
    resp = _post(server, json.dumps(["x"]))
    assert resp.status == 201


def test_upload_accepts_object_form(server):
    resp = _post(server, json.dumps({"clip_ids": ["x", "y"]}))
    assert resp.status == 201
    body = json.loads(resp.read())
    assert body["count"] == 2


def test_upload_rejects_empty_body(server):
    with pytest.raises(HTTPError) as e:
        _post(server, b"")
    assert e.value.code == 400


def test_upload_rejects_non_json(server):
    with pytest.raises(HTTPError) as e:
        _post(server, b"not json at all")
    assert e.value.code == 400


def test_upload_rejects_empty_list(server):
    with pytest.raises(HTTPError) as e:
        _post(server, json.dumps([]))
    assert e.value.code == 400


def test_upload_rejects_non_string_element(server):
    with pytest.raises(HTTPError) as e:
        _post(server, json.dumps(["ok", 42]))
    assert e.value.code == 400


def test_upload_rejects_too_long_element(server):
    with pytest.raises(HTTPError) as e:
        _post(server, json.dumps(["ok", "x" * 300]))
    assert e.value.code == 400


def test_get_clip_list_returns_stored(server):
    r = json.loads(_post(server, json.dumps(["z", "a", "m"])).read())
    body = _get_json(f"{server['url']}/clip_list?hash={r['hash']}")
    assert body["hash"] == r["hash"]
    assert body["count"] == 3
    assert body["clip_ids"] == ["a", "m", "z"]  # sorted


def test_get_clip_list_unknown_hash_404(server):
    with pytest.raises(HTTPError) as e:
        _get_json(f"{server['url']}/clip_list?hash=0000000000000000")
    assert e.value.code == 404


def test_get_clip_list_malformed_hash_400(server):
    with pytest.raises(HTTPError) as e:
        _get_json(f"{server['url']}/clip_list?hash=not-hex")
    assert e.value.code == 400


def test_get_clip_list_missing_hash_400(server):
    with pytest.raises(HTTPError) as e:
        _get_json(f"{server['url']}/clip_list")
    assert e.value.code == 400


def test_filter_applies_to_search_end_to_end(server):
    # Upload a list with 2 corpus clips + 1 nonexistent.
    r = json.loads(_post(
        server, json.dumps(["ds-A-clip-01", "ds-B-clip-02", "does-not-exist"])
    ).read())
    # The /clip_ids search endpoint already pipes through SearchPipeline,
    # which now applies clip_id_list_hash via ClipListSearch.search.
    qs = urlencode({"clip_id_list_hash": r["hash"], "project_source": "proj-A"})
    body = _get_json(f"{server['url']}/clip_ids?{qs}", cookie=_login(server))
    assert sorted(body["clip_ids"]) == ["ds-A-clip-01", "ds-B-clip-02"]


def test_filter_with_unknown_hash_returns_no_clips(server):
    qs = urlencode({"clip_id_list_hash": "0000000000000000", "project_source": "proj-A"})
    body = _get_json(f"{server['url']}/clip_ids?{qs}", cookie=_login(server))
    assert body["clip_ids"] == []
