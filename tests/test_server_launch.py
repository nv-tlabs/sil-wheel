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

"""End-to-end smoke test: build a minimal RequestHandler stack and verify
the HTTP server starts, serves a static page, and answers `/whoami`.

Heavy stores (FAISS, S3, predictions) are stubbed with MagicMock because the
endpoints we exercise here don't touch them. The point of this test is to
catch regressions in import paths, handler construction, and request
routing — not to validate any search backend.
"""
import json
import threading
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

# `scripts/` is on sys.path via pyproject.toml's [tool.pytest.ini_options].
import launch_server

from sil_wheel.search.search_pipeline import SearchPipeline


def _make_static_pages():
    root = Path(launch_server.__file__).resolve().parent.parent / "sil_wheel" / "app" / "static"
    return launch_server.StaticPages(
        {
            "/login": ("text/html", root / "html/login.html"),
            "/style.css": ("text/css", root / "css/style.css"),
            "/login.css": ("text/css", root / "css/login.css"),
            "/main.js": ("text/javascript", root / "js/main.js"),
        }
    )


@pytest.fixture()
def server(data_store, caption_db, users_db):
    """Start a RequestHandler-backed HTTP server on a random local port.

    Yields the base URL. Heavy stores are MagicMock; `data_store`,
    `caption_db`, and `users_db` come from the real fixtures.
    """
    embeddingsstore = MagicMock()
    clipembeddingsstore = MagicMock()
    captionembeddingsstore = MagicMock()
    trajectorystore = MagicMock()
    classifiersearch = MagicMock()
    clustersearch = MagicMock()
    cliplistsearch = MagicMock()
    metricstore = MagicMock()
    predictionsstore = MagicMock()
    autolabelsstore = MagicMock()
    bev_fetcher = MagicMock()
    wm_store = MagicMock()
    video_fetcher = MagicMock()

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
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _get(url):
    return urlopen(Request(url, method="GET"), timeout=5)


def _post(url, payload, cookie=None):
    headers = {"Content-Type": "text/plain"}
    if cookie is not None:
        headers["Cookie"] = cookie
    return urlopen(
        Request(url, data=payload.encode(), headers=headers, method="POST"),
        timeout=5,
    )


def _login_cookie(users_db):
    uid = users_db.create_user("alice", "pwd123")
    sid = users_db.create_session(uid)
    return f"{launch_server.SESSION_COOKIE}={sid}"


def test_login_page_served(server):
    resp = _get(f"{server}/login")
    assert resp.status == 200
    assert resp.headers.get("Content-Type") == "text/html"
    body = resp.read().lower()
    assert b"<html" in body or b"<!doctype" in body


def test_static_css(server):
    resp = _get(f"{server}/style.css")
    assert resp.status == 200
    assert resp.headers.get("Content-Type") == "text/css"


def test_favicon(server):
    resp = _get(f"{server}/favicon.ico")
    assert resp.status == 200


def test_whoami_unauthenticated(server):
    resp = _get(f"{server}/whoami")
    assert resp.status == 200
    payload = json.loads(resp.read().decode("utf-8"))
    assert payload == {"authenticated": False}


@pytest.mark.parametrize(
    "path",
    [
        "/videos",
        "/metrics",
        "/per_clip_metrics",
        "/predictions",
        "/current_search.csv",
        "/clip_ids",
        "/data_stats_list",
    ],
)
def test_endpoints_reject_unauthenticated(server, path):
    # These endpoints used to AttributeError on `user.id` for a missing
    # session cookie; they should now return a 403 JSON body via
    # _require_user().
    with pytest.raises(HTTPError) as excinfo:
        _get(f"{server}{path}")
    assert excinfo.value.code == 403


@pytest.mark.parametrize(
    "payload",
    [
        "add::ds-A-clip-05::ann-x::lbl::-1::-1::proj-A",
        "remove::ds-A-clip-01::ann-01::turn_left::-1::-1::proj-A",
        "verify::ds-A-clip-01::ann-01::turn_left::-1::-1::proj-A",
        "update_times::ds-A-clip-01::ann-01::turn_left::0::1::proj-A",
        "mass_label::lbl::ds-A-clip-01,ds-A-clip-02::proj-A",
        "upload_annotations::ds-A-clip-01::lbl::-1::-1::proj-A::",
        "upload_captions::model::src-A::",
        "reconstruction::ds-A-clip-01::NuRec",
    ],
    ids=lambda p: p.split("::")[0],
)
def test_core_post_rejects_unauthenticated(server, payload):
    # handle_core_post mutates annotations/captions or launches jobs, so
    # every core action must 403 without a session cookie (previously only
    # auto_label was gated).
    with pytest.raises(HTTPError) as excinfo:
        _post(f"{server}/", payload)
    assert excinfo.value.code == 403


def test_core_post_add_succeeds_authenticated(server, data_store, users_db):
    # A logged-in user still lands the write: the blanket gate must not
    # over-block the authenticated path.
    cookie = _login_cookie(users_db)
    resp = _post(
        f"{server}/",
        "add::ds-A-clip-05::ann-new::my_label::-1::-1::proj-A",
        cookie=cookie,
    )
    assert resp.status == 200
    row = data_store.conn.execute(
        "SELECT key, project FROM annotations WHERE uid = ?", ("ann-new",)
    ).fetchone()
    assert row is not None
    assert row["key"] == "my_label"
    assert row["project"] == "proj-A"
