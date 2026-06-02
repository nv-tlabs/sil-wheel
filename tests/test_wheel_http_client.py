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

"""Tests for WheelHTTPClient — remote client over a running wheel server.

Boots the same MagicMock-backed server fixture pattern used elsewhere,
authenticates via UsersDataStore, and asserts the remote client's results
match the local pipeline's for the same queries.
"""
import threading
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import launch_server
from sil_wheel.client import WheelClient
from sil_wheel.cluster_build import build_clustering_run
from sil_wheel.http_client import WheelHTTPClient
from sil_wheel.search.search_pipeline import SearchPipeline


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
def server_and_local(data_store, caption_db, users_db, tmp_path):
    """Run a real HTTP server backed by data_store + caption_db.

    Yields ``(remote_client, local_client, session_cookie)`` so each test
    can compare a remote query against the same query run in-process.
    """
    embeddingsstore = _passthrough()
    clipembeddingsstore = _passthrough()
    captionembeddingsstore = _passthrough()
    trajectorystore = _passthrough()
    classifiersearch = _passthrough()
    clustersearch = _passthrough()
    cliplistsearch = _passthrough()
    metricstore = _passthrough()
    predictionsstore = MagicMock()
    autolabelsstore = MagicMock()
    bev_fetcher = MagicMock()
    bev_fetcher.has_bev_index = MagicMock(return_value=False)
    metricstore.has_metrics_index = MagicMock(return_value=False)
    wm_store = _passthrough()
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
        clustering_dir=str(tmp_path / "clustering"),
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

    users_db.create_user("alice", "pwd123")

    remote = WheelHTTPClient(
        server_url=f"http://127.0.0.1:{port}",
        username="alice",
        password="pwd123",
    )
    local = WheelClient(pipeline=pipeline)

    try:
        yield remote, local
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_remote_search_caption_matches_local(server_and_local):
    """Remote search_caption returns the same clip_ids the local pipeline does."""
    remote, local = server_and_local
    r_remote = remote.search_caption("cyclist")
    r_local = local.search_caption("cyclist")
    assert r_remote.clip_ids == r_local.clip_ids


def test_remote_search_country_matches_local(server_and_local):
    remote, local = server_and_local
    r_remote = remote.search_country("DE")
    r_local = local.search_country("DE")
    assert sorted(r_remote.clip_ids) == sorted(r_local.clip_ids)


def test_remote_search_returns_empty_scores(server_and_local):
    """Remote search returns clip_ids but empty scores by design — the
    /clip_ids endpoint doesn't carry per-modality breakdowns."""
    remote, _ = server_and_local
    result = remote.search_caption("cyclist")
    assert result.scores == {}


def test_whoami_with_session(server_and_local):
    remote, _ = server_and_local
    info = remote.whoami()
    assert info.get("authenticated") is True


def test_whoami_without_session_unauthenticated(server_and_local):
    remote, _ = server_and_local
    anon = WheelHTTPClient(server_url=remote.server_url)
    info = anon.whoami()
    assert info == {"authenticated": False}


def test_constructor_requires_server_url():
    with pytest.raises(ValueError, match="server_url"):
        WheelHTTPClient(server_url="")


def test_constructor_requires_both_user_and_password():
    with pytest.raises(ValueError, match="username and password"):
        WheelHTTPClient(server_url="http://x:1", username="alice")
    with pytest.raises(ValueError, match="username and password"):
        WheelHTTPClient(server_url="http://x:1", password="pwd")


def test_login_with_bad_credentials_raises(server_and_local):
    remote, _ = server_and_local
    with pytest.raises(RuntimeError, match="invalid credentials"):
        WheelHTTPClient(
            server_url=remote.server_url,
            username="alice",
            password="wrong",
        )


def test_search_from_url_full_url_matches_kwargs(server_and_local):
    """A copy-pasted browser URL produces the same result as the kwargs form."""
    remote, _ = server_and_local
    url = f"{remote.server_url}/?search=cyclist"
    r_url = remote.search_from_url(url)
    r_kwargs = remote.search(search="cyclist")
    assert r_url.clip_ids == r_kwargs.clip_ids


def test_search_from_url_query_only(server_and_local):
    remote, _ = server_and_local
    r_url = remote.search_from_url("search_country=DE")
    r_kwargs = remote.search_country("DE")
    assert sorted(r_url.clip_ids) == sorted(r_kwargs.clip_ids)


def test_search_from_url_works_on_local_client(server_and_local):
    """search_from_url is on the base, so the local pipeline path also has it."""
    _, local = server_and_local
    r_url = local.search_from_url("?search_country=DE")
    r_kwargs = local.search_country("DE")
    assert sorted(r_url.clip_ids) == sorted(r_kwargs.clip_ids)


def test_search_from_url_handles_hash_fragment(server_and_local):
    """The wheel UI puts filters in the hash fragment, not the query string."""
    remote, _ = server_and_local
    hash_url = f"{remote.server_url}/#&search_country=DE&page=0"
    r_hash = remote.search_from_url(hash_url)
    r_kwargs = remote.search_country("DE")
    assert sorted(r_hash.clip_ids) == sorted(r_kwargs.clip_ids)


def test_helper_methods_inherited_from_base():
    """All search_* helpers are inherited; calling them shouldn't AttributeError."""
    c = WheelHTTPClient(server_url="http://x:1")
    for name in (
        "search_caption", "search_caption_embed", "search_semantic_text",
        "search_clip", "search_visual_text", "search_visual_image",
        "search_trajectory_pattern", "search_trajectory_shape",
        "search_classifier", "search_cluster", "search_country",
        "search_world_model",
    ):
        assert callable(getattr(c, name))


def test_upload_clustering_run_end_to_end(server_and_local, tmp_path):
    """Build a run locally, upload it via the HTTP client, verify it landed."""
    remote, _ = server_and_local
    embs = np.random.RandomState(0).randn(40, 4).astype(np.float32)
    clip_ids = [f"c{i}" for i in range(40)]
    run_dir = build_clustering_run(
        tmp_path / "src", embs, clip_ids, n_clusters=2, run_id="uploaded",
    )
    response = remote.upload_clustering_run(run_dir)
    assert response["run_id"] == "uploaded"
    assert "cluster_assignments.parquet" in response["files_written"]

    target = tmp_path / "clustering" / "uploaded"
    assert (target / "cluster_assignments.parquet").exists()
    assert (target / "metadata.json").exists()


def test_upload_run_id_collision_raises(server_and_local, tmp_path):
    remote, _ = server_and_local
    embs = np.random.RandomState(1).randn(20, 4).astype(np.float32)
    clip_ids = [f"x{i}" for i in range(20)]
    run_dir = build_clustering_run(
        tmp_path / "src2", embs, clip_ids, n_clusters=2, run_id="dup",
    )
    remote.upload_clustering_run(run_dir)
    with pytest.raises(RuntimeError, match="409"):
        remote.upload_clustering_run(run_dir)
    # overwrite=True succeeds
    response = remote.upload_clustering_run(run_dir, overwrite=True)
    assert response["run_id"] == "dup"
