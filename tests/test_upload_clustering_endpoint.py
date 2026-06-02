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

"""End-to-end tests for the /upload_clustering POST endpoint."""
import io
import json
import tarfile
import threading
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

import launch_server
from sil_wheel.cluster_build import build_clustering_run
from sil_wheel.search.search_pipeline import SearchPipeline
from sil_wheel.stores.cluster_search import ClusterSearch


def _make_static_pages():
    root = Path(launch_server.__file__).resolve().parent.parent / "sil_wheel" / "app" / "static"
    return launch_server.StaticPages({
        "/login": ("text/html", root / "html/login.html"),
    })


def _make_run_tarball(run_dir: Path) -> bytes:
    """Return a tar.gz of run_dir's contents (flat, no parent prefix)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in sorted(run_dir.iterdir()):
            tf.add(p, arcname=p.name)
    return buf.getvalue()


@pytest.fixture()
def clustering_dir(tmp_path):
    d = tmp_path / "clustering"
    d.mkdir()
    return d


@pytest.fixture()
def built_run(tmp_path):
    """A freshly-built valid run directory to upload from."""
    embs = np.random.RandomState(0).randn(80, 4).astype(np.float32)
    clip_ids = [f"clip-{i:03d}" for i in range(80)]
    return build_clustering_run(
        tmp_path / "src", embs, clip_ids, n_clusters=3, run_id="src_run",
    )


@pytest.fixture()
def server(data_store, caption_db, users_db, clustering_dir):
    """HTTP server with a real ClusterSearch over `clustering_dir`."""
    embeddingsstore = MagicMock()
    clipembeddingsstore = MagicMock()
    captionembeddingsstore = MagicMock()
    trajectorystore = MagicMock()
    classifiersearch = MagicMock()
    metricstore = MagicMock()
    predictionsstore = MagicMock()
    autolabelsstore = MagicMock()
    bev_fetcher = MagicMock()
    wm_store = MagicMock()
    video_fetcher = MagicMock()

    clustersearch = ClusterSearch(clustering_dir)
    cliplistsearch = MagicMock()

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
        clustering_dir=str(clustering_dir),
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
            "clustering_dir": clustering_dir,
            "clustersearch": clustersearch,
            "users_db": users_db,
        }
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _login_cookie(users_db) -> str:
    uid = users_db.create_user("alice", "pwd123")
    sid = users_db.create_session(uid)
    return f"{launch_server.SESSION_COOKIE}={sid}"


def _post_upload(server, run_id, body, overwrite=False, cookie=None):
    url = f"{server['url']}/upload_clustering?run_id={run_id}"
    if overwrite:
        url += "&overwrite=1"
    headers = {"Content-Type": "application/gzip"}
    if cookie is not None:
        headers["Cookie"] = cookie
    return urlopen(Request(url, data=body, headers=headers, method="POST"), timeout=10)


def test_upload_round_trip(server, built_run):
    cookie = _login_cookie(server["users_db"])
    body = _make_run_tarball(built_run)

    resp = _post_upload(server, "uploaded01", body, cookie=cookie)
    assert resp.status == 201
    payload = json.loads(resp.read().decode("utf-8"))
    assert payload["run_id"] == "uploaded01"
    assert payload["path"] == str(server["clustering_dir"] / "uploaded01")
    assert "cluster_assignments.parquet" in payload["files_written"]

    target = server["clustering_dir"] / "uploaded01"
    for name in ("cluster_assignments.parquet", "representative_by_cluster.json",
                 "umap.json", "metadata.json", "centroids.npy"):
        assert (target / name).exists(), f"missing {name}"

    src_centroids = np.load(built_run / "centroids.npy")
    dst_centroids = np.load(target / "centroids.npy")
    np.testing.assert_array_equal(src_centroids, dst_centroids)


def test_collision_returns_409(server, built_run):
    cookie = _login_cookie(server["users_db"])
    body = _make_run_tarball(built_run)
    _post_upload(server, "dup", body, cookie=cookie)

    with pytest.raises(HTTPError) as excinfo:
        _post_upload(server, "dup", body, cookie=cookie)
    assert excinfo.value.code == 409

    resp = _post_upload(server, "dup", body, overwrite=True, cookie=cookie)
    assert resp.status == 201


def test_invalid_run_id_rejected(server, built_run):
    cookie = _login_cookie(server["users_db"])
    body = _make_run_tarball(built_run)

    with pytest.raises(HTTPError) as excinfo:
        _post_upload(server, "../etc", body, cookie=cookie)
    assert excinfo.value.code == 400
    assert not any(server["clustering_dir"].iterdir())


def test_invalid_parquet_schema_rejected(server, built_run, tmp_path):
    """Tarball where parquet has the wrong columns → 400, no run dir written."""
    cookie = _login_cookie(server["users_db"])

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    for name in ("representative_by_cluster.json", "umap.json", "metadata.json",
                 "centroids.npy"):
        (bad_dir / name).write_bytes((built_run / name).read_bytes())
    import pandas as pd
    pd.DataFrame({"foo": ["a", "b"], "bar": [1, 2]}).to_parquet(
        bad_dir / "cluster_assignments.parquet", index=False,
    )

    body = _make_run_tarball(bad_dir)
    with pytest.raises(HTTPError) as excinfo:
        _post_upload(server, "bad01", body, cookie=cookie)
    assert excinfo.value.code == 400
    assert not (server["clustering_dir"] / "bad01").exists()
    assert not any(p.name.startswith(".tmp_bad01") for p in server["clustering_dir"].iterdir())


def test_unauthenticated_rejected(server, built_run):
    body = _make_run_tarball(built_run)
    with pytest.raises(HTTPError) as excinfo:
        _post_upload(server, "anon01", body, cookie=None)
    assert excinfo.value.code == 403


def test_upload_invalidates_cluster_search_cache(server, built_run, tmp_path):
    cookie = _login_cookie(server["users_db"])

    body = _make_run_tarball(built_run)
    _post_upload(server, "warmcache", body, cookie=cookie)

    cs = server["clustersearch"]
    cs._load_run("warmcache")
    assert "warmcache" in cs._cache

    embs = np.random.RandomState(7).randn(40, 4).astype(np.float32)
    clip_ids = [f"NEW-{i}" for i in range(40)]
    new_run = build_clustering_run(
        tmp_path / "src2", embs, clip_ids, n_clusters=2, run_id="warmcache",
    )
    body2 = _make_run_tarball(new_run)
    _post_upload(server, "warmcache", body2, overwrite=True, cookie=cookie)

    assert "warmcache" not in cs._cache
    df = cs._load_run("warmcache")
    assert all(c.startswith("NEW-") for c in df["clip_id"])
