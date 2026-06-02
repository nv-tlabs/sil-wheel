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

"""End-to-end tests for the /upload_classifier POST endpoint."""
import io
import json
import tarfile
import threading
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import faiss
import numpy as np
import pytest

import launch_server
from sil_wheel.classifier_build import build_classifier_run
from sil_wheel.search.search_pipeline import SearchPipeline
from sil_wheel.stores.classifier_search import ClassifierSearch


def _make_static_pages():
    root = Path(launch_server.__file__).resolve().parent.parent / "sil_wheel" / "app" / "static"
    return launch_server.StaticPages({
        "/login": ("text/html", root / "html/login.html"),
    })


def _make_run_tarball(run_dir: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in sorted(run_dir.iterdir()):
            tf.add(p, arcname=p.name)
    return buf.getvalue()


def _build_index(embeddings):
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


@pytest.fixture()
def classifier_dir(tmp_path):
    d = tmp_path / "classifiers"
    d.mkdir()
    return d


@pytest.fixture()
def built_run(tmp_path):
    rng = np.random.default_rng(1)
    pos_x = rng.normal(loc=5.0, scale=0.2, size=(20, 4)).astype(np.float32)
    neg_x = rng.normal(loc=-5.0, scale=0.2, size=(20, 4)).astype(np.float32)
    other = rng.normal(loc=0.0, scale=1.0, size=(20, 4)).astype(np.float32)
    embeddings = np.vstack([pos_x, neg_x, other]).astype(np.float32)
    clip_ids = (
        [f"pos-{i:02d}" for i in range(20)]
        + [f"neg-{i:02d}" for i in range(20)]
        + [f"other-{i:02d}" for i in range(20)]
    )
    index = _build_index(embeddings)
    clip_to_index = {c: i for i, c in enumerate(clip_ids)}
    return build_classifier_run(
        output_dir=tmp_path / "src",
        positive_clips=clip_ids[:20],
        negative_clips=clip_ids[20:40],
        positive_features=pos_x,
        negative_features=neg_x,
        corpus_items=list(clip_to_index.items()),
        features_index=index,
        embed_type="cosmos",
        positive_labels=["snow"],
        negative_labels=[],
        trained_by="alice",
        save_threshold=0.0,
        max_clips=-1,
        run_id="src_run",
    )


@pytest.fixture()
def server(data_store, caption_db, users_db, classifier_dir):
    """HTTP server with a real ClassifierSearch over `classifier_dir`."""
    embeddingsstore = MagicMock()
    clipembeddingsstore = MagicMock()
    captionembeddingsstore = MagicMock()
    trajectorystore = MagicMock()
    metricstore = MagicMock()
    predictionsstore = MagicMock()
    autolabelsstore = MagicMock()
    bev_fetcher = MagicMock()
    wm_store = MagicMock()
    video_fetcher = MagicMock()
    clustersearch = MagicMock()
    cliplistsearch = MagicMock()

    classifiersearch = ClassifierSearch(classifier_dir)

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
        classifier_dir=str(classifier_dir),
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
            "classifier_dir": classifier_dir,
            "classifiersearch": classifiersearch,
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
    url = f"{server['url']}/upload_classifier?run_id={run_id}"
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
    assert payload["path"] == str(server["classifier_dir"] / "uploaded01")
    assert "predicted_scores.json" in payload["files_written"]

    target = server["classifier_dir"] / "uploaded01"
    for name in (
        "metadata.json",
        "LR_weights.pkl",
        "predicted_scores.json",
        "positive_clips.json",
        "negative_clips.json",
    ):
        assert (target / name).exists(), f"missing {name}"


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
    assert not any(server["classifier_dir"].iterdir())


def test_invalid_metadata_rejected(server, built_run, tmp_path):
    """Tarball whose metadata.json is missing required keys → 400, no run dir written."""
    cookie = _login_cookie(server["users_db"])

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    for name in (
        "LR_weights.pkl", "predicted_scores.json",
        "positive_clips.json", "negative_clips.json",
    ):
        (bad_dir / name).write_bytes((built_run / name).read_bytes())
    (bad_dir / "metadata.json").write_text(json.dumps({"foo": "bar"}))

    body = _make_run_tarball(bad_dir)
    with pytest.raises(HTTPError) as excinfo:
        _post_upload(server, "bad01", body, cookie=cookie)
    assert excinfo.value.code == 400
    assert not (server["classifier_dir"] / "bad01").exists()
    assert not any(p.name.startswith(".tmp_bad01") for p in server["classifier_dir"].iterdir())


def test_unauthenticated_rejected(server, built_run):
    body = _make_run_tarball(built_run)
    with pytest.raises(HTTPError) as excinfo:
        _post_upload(server, "anon01", body, cookie=None)
    assert excinfo.value.code == 403


def test_upload_invalidates_classifier_cache(server, built_run, tmp_path):
    cookie = _login_cookie(server["users_db"])

    body = _make_run_tarball(built_run)
    _post_upload(server, "warmcache", body, cookie=cookie)

    cs = server["classifiersearch"]
    cs.load_scores("warmcache")
    assert "warmcache" in cs._score_cache

    # Replace with a smaller scores set; uploading must invalidate the cache.
    rng = np.random.default_rng(7)
    pos_x = rng.normal(loc=5.0, scale=0.2, size=(5, 4)).astype(np.float32)
    neg_x = rng.normal(loc=-5.0, scale=0.2, size=(5, 4)).astype(np.float32)
    embeddings = np.vstack([pos_x, neg_x]).astype(np.float32)
    clip_ids = [f"NEW-pos-{i}" for i in range(5)] + [f"NEW-neg-{i}" for i in range(5)]
    index = _build_index(embeddings)
    clip_to_index = {c: i for i, c in enumerate(clip_ids)}
    new_run = build_classifier_run(
        output_dir=tmp_path / "src2",
        positive_clips=clip_ids[:5],
        negative_clips=clip_ids[5:],
        positive_features=pos_x,
        negative_features=neg_x,
        corpus_items=list(clip_to_index.items()),
        features_index=index,
        embed_type="cosmos",
        positive_labels=["snow"],
        negative_labels=[],
        trained_by="alice",
        save_threshold=0.0,
        max_clips=-1,
        run_id="warmcache",
    )
    body2 = _make_run_tarball(new_run)
    _post_upload(server, "warmcache", body2, overwrite=True, cookie=cookie)

    assert "warmcache" not in cs._score_cache
    clip_ids_loaded, _ = cs.load_scores("warmcache")
    assert all(c.startswith("NEW-") for c in clip_ids_loaded.tolist())
