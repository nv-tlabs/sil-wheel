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

"""HTTP-only programmatic Wheel client.

``WheelHTTPClient`` exposes the same search modalities as ``WheelClient``
but talks to a *running* wheel server over HTTP instead of running the
pipeline in-process. It needs no local data — just a server URL and, for
authenticated endpoints, a username/password.

Typical usage::

    from sil_wheel.http_client import WheelHTTPClient

    client = WheelHTTPClient(
        server_url="http://wheel-host:8012",
        username="alice",
        password="...",
    )
    result = client.search_caption("intersection")
"""
from __future__ import annotations

import http.cookiejar
import io
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request

from sil_wheel.client import WheelClientBase, WheelSearchResult
from sil_wheel.stores.search_utils import SearchFilters


class WheelHTTPClient(WheelClientBase):
    """Remote Wheel API: thin wrapper around a running wheel server.

    Search calls hit ``GET /clip_ids`` and return the ranked clip list.
    Per-clip score breakdowns aren't carried over the wire today — the
    ``WheelSearchResult.scores`` field will be empty for remote results.
    """

    def __init__(
        self,
        server_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 30.0,
    ):
        if not server_url:
            raise ValueError("server_url is required")
        if (username is None) != (password is None):
            raise ValueError(
                "username and password must be provided together"
            )
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._cookies = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self._cookies)
        )
        if username and password:
            self.login(username, password)

    def _request(self, path: str, query: dict, method: str = "GET",
                 body: Optional[bytes] = None,
                 extra_headers: Optional[dict] = None):
        url = f"{self.server_url}{path}"
        if query:
            url = f"{url}?{urlencode(query, doseq=True)}"
        req = Request(
            url, data=body, headers=dict(extra_headers or {}), method=method,
        )
        return self._opener.open(req, timeout=self.timeout)

    def login(self, username: str, password: str) -> None:
        """POST ``user_login::user::pwd`` and capture the session cookie.

        Subsequent requests on this client will automatically send the
        cookie. Raises ``RuntimeError`` on bad credentials.
        """
        body = f"user_login::{username}::{password}".encode("utf-8")
        try:
            with self._request("/login", {}, method="POST", body=body):
                pass
        except HTTPError as e:
            if e.code == 403:
                raise RuntimeError("login failed: invalid credentials")
            raise

    def _get_json(self, path: str, query: Optional[dict] = None) -> dict:
        with self._request(path, query or {}) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _search_with_query(self, query: dict) -> WheelSearchResult:
        """Run a search via ``GET /clip_ids``.

        Returns the full ranked clip list. ``scores`` is empty — the
        endpoint doesn't carry per-modality breakdowns. If a use case for
        scores arises, the endpoint can grow a ``with_scores=1`` opt-in.
        """
        filters = SearchFilters.from_query(query)
        payload = self._get_json("/clip_ids", query)
        return WheelSearchResult(
            clip_ids=payload.get("clip_ids", []),
            scores={},
            filters=filters,
        )

    def whoami(self) -> dict:
        """``GET /whoami`` — sanity check that login worked."""
        return self._get_json("/whoami")

    def upload_clustering_run(
        self,
        run_dir,
        run_id: Optional[str] = None,
        overwrite: bool = False,
    ) -> dict:
        """Tar+gzip ``run_dir``'s contents and POST to ``/upload_clustering``.

        ``run_id`` defaults to ``Path(run_dir).name``. Returns the parsed
        JSON response on success; raises ``RuntimeError`` with the
        server's error message on non-2xx.
        """
        return self._upload_run("/upload_clustering", run_dir, run_id, overwrite)

    def upload_classifier_run(
        self,
        run_dir,
        run_id: Optional[str] = None,
        overwrite: bool = False,
    ) -> dict:
        """Tar+gzip ``run_dir``'s contents and POST to ``/upload_classifier``.

        ``run_id`` defaults to ``Path(run_dir).name``. Returns the parsed
        JSON response on success; raises ``RuntimeError`` with the
        server's error message on non-2xx.
        """
        return self._upload_run("/upload_classifier", run_dir, run_id, overwrite)

    def upload_clip_list(self, clip_ids) -> dict:
        """POST a list of clip_ids to ``/upload_clip_list``.

        Returns ``{"hash": str, "count": int, "created": bool}``. The
        endpoint is content-addressed and unauthenticated: re-uploading
        the same content returns the same hash with ``created=False``.
        Pair with :meth:`search_clip_list` to filter searches.
        """
        body = json.dumps(list(clip_ids)).encode("utf-8")
        try:
            with self._request(
                "/upload_clip_list", {}, method="POST",
                body=body,
                extra_headers={"Content-Type": "application/json"},
            ) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            try:
                err = json.loads(e.read().decode("utf-8")).get("error", str(e))
            except Exception:
                err = str(e)
            raise RuntimeError(f"/upload_clip_list failed ({e.code}): {err}")

    def _upload_run(self, endpoint, run_dir, run_id, overwrite):
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run dir not found: {run_dir}")
        if run_id is None:
            run_id = run_dir.name

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for p in sorted(run_dir.iterdir()):
                if p.is_file():
                    tf.add(p, arcname=p.name)

        query = {
            "run_id": [run_id],
            "overwrite": ["1" if overwrite else "0"],
        }
        try:
            with self._request(
                endpoint, query, method="POST",
                body=buf.getvalue(),
                extra_headers={"Content-Type": "application/gzip"},
            ) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            try:
                err = json.loads(e.read().decode("utf-8")).get("error", str(e))
            except Exception:
                err = str(e)
            raise RuntimeError(f"{endpoint} failed ({e.code}): {err}")
