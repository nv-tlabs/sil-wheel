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

import argparse
import base64
import concurrent.futures
import csv
import datetime
import gzip
import io
import json
import logging
import math
import orjson
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from functools import partial
from pathlib import Path
from socketserver import TCPServer, ThreadingMixIn
from threading import RLock
from urllib import parse
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import boto3
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError
from RangeHTTPServer import RangeRequestHandler
from sil_wheel.stores.autolabels_store import AutolabelsDataStore
from sil_wheel.classifier_build import (
    load_lr_weights,
    validate_run_dir as validate_classifier_run_dir,
)
from sil_wheel.cluster_build import validate_run_dir as validate_cluster_run_dir
from sil_wheel.stores.classifier_search import ClassifierSearch
from sil_wheel.stores.clip_list_search import ClipListSearch
from sil_wheel.stores.cluster_search import ClusterSearch
from sil_wheel.stores.caption_embeddings_store import CaptionEmbeddingsStore
from sil_wheel.stores.cluster_topics import read_topics
from sil_wheel.stores.cosmos_embeddings_store import CosmosEmbeddingsStore
from sil_wheel.stores.models_with_metrics_store import ModelsWithMetricsDataStore
from sil_wheel.stores.predictions_store import PredictionsDataStore
from sil_wheel.stores.search_utils import SearchFilters, project_dict, rrf_rank
from sil_wheel.stores.sqlite_caption_store import FTSCaptionStore as CaptionStore
from sil_wheel.stores.sqlite_data_store import SQLiteDataStore as DataStore
from sil_wheel.stores.time_utils import Timer
from sil_wheel.stores.trajectory_store import TrajectoryStore
from sil_wheel.stores.users_data_store import UsersDataStore
from sil_wheel.stores.utils import LRUDict
from sil_wheel.stores.visual_embeddings_store import Florence2SigCLIPEmbeddingStore
from sil_wheel.search.search_pipeline import SearchPipeline
from sil_wheel.stores.arena_store import ArenaStore
from sil_wheel.stores.wm_store import WMStore
from sil_wheel.llm.query_rewriter import QueryRewriter
from sil_wheel.llm.vlm_judge import VLMJudge
from sil_wheel.app.slack import SlackNotifier, load_slack_config
from sil_wheel.app.sheets_client import append_to_spreadsheet
from sil_wheel.app.arena_handler import ArenaHandlerMixin
from sil_wheel.app.websocket_utils import run_ws_server, ws_broadcast_threadsafe


def log_rss(label):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                kb = int(line.split()[1])
                print(f"[rss] {label}: {kb / 1024 / 1024:.2f} GB")
                return


NUM_VIDEOS_PER_PAGE = 6
VLM_JUDGE_RANKED_IDS_LIMIT = 1000

LABEL_TYPES = sorted(["manual", "autolabel"])

SESSION_COOKIE = "session_id"


def configure_logging_and_stats(server_cfg):
    """Create log files based on the timestamp that the server was launched."""
    log_dir = server_cfg.get("log_dir", "/tmp/logs")
    stats_dir = server_cfg.get("datasets_stats_dir", "/tmp/dataset_stats/")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(stats_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"server_logs_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}.txt",
    )
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        force=True,
    )
    return log_dir, stats_dir


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config


def apply_overrides(config: dict, overrides: list | None = None):
    """Override the default arguments in the config"""
    if overrides is None:
        overrides = []

    for item in overrides:
        key_path, _, value = item.partition("=")
        keys = key_path.split(".")
        node = config
        for k in keys[:-1]:
            node = node[k]
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            for cast in (int, float):
                try:
                    value = cast(value)
                    break
                except ValueError:
                    pass
        node[keys[-1]] = value


class InfEncoder(json.JSONEncoder):
    def encode(self, obj):
        obj = self._replace_inf(obj)
        return super().encode(obj)

    def _replace_inf(self, obj):
        if isinstance(obj, float) and math.isinf(obj):
            return str(obj)
        if isinstance(obj, dict):
            return {k: self._replace_inf(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._replace_inf(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._replace_inf(v) for v in obj)

        else:
            return obj


def unique_id(length=10):
    x = [random.randint(0, 35) for _ in range(length)]
    return "".join(
        [chr((xi < 10) * (48 + xi) + (xi >= 10) * (87 + xi)) for xi in x]
    )


def load_favicon(img_path):
    favicon_path = Path(__file__).parent.parent / "sil_wheel" / "app" / "static" / img_path
    if (img := Path(favicon_path)).exists():
        return (unique_id(), "image/png", img.open("rb").read())
    else:
        print(f"Using default favicon because {img_path} does not exist")
        img = """<svg xmlns="http://www.w3.org/2000/svg">
          <text y=\"32\" font-size=\"32\">🏠</text>
        </svg>
        """
        return (unique_id(), "image/svg+xml", img.encode())


class StaticPages:
    def __init__(self, files, devmode=True):
        self.files = files
        self.cache = {}
        self.devmode = devmode

    def __contains__(self, path):
        return path in self.files

    def load(self, path):
        self.cache[path] = (
            unique_id(),
            open(self.files[path][1], "rb").read(),
        )

    def serve(self, handler, path):
        if path not in self.files:
            handler.send_error(404, "File not found")
            return

        if path not in self.cache or self.devmode:
            self.load(path)

        if not self.devmode and (etag := handler.headers.get("If-None-Match")):
            if etag == self.cache[path][0]:
                handler.send_response(304)
                handler.end_headers()
                return

        handler.send_response(200)
        handler.send_header("Content-Type", self.files[path][0])
        if self.devmode:
            handler.send_header(
                "Cache-Control", "no-cache, no-store, must-revalidate"
            )
        else:
            handler.send_header("ETag", self.cache[path][0])
            handler.send_header(
                "Cache-Control", "max-age=3600, public, must-revalidate"
            )
        handler.end_headers()
        handler.wfile.write(self.cache[path][1])


class S3Fetcher:
    """Streams files from the configured S3-compatible object store.

    This backend serves S3 objects directly to HTTP clients, including support
    for byte-range requests used by video players during seeking. It expects
    callers to pass normalized S3 keys rather than URLs or filesystem paths.

    If the required AWS profile is unavailable, the fetcher stays constructible
    but marks itself as disabled; requests then fail with 503 instead of
    preventing the whole server from starting.
    """
    def __init__(self, bucket: str, endpoint: str | None = None, profile: str = "sil-wheel"):
        self.bucket = bucket
        self.client = None
        try:
            sess = boto3.Session(profile_name=profile, region_name="us-east-1")
            self.client = sess.client(
                "s3",
                config=Config(
                    max_pool_connections=50,
                    read_timeout=30,
                    connect_timeout=5,
                ),
                endpoint_url=endpoint,
            )
        except Exception as e:
            print(
                f"[S3Fetcher] AWS profile {profile!r} not configured ({e}); "
                "S3 keys will 503."
            )

    def serve(self, handler, key: str, headers: dict):
        if self.client is None:
            handler.send_error(503, "S3 backend not configured")
            return
        try:
            if "Range" in handler.headers:
                resp = self.client.get_object(
                    Bucket=self.bucket,
                    Key=key,
                    Range=handler.headers["Range"],
                )
                handler.send_response(206)
                for k, v in headers.items():
                    handler.send_header(k, v)
                handler.send_header("Content-Range", resp["ContentRange"])
                handler.send_header("Content-Length", resp["ContentLength"])
                handler.end_headers()
                shutil.copyfileobj(resp["Body"], handler.wfile)
            else:
                handler.send_response(200)
                for k, v in headers.items():
                    handler.send_header(k, v)
                handler.end_headers()
                self.client.download_fileobj(self.bucket, key, handler.wfile)
        except (BrokenPipeError, ConnectionResetError):
            # Client closed mid-stream — normal for video players that issue
            # range requests and close as soon as they have enough data.
            pass
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            handler.log_message("S3 error fetching %s: %s", key, code)
            handler.send_error(404, "File not found")

    def get_bytes(self, key: str) -> bytes:
        if self.client is None:
            raise RuntimeError("S3 client not configured for this server")
        resp = self.client.get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read()


class LocalFileFetcher:
    """Streams files from the local filesystem to HTTP clients.

    This backend is used for absolute filesystem paths and implements HTTP
    byte-range support so local videos can be played and seeked efficiently
    in the browser.
    """
    def serve(self, handler, fs_path: str, headers: dict):
        try:
            self._serve_with_range(handler, fs_path, headers)
        except FileNotFoundError:
            handler.send_error(404, "File not found")

    def get_bytes(self, fs_path: str) -> bytes:
        with open(fs_path, "rb") as f:
            return f.read()

    @staticmethod
    def _serve_with_range(handler, fs_path: str, headers: dict):
        f = open(fs_path, "rb")
        try:
            size = os.fstat(f.fileno()).st_size
            range_header = handler.headers.get("Range")
            if range_header and range_header.startswith("bytes="):
                start, _, end = range_header[6:].partition("-")
                start = int(start) if start else 0
                end = int(end) if end else size - 1
                end = min(end, size - 1)
                if start > end or start >= size:
                    handler.send_response(416)
                    handler.send_header("Content-Range", f"bytes */{size}")
                    handler.end_headers()
                    return
                length = end - start + 1
                handler.send_response(206)
                for k, v in headers.items():
                    handler.send_header(k, v)
                handler.send_header("Accept-Ranges", "bytes")
                handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                handler.send_header("Content-Length", str(length))
                handler.end_headers()
                f.seek(start)
                LocalFileFetcher._copy_n(f, handler.wfile, length)
            else:
                handler.send_response(200)
                for k, v in headers.items():
                    handler.send_header(k, v)
                handler.send_header("Accept-Ranges", "bytes")
                handler.send_header("Content-Length", str(size))
                handler.end_headers()
                shutil.copyfileobj(f, handler.wfile)
        except (BrokenPipeError, ConnectionResetError):
            # Browser closed mid-stream — normal for <video> seeking.
            pass
        finally:
            f.close()

    @staticmethod
    def _copy_n(src, dst, n: int, chunk: int = 64 * 1024):
        while n > 0:
            buf = src.read(min(chunk, n))
            if not buf:
                break
            dst.write(buf)
            n -= len(buf)


class BaseFetcher:
    """Shared dispatch layer for serving data from either local disk or S3.

    Subclasses translate domain-specific inputs, such as a video path or clip ID,
    into a concrete storage key plus response headers. BaseFetcher then routes
    absolute paths to the local backend and relative keys to the S3 backend.

    This keeps storage-specific streaming logic separate from the logic that
    knows how to construct paths for videos, BEV files, or other data.
    """

    def __init__(self, bucket: str, endpoint: str | None = None, profile: str = "sil-wheel"):
        self.s3 = S3Fetcher(bucket, endpoint, profile)
        self.local = LocalFileFetcher()

    def get_key(self, handler, path) -> tuple[str, dict]:
        raise NotImplementedError

    @staticmethod
    def _is_local_key(key: str) -> bool:
        return key.startswith("/")

    def serve(self, handler, path):
        key, headers = self.get_key(handler, path)
        if self._is_local_key(key):
            self.local.serve(handler, key, headers)
        else:
            self.s3.serve(handler, key, headers)

    def get_bytes(self, path) -> bytes:
        key, _ = self.get_key(None, path)
        if self._is_local_key(key):
            return self.local.get_bytes(key)
        return self.s3.get_bytes(key)


class VideoFetcher(BaseFetcher):
    """Resolves video paths and serves MP4 files from local disk or S3.

    Video paths may come as absolute local paths or as S3-style relative keys.
    This fetcher normalizes those paths and attaches the correct MP4 content
    type before delegating serving to BaseFetcher.
    """

    def get_key(self, handler, video_path):
        # video_paths.path holds either an absolute local path (passed through
        # unchanged) or an S3-key-shaped relative path (normalized: URL-decode,
        # strip leading '/' and './').
        s = parse.unquote(str(video_path))
        if s.startswith("/"):
            return s, {"Content-Type": "video/mp4"}
        key = s.lstrip("/")
        while key.startswith("./"):
            key = key[2:]
        return key, {"Content-Type": "video/mp4"}


class BEVFetcher(BaseFetcher):
    """Resolves and serves per-clip BEV msgpack data from S3.

    BEV files are addressed by clip ID using the fixed
    ``bev_data/v0/{clip_id}.msgpack`` key layout. Optionally, this fetcher can
    load a precomputed set of clip IDs with BEV data and use it as a search
    filter so queries can be restricted to clips with available BEV outputs.
    """

    def __init__(self, bucket, index_dir: str | None = None, endpoint: str | None = None, profile: str = "sil-wheel"):
        super().__init__(bucket, endpoint, profile)
        self.clips_with_bev = None
        if index_dir is not None:
            index_path = Path(index_dir) / "clips_with_bev_set.pkl"
            try:
                with open(index_path, "rb") as f:
                    self.clips_with_bev = set(pickle.load(f))
            except Exception as e:
                print(f"Warning: failed to load {index_path}: {e}")

    def get_key(self, handler, clip_id):
        return (
            f"bev_data/v0/{clip_id}.msgpack",
            {"Content-Type": "application/octet-stream", "Content-Encoding": "msgpack"},
        )

    def search(self, filters, current_results):
        if filters.with_bev:
            current_results = project_dict(current_results, self.clips_with_bev)
        return current_results

    def get_bev_clip_ids(self):
        return self.clips_with_bev

    def has_bev_index(self):
        return self.clips_with_bev is not None


def embed_dir_is_populated(path) -> bool:
    """Return True if the dir has any artifacts the embedding stores read."""
    if path is None:
        return False
    p = Path(path)
    if not p.exists():
        return False
    suffixes = ("*.parquet", "*.pkl", "*.npy", "*.index")
    return any(next(p.rglob(s), None) is not None for s in suffixes)


class NullStore:
    """Decorator on data stores that makes them optional"""
    def __init__(self, store):
        self.store = store

    def search(self, filters, current_results):
        if self.store is None:
            return current_results
        return self.store.search(filters, current_results)

    def invalidate(self, *args, **kwargs):
        if self.store is not None:
            self.store.invalidate(*args, **kwargs)


class EmptyEmbedStore:
    """Lightweight stand-in for an embedding store when no embeddings exist.

    This object implements the same interface as the real embedding stores but
    performs no model loading, indexing, or search. It allows the server and
    search pipeline to start normally when a particular embedding directory is
    missing or empty.

    Search methods return unchanged or empty results, while encoding methods
    return ``None``.
    """
    def __init__(self, path_to_embeddings, *_args, **_kwargs):
        self.path_to_embeddings = path_to_embeddings
        self._tag = "empty"
        self._index_tag = "empty"
        self.features_index = None
        self.clip_row_map = None
        self.clips_to_index = {}
        self.searches = LRUDict(size=1)
        self.uploaded_image_features = LRUDict(size=1)
        self.uploaded_features = None

    def warmup(self):
        return None

    def search(self, filters, current_results):
        return current_results

    def has_embeddings(self, *_args, **_kwargs):
        return False

    def append_embeddings_parquet(self, *_args, **_kwargs):
        return 0

    def search_with_text(self, *_args, **_kwargs):
        return []

    def search_with_video_clip(self, *_args, **_kwargs):
        return []

    def search_with_image(self, *_args, **_kwargs):
        return []

    def encode_text(self, *_args, **_kwargs):
        return None

    def compute_features(self, *_args, **_kwargs):
        return None

    def compute_image_features(self, *_args, **_kwargs):
        return None


class JobRegistry:
    """Thread-safe registry for subprocess jobs launched by the server.

    The registry tracks active jobs by label, stores their command and start
    time, and removes completed processes when queried. It is used to prevent
    duplicate jobs and to report which long-running tasks are still active.
    """

    def __init__(self):
        self._lock = RLock()
        self._jobs: dict[str, dict] = {}

    def start(self, label: str, proc: subprocess.Popen, cmd: list[str]):
        info = {
            "proc": proc,
            "cmd": list(cmd) if cmd is not None else [],
            "started_at": float(time.time()),
        }
        with self._lock:
            self._jobs[label] = info
        return info

    def is_running(self, label: str):
        with self._lock:
            info = self._jobs.get(label)
        if not info:
            return False
        status = info["proc"].poll()
        if status is not None:
            # purge dead entry
            with self._lock:
                self._jobs.pop(label, None)
            return False
        return True

    def running_keys(self):
        with self._lock:
            all_labels = list(self._jobs.keys())
            return sorted(
                [label for label in all_labels if self.is_running(label)]
            )


class NurecJobRegistry:
    def __init__(self):
        self._lock = RLock()
        self._job = None
        self._started = 0
        self._nurec_root = (
            Path(__file__).resolve().parent.parent / "nre-nrm-mgpu-demo"
        )
        self._kill_script = str(self._nurec_root / "kill_viewer.sh")
        self._launch_script = str(self._nurec_root / "launch_viewer.sh")

        self._background_thread = threading.Thread(target=self._kill_on_timeout)
        self._stop_bg_thread = False
        self._background_thread.start()

    def _kill_on_timeout(self):
        timeout = 5 * 60
        while not self._stop_bg_thread:
            with self._lock:
                if (
                    self._job is not None
                    and time.time() - self._started > timeout
                ):
                    try:
                        self.kill_previous()
                    except:
                        # Failed to kill it not sure what we can do ...
                        pass
            time.sleep(1.0)

    def kill_previous(self):
        with self._lock:
            if self._job is None:
                return

            if self._job.poll() is None:
                subprocess.call(["bash", self._kill_script])
                try:
                    self._job.wait(5)
                except subprocess.TimeoutExpired:
                    if self._job.poll() is None:
                        raise RuntimeError("Failed to kill the remote viewer")
                self._job = None

    def launch(self, clip_id):
        with self._lock:
            arg = f"lidar-model-static-full/{clip_id}"
            cmd = ["bash", self._launch_script, arg]
            print(cmd)

            self._job = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )
            self._started = time.time()

    def join(self):
        self._stop_bg_thread = True
        self.kill_previous()
        self._background_thread.join()


class ThreadingTCPServer(ThreadingMixIn, TCPServer):
    allow_reuse_address = True


class RequestHandler(ArenaHandlerMixin, RangeRequestHandler):
    def __init__(
        self,
        request,
        client_address,
        server,
        datastore=None,
        captionstore=None,
        trajectorystore=None,
        embeddingsstore=None,
        clipembeddingsstore=None,
        captionembeddingsstore=None,
        wm_store=None,
        classifier_dir=None,
        classifiersearch=None,
        metricstore=None,
        predictionsstore=None,
        autolabelsstore=None,
        search_pipeline=None,
        favicon=None,
        static_pages=None,
        video_fetcher=None,
        usersstore=None,
        classifier_jobs=None,
        nurec_job=None,
        clips_to_apis=None,
        bev_fetcher=None,
        clustering_jobs=None,
        clustering_dir=None,
        clustersearch=None,
        cliplistsearch=None,
        slack_notifier=None,
        bug_report_spreadsheet_id=None,
        bug_report_credential_path=None,
        rewriter=None,
        arena_store=None,
        vlm_judge=None,
        vlm_judge_workers=16,
        agent_url=None,
        log_dir=None,
        stats_dir=None,
    ):
        self.datastore = datastore
        self.captionstore = captionstore
        self.trajectorystore = trajectorystore
        self.embeddingsstore = embeddingsstore
        self.clipembeddingsstore = clipembeddingsstore
        self.captionembeddingsstore = captionembeddingsstore
        self.wm_store = wm_store
        self.classifier_dir = classifier_dir
        self.classifiersearch = classifiersearch
        self.clustersearch = clustersearch
        self.metricstore = metricstore
        self.predictions_store = predictionsstore
        self.autolabels_store = autolabelsstore
        self.usersstore = usersstore
        self.classifier_jobs = classifier_jobs
        self.nurec_job = nurec_job
        self.search_pipeline = search_pipeline
        self.favicon = favicon
        self.static_pages = static_pages
        self.video_fetcher = video_fetcher
        self.bev_fetcher = bev_fetcher
        self.clips_to_apis = clips_to_apis
        self.clustering_jobs = clustering_jobs
        self.clustering_dir = clustering_dir
        self.cliplistsearch = cliplistsearch
        self.slack_notifier = slack_notifier
        self.bug_report_spreadsheet_id = bug_report_spreadsheet_id
        self.bug_report_credential_path = bug_report_credential_path
        self.rewriter = rewriter
        self.arena_store = arena_store
        self.vlm_judge = vlm_judge
        self.vlm_judge_workers = vlm_judge_workers
        self.agent_url = agent_url
        self.log_dir = log_dir
        self.stats_dir = stats_dir

        self.timers = Timer()
        self.directory = None
        super().__init__(request, client_address, server)

    def _get_session_id(self):
        cookie_data = self.headers.get("Cookie", "").split(";")
        session_cookie = f"{SESSION_COOKIE}="
        for d in cookie_data:
            d = d.strip()
            if d.startswith(session_cookie):
                k, v = d.split("=", 1)
                return v
        return None

    def log_message(self, format, *args):
        "Prepends `[user=<username>]` to every server log message"
        user = self._current_user()
        uname = getattr(user, "username", None) if user else None
        prefix = f"[user={uname}]" if uname else "[user=anonymous]"
        super().log_message(prefix + " " + format, *args)

        logging.info(prefix + " " + (format % args))

    def _current_user(self):
        sid = self._get_session_id()
        if not sid:
            return None
        return self.usersstore.get_user_by_session(sid)

    def _require_user(self, redirect=False):
        """Return the authenticated user, or None after writing an
        unauthorized response. Use ``redirect=True`` for page handlers
        that should bounce to /login; the default writes a 403 JSON body
        suitable for fetch/XHR endpoints."""
        user = self._current_user()
        if user is not None:
            return user
        if redirect:
            self.send_response(302)
            self.send_header("Location", "/login")
            self.end_headers()
        else:
            self._send_json({"error": "unauthorized"}, status=403)
        return None

    def _send_json(self, obj, status=200, headers=None):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        if headers:
            for k, v in headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode("utf-8"))

    def _handle_upload_clustering(self, parsed_path):
        """Accept a tar.gz of a clustering run directory and install it under
        ``clustering_dir/<run_id>/``. See WheelClient.upload_clustering_run."""
        user = self._current_user()
        if user is None:
            return self._send_json({"error": "unauthorized"}, status=403)

        if not self.clustering_dir:
            return self._send_json(
                {"error": "clustering_dir not configured"}, status=500
            )

        qs = parse.parse_qs(parsed_path.query)
        run_id = (qs.get("run_id") or [""])[0]
        overwrite = (qs.get("overwrite") or ["0"])[0] == "1"

        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", run_id):
            return self._send_json({"error": "invalid run_id"}, status=400)

        clustering_dir = Path(self.clustering_dir).resolve()
        final_dir = (clustering_dir / run_id).resolve()
        if final_dir.parent != clustering_dir:
            return self._send_json({"error": "invalid run_id"}, status=400)

        if final_dir.exists() and not overwrite:
            return self._send_json(
                {"error": f"run_id {run_id!r} already exists"}, status=409
            )

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return self._send_json({"error": "empty body"}, status=400)
        body = self.rfile.read(content_length)

        clustering_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(
            prefix=f".tmp_{run_id}_", dir=clustering_dir,
        ))
        try:
            with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tf:
                tf.extractall(tmp_dir, filter="data")
            validate_cluster_run_dir(tmp_dir)
        except (tarfile.TarError, ValueError) as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return self._send_json({"error": str(e)}, status=400)
        except Exception as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return self._send_json(
                {"error": f"upload failed: {e}"}, status=500
            )

        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.rename(tmp_dir, final_dir)
        if self.clustersearch is not None:
            self.clustersearch.invalidate(run_id)

        files_written = sorted(p.name for p in final_dir.iterdir() if p.is_file())
        return self._send_json(
            {
                "run_id": run_id,
                "path": str(final_dir),
                "files_written": files_written,
            },
            status=201,
        )

    def _handle_upload_classifier(self, parsed_path):
        """Accept a tar.gz of a classifier run directory and install it under
        ``classifier_dir/<run_id>/``. See WheelHTTPClient.upload_classifier_run."""
        user = self._current_user()
        if user is None:
            return self._send_json({"error": "unauthorized"}, status=403)

        if not self.classifier_dir:
            return self._send_json(
                {"error": "classifier_dir not configured"}, status=500
            )

        qs = parse.parse_qs(parsed_path.query)
        run_id = (qs.get("run_id") or [""])[0]
        overwrite = (qs.get("overwrite") or ["0"])[0] == "1"

        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", run_id):
            return self._send_json({"error": "invalid run_id"}, status=400)

        classifier_dir = Path(self.classifier_dir).resolve()
        final_dir = (classifier_dir / run_id).resolve()
        if final_dir.parent != classifier_dir:
            return self._send_json({"error": "invalid run_id"}, status=400)

        if final_dir.exists() and not overwrite:
            return self._send_json(
                {"error": f"run_id {run_id!r} already exists"}, status=409
            )

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return self._send_json({"error": "empty body"}, status=400)
        body = self.rfile.read(content_length)

        classifier_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(
            prefix=f".tmp_{run_id}_", dir=classifier_dir,
        ))
        try:
            with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tf:
                tf.extractall(tmp_dir, filter="data")
            validate_classifier_run_dir(tmp_dir)
        except (tarfile.TarError, ValueError) as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return self._send_json({"error": str(e)}, status=400)
        except Exception as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return self._send_json(
                {"error": f"upload failed: {e}"}, status=500
            )

        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.rename(tmp_dir, final_dir)
        if self.classifiersearch is not None:
            self.classifiersearch.invalidate(run_id)

        files_written = sorted(p.name for p in final_dir.iterdir() if p.is_file())
        return self._send_json(
            {
                "run_id": run_id,
                "path": str(final_dir),
                "files_written": files_written,
            },
            status=201,
        )

    def _handle_upload_clip_list(self, parsed_path):
        if self.cliplistsearch is None:
            return self._send_json(
                {"error": "clip_list_search not configured"}, status=500
            )
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return self._send_json({"error": "empty body"}, status=400)
        body = self.rfile.read(content_length)
        try:
            payload = orjson.loads(body)
        except orjson.JSONDecodeError as e:
            return self._send_json(
                {"error": f"invalid JSON: {e}"}, status=400
            )
        if isinstance(payload, dict):
            payload = payload.get("clip_ids")
        if not isinstance(payload, list):
            return self._send_json(
                {"error": "expected JSON list or {clip_ids: [...]}"},
                status=400,
            )
        for cid in payload:
            if not isinstance(cid, str) or not cid or len(cid) > 256:
                return self._send_json(
                    {"error": "clip_ids must be non-empty strings <= 256 chars"},
                    status=400,
                )
        try:
            hash, created = self.cliplistsearch.save(payload)
        except ValueError as e:
            return self._send_json({"error": str(e)}, status=400)
        return self._send_json(
            {"hash": hash, "count": len(set(payload)), "created": created},
            status=201,
        )

    def _clip_list_hash_for(self, path):
        """Return the clip-list hash for a classifier run's training set
        so the UI's "View Positives/Negatives" can apply it as a
        ``clip_id_list_hash=...`` filter. ``None`` if no clips were
        captured for this run (legacy / pending / failed)."""
        if not path.exists():
            return None
        with open(path, "rb") as f:
            clip_ids = orjson.loads(f.read())
        if not clip_ids:
            return None
        h, _ = self.cliplistsearch.save(clip_ids)
        return h

    def _embed_store_path_and_tag(self, embed_type_str):
        if embed_type_str == "caption":
            return (
                self.captionembeddingsstore.path_to_embeddings,
                self.captionembeddingsstore._tag,
            )
        if embed_type_str == "visual":
            return (
                self.clipembeddingsstore.path_to_embeddings,
                self.clipembeddingsstore._tag,
            )
        return (
            self.embeddingsstore.path_to_embeddings,
            self.embeddingsstore._index_tag,
        )

    def validate_search_vlm(self, query: str, clip_ids: list) -> tuple:
        """Run VLM judge video-query match on clip_ids. Callable from server code.

        Returns:
            (results, None) with results a list of dicts with clip_id, match, reasoning, analysis;
            or (None, error_message) if judge is not available.
        """
        clip_ids = [x.strip() for x in clip_ids if (x and str(x).strip())][
            :VLM_JUDGE_RANKED_IDS_LIMIT
        ]
        if not clip_ids:
            return ([], None)
        workers = getattr(self, "vlm_judge_workers", 16)
        if self.vlm_judge is not None:
            def do_one(cid):
                try:
                    r = self.vlm_judge.match_query(cid, query)
                    return {
                        "clip_id": r["clip_id"],
                        "match": r.get("match", False),
                        "reasoning": r.get("reasoning", r.get("reason", r.get("analysis", ""))),
                        "analysis": r.get("analysis", ""),
                    }
                except Exception as e:
                    self.log_message("validate_search clip %s: %s", cid, e)
                    return {"clip_id": cid, "match": False, "reasoning": str(e), "analysis": ""}
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(do_one, clip_ids))
            return (results, None)
        return (
            None,
            "VLM Judge not available (set OPENAI_API_KEY)",
        )

    def _notify_access_request(
        self, username: str, email: str, reason: str, req_id: int
    ):
        if self.slack_notifier is None:
            return

        result = self.slack_notifier.notify_access_request(
            username=username,
            email=email,
            reason=reason,
            req_id=req_id,
        )

        sent_to = result.get("sent_to", [])
        errors = result.get("errors", [])
        if sent_to:
            self.log_message(
                "Slack notification sent for request %s to %s",
                req_id,
                ",".join(sent_to),
            )
        if errors:
            self.log_message(
                "Slack notification errors for request %s: %s",
                req_id,
                " | ".join(errors),
            )

    def invalidate_cache_annotation(self, *labels):
        self.search_pipeline.invalidate_annotation(*labels)

    def invalidate_cache_times(self):
        self.search_pipeline.invalidate_times()

    def invalidate_cache_comments(self):
        self.search_pipeline.invalidate_comments()

    def invalidate_cache_classifier(self, run_id):
        self.search_pipeline.invalidate_classifier(run_id)

    def _serve_stats_file(self, url_path, url_prefix, subdir):
        rel = url_path[len(url_prefix):]
        rel = rel.replace("\\", "/").lstrip("/")
        path = (Path(self.stats_dir) / subdir / rel).resolve()
        base = (Path(self.stats_dir) / subdir).resolve()
        if (
            not str(path).startswith(str(base))
            or not path.exists()
            or not path.is_file()
        ):
            return self.send_error(404, "Not found")
        if path.suffix.lower() == ".png":
            ctype = "image/png"
        elif path.suffix.lower() == ".json":
            ctype = "application/json"
        else:
            ctype = "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "max-age=60, public")
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def search(self, filters):
        return self.search_pipeline.search(filters)

    def do_GET(self):
        parsed_path = parse.urlparse(self.path)
        parsed_qs = parse.parse_qs(parsed_path.query)

        # Auth-aware endpoints and gating
        if parsed_path.path == "/whoami":
            user = self._current_user()
            if user is None:
                return self._send_json({"authenticated": False})
            return self._send_json(
                {
                    "authenticated": True,
                    "user": {
                        "username": user.username,
                        "email": user.email,
                        "role": user.role,
                    },
                }
            )

        if parsed_path.path == "/agent_url":
            user = self._require_user()
            if user is None:
                return
            return self._send_json({"agent_url": self.agent_url})

        if parsed_path.path == "/admin_data":
            user = self._current_user()
            if user is None or user.role != "admin":
                return self._send_json({"error": "forbidden"}, status=403)
            data = {
                "users": self.usersstore.list_users(),
                # Only show non-rejected (pending) requests in the admin console list
                "requests": self.usersstore.list_access_requests("pending"),
                "dataset_options": self.datastore.data_source_options,
            }
            return self._send_json(data)

        if parsed_path.path in (
            "/",
            "/leaderboard",
            "/arena",
            "/data_stats",
            "/policy_predictions",
        ):
            user = self._current_user()
            if user is None:
                self.send_response(302)
                self.send_header("Location", "/login")
                self.end_headers()
                return

        if parsed_path.path == "/admin":
            user = self._current_user()
            if user is None or user.role != "admin":
                self.send_response(302)
                self.send_header("Location", "/login")
                self.end_headers()
                return

        # Admin-only usage stats page
        if parsed_path.path == "/admin_stats":
            user = self._current_user()
            if user is None or user.role != "admin":
                self.send_response(302)
                self.send_header("Location", "/login")
                self.end_headers()
                return

        if parsed_path.path == "/rewrite":
            query = parsed_qs.get("query", [None])[0]
            if not query or not self.rewriter:
                return self._send_json({"rewrites": []})
            try:
                result = self.rewriter.rewrite_query(query)
                self.log_message("Query rewrite: query=%s", query)
                return self._send_json({"rewrites": result.queries})
            except Exception as e:
                print(f"[Rewrite endpoint] FAILED: {e}")
                return self._send_json({"rewrites": []})

        if parsed_path.path == "/videos":
            # Limit datasource options to those allowed for the current user
            user = self._require_user()
            if user is None:
                return
            ds_options = self.usersstore.get_allowed_datasources(user.id)

            filters = SearchFilters.from_query(
                parsed_qs, user_options={"data_source": ds_options}
            )
            clip_ids, results = self.search(filters)

            n = min(
                max(int(parsed_qs.get("n", [NUM_VIDEOS_PER_PAGE])[0]), 1), 20
            )
            num_videos = len(results)
            n_pages = (num_videos + n - 1) // n
            page = min(max(int(parsed_qs.get("page", [0])[0]), 0), n_pages)
            start = page * n
            stop = min(start + n, num_videos)

            self.send_response(200)
            self.send_header("Content-Type", "text/json")
            self.end_headers()

            project_source = filters.project_source
            segment = clip_ids[start:stop]
            clips = self.datastore.get_clips_dict(segment, project_source)
            clips = [clips[c] for c in segment]

            captions_by_clip = {
                cid: self.captionstore.get(cid) if self.captionstore else None
                for cid in segment
            }
            vlm_caption_scores = {}
            if self.vlm_judge:
                vlm_caption_scores = self.vlm_judge.get_caption_scores_for_videos(
                    captions_by_clip
                )

            # When a clustering run is active, look up each clip's
            # (cluster_id, distance) so the UI can render a "Show cluster"
            # affordance even before the user has narrowed to one cluster.
            cluster_for_clip = {}
            if filters.cluster_run_id and self.clustersearch:
                cluster_for_clip = self.clustersearch.cluster_for_clips(
                    list(segment), filters.cluster_run_id
                )

            judge_clip_ids = clip_ids[:VLM_JUDGE_RANKED_IDS_LIMIT]
            response = {
                "num_videos": num_videos,
                "page": page,
                "total": n_pages,
                "search_judge_clip_ids": judge_clip_ids,
                "vlm_judge_max_k": len(judge_clip_ids),
                "annotations_count": self.datastore.num_annotations(
                    project_source
                ),
                "manual_annotations_count": self.datastore.num_manual_annotations(
                    project_source
                ),
                "autolabel_annotations_count": self.datastore.num_autolabel_annotations(
                    project_source
                ),
                "n": n,
                "options": self.datastore.options(project_source),
                "metric_names": self.datastore.metric_names(project_source),
                "data_source_options": ds_options,
                "dataset_metadata": self.datastore.dataset_metadata(ds_options),
                "label_type_options": LABEL_TYPES,
                "project_options": self.datastore.project_options,
                "with_metrics_available": (
                    self.metricstore.has_metrics_index()
                    if self.metricstore else False
                ),
                "with_bev_available": (
                    self.bev_fetcher.has_bev_index()
                    if self.bev_fetcher else False
                ),
                "query_rewrite_available": self.rewriter is not None,
                "vlm_judge_available": self.vlm_judge is not None,
                **filters.to_dict(),
                "videos": [
                    {
                        "data_source": clip.data_source,
                        "annotations": clip.to_dict(),
                        "speed": (
                            self.trajectorystore.get_speed(clip_id)
                            if self.trajectorystore else None
                        ),
                        "acceleration": (
                            self.trajectorystore.get_acceleration(clip_id)
                            if self.trajectorystore else None
                        ),
                        "curvature": (
                            self.trajectorystore.get_curvature(clip_id)
                            if self.trajectorystore else None
                        ),
                        "jerk": (
                            self.trajectorystore.get_jerk(clip_id)
                            if self.trajectorystore else None
                        ),
                        "positions": (
                            self.trajectorystore.get_positions(clip_id)
                            if self.trajectorystore else None
                        ),
                        "captions": captions_by_clip[clip_id],
                        "vlm_caption_scores": vlm_caption_scores.get(clip_id, {}),
                        "country": self.datastore.get_country(clip_id),
                        "country_name": self.datastore.get_country_name(
                            clip_id
                        ),
                        "has_embeddings": self.embeddingsstore.has_embeddings(
                            clip_id
                        ),
                        "has_trajectories": (
                            self.trajectorystore.has_trajectories(clip_id)
                            if self.trajectorystore else False
                        ),
                        "semantic_video_score": results[
                            clip_id
                        ].semantic_search_clip_score,
                        "semantic_text_score": results[
                            clip_id
                        ].semantic_search_text_score,
                        "trajectory_shape_score": results[
                            clip_id
                        ].trajectory_shape_score,
                        "clip_score": results[clip_id].visual_search_score,
                        "clip_image_score": results[clip_id].visual_image_score,
                        "numeric_scores": results[clip_id].numeric_scores,
                        "cluster_distance_score": results[
                            clip_id
                        ].cluster_distance_score,
                        "caption_embed_score": results[
                            clip_id
                        ].caption_embed_score,
                        "classification_score": results[
                            clip_id
                        ].classifier_score,
                        "rrf_score": results[clip_id].rrf_score,
                        "sil_apis": self.clips_to_apis.get(clip_id, None),
                        "cluster_membership": (
                            {
                                "cluster_id": cluster_for_clip[clip_id][0],
                                "distance": cluster_for_clip[clip_id][1],
                            }
                            if clip_id in cluster_for_clip else None
                        ),
                        "comments": "",
                    }
                    for ii, (clip, clip_id) in enumerate(zip(clips, segment))
                ],
            }

            self.wfile.write(
                json.dumps(response, cls=InfEncoder).encode("utf-8")
            )

        elif parsed_path.path == "/clip_ids":
            # Lightweight programmatic search: returns the full ranked clip
            # list only. No pagination, no metadata. Programmatic callers
            # (WheelHTTPClient) use this instead of /videos.
            user = self._require_user()
            if user is None:
                return
            ds_options = self.usersstore.get_allowed_datasources(user.id)
            filters = SearchFilters.from_query(
                parsed_qs, user_options={"data_source": ds_options}
            )
            clip_ids, _ = self.search(filters)
            return self._send_json({"clip_ids": list(clip_ids)})

        elif parsed_path.path.startswith("/video/"):
            clip_id = parsed_path.path[7:-4]
            video_path = self.datastore.get_video_path(clip_id)
            if video_path is None:
                self.send_error(404, "File not found")
                return
            self.video_fetcher.serve(self, video_path)

        elif parsed_path.path.startswith("/depth_video/"):
            clip_id = parsed_path.path[13:-4]
            depth_path = self.autolabels_store.get_depth_path(clip_id)
            if depth_path is None:
                self.send_error(404, "File not found")
                return
            self.video_fetcher.serve(self, depth_path)

        elif parsed_path.path.startswith("/boxes_video/"):
            clip_id = parsed_path.path[13:-4]
            boxes_path = self.autolabels_store.get_boxes_path(clip_id)
            if boxes_path is None:
                self.send_error(404, "File not found")
                return
            self.video_fetcher.serve(self, boxes_path)

        elif parsed_path.path.startswith("/point_video/"):
            clip_id = parsed_path.path[13:-4]
            point_path = self.autolabels_store.get_pointmap_path(clip_id)
            if point_path is None:
                self.send_error(404, "File not found")
                return
            self.video_fetcher.serve(self, point_path)

        elif parsed_path.path.startswith("/mfmrh_video/"):
            clip_id = parsed_path.path[13:-4]
            mfmrh_path = self.autolabels_store.get_mfmr_path(clip_id)
            if mfmrh_path is None:
                self.send_error(404, "File not found")
                return
            self.video_fetcher.serve(self, mfmrh_path)

        elif parsed_path.path.startswith("/vipe_video/"):
            clip_id = parsed_path.path[12:-4]
            vipe_path = self.autolabels_store.get_vipe_path(clip_id)
            if vipe_path is None:
                self.send_error(404, "File not found")
                return
            self.video_fetcher.serve(self, vipe_path)

        elif parsed_path.path == "/metrics":
            # Limit datasource options to those allowed for the current user
            user = self._require_user()
            if user is None:
                return
            ds_options = self.usersstore.get_allowed_datasources(user.id)

            filters = SearchFilters.from_query(
                parsed_qs, user_options={"data_source": ds_options}
            )
            clip_ids, _ = self.search(filters)

            self.send_response(200)
            self.send_header("Content-Type", "text/json")
            self.end_headers()

            reduction = parsed_qs.get("reduction", [None])[0]
            if reduction is None:
                reduction = "mean"
            with_same_clips = parsed_qs.get("with_same_clips", [None])[0]
            if with_same_clips is not None:
                with_same_clips = True

            project_source = filters.project_source

            response = {
                "options": self.datastore.options(project_source),
                "metric_names": self.datastore.metric_names(project_source),
                "data_source_options": ds_options,
                "dataset_metadata": self.datastore.dataset_metadata(ds_options),
                "label_type_options": LABEL_TYPES,
                "project_options": self.datastore.project_options,
                **filters.to_dict(),
                "reduction": reduction,
                "with_same_clips": with_same_clips,
                "model_name": None,
                "models_by_leaderboard": self.metricstore.models_by_leaderboard,
                "query_rewrite_available": self.rewriter is not None,
                "metrics": {
                    model: self.metricstore.get_reduced_metrics(
                        model,
                        clip_ids,
                        hash_key=filters.key,
                        reduction=reduction,
                        with_same_clips=with_same_clips,
                    )
                    for model in self.metricstore.models
                },
            }

            self.wfile.write(
                json.dumps(response, cls=InfEncoder).encode("utf-8")
            )

        elif parsed_path.path == "/per_clip_metrics":
            user = self._require_user()
            if user is None:
                return
            ds_options = self.usersstore.get_allowed_datasources(user.id)

            filters = SearchFilters.from_query(
                parsed_qs, user_options={"data_source": ds_options}
            )
            clip_ids, _ = self.search(filters)

            self.send_response(200)
            self.send_header("Content-Type", "text/json")
            self.end_headers()

            response = {"metrics": [], "clips": [], "values": []}
            model_name = parsed_qs.get("model_name", [None])[0]
            with_same_clips = parsed_qs.get("with_same_clips", [None])[0]
            if with_same_clips is None:
                with_same_clips = False
            if model_name is not None and model_name in self.metricstore.models:
                response = self.metricstore.get_per_clip_metrics(
                    model_name, clip_ids, with_same_clips=with_same_clips
                )

            self.wfile.write(
                json.dumps(response, cls=InfEncoder).encode("utf-8")
            )

        elif parsed_path.path == "/predictions":
            user = self._require_user()
            if user is None:
                return
            ds_options = self.usersstore.get_allowed_datasources(user.id)

            filters = SearchFilters.from_query(
                parsed_qs, user_options={"data_source": ds_options}
            )

            self.send_response(200)
            self.send_header("Content-Type", "text/json")
            self.end_headers()

            clip_id = parsed_qs.get("clip_id", [None])[0]
            project_source = filters.project_source
            clips = self.datastore.get_clips_dict([clip_id], project_source)
            clip = clips[clip_id]
            print(f"predictions for {clip_id}")

            captions_by_clip = {clip_id: self.captionstore.get(clip_id)}
            vlm_caption_scores = (
                self.vlm_judge.get_caption_scores_for_videos(captions_by_clip)
                if self.vlm_judge
                else {}
            )
            response = {
                "annotations_count": self.datastore.num_annotations(
                    project_source
                ),
                "manual_annotations_count": self.datastore.num_manual_annotations(
                    project_source
                ),
                "autolabel_annotations_count": self.datastore.num_autolabel_annotations(
                    project_source
                ),
                "project_source": project_source,
                "project_options": self.datastore.project_options,
                "clip_id": clip_id,
                "options": self.datastore.options(project_source),
                "metric_names": self.datastore.metric_names(project_source),
                "vlm_judge_available": self.vlm_judge is not None,
                "videos": [
                    {
                        "data_source": clip.data_source,
                        "annotations": clip.to_dict(),
                        "captions": captions_by_clip[clip_id],
                        "vlm_caption_scores": vlm_caption_scores.get(clip_id, {}),
                        "country": self.datastore.get_country(clip_id),
                        "country_name": self.datastore.get_country_name(
                            clip_id
                        ),
                        "speed": self.trajectorystore.get_speed(clip_id),
                        "acceleration": self.trajectorystore.get_acceleration(
                            clip_id
                        ),
                        "curvature": self.trajectorystore.get_curvature(
                            clip_id
                        ),
                        "jerk": self.trajectorystore.get_jerk(clip_id),
                        "positions": self.trajectorystore.get_positions(
                            clip_id
                        ),
                    }
                ],
                "predictions": [
                    {
                        "model_name": model,
                        "pred_positions": self.predictions_store.get_pred_positions(
                            clip_id, model
                        ),
                        "gt_positions": self.predictions_store.get_gt_positions(
                            clip_id, model
                        ),
                        "pred_captions": self.predictions_store.get_pred_captions(
                            clip_id, model
                        ),
                    }
                    for model in self.predictions_store.get_models(clip_id)
                ],
                "metrics": [
                    {
                        model: self.metricstore.get_per_clip_metrics(
                            model, [clip_id]
                        )
                    }
                    for model in self.predictions_store.get_models(clip_id)
                ],
            }
            self.wfile.write(
                json.dumps(response, cls=InfEncoder).encode("utf-8")
            )

        elif parsed_path.path == "/full_metrics":
            clip_id = parsed_qs.get("clip_id", [None])[0]
            model_name = parsed_qs.get("model_name", ["ground_truth"])[0]

            if clip_id is None:
                self.send_error(400, "Missing clip_id parameter")
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/json")
            self.end_headers()

            full_metrics = self.metricstore.get_full_clip_metrics(
                model_name, clip_id
            )
            response = {
                "clip_id": clip_id,
                "model_name": model_name,
                **full_metrics,
            }

            self.wfile.write(json.dumps(response).encode("utf-8"))

        elif parsed_path.path == "/classifiers_status":
            projects = self.datastore.project_options

            runs = []
            classifier_runs = (
                self.classifiersearch.list_runs()
                if self.classifiersearch else []
            )
            for metadata in classifier_runs:
                run_id = metadata.get("run_id")
                if not run_id:
                    continue
                run_dir = Path(self.classifier_dir) / run_id
                is_done = (run_dir / "predicted_scores.json").exists()
                is_running = self.classifier_jobs.is_running(run_id)
                if is_done:
                    status = "done"
                elif is_running:
                    status = "pending"
                else:
                    status = metadata.get("status", "failed")
                    if status not in ("done", "pending", "failed"):
                        status = "failed"
                pos_hash = self._clip_list_hash_for(run_dir / "positive_clips.json")
                neg_hash = self._clip_list_hash_for(run_dir / "negative_clips.json")
                runs.append({
                    "run_id": run_id,
                    "status": status,
                    "embed_type": metadata.get("embed_type", "cosmos"),
                    "positive_labels": metadata.get("positive_labels", []),
                    "negative_labels": metadata.get("negative_labels", []),
                    "trained_by": metadata.get("trained_by", ""),
                    "n_positive_clips": metadata.get("n_positive_clips", 0),
                    "n_negative_clips": metadata.get("n_negative_clips", 0),
                    "started_at": metadata.get("started_at", 0),
                    "use_autolabels": metadata.get("use_autolabels", False),
                    "search_params": metadata.get("search_params", ""),
                    "positive_clip_list_hash": pos_hash,
                    "negative_clip_list_hash": neg_hash,
                })

            runs.sort(key=lambda r: r.get("started_at", 0), reverse=True)

            response = {
                "runs": runs,
                "pending": [
                    r["run_id"] for r in runs if r["status"] == "pending"
                ],
                "number_of_annotations": dict(
                    self.datastore.option_set(projects)
                ),
                "number_of_autolabelled_annotations": dict(
                    self.datastore.option_set_autolabelled(projects)
                ),
            }

            self.send_response(200)
            self.send_header("Content-Type", "text/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))

        elif parsed_path.path.startswith("/classifier/export/"):
            # Export classifier weights as JSON
            run_id = parse.unquote(
                parsed_path.path[len("/classifier/export/"):]
            ).strip("/")
            run_dir = Path(self.classifier_dir) / run_id
            weights_path = run_dir / "LR_weights.npz"
            metadata_path = run_dir / "metadata.json"

            if not weights_path.exists():
                self.send_error(404, f"Classifier run '{run_id}' not found")
                return

            try:
                coef, intercept = load_lr_weights(run_dir)
                metadata = {}
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)

                export_data = {
                    "run_id": run_id,
                    "version": 2,
                    "embed_type": metadata.get("embed_type"),
                    "positive_labels": metadata.get("positive_labels", []),
                    "negative_labels": metadata.get("negative_labels", []),
                    "trained_by": metadata.get("trained_by"),
                    "coefficients": coef.tolist(),
                    "intercept": intercept.tolist(),
                }

                json_bytes = json.dumps(export_data, indent=2).encode("utf-8")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="classifier_{run_id}.json"'
                )
                self.send_header("Content-Length", str(len(json_bytes)))
                self.end_headers()
                self.wfile.write(json_bytes)

            except Exception as e:
                self.send_error(500, f"Failed to export classifier: {str(e)}")

        elif parsed_path.path == "/clip_list":
            # Return the stored clip list for a hash. Used by the UI on
            # page load (URL has the hash, panel needs the count).
            qs = parse.parse_qs(parsed_path.query)
            hash = (qs.get("hash") or [""])[0]
            if not hash:
                return self._send_json(
                    {"error": "missing hash"}, status=400
                )
            try:
                clip_ids = self.cliplistsearch.load(hash)
            except ValueError as e:
                return self._send_json({"error": str(e)}, status=400)
            if clip_ids is None:
                return self._send_json(
                    {"error": f"unknown hash: {hash}"}, status=404
                )
            return self._send_json(
                {"hash": hash, "count": len(clip_ids), "clip_ids": clip_ids}
            )

        elif parsed_path.path == "/annotations.csv":
            # Stream CSV efficiently
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Encoding", "gzip")
            self.end_headers()

            gz = gzip.GzipFile(fileobj=self.wfile, mode="wb")
            writer_wrapper = io.TextIOWrapper(
                gz, encoding="utf-8", newline="", write_through=True
            )
            writer = csv.writer(writer_wrapper)
            writer.writerow(
                ["clip_id", "key", "start_time", "end_time", "label_type"]
            )

            query = (
                "SELECT c.clip_id, a.key, a.start_time, a.end_time, a.label_type "
                "FROM annotations a JOIN clips c ON c.clip_id = a.clip_id "
                "ORDER BY c.clip_id"
            )
            with self.datastore.lock:
                cur = self.datastore.conn.execute(query)
                for row in cur:
                    writer.writerow(
                        [
                            row["clip_id"],
                            row["key"],
                            row["start_time"],
                            row["end_time"],
                            row["label_type"],
                        ]
                    )

            writer_wrapper.flush()
            try:
                buf = writer_wrapper.detach()
            except Exception:
                buf = None
            if buf is not None:
                buf.close()

        elif parsed_path.path == "/annotations_summary.csv":
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.end_headers()

            writer_wrapper = io.TextIOWrapper(
                self.wfile, encoding="utf-8", newline="", write_through=True
            )
            writer = csv.writer(writer_wrapper)

            writer.writerow(
                [
                    "Category",
                    "# Manual Annotations",
                    "# Timed Annotations",
                    "# Autolabel Annotations",
                    "Total (manual + autolabel)",
                ]
            )

            summary = self.datastore.summarize_annotations()
            summary.sort(key=lambda x: x["key"])
            for row in summary:
                writer.writerow(
                    [
                        row["key"],
                        row["manual_count"],
                        row["with_time_count"],
                        row["autolabel_count"],
                        row["manual_count"] + row["autolabel_count"],
                    ]
                )

            writer_wrapper.flush()
            writer_wrapper.detach()

        elif parsed_path.path == "/current_search.csv":
            # Reject unauthenticated callers before any response bytes
            # are written, so the 403 stays well-formed.
            user = self._require_user()
            if user is None:
                return

            # Export current search results as a CSV file
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Encoding", "gzip")
            self.end_headers()

            gz = gzip.GzipFile(fileobj=self.wfile, mode="wb")
            writer_wrapper = io.TextIOWrapper(
                gz, encoding="utf-8", newline="", write_through=True
            )
            writer = csv.writer(writer_wrapper)

            ds_options = self.usersstore.get_allowed_datasources(user.id)

            # Build search filters from query parameters
            filters = SearchFilters.from_query(
                parsed_qs, user_options={"data_source": ds_options}
            )

            # Execute search to get matching clip IDs
            clip_ids, _ = self.search(filters)

            # Safety net against pathological queries
            if len(clip_ids) > 10000000:
                return

            # For very large result sets (>500k clips), skip the per-clip
            # data_source lookup and export clip_id only
            if len(clip_ids) > 500000:
                writer.writerow(["clip_id"])
                for clip_id in clip_ids:
                    writer.writerow([clip_id])
            # For large result sets (>10k clips), export simplified CSV with
            # only clip_id and data_source
            elif len(clip_ids) > 10000:
                clip_data_sources = self.datastore.get_clip_data_sources(clip_ids)
                print(f"Gathered the data_sources for {len(clip_ids)} clips")

                fieldnames = ["clip_id", "data_source"]
                writer.writerow(fieldnames)

                for clip_id in clip_ids:
                    data_source = clip_data_sources.get(clip_id, "")
                    writer.writerow([clip_id, data_source])
            else:
                # For smaller result sets, export detailed CSV with annotations
                clips = self.datastore.get_clips_dict(
                    clip_ids, filters.project_source
                )
                clips = [clips[c].to_dict() for c in clip_ids]
                fieldnames = [
                    "clip_id",
                    "data_source",
                    "project",
                    "label",
                    "start_time",
                    "end_time",
                    "label_type",
                    "value",
                ]

                writer.writerow(fieldnames)
                for clip in clips:
                    annotations = clip["annotations"]

                    # Export clips without annotations as a single row with empty annotation fields
                    if len(annotations) == 0:
                        writer.writerow(
                            [
                                clip["clip_id"],
                                clip["data_source"],
                                "",  # project
                                "",  # key
                                "",  # start_time
                                "",  # end_time
                                "",  # label_type
                                "",  # value
                            ]
                        )

                    # Export one row per annotation for clips with annotations
                    for ann in annotations:
                        writer.writerow(
                            [
                                clip["clip_id"],
                                clip["data_source"],
                                ann["project"],
                                ann["key"],
                                ann["start_time"],
                                ann["end_time"],
                                ann["label_type"],
                                ann["value"],
                            ]
                        )
            writer_wrapper.flush()
            try:
                buf = writer_wrapper.detach()
            except Exception:
                buf = None
            if buf is not None:
                buf.close()

        elif parsed_path.path in ("/favicon", "/favicon.ico"):
            self.send_response(200)
            self.send_header("ETag", self.favicon[0])
            self.send_header("Content-Type", self.favicon[1])
            self.send_header(
                "Cache-Control", "max-age=3600, public, must-revalidate"
            )
            self.end_headers()
            self.wfile.write(self.favicon[2])

        elif parsed_path.path.startswith("/api/bev/"):
            # BEV data endpoint: /api/bev/{clip_id}
            clip_id = parsed_path.path[len("/api/bev/") :]
            self.bev_fetcher.serve(self, clip_id)

        elif parsed_path.path == "/api/vlm_judge/caption_score":
            clip_id = parsed_qs.get("clip_id", [None])[0]
            caption = parsed_qs.get("caption", [None])[0]
            uid_raw = parsed_qs.get("uid", [None])[0]
            if not clip_id or not caption or uid_raw is None:
                return self._send_json(
                    {"error": "clip_id, caption and uid required"}, status=400
                )
            if self.vlm_judge is not None:
                try:
                    result = self.vlm_judge.score_caption(clip_id, caption, int(uid_raw))
                    if "error" in result:
                        return self._send_json(result, status=400)
                    return self._send_json(result)
                except Exception as e:
                    self.log_message("VLM Judge error: %s", e)
                    return self._send_json({"error": str(e)}, status=500)
            return self._send_json(
                {"error": "VLM Judge not configured (please set OPENAI_API_KEY)"},
                status=503,
            )

        elif parsed_path.path == "/api/vlm_judge/validate_search":
            query = parsed_qs.get("search", [None])[0] or parsed_qs.get("query", [None])[0]
            clip_ids_raw = parsed_qs.get("clip_ids", [None])[0] or parsed_qs.get("clip_id", [""])[0]
            if not query or not clip_ids_raw:
                return self._send_json(
                    {"error": "search and clip_ids required"},
                    status=400,
                )
            clip_ids = [x.strip() for x in clip_ids_raw.split(",") if x.strip()]
            results, err = self.validate_search_vlm(query, clip_ids)
            if err is not None:
                return self._send_json({"error": err}, status=503)
            return self._send_json({"results": results})

        elif parsed_path.path == "/api/vlm_judge/status":
            if self.vlm_judge is not None:
                return self._send_json({
                    "enabled": True,
                    "healthy": True,
                    "in_process": True,
                })
            return self._send_json({"enabled": False})

        elif parsed_path.path == "/admin_stats_data":
            # Admin-only endpoint: runs analyze_logs.py and returns JSON summary
            user = self._current_user()
            if user is None or user.role != "admin":
                return self._send_json({"error": "forbidden"}, status=403)

            try:
                # Temporary output locations for analyzer
                tmp_dir = tempfile.mkdtemp(prefix="admin_stats_")
                json_out = Path(tmp_dir) / "summary.json"

                analyzer = str(Path(__file__).parent / "analyze_logs.py")

                # Launch analyzer as a subprocess each time, per requirements
                cmd = [
                    sys.executable,
                    analyzer,
                    self.log_dir,
                    tmp_dir,
                    "--json-out",
                    str(json_out),
                    "-r",
                ]
                print(cmd)
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    text=True,
                )
                if proc.returncode != 0:
                    self.log_message(
                        "analyze_logs.py failed: rc=%d stderr=%s",
                        proc.returncode,
                        proc.stderr[-200:].replace("\n", " "),
                    )
                    return self._send_json(
                        {
                            "error": "analysis_failed",
                            "detail": "See server logs",
                        },
                        status=500,
                    )

                # Read JSON summary and return
                try:
                    with open(json_out, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                except Exception as e:
                    self.log_message("Failed reading analyzer JSON: %s", str(e))
                    return self._send_json(
                        {"error": "invalid_summary", "detail": str(e)},
                        status=500,
                    )

                return self._send_json({"summary": summary})
            except Exception as e:
                self.log_message("Exception running analyzer: %s", str(e))
                return self._send_json(
                    {"error": "exception", "detail": str(e)}, status=500
                )

        elif parsed_path.path == "/data_stats_list":
            user = self._require_user()
            if user is None:
                return

            ds_options = self.usersstore.get_allowed_datasources(user.id)
            base_traj = Path(self.stats_dir) / "trajectory_stats"
            base_data = Path(self.stats_dir) / "dataset_stats"

            entries = []
            for ds in ds_options:
                slug = (
                    re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(ds)).strip("_").lower()
                )

                vis = base_traj / f"trajectory_visualization_{slug}.png"
                stats = base_traj / f"trajectory_statistics_{slug}.png"
                per_clip = base_traj / f"trajectory_per_clip_stats_{slug}.png"
                summary = base_traj / f"trajectory_summary_{slug}.json"

                traj_pngs = (
                    [p for p in base_traj.glob(f"*{slug}*.png")]
                    if base_traj.exists()
                    else []
                )
                traj_jsons = (
                    [p for p in base_traj.glob(f"*{slug}*.json")]
                    if base_traj.exists()
                    else []
                )
                data_pngs = (
                    [p for p in base_data.glob(f"*{slug}*.png")]
                    if base_data.exists()
                    else []
                )
                data_jsons = (
                    [p for p in base_data.glob(f"*{slug}*.json")]
                    if base_data.exists()
                    else []
                )

                has_traj = (
                    vis.exists()
                    and stats.exists()
                    and per_clip.exists()
                    and summary.exists()
                )
                has_any = (
                    has_traj
                    or traj_pngs
                    or traj_jsons
                    or data_pngs
                    or data_jsons
                )
                if not has_any:
                    continue

                entry = {"dataset": ds}
                if has_traj:
                    entry.update(
                        {
                            "visualization": f"/trajectory_stats/{vis.name}",
                            "statistics": f"/trajectory_stats/{stats.name}",
                            "per_clip": f"/trajectory_stats/{per_clip.name}",
                            "summary": f"/trajectory_stats/{summary.name}",
                        }
                    )
                if traj_pngs:
                    entry["trajectory_pngs"] = [
                        f"/trajectory_stats/{p.name}" for p in traj_pngs
                    ]
                if traj_jsons:
                    entry["trajectory_jsons"] = [
                        f"/trajectory_stats/{p.name}" for p in traj_jsons
                    ]

                if data_pngs:
                    entry["data_pngs"] = [
                        f"/dataset_stats/{p.name}" for p in data_pngs
                    ]
                if data_jsons:
                    entry["data_jsons"] = [
                        f"/dataset_stats/{p.name}" for p in data_jsons
                    ]

                data_summary = base_data / f"data_stats_summary_{slug}.json"
                data_plot = base_data / f"labels_barplot_{slug}.png"
                if data_summary.exists():
                    entry["data_summary"] = (
                        f"/dataset_stats/{data_summary.name}"
                    )
                if data_plot.exists():
                    entry["data_plot"] = f"/dataset_stats/{data_plot.name}"

                entries.append(entry)

            # Follow videos endpoint style: set headers explicitly
            self.send_response(200)
            self.send_header("Content-Type", "text/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"datasets": entries}, cls=InfEncoder).encode(
                    "utf-8"
                )
            )

        elif parsed_path.path.startswith("/trajectory_stats/"):
            self._serve_stats_file(
                parsed_path.path, "/trajectory_stats/", "trajectory_stats"
            )

        elif parsed_path.path.startswith("/dataset_stats/"):
            self._serve_stats_file(
                parsed_path.path, "/dataset_stats/", "data_stats"
            )

        elif parsed_path.path == "/clustering_status":
            clustering_dir = Path(self.clustering_dir)
            runs = []
            for subdir in clustering_dir.iterdir():
                if not subdir.is_dir():
                    continue
                meta_path = subdir / "metadata.json"
                if not meta_path.exists():
                    continue
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                except Exception:
                    continue
                run_id = subdir.name
                is_done = (
                    (subdir / "representative_by_cluster.json").exists()
                    and (subdir / "umap.json").exists()
                )
                is_running = self.clustering_jobs.is_running(run_id)
                if is_done:
                    status = "done"
                elif is_running:
                    status = "pending"
                else:
                    status = meta.get("status", "failed")
                    if status not in ("done", "pending", "failed"):
                        status = "failed"
                runs.append({
                    "run_id": run_id,
                    "status": status,
                    "n_clusters": meta.get("n_clusters", 0),
                    "n_clips": meta.get("n_input_clips", 0),
                    "search_params": meta.get("search_params", ""),
                    "started_at": meta.get("started_at", 0),
                    "embed_type": meta.get("embed_type", "cosmos"),
                })

            runs.sort(key=lambda r: r.get("started_at", 0), reverse=True)
            return self._send_json({"runs": runs})

        elif parsed_path.path == "/clustering_results":
            run_id = parsed_qs.get("run_id", [None])[0]
            if not run_id or not self.clustering_dir:
                return self._send_json({"error": "missing run_id"}, status=404)
            run_dir = Path(self.clustering_dir) / run_id
            clusters_path = run_dir / "representative_by_cluster.json"
            umap_path = run_dir / "umap.json"
            meta_path = run_dir / "metadata.json"
            if not clusters_path.exists():
                return self._send_json({"error": "run not done"}, status=404)
            try:
                with open(clusters_path, "r") as f:
                    clusters = json.load(f)
                # Compute per-cluster representatives from the on-disk
                # assignments parquet. Old runs without a readable parquet
                # fall back to no preview rather than erroring.
                try:
                    reps = self.clustersearch.representatives(run_id)
                except Exception:
                    reps = {}
                # Strip clip_ids/distances if present (old-format runs);
                # keep cluster_size and add representative_clip_id.
                clusters = {
                    cid: {
                        "cluster_size": v.get("cluster_size", 0),
                        "representative_clip_id": reps.get(int(cid)),
                    }
                    for cid, v in clusters.items()
                }
                umap = {}
                if umap_path.exists():
                    with open(umap_path, "r") as f:
                        umap = json.load(f)
                metadata = {}
                if meta_path.exists():
                    with open(meta_path, "r") as f:
                        metadata = json.load(f)
                topics_payload = read_topics(run_dir)
                topics = topics_payload.get("topics", {})
                topics_meta = {
                    k: v for k, v in topics_payload.items() if k != "topics"
                }
                return self._send_json({
                    "run_id": run_id,
                    "clusters": clusters,
                    "umap": umap,
                    "metadata": metadata,
                    "topics": topics,
                    "topics_meta": topics_meta,
                })
            except Exception as e:
                return self._send_json(
                    {"error": str(e)}, status=500
                )

        elif parsed_path.path == "/cluster_members":
            run_id = parsed_qs.get("run_id", [None])[0]
            cluster_id = parsed_qs.get("cluster_id", [None])[0]
            if not run_id or cluster_id is None:
                return self._send_json(
                    {"error": "missing run_id/cluster_id"}, status=400
                )
            try:
                clip_ids, distances = self.clustersearch.members(
                    run_id, int(cluster_id)
                )
                return self._send_json(
                    {"clip_ids": clip_ids, "distances": distances}
                )
            except Exception as e:
                return self._send_json({"error": str(e)}, status=500)

        elif parsed_path.path == "/closest_clusters":
            run_id = parsed_qs.get("run_id", [None])[0]
            query = parsed_qs.get("query", [None])[0]
            try:
                k = int(parsed_qs.get("k", ["10"])[0])
            except ValueError:
                return self._send_json({"error": "k must be int"}, status=400)
            if not run_id or not query:
                return self._send_json(
                    {"error": "missing run_id or query"}, status=400
                )
            metadata_path = (
                Path(self.clustering_dir) / run_id / "metadata.json"
            )
            if not metadata_path.exists():
                return self._send_json({"error": "run not found"}, status=404)
            with open(metadata_path) as f:
                embed_type = json.load(f).get("embed_type", "cosmos")
            encoder = {
                "cosmos":  self.embeddingsstore,
                "caption": self.captionembeddingsstore,
                "visual":  self.clipembeddingsstore,
            }.get(embed_type)
            if encoder is None:
                return self._send_json(
                    {"error": f"unknown embed_type: {embed_type}"}, status=400
                )
            try:
                query_embedding = encoder.encode_text(query)
            except Exception as e:
                return self._send_json(
                    {"error": f"encoding failed: {e}"}, status=500
                )
            rankings = self.clustersearch.closest_clusters(
                query_embedding, run_id, k=k
            )
            payload = {
                "run_id": run_id,
                "query": query,
                "results": [
                    {"cluster_id": cid, "distance": dist}
                    for cid, dist in rankings
                ],
            }
            if not rankings:
                payload["warning"] = (
                    "centroids not persisted for this run; re-run "
                    "clustering to enable cluster-by-query"
                )
            return self._send_json(payload)

        elif parsed_path.path.startswith("/arena/") and self.arena_store is not None:
            self.handle_arena_get(parsed_path, parsed_qs)

        else:
            self.static_pages.serve(self, parsed_path.path)

    def handle_core_post(self, action, parts):
        if action == "add":
            self.timers.tic()

            video, uid, label, start_time, end_time, project = parts[1:]
            start_time = float(start_time)
            end_time = float(end_time)
            print(action, video, uid, label, start_time, end_time, project)
            self.datastore.add(
                video, uid, label, label_type="manual", project=project
            )
            self.invalidate_cache_annotation(label)

            self.log_message(
                "Adding label %s for project %s took %f",
                label,
                project,
                self.timers.toc(),
            )

        elif action == "update_times":
            self.timers.tic()

            video, uid, label, start_time, end_time, project = parts[1:]
            start_time = float(start_time)
            end_time = float(end_time)
            print(action, video, uid, label, start_time, end_time)
            self.datastore.update_times(
                video, uid, label, start_time, end_time, project=project
            )
            self.invalidate_cache_times()

            self.log_message(
                "Updating times for label %s for project %s took %f",
                label,
                project,
                self.timers.toc(),
            )

        elif action == "remove":
            self.timers.tic()

            video, uid, label, start_time, end_time, project = parts[1:]
            print(action, video, uid, label, start_time, end_time)
            self.datastore.remove(video, uid, label, project)
            self.invalidate_cache_annotation(label)

            self.log_message(
                "Removing label %s for project %s took %f",
                label,
                project,
                self.timers.toc(),
            )

        elif action == "verify":
            self.timers.tic()

            video, uid, label, start_time, end_time, project = parts[1:]
            print(action, video, uid, label, project)
            self.datastore.verify(uid, project)
            self.invalidate_cache_annotation(label)

            self.log_message(
                "Verifying label %s for project %s took %f",
                label,
                project,
                self.timers.toc(),
            )


        elif action == "auto_label":
            self.timers.tic()

            path, label, auto_label_type, npages, n_clips, project_to_write = parts[1:]
            parsed_path = parse.urlparse(path)
            parsed_qs = parse.parse_qs(parsed_path.query)

            # Limit datasource options to those allowed for the current user
            user = self._current_user()
            ds_options = self.usersstore.get_allowed_datasources(user.id)

            filters = SearchFilters.from_query(
                parsed_qs, user_options={"data_source": ds_options}
            )
            clip_ids, _ = self.search(filters)
            if npages != "":
                clip_ids = clip_ids[: NUM_VIDEOS_PER_PAGE * (int(npages) + 1)]
            elif n_clips != "":
                clip_ids = clip_ids[: int(n_clips)]
            with self.datastore.lock:
                if auto_label_type in ["replace", "clear"]:
                    self.datastore.remove_autolabel(label, project_to_write)
                if auto_label_type in ["union", "replace"]:
                    self.datastore.add_many(
                        clip_ids,
                        label,
                        project=project_to_write,
                        label_type="autolabel",
                    )
            self.invalidate_cache_annotation(label)

            self.log_message(
                "Autolabel pages=%s clips=%s label=%s project=%s took %f",
                npages,
                n_clips,
                label,
                project_to_write,
                self.timers.toc(),
            )

        elif action in ["rename_label", "merge_label"]:
            old_label, new_label, project_to_write = parts[1:]
            old_label = old_label.split(",")
            self.datastore.rename(old_label, new_label, project_to_write)
            self.invalidate_cache_annotation(*old_label, new_label)

        elif action == "delete_label":
            label, _, project_to_write = parts[1:]
            self.datastore.remove_label(label, project_to_write)
            self.invalidate_cache_annotation(label)

        elif action == "mass_label":
            self.timers.tic()

            label, clip_ids, project_to_write = parts[1:4]
            label_type = parts[4] if len(parts) > 4 else "manual"
            clip_ids_to_label = clip_ids.split(",")
            self.datastore.add_many(
                clip_ids_to_label,
                label,
                project=project_to_write,
                start_time=-1,
                end_time=-1,
                label_type=label_type,
            )
            self.invalidate_cache_annotation(label)

            self.log_message(
                f"Labelling {len(clip_ids_to_label)} took %f", self.timers.toc()
            )

        elif action == "upload_annotations":
            rest = parts[1:]
            self.timers.tic()
            (
                clip_ids,
                annotations,
                start_time,
                end_time,
                project_to_write,
                values,
            ) = parts[1:]
            clip_ids = clip_ids.split(",")
            annotations = annotations.split(",")
            start_time = start_time.split(",")
            end_time = end_time.split(",")
            values = values.split(",")
            # Update "" with None and str with float
            values = [None if v == "" else float(v) for v in values]

            self.datastore.add_many(
                clip_ids,
                annotations,
                project=project_to_write,
                start_time=start_time,
                end_time=end_time,
                value=values,
                label_type="manual",
            )
            self.log_message(
                f"Uploading {len(annotations)} took %f", self.timers.toc()
            )
            labels = set(annotations)
            for label in labels:
                self.invalidate_cache_annotation(label)

        elif action == "upload_captions":
            rest = parts[1:]
            self.timers.tic()
            model_name = rest[0]
            dataset_name = rest[1]
            raw_bytes = base64.b64decode(rest[2].encode("ascii"))
            # Read parquet from bytes
            df = pd.read_parquet(io.BytesIO(raw_bytes))
            self.captionstore.insert_from_dataframe(
                df, model_name.strip(), dataset_name.strip()
            )
            self.log_message(
                f"Uploading {len(df)} for {len(set(df['clip_id']))} took %f",
                self.timers.toc(),
            )

        elif action == "reconstruction":
            clip_id, method = parts[1], parts[2]
            # Normalize method (strip optional emoji prefix)
            method = method.strip()
            if method.startswith("⚡"):
                method = method.lstrip("⚡").strip()

            self.log_message(
                "Reconstruction action requested: clip_id=%s method=%s",
                clip_id,
                method,
            )
            if method == "InstantNuRec":
                # Mimic mock-server submit: print & broadcast
                print(
                    f"SUBMITTED (reconstruction): clip_id={clip_id} method={method}"
                )
                ws_broadcast_threadsafe(
                    {
                        "type": "broadcast",
                        "data": f"/path/to/datasets/ncore/lidar-model-static-full-nrm/{clip_id}/{clip_id}.json",
                    }
                )
            elif method == "NuRec":
                self.nurec_job.kill_previous()
                self.nurec_job.launch(clip_id)
            else:
                self.log_message(
                    "Reconstruction method not implemented: %s", method
                )

    def do_POST(self):
        if self.path == "/upload_video":
            user = self._current_user()
            if user is None:
                self.send_response(403)
                self.end_headers()
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            video_bytes = self.rfile.read(content_length)
            self.embeddingsstore.uploaded_features = (
                self.embeddingsstore.compute_features(video_bytes)
            )
            return self._send_json({"upload_id": "__uploaded_video__"})

        if self.path == "/upload_image":
            user = self._current_user()
            if user is None:
                self.send_response(403)
                self.end_headers()
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            image_bytes = self.rfile.read(content_length)
            upload_id = str(uuid.uuid4())
            self.clipembeddingsstore.uploaded_image_features[upload_id] = (
                self.clipembeddingsstore.compute_image_features(image_bytes)
            )
            return self._send_json({"upload_id": upload_id})

        parsed_path = parse.urlparse(self.path)
        if parsed_path.path == "/upload_clustering":
            return self._handle_upload_clustering(parsed_path)
        if parsed_path.path == "/upload_classifier":
            return self._handle_upload_classifier(parsed_path)
        if parsed_path.path == "/upload_clip_list":
            return self._handle_upload_clip_list(parsed_path)

        payload = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        parts = payload.decode().split("::") if payload else [""]
        action = parts[0]

        # Auth actions
        if action == "user_login":
            try:
                username, password = parts[1], parts[2]
            except Exception:
                self.send_response(400)
                self.end_headers()
                return
            user = self.usersstore.verify_credentials(username, password)
            if user is None:
                self.send_response(403)
                self.end_headers()
                return
            sid = self.usersstore.create_session(user.id)
            self.send_response(200)
            self.send_header(
                "Set-Cookie",
                f"{SESSION_COOKIE}={sid}; Path=/; HttpOnly",
            )
            self.end_headers()
            return

        if action == "logout":
            sid = self._get_session_id()
            if sid:
                self.usersstore.delete_session(sid)
            self.send_response(200)
            self.send_header(
                "Set-Cookie",
                f"{SESSION_COOKIE}=deleted; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT",
            )
            self.end_headers()
            return

        if action == "request_access":
            try:
                username, password, email, reason = (
                    parts[1],
                    parts[2],
                    parts[3],
                    parts[4],
                )
            except Exception:
                self.send_response(400)
                self.end_headers()
                return
            try:
                req_id = self.usersstore.create_access_request(
                    username, email, reason, password
                )
            except ValueError:
                self.send_response(400)
                self.end_headers()
                return
            self._notify_access_request(username, email, reason, req_id)
            self.send_response(200)
            self.end_headers()
            return

        # Admin-only actions
        if action.startswith("admin_"):
            user = self._current_user()
            if user is None or user.role != "admin":
                self.send_response(403)
                self.end_headers()
                return

            if action == "admin_create_user":
                try:
                    username, password, email, role = parts[1:5]
                except Exception:
                    self.send_response(400)
                    self.end_headers()
                    return
                # Disallow creating a user without a non-empty password
                if not isinstance(password, str) or not password.strip():
                    self.send_response(400)
                    self.end_headers()
                    return
                try:
                    self.usersstore.create_user(
                        username=username,
                        password=password,
                        email=email if email else None,
                        role=role or "user",
                    )
                except ValueError:
                    # Validation failure (e.g., missing password)
                    self.send_response(400)
                    self.end_headers()
                    return
                except Exception:
                    self.send_response(409)
                    self.end_headers()
                    return
                self.send_response(200)
                self.end_headers()
                return

            if action == "admin_set_user_role":
                try:
                    username, role = parts[1], parts[2]
                except Exception:
                    self.send_response(400)
                    self.end_headers()
                    return
                ok = self.usersstore.set_user_role(username, role)
                self.send_response(200 if ok else 404)
                self.end_headers()
                return

            # admin_set_user_active removed (no is_active)

            if action == "admin_set_request_status":
                try:
                    rid, status = int(parts[1]), parts[2]
                except Exception:
                    self.send_response(400)
                    self.end_headers()
                    return
                if status == "approved":
                    ok, _gen_pass = self.usersstore.approve_request(rid)
                elif status == "rejected":
                    ok = self.usersstore.reject_request(rid)
                else:
                    ok = False
                self.send_response(200 if ok else 404)
                self.end_headers()
                return

            if action == "admin_delete_user":
                try:
                    username = parts[1]
                except Exception:
                    self.send_response(400)
                    self.end_headers()
                    return
                ok = self.usersstore.delete_user(username)
                self.send_response(200 if ok else 404)
                self.end_headers()
                return

            if action == "admin_update_user":
                try:
                    username, email, role, password = parts[1:5]
                    ds_csv = parts[5] if len(parts) > 5 else ""
                except Exception:
                    self.send_response(400)
                    self.end_headers()
                    return
                email = email or None
                role = role or None
                password = password or None
                datasources = (
                    [ds.strip() for ds in ds_csv.split(",") if ds.strip()]
                    if ds_csv is not None and ds_csv != ""
                    else None
                )
                ok = self.usersstore.update_user(
                    username,
                    email=email,
                    role=role,
                    password=password,
                    datasources=datasources,
                )
                self.send_response(200 if ok else 404)
                self.end_headers()
                return

        if action == "run_clustering":
            path, n_clusters_str, spherical_str, max_points_per_centroid_str, embed_type_str = parts[1:]

            # Reconstruct search filters from the encoded /videos?... path and
            # fetch matching clip_ids — same pattern as auto_label
            user = self._current_user()
            ds_opts = self.usersstore.get_allowed_datasources(user.id) if user else []
            parsed_path = parse.urlparse(path)
            parsed_qs = parse.parse_qs(parsed_path.query)
            filters = SearchFilters.from_query(parsed_qs, user_options={"data_source": ds_opts})
            clip_ids, _ = self.search(filters)

            # Write clip_ids to a temp file so the subprocess can load them
            tmp = tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json"
            )
            json.dump(clip_ids, tmp)
            tmp.close()

            # Create a unique output directory and persist run metadata
            run_id = unique_id()
            out_dir = Path(self.clustering_dir) / run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "metadata.json", "w") as f:
                json.dump({
                    "run_id": run_id,
                    "n_clusters": int(n_clusters_str),
                    "spherical_kmeans": spherical_str == "true",
                    "max_points_per_centroid": int(max_points_per_centroid_str),
                    "n_input_clips": len(clip_ids),
                    "started_at": time.time(),
                    "search_params": parsed_path.query,
                    "embed_type": embed_type_str,
                }, f)

            # Launch clustering as a detached subprocess; done-detection is
            # file-based (both representative_by_cluster.json and umap.json
            # must exist for the run to be considered complete)
            script = (
                Path(__file__).resolve().parent
                / "cluster_clips_and_select.py"
            )
            embed_path, index_tag = self._embed_store_path_and_tag(embed_type_str)

            cmd = [
                "python", str(script),
                str(out_dir),
                str(embed_path),
                n_clusters_str,
                "--path_to_clip_ids", tmp.name,
                "--embed_type", embed_type_str,
                "--index_tag", index_tag,
            ]
            if spherical_str == "true":
                cmd.append("--spherical_kmeans")
            cmd += ["--max_points_per_centroid", max_points_per_centroid_str]

            # Topic extraction now runs as the final step of the clustering
            # subprocess (so the server never blocks on it at view time).
            # Skipped if the captionstore has no on-disk SQLite path.
            captions_db = getattr(self.captionstore, "db_path", None)
            if captions_db:
                cmd += ["--captions_db", str(captions_db)]
            print(cmd)

            # Stream stdout+stderr to a log file in the run dir so failed
            # subprocesses leave a usable traceback (otherwise the click
            # silently does nothing). Opened in line-buffered mode.
            log_fp = open(out_dir / "subprocess.log", "w", buffering=1)
            proc = subprocess.Popen(
                cmd,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )
            self.clustering_jobs.start(run_id, proc, cmd)

            self._send_json({"run_id": run_id})
            return

        if action == "delete_clustering":
            run_id = parts[1]
            run_dir = Path(self.clustering_dir) / run_id
            # Guard against path traversal
            if run_dir.parent.resolve() != Path(self.clustering_dir).resolve():
                self._send_json({"ok": False}, status=400)
                return
            if run_dir.exists():
                shutil.rmtree(run_dir)
            self._send_json({"ok": True})
            return

        if action == "train_classifier":
            label, n_neg, neg_labels, use_autolabels, n_pos, embed_type_str = parts[1:]
            self.timers.tic()

            user = self._current_user()
            if user is None:
                self._send_json({"error": "unauthorized"}, status=403)
                return
            trained_by = user.username

            positive_labels = label.split("&&")
            negative_label_list = neg_labels.split(",") if neg_labels else []
            keys = positive_labels + negative_label_list
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", delete=False, suffix=".json"
            ) as tmp_fp:
                self.datastore.export_to_json(tmp_fp, keys=keys)
                path_to_annotations = tmp_fp.name
            self.log_message(
                "Exporting the annotations as json took %f", self.timers.toc()
            )

            # Pre-stamp metadata.json so the run shows up in
            # /classifiers_status as "pending" before the subprocess
            # finishes; classifier_build.write_metadata uses setdefault
            # so the subprocess won't overwrite these fields.
            run_id = unique_id()
            run_dir = Path(self.classifier_dir) / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / "metadata.json", "w") as f:
                json.dump({
                    "run_id": run_id,
                    "embed_type": embed_type_str,
                    "positive_labels": positive_labels,
                    "negative_labels": negative_label_list,
                    "trained_by": trained_by,
                    "use_autolabels": use_autolabels == "true",
                    "started_at": time.time(),
                    "status": "pending",
                    "n_positive_clips": 0,
                    "n_negative_clips": 0,
                }, f)

            train_script = str(
                Path(__file__).resolve().parent / "train_classifier.py"
            )
            embed_path, index_tag = self._embed_store_path_and_tag(embed_type_str)

            cmd = [
                "python",
                train_script,
                self.classifier_dir,
                str(path_to_annotations),
                label,
                str(embed_path),
                "--trained_by", trained_by,
                "--run_id", run_id,
                "--n_negative_samples", str(n_neg),
                "--embed_type", embed_type_str,
                "--index_tag", index_tag,
            ]
            if use_autolabels == "true":
                cmd.append("--use_autolabels")
            if neg_labels != "":
                cmd += ["--negative_labels", str(neg_labels)]
            if n_pos:
                cmd += ["--n_positive_samples", str(n_pos)]

            print(cmd)
            log_fp = open(run_dir / "subprocess.log", "w", buffering=1)
            proc = subprocess.Popen(
                cmd,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )
            time.sleep(0.25)
            self.classifier_jobs.start(run_id, proc, cmd)
            self.invalidate_cache_classifier(run_id)
            self.log_message(
                "Classifier training started: run_id=%s label=%s embed_type=%s n_pos=%s n_neg=%s trained_by=%s",
                run_id, label, embed_type_str, n_pos, n_neg, trained_by,
            )
            self._send_json({"run_id": run_id})
            return

        if action == "delete_classifier_run":
            run_id = parts[1]
            run_dir = Path(self.classifier_dir) / run_id
            if run_dir.parent.resolve() != Path(self.classifier_dir).resolve():
                self._send_json({"ok": False}, status=400)
                return
            if run_dir.exists():
                shutil.rmtree(run_dir)
            if self.classifiersearch is not None:
                self.classifiersearch.invalidate(run_id)
            self._send_json({"ok": True})
            return

        if action == "report_bug":
            title, description, user_agent = parts[1], parts[2], parts[3]
            page_url = parts[4] if len(parts) > 4 else ""
            user = self._current_user()
            username = getattr(user, "username", "anonymous") if user else "anonymous"
            timestamp = datetime.datetime.now().isoformat(timespec="seconds")
            row = [timestamp, username, title, description, user_agent, page_url]
            if self.bug_report_spreadsheet_id and self.bug_report_credential_path:
                try:
                    append_to_spreadsheet(
                        self.bug_report_spreadsheet_id,
                        "Bug Reports",
                        [row],
                        self.bug_report_credential_path,
                    )
                    self._send_json({"ok": True})
                except Exception as exc:
                    self.log_message("Bug report submission failed: %s", str(exc))
                    self._send_json({"ok": False, "error": str(exc)})
            else:
                self.log_message("Bug report (no Sheets configured): %s | %s", title, description)
                self._send_json({"ok": True})
            return

        if action.startswith("arena_") and self.arena_store is not None:
            self.handle_arena_post(action, parts)
            return

        # auto_label scopes its work to the caller's allowed datasources,
        # so the writer must be authenticated.
        if action == "auto_label" and self._require_user() is None:
            return

        # Core actions
        self.handle_core_post(action, parts)
        self.send_response(200)
        self.end_headers()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Server for annotating with login"
    )
    parser.add_argument(
        "config_file",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        metavar="KEY=VALUE",
        help="Override config values using dot-notation, e.g. server.bindto=0.0.0.0:8000",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config_file)
    apply_overrides(config, args.override)
    datastores_cfg = config["datastores"]
    server_cfg = config["server"]
    log_dir, stats_dir = configure_logging_and_stats(server_cfg)
    # S3 connection settings come from config (not hardcoded) so they can't be
    # baked into / sanitized out of the code; the fetchers below read them.
    s3_endpoint = server_cfg.get("s3_endpoint")
    s3_profile = server_cfg.get("s3_profile", "sil-wheel")

    print(f"Loading clips_to_apis from {config['clips_to_sil_apis']}")
    with open(config["clips_to_sil_apis"], "r") as f:
        clips_to_apis = json.load(f)
    autolabelsstore = AutolabelsDataStore(clips_to_apis)

    bev_cfg = datastores_cfg.get("bev_store")
    if bev_cfg:
        print(f"Initializing BEV fetcher with bucket: {bev_cfg['s3_bucket']}")
        bev_fetcher = BEVFetcher(
            bev_cfg["s3_bucket"],
            index_dir=bev_cfg["metrics_index_dir"],
            endpoint=s3_endpoint,
            profile=s3_profile,
        )
    else:
        bev_fetcher = None
        print("bev_store not configured, skipping")

    log_rss("startup")

    print(f"Loading datastore from {datastores_cfg['annotations_db']}")
    datastore = DataStore(
        datastores_cfg["annotations_db"], clips_to_apis=clips_to_apis
    )
    log_rss("after sqlite")

    cosmos_cfg = datastores_cfg.get("cosmos_embed_store") or {}
    cosmos_dir = cosmos_cfg.get("embeddings_dir")
    if embed_dir_is_populated(cosmos_dir):
        print(f"Loading embeddingsstore from {cosmos_dir}")
        embeddingsstore = CosmosEmbeddingsStore(
            cosmos_dir,
            index_spec=cosmos_cfg["index_spec"],
            mmap=cosmos_cfg.get("mmap", False),
        )
    else:
        print(f"[cosmos] No embeddings under {cosmos_dir}; using empty stub (no model loaded)")
        embeddingsstore = EmptyEmbedStore(cosmos_dir)
    log_rss("after cosmos")

    clip_embed_cfg = datastores_cfg.get("visual_embed_store") or {}
    clip_embed_dir = clip_embed_cfg.get("embeddings_dir")
    if embed_dir_is_populated(clip_embed_dir):
        print(f"Loading visual embeddingsstore from {clip_embed_dir}")
        clipembeddingsstore = Florence2SigCLIPEmbeddingStore(
            clip_embed_dir,
            index_spec=clip_embed_cfg.get("index_spec", "IVF4096,PQ64x8"),
            mmap=clip_embed_cfg.get("mmap", False),
            siglip_model=clip_embed_cfg.get("siglip_model", "google/siglip2-base-patch16-224"),
        )
    else:
        print(f"[visual] No embeddings under {clip_embed_dir}; using empty stub (no SigLIP loaded)")
        clipembeddingsstore = EmptyEmbedStore(clip_embed_dir)
    log_rss("after visual")

    caption_embed_cfg = datastores_cfg.get("caption_embed_store") or {}
    caption_embed_dir = caption_embed_cfg.get("embeddings_dir")
    if embed_dir_is_populated(caption_embed_dir):
        print(f"Loading captionembeddingsstore from {caption_embed_dir}")
        captionembeddingsstore = CaptionEmbeddingsStore(
            caption_embed_dir,
            index_spec=caption_embed_cfg.get("index_spec", "IVF4096,PQ128x8"),
            mmap=caption_embed_cfg.get("mmap", False),
            embedding_model=caption_embed_cfg.get(
                "embedding_model", "Qwen/Qwen3-Embedding-8B"
            ),
        )
    else:
        print(f"[caption] No embeddings under {caption_embed_dir}; using empty stub (query model not loaded)")
        captionembeddingsstore = EmptyEmbedStore(caption_embed_dir)
    log_rss("after caption")

    traj_cfg = datastores_cfg.get("trajectory_store")
    if traj_cfg:
        print(f"Loading trajectorystore from {traj_cfg['trajectory_dir']}")
        trajectorystore = TrajectoryStore(
            traj_cfg["trajectory_dir"],
            server_cfg["debug"],
            index_spec=traj_cfg.get("index_spec"),
        )
    else:
        trajectorystore = None
        print("trajectory_store not configured, skipping")
    log_rss("after trajectory")

    wm_cfg = datastores_cfg.get("wm_store")
    if wm_cfg:
        print(f"Loading wm_store from {wm_cfg['data_file']}")
        wm_store = WMStore(wm_cfg["data_file"])
        log_rss("after wm")
    else:
        wm_store = None
        print("wm_store not configured, skipping")

    captions_db = datastores_cfg.get("captions_db")
    if captions_db:
        print(f"Loading captionstore from {captions_db}")
        captionstore = CaptionStore(captions_db)
        log_rss("after caption_store")
    else:
        captionstore = None
        print("captions_db not configured, skipping")

    pred_cfg = datastores_cfg.get("predictions_store")
    if pred_cfg:
        metrics_index_dir = bev_cfg["metrics_index_dir"] if bev_cfg else None
        print(f"Loading metricstore from {pred_cfg['predictions_dir']}")
        metricstore = ModelsWithMetricsDataStore(
            pred_cfg["predictions_dir"], index_dir=metrics_index_dir
        )
        print(f"Loading predictionsstore from {pred_cfg['predictions_dir']}")
        predictionsstore = PredictionsDataStore(pred_cfg["predictions_dir"])
        log_rss("after predictions")
    else:
        metricstore = None
        predictionsstore = None
        print("predictions_store not configured, skipping")

    # Warm up CUDA kernels, model weights, and FAISS caches so the first
    # user query doesn't eat the cold-start tax.
    print("Warming up FAISS-backed stores")
    embeddingsstore.warmup()
    captionembeddingsstore.warmup()
    clipembeddingsstore.warmup()
    log_rss("after warmup")

    classifier_cfg = datastores_cfg.get("classifier_search")
    classifier_dir = classifier_cfg["classifier_dir"] if classifier_cfg else None
    if classifier_dir:
        classifiersearch = ClassifierSearch(classifier_dir)
    else:
        classifiersearch = None
        print("classifier_search not configured, skipping")
    usersstore = UsersDataStore(datastores_cfg["users_db"])
    slack_notifier = None
    slack_profile = "sil-wheel"
    slack_timeout_s = 5.0

    try:
        slack_cfg = load_slack_config()
        if slack_cfg is None:
            print(f"Slack notifications disabled (profile={slack_profile})")
        else:
            slack_notifier = SlackNotifier(slack_cfg, timeout_s=slack_timeout_s)
            destinations = []
            if slack_cfg.channel_id:
                destinations.append("channel")
            if slack_cfg.dm_user_ids:
                destinations.append(f"dm({len(slack_cfg.dm_user_ids)})")
            print(
                "Slack notifications enabled "
                + f"(profile={slack_cfg.profile}, destinations={','.join(destinations)})"
            )
    except Exception as exc:
        print(
            "Slack notifications disabled due to configuration error: "
            + str(exc)
        )

    bug_report_cfg = server_cfg.get("bug_report", {})
    bug_report_spreadsheet_id = bug_report_cfg.get("spreadsheet_id")
    bug_report_credential_path = bug_report_cfg.get("credential_path")
    if bug_report_spreadsheet_id and bug_report_credential_path:
        print(f"Bug report submissions enabled (spreadsheet_id={bug_report_spreadsheet_id})")
    else:
        print("Bug report submissions disabled (no spreadsheet_id or credential_path in config)")

    llm_provider = server_cfg.get("llm_provider", "auto")
    rewriter = QueryRewriter(provider=llm_provider)

    # Static pages (extend with login)
    root = Path(__file__).parent.parent / "sil_wheel" / "app" / "static"
    static_pages = StaticPages(
        {
            "/": ("text/html", root / "html/annotation_page.html"),
            "/leaderboard": ("text/html", root / "html/leaderboard.html"),
            "/policy_predictions": (
                "text/html",
                root / "html/policy_predictions.html",
            ),
            "/admin": ("text/html", root / "html/admin.html"),
            "/admin_stats": ("text/html", root / "html/admin_stats.html"),
            "/data_stats": ("text/html", root / "html/data_stats.html"),
            "/login": ("text/html", root / "html/login.html"),
            "/style.css": ("text/css", root / "css/style.css"),
            "/leaderboard.css": ("text/css", root / "css/leaderboard.css"),
            "/admin.css": ("text/css", root / "css/admin.css"),
            "/admin_stats.css": ("text/css", root / "css/admin_stats.css"),
            "/data_stats.css": ("text/css", root / "css/data_stats.css"),
            "/predictions.css": ("text/css", root / "css/predictions.css"),
            "/login.css": ("text/css", root / "css/login.css"),
            "/main.js": ("text/javascript", root / "js/main.js"),
            "/clustering.js": ("text/javascript", root / "js/clustering.js"),
            "/annotation.js": ("text/javascript", root / "js/annotation.js"),
            "/predictions.js": ("text/javascript", root / "js/predictions.js"),
            "/leaderboard.js": ("text/javascript", root / "js/leaderboard.js"),
            "/world_model.js": ("text/javascript", root / "js/world_model.js"),
            "/admin.js": ("text/javascript", root / "js/admin.js"),
            "/admin_stats.js": ("text/javascript", root / "js/admin_stats.js"),
            "/data_stats.js": ("text/javascript", root / "js/data_stats.js"),
            "/login.js": ("text/javascript", root / "js/login.js"),
            "/msgpack.min.js": ("text/javascript", root / "js/msgpack.min.js"),
            "/bev-utils.js": ("text/javascript", root / "js/bev/bev-utils.js"),
            "/bev-binary-utils.js": (
                "text/javascript",
                root / "js/bev/bev-binary-utils.js",
            ),
            "/bev-renderer.js": (
                "text/javascript",
                root / "js/bev/bev-renderer.js",
            ),
            "/bev-viewer.js": (
                "text/javascript",
                root / "js/bev/bev-viewer.js",
            ),
            "/metrics-viewer.js": (
                "text/javascript",
                root / "js/metrics-viewer.js",
            ),
            "/arena": ("text/html", root / "html/arena.html"),
            "/arena.css": ("text/css", root / "css/arena.css"),
            "/arena.js": ("text/javascript", root / "js/arena.js"),
            "/docs_chatbot.css": (
                "text/css",
                root / "css/docs_chatbot.css",
            ),
            "/docs_chatbot.js": (
                "text/javascript",
                root / "js/docs_chatbot.js",
            ),
        }
    )

    address = server_cfg["bindto"].split(":")
    address[1] = int(address[1])
    address = tuple(address)

    print(f"Starting WebSocket server")
    # Start WebSocket server in background thread
    websocket_thread = threading.Thread(target=run_ws_server, daemon=True)
    websocket_thread.start()

    nurec_job = NurecJobRegistry()
    video_fetcher = VideoFetcher("processed_data", endpoint=s3_endpoint, profile=s3_profile)
    vlm_judge = None
    vlm_provider = server_cfg.get("vlm_provider", "auto")
    try:
        vlm_judge = VLMJudge(
            datastore=datastore,
            video_fetcher=video_fetcher,
            provider=vlm_provider,
        )
        logging.info("VLM Judge enabled (provider=%s)", vlm_provider)
    except Exception as exc:
        print(f"VLM Judge disabled: {exc}")

    arena_store = None
    if "arena_db" in datastores_cfg:
        arena_db_path = datastores_cfg["arena_db"]
        arena_s3_sess = boto3.Session(profile_name=s3_profile, region_name="us-east-1")
        arena_s3_client = arena_s3_sess.client(
            "s3",
            config=Config(max_pool_connections=10, read_timeout=30, connect_timeout=5),
            endpoint_url=s3_endpoint,
        )
        arena_store = ArenaStore(arena_db_path, arena_s3_client, "processed_data")
        print(f"Arena store initialized: {arena_db_path}")

    cluster_cfg = datastores_cfg.get("cluster_search")
    clustering_dir = cluster_cfg["clustering_dir"] if cluster_cfg else None
    if clustering_dir:
        clustersearch = ClusterSearch(clustering_dir)
    else:
        clustersearch = None
        print("cluster_search not configured, skipping")
    clip_list_cfg = datastores_cfg.get("clip_list_search")
    if clip_list_cfg:
        cliplistsearch = ClipListSearch(clip_list_cfg["clip_lists_dir"])
    else:
        cliplistsearch = None
        print("clip_list_search not configured, skipping")
    search_pipeline = SearchPipeline(
        datastore=datastore,
        captionstore=NullStore(captionstore),
        captionembeddingsstore=captionembeddingsstore,
        embeddingsstore=embeddingsstore,
        clipembeddingsstore=clipembeddingsstore,
        classifiersearch=NullStore(classifiersearch),
        clustersearch=NullStore(clustersearch),
        cliplistsearch=NullStore(cliplistsearch),
        trajectorystore=NullStore(trajectorystore),
        metricstore=NullStore(metricstore),
        bev_fetcher=NullStore(bev_fetcher),
        wm_store=NullStore(wm_store),
    )

    Handler = partial(
        RequestHandler,
        datastore=datastore,
        captionstore=captionstore,
        trajectorystore=trajectorystore,
        embeddingsstore=embeddingsstore,
        clipembeddingsstore=clipembeddingsstore,
        captionembeddingsstore=captionembeddingsstore,
        classifiersearch=classifiersearch,
        wm_store=wm_store,
        classifier_dir=classifier_dir,
        metricstore=metricstore,
        predictionsstore=predictionsstore,
        autolabelsstore=autolabelsstore,
        search_pipeline=search_pipeline,
        favicon=load_favicon("images/car.png"),
        static_pages=static_pages,
        video_fetcher=video_fetcher,
        usersstore=usersstore,
        classifier_jobs=JobRegistry(),
        nurec_job=nurec_job,
        clips_to_apis=clips_to_apis,
        bev_fetcher=bev_fetcher,
        clustering_jobs=JobRegistry(),
        clustering_dir=clustering_dir,
        clustersearch=clustersearch,
        cliplistsearch=cliplistsearch,
        slack_notifier=slack_notifier,
        bug_report_spreadsheet_id=bug_report_spreadsheet_id,
        bug_report_credential_path=bug_report_credential_path,
        rewriter=rewriter,
        arena_store=arena_store,
        vlm_judge=vlm_judge,
        vlm_judge_workers=server_cfg.get("vlm_judge_workers", 20),
        agent_url=server_cfg.get("agent_url", ""),
        log_dir=log_dir,
        stats_dir=stats_dir,
    )

    with ThreadingTCPServer(address, Handler) as httpd:
        print(f"Listening at {server_cfg['bindto']}")
        httpd.serve_forever()

    nurec_job.join()


if __name__ == "__main__":
    main()
