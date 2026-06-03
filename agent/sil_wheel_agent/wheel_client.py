#!/usr/bin/env python3
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

"""
SIL Wheel API Client — programmatic access to all SIL Wheel features.

Supports all 14 composable search modes + VLM Judge:
  - Caption/text search (FTS5 on Qwen captions, with LLM query rewriting)
  - Cosmos embedding similarity (clip-to-clip, text-to-clip)
  - CLIP visual similarity (text-to-clip)
  - Trajectory shape similarity (clip-to-clip, 20s/10s/5s windows)
  - Trajectory predicate search (speed, curvature, acceleration expressions)
  - Classifier score filtering (93+ trained labels on production)
  - Cluster search (K-means over Cosmos embeddings, with TF-IDF topic
    keywords + optional LLM theme descriptions per cluster)
  - World model object search (class, count, distance, angle)
  - Label/annotation filtering (manual + autolabels, AND/OR logic)
  - Numeric metric filtering
  - Country/driving-side filtering
  - SIL API applicability filtering
  - Comment search
  - VLM Judge (caption scoring, search validation via vision-language model)

Usage as library:
    from src.wheel_client import WheelClient

    client = WheelClient()  # loads credentials from .env
    client.login()

    # Search methods return (total, results) — except semantic_search_by_clip (list only)
    total, results = client.semantic_search_by_text("rainy highway with trucks at night")

    # Caption search (fast)
    total, results = client.caption_search("construction zone")

    # Compose multiple filters
    total, results = client.search(
        data_source="MADS",
        search="tunnel",
        classifier_select="interesting",
        probability_threshold=0.5,
    )

    # Export clip IDs
    clip_ids = client.export_search_clip_ids(data_source="MADS", search="tunnel")

Usage as CLI:
    python sil_wheel_agent/wheel_client.py info
    python sil_wheel_agent/wheel_client.py check
    python sil_wheel_agent/wheel_client.py search --caption "construction zone" -n 10
    python sil_wheel_agent/wheel_client.py search --semantic-text "rainy highway" --data-source MADS
    python sil_wheel_agent/wheel_client.py export --caption "tunnel" -o tunnel_clips.txt
    python sil_wheel_agent/wheel_client.py metrics
    python sil_wheel_agent/wheel_client.py label --clips ids.txt --label "Snow"
    python sil_wheel_agent/wheel_client.py scenario "construction zone in rain" --data-source MADS
    python sil_wheel_agent/wheel_client.py inventory --data-source MADS
    python sil_wheel_agent/wheel_client.py similar dd87da72-... -n 20
    python sil_wheel_agent/wheel_client.py expand --clips seeds.txt -o expanded.txt --max-total 500
    python sil_wheel_agent/wheel_client.py merge-clips urban.txt weather.txt -o combined.txt --mode union
    python sil_wheel_agent/wheel_client.py lookup --clips ids.txt -n 10
    python sil_wheel_agent/wheel_client.py vlm-judge --status
    python sil_wheel_agent/wheel_client.py vlm-judge --score-clip CLIP_ID
    python sil_wheel_agent/wheel_client.py vlm-judge --score-caption CLIP_ID --caption "A car driving"
    python sil_wheel_agent/wheel_client.py vlm-judge --validate "rain" --clips ids.txt
    python sil_wheel_agent/wheel_client.py clusters                          # list runs
    python sil_wheel_agent/wheel_client.py clusters --run-id RUN --top-k 10  # top clusters w/ topics
    python sil_wheel_agent/wheel_client.py clusters --run-id RUN --keyword pedestrian
    python sil_wheel_agent/wheel_client.py version
    python sil_wheel_agent/wheel_client.py version --update
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import os
import random as _random_mod
import re as _re_mod
import string as _string_mod
import sys
import threading
import time as _time_mod
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import difflib
import zlib

import requests
from urllib.parse import quote, urlparse

_zlib_error = zlib.error

try:
    from dotenv import load_dotenv
    _dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_dotenv_path)
except ImportError:
    pass

PROD_SERVER = "http://localhost:8000"
DEV_SERVER = "http://localhost:8018"

SDK_VERSION = "1.8.2"
_SKILL_URL = os.environ.get("WHEEL_SKILL_URL", "https://raw.githubusercontent.com/nv-tlabs/sil-wheel/main/agent/SKILL.md")

# Exact clip-ID shapes the Wheel server indexes:
#   Plain MADS:  "<uuid>"                          (36 chars; e.g. "955a526c-a388-11ec-a932-00044bf65dfd")
#   MADS-1M:     "<uuid>_<start_us>_<end_us>"     (UUID + two microsecond integers)
# Used by ``search()`` to detect when an agent passes an exact key as
# ``search_clipid`` and emit a guidance ``UserWarning`` recommending
# ``lookup_clip()`` (clearer intent + cleaner ``SearchResult | None`` return
# type). Both routes hit the same server endpoint — see
# ``knowledge/anti-patterns.md`` for the rationale.
_UUID_PATTERN = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_EXACT_CLIP_ID_RE = _re_mod.compile(rf"^{_UUID_PATTERN}(?:_\d+_\d+)?$")

# Thread-local guard so ``lookup_clip()`` (which itself calls
# ``search(search_clipid=...)`` internally) doesn't trigger the
# "prefer lookup_clip" guidance warning on itself — that would be both
# noisy and contradictory, especially in batch contexts where
# ``lookup_clips_batch`` would emit one warning per missing clip.
_lookup_clip_in_progress = threading.local()


# Per-classifier list of data sources known to NOT have inferences. Built
# empirically from probes during the May-2026 sessions. Format:
# ``{"<Classifier label>": frozenset({"<unindexed data_source>", ...}), ...}``.
# Used by ``WheelClient.search()`` for the preflight warning when a caller
# composes ``classifier_select=<label>`` with ``data_source=<unindexed>``.
#
# Why this exists: ``classifier_search("Change lane to the right",
# data_source="MADS-1M")`` returns 0 silently — but the same call against
# AV V2 train returns 1.4M. The classifier IS trained, but inferences only
# ever ran on AV. We can't probe per-call (that would defeat the warning's
# purpose), so we maintain a small known-bad list. To extend, run
# ``client.get_classifier_coverage(label)`` and add zeros to the set.
_CLASSIFIER_KNOWN_MISSING: dict[str, frozenset[str]] = {
    "Change lane to the right": frozenset({"MADS-1M", "MADS"}),
    # NOTE: "Change lane to the left" works on MADS-1M (266K matches),
    # but "Change lane to the right" does not. Inference coverage is
    # asymmetric and unpredictable per label.
}


def _strict_mode_enabled() -> bool:
    """Return True when the agent has opted into strict mode.

    In strict mode, certain SDK warnings (silent-bug-class) become errors —
    valuable for autonomous agents that don't read warning logs and would
    otherwise conclude "no clips" when they were actually doing something
    slightly wrong. Set ``WHEEL_STRICT=1`` (or any truthy value) to enable.
    """
    return os.environ.get("WHEEL_STRICT", "").strip().lower() in (
        "1", "true", "yes", "on", "strict",
    )


def _default_server() -> str:
    return os.environ.get("WHEEL_SERVER_URL", PROD_SERVER)


def _first_of(d: dict, *keys):
    """Return value of the first key present in d with a non-None value."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _safe_int(value, default: int = 0) -> int:
    """Coerce ``value`` to int, returning ``default`` on any conversion failure.

    Use whenever JSON-derived data may be missing, ``None``, a non-numeric
    string, or otherwise malformed. ``int(x or 0)`` is NOT a substitute —
    truthy non-numeric strings (e.g. ``"abc"``) crash with ``ValueError``.

    Catches ``TypeError`` (None/list/dict), ``ValueError`` (non-numeric
    strings), and ``OverflowError`` (``float('inf')``, ``float('-inf')``).
    NaN coerces to 0 via the explicit ``isnan`` check.
    """
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _timeout_response(url: str, error: Exception) -> requests.Response:
    """Create a synthetic response for network/timeout errors."""
    resp = requests.Response()
    resp.status_code = 504
    error_type = "timeout" if isinstance(error, requests.exceptions.Timeout) else "network_error"
    resp._content = json.dumps({"error": error_type, "url": url, "detail": str(error)}).encode("utf-8")
    resp.encoding = "utf-8"
    resp.headers["Content-Type"] = "application/json"
    return resp


@dataclass
class SearchResult:
    """A clip returned from a SIL Wheel search.

    Trajectory data — sampling rates:

    Two trajectory representations come back with most search results, at
    *different* sampling rates. This is easy to miss and breaks any code
    that integrates them assuming a common ``dt``.

    - ``speed``, ``acceleration``, ``curvature``, ``jerk``:
      dense ego-state traces (typically ~351 samples over a 10s MADS-1M
      clip, i.e. **~35 Hz**). These are the unsigned-magnitude curvature
      values reflecting the actual recorded driving behaviour.

    - ``positions``:
      sparse ego XY positions (typically ~36 samples over the same 10s
      clip, i.e. **~3.6 Hz**). Designed for trajectory **shape** analysis
      (start↔end distance, path length, cumulative heading from segment
      vectors) — not for high-rate ego-state estimation.

    When you need cumulative *signed* heading change, compute it
    geometrically from ``positions`` (the curvature trace is unsigned).
    When you need timed speed/curvature, use those traces directly and
    compute ``dt = clip_duration_s / (len(trace) - 1)`` per clip.
    """
    clip_id: str
    data_source: str = ""
    annotations: list[dict] = field(default_factory=list)
    speed: list[float] = field(default_factory=list)
    acceleration: list[float] = field(default_factory=list)
    curvature: list[float] = field(default_factory=list)
    jerk: list[float] = field(default_factory=list)
    has_trajectories: bool = False
    has_embeddings: bool = False
    captions: dict = field(default_factory=dict)
    semantic_clip_score: float | None = None
    semantic_text_score: float | None = None
    visual_score: float | None = None
    visual_image_score: float | None = None
    trajectory_score: float | None = None
    classifier_score: float | None = None
    cluster_distance_score: float | None = None
    caption_embed_score: float | None = None
    rrf_score: float | None = None
    numeric_scores: dict | None = None
    country: str = ""
    country_name: str = ""
    positions: list | None = None
    sil_apis: list[str] = field(default_factory=list)
    comments: str = ""
    vlm_caption_scores: dict | None = None
    # Server enriches /videos results with per-clip cluster context when
    # `cluster_run_id` is in the search filter (server commit d7033ede,
    # "Refactor cluster selection"). Shape: {"cluster_id": int, "distance": float}.
    # None if no cluster filter is active or the clip isn't in the run.
    cluster_membership: dict | None = None

    @property
    def caption_text(self) -> str:
        """Extract the first caption string from the captions dict."""
        if not self.captions:
            return ""
        for val in self.captions.values():
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict) and item.get("caption"):
                        return item["caption"]
                    if isinstance(item, str) and item:
                        return item
            if isinstance(val, str) and val:
                return val
        return ""

    @property
    def all_captions(self) -> list[str]:
        """Extract all caption strings from all models."""
        caps = []
        if not self.captions:
            return caps
        for val in self.captions.values():
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict) and item.get("caption"):
                        caps.append(item["caption"])
                    elif isinstance(item, str) and item:
                        caps.append(item)
            elif isinstance(val, str) and val:
                caps.append(val)
        return caps

    @property
    def scores_dict(self) -> dict[str, float | None]:
        """Return all scores as a flat dict for analysis/DataFrame construction."""
        d: dict[str, float | None] = {
            "semantic_clip": self.semantic_clip_score,
            "semantic_text": self.semantic_text_score,
            "visual": self.visual_score,
            "visual_image": self.visual_image_score,
            "trajectory": self.trajectory_score,
            "classifier": self.classifier_score,
            "cluster_distance": self.cluster_distance_score,
            "caption_embed": self.caption_embed_score,
            "rrf": self.rrf_score,
        }
        if self.numeric_scores:
            for k, v in self.numeric_scores.items():
                d[f"metric:{k}"] = v
        if self.vlm_caption_scores and isinstance(self.vlm_caption_scores, dict):
            scores = self.vlm_caption_scores.get("scores", self.vlm_caption_scores)
            if isinstance(scores, dict):
                for k, v in scores.items():
                    if isinstance(v, (int, float)):
                        d[f"vlm:{k}"] = v
        return d

    @property
    def trajectory_distance(self) -> float | None:
        """Trajectory shape distance to the seed (lower = more similar).

        This is the **same** value as :attr:`trajectory_score`, exposed
        under the more honest name. ``trajectory_score`` was historically
        named like a similarity but is actually an L2 distance — sorting
        descending puts the LEAST similar clips first. Prefer this name
        when intent matters.
        """
        return self.trajectory_score

    @property
    def trajectory_similarity(self) -> float | None:
        """Monotone-decreasing transform of trajectory_distance (higher = better).

        Computed as ``1.0 / (1.0 + trajectory_distance)``. Useful when you
        want to combine trajectory matches with other similarity-style
        scores (Cosmos / CLIP / classifier) which are all higher-is-better.

        Sort descending by ``trajectory_similarity`` to get the most
        similar clips first; equivalent to sorting ascending by
        ``trajectory_distance``.
        """
        d = self.trajectory_score
        if d is None or not math.isfinite(d) or d < 0:
            return None
        return 1.0 / (1.0 + d)

    @property
    def cluster_similarity(self) -> float | None:
        """Higher-is-better transform of cluster_distance_score."""
        d = self.cluster_distance_score
        if d is None or not math.isfinite(d) or d < 0:
            return None
        return 1.0 / (1.0 + d)

    @property
    def best_score(self) -> float | None:
        """Return the most relevant finite non-None score for ranking display.

        Priority order: classifier > semantic_text > semantic_clip >
        caption_embed > visual_image > visual > trajectory_similarity >
        cluster_similarity. (When ``rank_mode='rrf'`` was used,
        ``rrf_score`` takes precedence — that's the fused score the server
        actually used to rank.)

        **CRITICAL** (v1.8.2 fix): trajectory and cluster scores are
        DISTANCES (lower = more similar). Earlier versions of this
        property returned the raw distance which made
        ``sort(key=lambda r: -r.best_score)`` (descending) put the LEAST
        similar clips first. The property now returns the *similarity*
        transform (``1.0 / (1.0 + distance)``) for these two fields, so
        every component of ``best_score`` is uniformly higher-is-better.

        See ``knowledge/anti-patterns.md § 12``.
        """
        for s in (self.rrf_score,
                  self.classifier_score, self.semantic_text_score,
                  self.semantic_clip_score, self.caption_embed_score,
                  self.visual_image_score, self.visual_score,
                  self.trajectory_similarity, self.cluster_similarity):
            if s is not None and math.isfinite(s):
                return s
        return None

    @property
    def caption(self) -> str:
        """Alias for caption_text — the first caption string from this clip."""
        return self.caption_text

    @property
    def score(self) -> float | None:
        """Alias for best_score — the most relevant score for ranking."""
        return self.best_score

    @property
    def annotation_labels(self) -> list[str]:
        """Unique annotation label names on this clip."""
        seen: set[str] = set()
        out: list[str] = []
        for a in self.annotations:
            if isinstance(a, dict):
                lbl = a.get("label", "")
                if lbl and lbl not in seen:
                    seen.add(lbl)
                    out.append(lbl)
        return out

    @property
    def timed_annotations(self) -> list[dict]:
        """Annotations with temporal bounds (start_time and/or end_time).

        Each dict has keys: label, start_time, end_time, project, label_type, value.
        Useful for identifying temporal regions of interest within a clip.
        """
        return [
            a for a in self.annotations
            if isinstance(a, dict) and (a.get("start_time") is not None
                                        or a.get("end_time") is not None)
        ]

    def to_dict(self) -> dict:
        """Convert to a plain dict for JSON serialization or DataFrame construction."""
        d = asdict(self)
        d["caption_text"] = self.caption_text
        d["best_score"] = self.best_score
        d["annotation_labels"] = self.annotation_labels
        return d

    @classmethod
    def from_video_dict(cls, v: dict) -> SearchResult:
        ann = v.get("annotations") or {}
        if not isinstance(ann, dict):
            ann = {}

        def _pick(key: str, default=""):
            """Return ann[key] if not None, else v[key] if not None, else default."""
            val = ann.get(key)
            if val is not None:
                return val
            val = v.get(key)
            if val is not None:
                return val
            return default

        def _or(val, default):
            return val if val is not None else default

        return cls(
            clip_id=str(_pick("clip_id", "")),
            data_source=_pick("data_source", ""),
            annotations=_or(ann.get("annotations"), []),
            speed=_or(v.get("speed"), []),
            acceleration=_or(v.get("acceleration"), []),
            curvature=_or(v.get("curvature"), []),
            jerk=_or(v.get("jerk"), []),
            has_trajectories=bool(v.get("has_trajectories")),
            has_embeddings=bool(v.get("has_embeddings")),
            captions=v.get("captions") if isinstance(v.get("captions"), dict) else {},
            semantic_clip_score=_first_of(v, "semantic_search_clip_score", "semantic_video_score"),
            semantic_text_score=_first_of(v, "semantic_search_text_score", "semantic_text_score"),
            visual_score=v.get("clip_score"),
            # Server emits clip_image_score for image-based CLIP search
            # (visual_search_image_id) — was being silently dropped pre-this-fix.
            visual_image_score=v.get("clip_image_score"),
            trajectory_score=_first_of(v, "trajectory_score", "trajectory_shape_score"),
            classifier_score=_first_of(v, "classifier_score", "classification_score"),
            cluster_distance_score=v.get("cluster_distance_score"),
            # Server emits caption_embed_score for caption_embedding_search —
            # was being silently dropped pre-this-fix; sort orders & best_score
            # were wrong for that mode as a result.
            caption_embed_score=v.get("caption_embed_score"),
            # Server emits rrf_score when rank_mode='rrf' is requested —
            # was being silently dropped pre-this-fix; ranking was wrong.
            rrf_score=v.get("rrf_score"),
            numeric_scores=v.get("numeric_scores"),
            country=_or(v.get("country"), ""),
            country_name=_or(v.get("country_name"), ""),
            positions=v.get("positions"),
            sil_apis=_or(v.get("sil_apis"), []),
            comments=_or(v.get("comments"), ""),
            vlm_caption_scores=v.get("vlm_caption_scores"),
            # Server commit d7033ede ("Refactor cluster selection") emits
            # cluster_membership on /videos when cluster_run_id is in the
            # search filter. Shape: {"cluster_id": int, "distance": float}.
            cluster_membership=(
                v.get("cluster_membership")
                if isinstance(v.get("cluster_membership"), dict)
                else None
            ),
        )


@dataclass
class ScoredClipID:
    """Clip ID paired with multi-dimensional relevance scores.

    Downstream pipelines (e.g., training data selection) can use these scores
    for smooth weighting — e.g., how strongly a clip matches "urban", "weather",
    "interesting", or any other search dimension.

    The agent produces these scores; the training pipeline decides how to use
    them (e.g., as mixture weights, importance sampling, curriculum ordering).
    """
    clip_id: str
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def aggregate_score(self) -> float:
        """Mean of all finite scores. 0.0 if no scores."""
        vals = [v for v in self.scores.values() if isinstance(v, (int, float)) and math.isfinite(v)]
        return sum(vals) / len(vals) if vals else 0.0

    @staticmethod
    def from_search_result(r: SearchResult, dimension: str = "") -> ScoredClipID:
        """Create from a SearchResult, extracting all available scores."""
        scores: dict[str, float] = {}
        for name, val in [
            ("semantic_clip", r.semantic_clip_score),
            ("semantic_text", r.semantic_text_score),
            ("visual", r.visual_score),
            ("trajectory", r.trajectory_score),
            ("classifier", r.classifier_score),
            ("cluster_distance", r.cluster_distance_score),
        ]:
            if val is not None and math.isfinite(val):
                key = f"{dimension}:{name}" if dimension else name
                scores[key] = val
        if r.numeric_scores:
            for k, v in r.numeric_scores.items():
                if isinstance(v, (int, float)) and math.isfinite(v):
                    key = f"{dimension}:metric:{k}" if dimension else f"metric:{k}"
                    scores[key] = v
        return ScoredClipID(clip_id=r.clip_id, scores=scores)


class WheelZeroResultError(RuntimeError):
    """Raised in strict mode (``WHEEL_STRICT=1``) when a search returns 0
    results in a way that's likely a silent footgun rather than a true
    empty set.

    The plain warning is converted to this exception so autonomous agents
    that ignore stderr stop and surface the failure instead of concluding
    "no clips" when they were actually doing something slightly wrong
    (composed-search trap, exact-clip-key footgun, FTS5 AND-of-words on
    natural language, etc.). See ``knowledge/anti-patterns.md``.

    Always carries:
    - ``kind``: machine-readable category (``"composed_search"``,
      ``"exact_clip_id_lookup"``, ``"caption_fts5_and"``, …)
    - ``message``: human-readable explanation including the recommended
      remediation
    """

    def __init__(self, message: str, *, kind: str = "unknown") -> None:
        super().__init__(message)
        self.kind = kind


class WheelAuthenticationError(RuntimeError):
    """Raised when ``WheelClient.login(raise_on_failure=True)`` cannot
    authenticate after all retries.

    Carries:
    - ``reason``: machine-readable category (``"timeout"``,
      ``"connection_error"``, ``"bad_credentials"``, ``"missing_credentials"``,
      ``"network_down"``, ``"wheel_down"``)
    - ``attempts``: number of login attempts made before giving up
    - ``last_error``: the underlying exception or status, if any

    The single most common silent-bug failure mode this prevents: ``login()``
    returning ``False`` (because the server timed out or VPN dropped) while
    the caller proceeds to call ``search()``. Unauthenticated searches return
    ``total=0, results=[]`` — indistinguishable from "no matches" — and the
    ``WHEEL_STRICT`` warnings never fire. With ``raise_on_failure=True`` the
    caller is forced to handle the auth failure or abort.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str = "unknown",
        attempts: int = 1,
        last_error: Any = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.attempts = attempts
        self.last_error = last_error


class WheelClient:
    """Programmatic client for the SIL Wheel HTTP API.

    Credentials are loaded from environment variables (set in .env):
      - WHEEL_USERNAME: NVIDIA username
      - WHEEL_PASSWORD: SIL Wheel password
      - WHEEL_SERVER_URL: server URL (defaults to production)
    """

    def __init__(self, base_url: str | None = None, timeout: int = 120):
        self.base_url = (base_url or _default_server()).rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        from requests.adapters import HTTPAdapter
        adapter = HTTPAdapter(pool_connections=1, pool_maxsize=16)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self._authenticated = False
        self._username: str | None = None
        self._password: str | None = None
        self._cache: dict[str, tuple[float, Any]] = {}
        self._cache_ttl: float = 300  # 5 minutes
        self._auth_lock = threading.Lock()
        self._auth_generation = 0
        self._cache_locks: dict[str, threading.Lock] = {}
        self._cache_meta_lock = threading.Lock()
        self._export_lock = threading.Lock()
        self._search_error_lock = threading.Lock()
        self.last_search_error: str | None = None

    def close(self):
        """Close the underlying HTTP session, clear stored credentials and cache."""
        self.session.close()
        self._authenticated = False
        self._username = None
        self._password = None
        self._cache.clear()
        self._cache_locks.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _get_cached(self, key: str, fetcher) -> Any:
        """Return cached value if fresh, otherwise call fetcher and cache.

        Thread-safe: uses per-key locking to prevent cache stampede.
        Does not cache error responses (re-fetched on next call).
        Evicts expired entries on write to bound memory usage.
        """
        with self._cache_meta_lock:
            entry = self._cache.get(key)
        if entry is not None:
            ts, val = entry
            if _time_mod.time() - ts < self._cache_ttl:
                return val
        with self._cache_meta_lock:
            if key not in self._cache_locks:
                self._cache_locks[key] = threading.Lock()
            lock = self._cache_locks[key]
        with lock:
            entry = self._cache.get(key)
            if entry is not None:
                ts, val = entry
                if _time_mod.time() - ts < self._cache_ttl:
                    return val
            val = fetcher()
            if isinstance(val, dict) and "error" in val:
                return val
            with self._cache_meta_lock:
                now = _time_mod.time()
                self._cache[key] = (now, val)
                expired = [k for k, (t, _) in self._cache.items()
                           if now - t >= self._cache_ttl]
                for k in expired:
                    del self._cache[k]
                self._prune_cache_locks()
            return val

    def _prune_cache_locks(self) -> None:
        """Remove locks for keys no longer in cache. Must be called under _cache_meta_lock."""
        stale = [k for k in self._cache_locks if k not in self._cache]
        for k in stale:
            del self._cache_locks[k]

    def invalidate_cache(self, key: str | None = None):
        """Clear cached data. Pass key for specific item, None for all."""
        with self._cache_meta_lock:
            if key is None:
                self._cache.clear()
                self._cache_locks.clear()
            else:
                self._cache.pop(key, None)
                self._cache_locks.pop(key, None)

    @classmethod
    def dev(cls, timeout: int = 120) -> WheelClient:
        """Create a client pointed at the dev server. Loads credentials from .env.

        The dev server (localhost:8018) is only reachable from NVIDIA
        internal network or VPN. Returns an unauthenticated client if the
        dev server URL is not configured.
        """
        url = os.environ.get("WHEEL_DEV_URL", DEV_SERVER)
        return cls(base_url=url, timeout=timeout)

    def login(
        self,
        username: str | None = None,
        password: str | None = None,
        *,
        raise_on_failure: bool = False,
        retries: int | None = None,
        backoff: float = 5.0,
    ) -> bool:
        """Authenticate with the SIL Wheel server.

        Args:
            username: NVIDIA username. Defaults to WHEEL_USERNAME env var
                      (or WHEEL_DEV_USER for dev server).
            password: SIL Wheel password. Defaults to WHEEL_PASSWORD env var
                      (or WHEEL_DEV_PASSWORD for dev server).
            raise_on_failure: when True (or env ``WHEEL_LOGIN_RAISE=1``),
                              raise ``WheelAuthenticationError`` instead of
                              returning ``False`` after retries exhausted.
                              Strongly recommended for autonomous loops —
                              prevents the silent ``total=0`` failure mode
                              where the caller forgets to check the return
                              value and downstream searches go through
                              unauthenticated.
            retries: number of additional retry attempts on transient
                     network errors / timeouts (default: env
                     ``WHEEL_LOGIN_RETRIES`` if set, else 0). Each retry
                     sleeps ``backoff * (attempt^1.5)`` seconds.
            backoff: base backoff in seconds between retries (default 5s).

        Returns:
            ``True`` on success.
            ``False`` on failure (only if ``raise_on_failure`` is False).
        """
        env_raise = os.environ.get("WHEEL_LOGIN_RAISE", "").lower() in ("1", "true", "yes")
        raise_on_failure = raise_on_failure or env_raise

        if retries is None:
            try:
                retries = int(os.environ.get("WHEEL_LOGIN_RETRIES", "0"))
            except ValueError:
                retries = 0
        retries = max(0, int(retries))

        is_dev = DEV_SERVER in self.base_url or (bool(os.environ.get("WHEEL_DEV_URL", "")) and os.environ.get("WHEEL_DEV_URL", "").rstrip("/") in self.base_url)
        if username is None:
            if is_dev:
                username = os.environ.get("WHEEL_DEV_USER",
                           os.environ.get("WHEEL_USERNAME", ""))
            else:
                username = os.environ.get("WHEEL_USERNAME", "")
        if password is None:
            if is_dev:
                password = os.environ.get("WHEEL_DEV_PASSWORD",
                           os.environ.get("WHEEL_PASSWORD", ""))
            else:
                password = os.environ.get("WHEEL_PASSWORD", "")

        if not username or not password:
            msg = "WHEEL_USERNAME and WHEEL_PASSWORD must be set in .env"
            print(f"Error: {msg}", file=sys.stderr)
            if raise_on_failure:
                raise WheelAuthenticationError(
                    msg, reason="missing_credentials", attempts=0,
                )
            return False

        try:
            self._validate_no_protocol_chars(username, password)
        except ValueError as e:
            msg = ("Invalid characters in username or password "
                   "(do not use :: or control characters)")
            print(f"Error: {msg}", file=sys.stderr)
            if raise_on_failure:
                raise WheelAuthenticationError(
                    msg, reason="bad_credentials", attempts=0, last_error=e,
                )
            return False

        self._username = username
        self._password = password

        last_err: Any = None
        last_reason: str = "unknown"
        last_status: int | None = None
        attempts = 0
        for attempt in range(retries + 1):
            attempts = attempt + 1
            try:
                resp = self.session.post(
                    f"{self.base_url}/",
                    data=f"user_login::{username}::{password}",
                    headers={"Content-Type": "text/plain"},
                    timeout=self.timeout,
                    allow_redirects=False,
                )
            except requests.exceptions.ReadTimeout as e:
                last_err = e
                last_reason = "timeout"
                print(
                    f"Error: Login timed out after {self.timeout}s connecting to "
                    f"{self.base_url} (attempt {attempts}/{retries + 1}). "
                    "The server may be down or under heavy load.",
                    file=sys.stderr,
                )
                if attempt < retries:
                    _time_mod.sleep(backoff * ((attempt + 1) ** 1.5))
                    continue
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_err = e
                last_reason = "connection_error"
                print(
                    f"Error: Cannot connect to {self.base_url} — "
                    f"{type(e).__name__} (attempt {attempts}/{retries + 1}). "
                    "Check your network connection and VPN. Use "
                    "client.check_connection() to diagnose.",
                    file=sys.stderr,
                )
                if attempt < retries:
                    _time_mod.sleep(backoff * ((attempt + 1) ** 1.5))
                    continue
                break

            last_status = resp.status_code
            self._authenticated = "session_id" in self.session.cookies
            if not self._authenticated:
                self._authenticated = resp.status_code in (200, 302)
            if self._authenticated:
                return True

            # got a response but auth failed — almost certainly bad credentials.
            # Don't retry: same creds will fail the same way.
            last_reason = "bad_credentials"
            last_err = f"HTTP {resp.status_code}"
            break

        if raise_on_failure:
            # When connection/timeout was the cause, refine the reason via the
            # health-probe path so the error message is actionable.
            if last_reason in ("timeout", "connection_error"):
                try:
                    diag = self.check_connection(timeout=6, probe_vpn=True)
                    verdict = diag.get("verdict") or diag.get("diagnosis")
                    if verdict == "network_down":
                        last_reason = "network_down"
                    elif verdict in ("wheel_down", "server_down",
                                     "server_unresponsive"):
                        last_reason = "wheel_down"
                except Exception:
                    pass
            raise WheelAuthenticationError(
                f"login() failed after {attempts} attempt(s): reason={last_reason}, "
                f"last_error={last_err!r}",
                reason=last_reason,
                attempts=attempts,
                last_error=last_err,
            )
        return False

    @property
    def is_production(self) -> bool:
        return os.environ.get("WHEEL_READONLY", "").strip().lower() in ("1", "true", "yes")

    #: Reachability probe used to disambiguate "VPN down" from
    #: "sil-wheel down". Must be a reliable, always-up host.
    #: (so a public-internet host doesn't false-positive). Override via the
    #: ``WHEEL_VPN_PROBE_URL`` env var if this URL ever stops working.
    _VPN_PROBE_URL = "https://github.com"

    def check_connection(
        self,
        timeout: int = 10,
        *,
        probe_vpn: bool = True,
    ) -> dict:
        """Quick connectivity check — returns status without full auth.

        When ``probe_vpn=True`` (default), an additional probe to an
        reliable public host (default ``github.com/nv-tlabs/sil-wheel``,
        configurable via env ``WHEEL_VPN_PROBE_URL``) runs in parallel.
        This disambiguates **VPN-down** (both probes fail) from
        **sil-wheel-down** (wheel probe fails, VPN probe succeeds), which
        the previous "vpn_or_network" diagnosis conflated.

        Returns:
            dict with keys:
              - ``reachable`` (bool)               — is the wheel server reachable?
              - ``authenticated`` (bool)           — is the session authenticated?
              - ``latency_ms`` (int)               — wheel-server probe latency
              - ``status_code`` (int, optional)    — wheel-server HTTP status
              - ``error`` (str, optional)          — wheel-server error message
              - ``diagnosis`` (str)                — one of:
                  * ``"healthy"``           — wheel reachable
                  * ``"server_unresponsive"`` — wheel timed out
                  * ``"ssl_error"``         — TLS/cert problem
                  * ``"server_down"``       — high-latency connect refused
                  * ``"network_down"``          — VPN probe also failed (NEW)
                  * ``"wheel_down"``    — VPN works, wheel doesn't (NEW)
                  * ``"vpn_or_network"``    — probe_vpn=False fallback
                  * ``"network_error"``     — other exception
              - ``vpn_probe`` (dict, optional)     — sub-result for the VPN
                probe: ``{ok: bool, latency_ms: int, error?: str}``
        """
        t0 = _time_mod.time()
        wheel_result: dict
        try:
            resp = self.session.get(f"{self.base_url}/whoami", timeout=timeout)
            latency = int((_time_mod.time() - t0) * 1000)
            try:
                data = resp.json() if resp.status_code == 200 else {}
            except (json.JSONDecodeError, ValueError):
                data = {}
            return {
                "reachable": True,
                "authenticated": data.get("authenticated", False),
                "latency_ms": latency,
                "status_code": resp.status_code,
                "diagnosis": "healthy",
            }
        except requests.exceptions.ReadTimeout:
            wheel_result = {
                "reachable": False,
                "error": f"Timeout after {timeout}s",
                "latency_ms": timeout * 1000,
                "diagnosis": "server_unresponsive",
            }
        except requests.exceptions.SSLError as e:
            wheel_result = {
                "reachable": False,
                "error": f"SSL error: {e}",
                "latency_ms": int((_time_mod.time() - t0) * 1000),
                "diagnosis": "ssl_error",
            }
        except requests.exceptions.ConnectionError:
            latency = int((_time_mod.time() - t0) * 1000)
            wheel_result = {
                "reachable": False,
                "error": "Connection refused",
                "latency_ms": latency,
                "diagnosis": ("vpn_or_network" if latency < 2000 else "server_down"),
            }
        except requests.exceptions.RequestException as e:
            wheel_result = {
                "reachable": False,
                "error": f"{type(e).__name__}: {e}",
                "latency_ms": int((_time_mod.time() - t0) * 1000),
                "diagnosis": "network_error",
            }

        if probe_vpn:
            vpn = self._probe_vpn(timeout=timeout)
            wheel_result["vpn_probe"] = vpn
            if wheel_result["diagnosis"] in ("vpn_or_network", "server_down",
                                             "server_unresponsive", "network_error"):
                # Disambiguate: if the VPN probe succeeds, the *wheel* is down.
                # If both fail, the VPN/internal network is the problem.
                if vpn.get("ok"):
                    wheel_result["diagnosis"] = "wheel_down"
                else:
                    wheel_result["diagnosis"] = "network_down"

        return wheel_result

    def _probe_vpn(self, timeout: int = 10) -> dict:
        """Probe an external sibling URL to determine if VPN is up.

        Used by ``check_connection(probe_vpn=True)`` to disambiguate
        "VPN-down" from "sil-wheel-down". We use a *new* short-lived
        ``requests.Session`` so cookies / auth state from the wheel session
        cannot influence the probe.
        """
        url = os.environ.get("WHEEL_VPN_PROBE_URL", self._VPN_PROBE_URL)
        t0 = _time_mod.time()
        try:
            with requests.Session() as sess:
                resp = sess.get(url, timeout=timeout, allow_redirects=False)
            return {
                "url": url,
                "ok": True,
                "status_code": resp.status_code,
                "latency_ms": int((_time_mod.time() - t0) * 1000),
            }
        except requests.exceptions.RequestException as e:
            return {
                "url": url,
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "latency_ms": int((_time_mod.time() - t0) * 1000),
            }

    def wait_for_server(
        self,
        max_wait: int = 1800,
        interval: int = 90,
        probe_vpn: bool = True,
        verbose: bool = True,
    ) -> dict:
        """Block until the SIL Wheel server is reachable (or ``max_wait`` elapses).

        Repeatedly calls ``check_connection`` with backoff-friendly interval.
        Useful in autonomous loops that hit transient outages — instead of
        bailing on the first failure, they can ``client.wait_for_server()``
        and resume.

        Args:
            max_wait: maximum seconds to wait before giving up (default 30min).
            interval: seconds between probes (default 90s).
            probe_vpn: pass through to ``check_connection``.
            verbose: print one summary line per probe attempt.

        Returns:
            The final ``check_connection`` dict (``diagnosis == "healthy"``
            on success).

        Raises:
            TimeoutError: if ``max_wait`` elapses without the server
                          becoming reachable.
        """
        start = _time_mod.time()
        attempt = 0
        last: dict = {}
        while _time_mod.time() - start < max_wait:
            attempt += 1
            last = self.check_connection(timeout=10, probe_vpn=probe_vpn)
            if verbose:
                diag = last.get("diagnosis", "unknown")
                elapsed = int(_time_mod.time() - start)
                print(
                    f"[wait_for_server] attempt {attempt} t={elapsed:>4}s  "
                    f"diagnosis={diag}  reachable={last.get('reachable')}",
                    flush=True,
                )
            if last.get("reachable"):
                return last
            _time_mod.sleep(interval)
        raise TimeoutError(
            f"wait_for_server timed out after {max_wait}s ({attempt} attempts); "
            f"last diagnosis={last.get('diagnosis')!r}"
        )

    _NETWORK_ERRORS = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.TooManyRedirects,
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ContentDecodingError,
    )

    def _relogin(self) -> bool:
        """Attempt silent re-authentication using stored credentials.

        Thread-safe: uses a lock so only one thread re-authenticates at a time.
        """
        with self._auth_lock:
            if not self._username or not self._password:
                return False
            try:
                resp = self.session.post(
                    f"{self.base_url}/",
                    data=f"user_login::{self._username}::{self._password}",
                    headers={"Content-Type": "text/plain"},
                    timeout=self.timeout,
                    allow_redirects=False,
                )
            except self._NETWORK_ERRORS:
                return False
            self._authenticated = "session_id" in self.session.cookies
            if not self._authenticated:
                self._authenticated = resp.status_code in (200, 302)
            if self._authenticated:
                self._auth_generation += 1
            return self._authenticated

    _RETRYABLE_STATUS = frozenset({429, 502, 503})
    max_retries: int = 1

    def _get(self, endpoint: str, params: dict | None = None, **kwargs) -> requests.Response:
        """HTTP GET with auto-retry on transient errors and auto re-login on 401/403."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        timeout = kwargs.pop("timeout", self.timeout)
        max_retries = max(self.max_retries, 0)
        resp = None

        for attempt in range(max_retries + 1):
            gen_before = self._auth_generation
            try:
                resp = self.session.get(url, params=params, timeout=timeout, **kwargs)
            except self._NETWORK_ERRORS as e:
                if attempt < max_retries:
                    _time_mod.sleep(min(2 ** attempt + _random_mod.random(), 8))
                    continue
                return _timeout_response(url, e)

            if resp.status_code in (401, 403) and self._authenticated:
                if self._auth_generation == gen_before:
                    if not self._relogin():
                        return resp
                try:
                    resp = self.session.get(url, params=params, timeout=timeout, **kwargs)
                except self._NETWORK_ERRORS as e:
                    return _timeout_response(url, e)

            if resp.status_code not in self._RETRYABLE_STATUS or attempt >= max_retries:
                return resp
            _time_mod.sleep(min(2 ** attempt + _random_mod.random(), 8))

        return resp  # type: ignore[return-value]

    def _post(self, endpoint: str, data: str | dict | None = None, **kwargs) -> requests.Response:
        """HTTP POST with auto re-login on 401/403. No retry (POSTs may not be idempotent)."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        timeout = kwargs.pop("timeout", self.timeout)
        headers = kwargs.pop("headers", None)
        if isinstance(data, str) and not headers:
            headers = {"Content-Type": "text/plain"}
        gen_before = self._auth_generation
        try:
            resp = self.session.post(url, data=data, headers=headers, timeout=timeout, **kwargs)
        except self._NETWORK_ERRORS as e:
            return _timeout_response(url, e)
        if resp.status_code in (401, 403) and self._authenticated:
            if self._auth_generation == gen_before:
                if not self._relogin():
                    return resp
            try:
                resp = self.session.post(url, data=data, headers=headers, timeout=timeout, **kwargs)
            except self._NETWORK_ERRORS as e:
                return _timeout_response(url, e)
        return resp

    def logout(self) -> bool:
        """End the current session and clear stored credentials."""
        try:
            self._post("", data="logout")
        except Exception:
            pass
        self.session.cookies.clear()
        self._authenticated = False
        self._username = None
        self._password = None
        return True

    @staticmethod
    def load_clip_ids(filepath: str) -> list[str]:
        """Load clip IDs from a text file (one per line) or JSON file.

        JSON files can be a list of IDs or a dict with IDs as keys.
        """
        p = Path(filepath).expanduser()
        if p.suffix == ".json":
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return [str(k) for k in data.keys()]
            if isinstance(data, list):
                out: list[str] = []
                for item in data:
                    if isinstance(item, dict):
                        cid = item.get("clip_id") or item.get("id")
                        if not cid and item:
                            cid = next(iter(item.values()))
                        if cid:
                            out.append(str(cid))
                        continue
                    else:
                        out.append(str(item))
                return out
            raise ValueError(f"Expected list or dict in {filepath}, got {type(data).__name__}")
        with open(p, encoding="utf-8-sig") as f:
            return [line.strip() for line in f if line.strip()]

    # ── Server Info ──────────────────────────────────────────────────

    def whoami(self) -> dict:
        """Get current user info and authentication status."""
        resp = self._get("whoami")
        try:
            return resp.json()
        except (json.JSONDecodeError, ValueError):
            return {"authenticated": False, "error": resp.status_code}

    def get_data_sources(self) -> list[str]:
        """List available data source names (e.g. 'MADS', 'MADS-1M', 'Alpamayo').

        For full per-dataset stats (trajectory plots, annotation counts),
        use get_data_stats() instead.
        """
        stats = self.get_data_stats()
        return [s["dataset"] for s in stats if isinstance(s, dict) and s.get("dataset")]

    def get_classifiers(self) -> dict:
        """List all classifier labels (trained and untrained). Cached for 5 min."""
        return self._get_cached("classifiers", self._fetch_classifiers)

    def list_classifier_names(
        self,
        include_untrained: bool = False,
        embed_type: str | None = None,
    ) -> list[str]:
        """Return a flat list of classifier label names.

        By default returns trained classifiers from BOTH embedding types
        (``cosmos`` and ``caption``), deduplicated, sorted. Pass
        ``include_untrained=True`` to include pending/untrained labels.
        Pass ``embed_type='cosmos'`` or ``'caption'`` to filter to one
        embedding backend.

        The returned names can be passed directly to
        :meth:`classifier_search`. (Classifier names are case-sensitive on
        the server — use :meth:`resolve_classifier_name` if you have an
        approximate query.)

        Server schema: the ``/classifiers_status`` endpoint returns
        ``trained_by_type: {cosmos: [...], caption: [...]}`` (renamed
        from a flat ``trained`` list in 2026-Q2). We handle both shapes
        for back-compat.
        """
        data = self.get_classifiers()
        names: list[str] = []
        # New schema (2026-Q2+): trained_by_type
        tbt = data.get("trained_by_type") or {}
        if isinstance(tbt, dict):
            if embed_type is not None:
                names.extend(
                    n for n in (tbt.get(embed_type) or []) if isinstance(n, str)
                )
            else:
                seen: set[str] = set()
                for et in ("cosmos", "caption"):
                    for n in (tbt.get(et) or []):
                        if isinstance(n, str) and n not in seen:
                            seen.add(n)
                            names.append(n)
        # Legacy schema fallback (pre-2026-Q2): flat "trained" list.
        legacy_trained = data.get("trained")
        if not names and isinstance(legacy_trained, list):
            names = [l for l in legacy_trained if isinstance(l, str)]
        # Sort trained classifiers alphabetically for deterministic display.
        names = sorted(names)
        if include_untrained:
            # Untrained appended AFTER trained (also sorted) so the agent
            # can visually separate "use these now" vs "needs training".
            untrained = sorted(
                n for n in (data.get("untrained") or []) if isinstance(n, str)
            )
            names = names + untrained
        return names

    def list_classifiers(
        self,
        include_untrained: bool = False,
        embed_type: str | None = None,
    ) -> list[str]:
        """Alias for :meth:`list_classifier_names` — follows the ``list_*`` convention.

        Agents familiar with ``list_data_sources()`` / ``list_communities()``
        often guess ``list_classifiers()`` first. This alias short-circuits that
        ~2-minute discovery cost. For the full dict structure (trained,
        untrained, annotation counts, per-embed-type breakdown), use
        :meth:`get_classifiers`.
        """
        return self.list_classifier_names(
            include_untrained=include_untrained, embed_type=embed_type,
        )

    def get_classifier_embed_type(self, label: str) -> str | None:
        """Return the embed type (``"cosmos"`` or ``"caption"``) for a trained classifier.

        Returns ``None`` if the label is not trained on either backend.
        Useful for :meth:`export_classifier_weights` which requires
        ``embed_type``.
        """
        data = self.get_classifiers()
        tbt = data.get("trained_by_type") or {}
        if isinstance(tbt, dict):
            for et in ("cosmos", "caption"):
                if label in (tbt.get(et) or []):
                    return et
        return None

    def _fetch_classifiers(self) -> dict:
        resp = self._get("classifiers_status")
        try:
            return resp.json()
        except (json.JSONDecodeError, ValueError):
            return {"trained": [], "untrained": [], "error": resp.status_code}

    def get_leaderboard(self) -> dict:
        """Get all model leaderboards with aggregate metrics.

        Delegates to get_metrics() — both hit the same /metrics endpoint.
        """
        return self.get_metrics()

    # ── Core Search ──────────────────────────────────────────────────

    def search(
        self,
        page: int = 0,
        n: int = 20,
        data_source: str | None = None,
        project_source: str | None = None,
        label_filter: str | None = None,
        filter_mode: str | None = None,
        semantic_search_clipid: str | None = None,
        semantic_search_text: str | None = None,
        visual_search_text: str | None = None,
        trajectory_shape_clipid: str | None = None,
        trajectory_shape_start_t: float | None = None,
        trajectory_shape_end_t: float | None = None,
        trajectory_pattern: str | None = None,
        search_speed: str | None = None,
        with_ego_data: bool | None = None,
        classifier_select: str | None = None,
        probability_threshold: float | None = None,
        probability_expression: str | None = None,
        search: str | None = None,
        query_rewrite: bool | None = None,
        search_country: str | None = None,
        left_hand_driving: bool | None = None,
        wm_class_name: str | None = None,
        wm_min_count: int | None = None,
        wm_max_count: int | None = None,
        wm_max_dist: float | None = None,
        wm_min_time: float | None = None,
        wm_angle_range: str | None = None,
        with_metrics: bool | None = None,
        with_bev: bool | None = None,
        numeric_filter: str | None = None,
        search_clipid: str | None = None,
        labels_to_exclude: str | None = None,
        label_types: str | None = None,
        search_comments: str | None = None,
        cluster_run_id: str | None = None,
        cluster_id: str | None = None,
        sil_apis: str | None = None,
        without_ann: bool | None = None,
        times: bool | None = None,
        extra_queries: str | None = None,
        caption_embed_search_text: str | None = None,
        caption_embed_extra_queries: str | None = None,
        semantic_extra_queries: str | None = None,
        visual_extra_queries: str | None = None,
        visual_search_image_id: str | None = None,
        classifier_embed_type: str | None = None,
        rank_mode: str | None = None,
        **extra_params,
    ) -> tuple[int, list[SearchResult]]:
        """Unified search combining ALL SIL Wheel filter modes.

        All parameters are composable — combine any subset in a single call.
        Returns (total_count, list_of_SearchResult).

        The server returns at most 20 results per page. ``n`` is clamped to
        [1, 20]; values outside this range are silently adjusted. For larger
        result sets use ``search_all_pages()`` or ``export_search_clip_ids()``.

        Args:
            n: Results per page, clamped to [1, 20] (server page size limit).
            filter_mode: 'any' (OR, default) or 'all' (AND) for annotation labels.
            query_rewrite: If True, enables LLM-based caption search expansion
                           (server must have NV_INFERENCE_API_KEY configured).
            without_ann: If True, only return clips without annotations.
            times: If True, only return clips with timed annotations.
            extra_queries: Additional caption queries separated by '||' for
                           multi-query caption search.
            search_clipid: Filter by clip ID. For exact-key retrieval prefer
                           ``lookup_clip(clip_id) -> SearchResult | None`` —
                           it expresses intent more clearly. If a 0-result
                           response comes back for a key-shaped value, a
                           ``UserWarning`` recommends ``lookup_clip``.
        """
        if n > 20:
            warnings.warn(
                f"n={n} exceeds server page size of 20; clipped. "
                "Use search_all_pages() or export_search_clip_ids() for more.",
                stacklevel=2,
            )

        # Preflight: trajectory predicates are NOT indexed on a known set
        # of data sources. Warn so callers don't waste 50-150s on a search
        # that's guaranteed to return 0. v1.8.2.
        if (
            (search_speed is not None or trajectory_pattern is not None)
            and data_source in self._TRAJECTORY_UNINDEXED_SOURCES
        ):
            warnings.warn(
                f"trajectory predicate / pattern with data_source={data_source!r} "
                "will silently return 0 — trajectory predicates are not indexed on "
                "this source (and the request takes ~50-150s before failing). Use "
                "'AV V2 train', 'AV V1 train', 'AV V2.2. train', 'Waymo train', or "
                "a celsius dataset. See knowledge/feature-compat.md.",
                UserWarning,
                stacklevel=2,
            )

        # Preflight: classifier inferences are not uniform across data sources.
        # The expensive case is `Change lane to the right` returning 1.8M
        # globally but 0 on MADS-1M. We can't probe coverage cheaply per
        # call (would defeat the warning's point), but we can at least flag
        # the most common foot-gun: a labels-cosmos classifier running
        # against a data source known to be missing inferences for it.
        if (
            classifier_select is not None
            and data_source in _CLASSIFIER_KNOWN_MISSING.get(classifier_select, ())
        ):
            warnings.warn(
                f"classifier_search({classifier_select!r}, data_source="
                f"{data_source!r}) is known to silently return 0 — that "
                "classifier has no inferences on this data source. Try "
                "data_source='AV V2 train' or use "
                f"client.get_classifier_coverage({classifier_select!r}) to map "
                "the per-data-source counts. See knowledge/feature-compat.md.",
                UserWarning,
                stacklevel=2,
            )

        # Translate client-friendly classifier threshold to server's expression
        # grammar (server expects ``probability_expression``, not
        # ``probability_threshold``; the latter is silently ignored — that's
        # the historical "broken classifier" bug). See knowledge/anti-patterns.md.
        prob_expr = probability_expression
        if prob_expr is None and probability_threshold is not None:
            prob_expr = f"p > {probability_threshold}"
        params: dict[str, Any] = {"page": max(page, 0), "n": max(1, min(n, 20))}
        _maybe = {
            "data_source": data_source,
            "project_source": project_source,
            "filter": label_filter,
            "filter_mode": filter_mode,
            "semantic_search_clipid": semantic_search_clipid,
            "semantic_search_text": semantic_search_text,
            "visual_search_text": visual_search_text,
            "trajectory_shape_clipid": trajectory_shape_clipid,
            "trajectory_shape_start_t": trajectory_shape_start_t,
            "trajectory_shape_end_t": trajectory_shape_end_t,
            "trajectory_pattern": trajectory_pattern,
            "search_speed": search_speed,
            "with_ego_data": "true" if with_ego_data else None,
            "classifier_select": classifier_select,
            # Server's URL param is ``probability_expression``; we accept
            # either ``probability_threshold`` (numeric, sugar) or
            # ``probability_expression`` (full expression).
            "probability_expression": prob_expr,
            "search": search,
            "query_rewrite": "true" if query_rewrite else None,
            "search_country": search_country,
            "left_hand_driving": "true" if left_hand_driving else None,
            "wm_class_name": wm_class_name,
            "wm_min_count": wm_min_count,
            "wm_max_count": wm_max_count,
            "wm_max_dist": wm_max_dist,
            "wm_min_time": wm_min_time,
            "wm_angle_range": wm_angle_range,
            "with_metrics": "true" if with_metrics else None,
            "with_bev": "true" if with_bev else None,
            "numeric_filter": numeric_filter,
            "search_clipid": search_clipid,
            "labels_to_exclude": labels_to_exclude,
            "label_types": label_types,
            "search_comments": search_comments,
            "cluster_run_id": cluster_run_id,
            "cluster_id": cluster_id,
            "sil_apis": sil_apis,
            "without_ann": "true" if without_ann else None,
            "times": "true" if times else None,
            # Server renamed bare ``extra_queries`` → ``caption_extra_queries``
            # somewhere in 2026-Q1. The bare name is silently dropped — the
            # historical "OR semantics broken" bug for ``caption_search_any``.
            "caption_extra_queries": extra_queries,
            # Per-mode multi-query support (each = "||"-separated list);
            # server applies OR semantics per mode independently of others.
            "caption_embed_extra_queries": caption_embed_extra_queries,
            "semantic_extra_queries": semantic_extra_queries,
            "visual_extra_queries": visual_extra_queries,
            "caption_embed_search": caption_embed_search_text,
            # Visual search by uploaded image ID (server stores the upload
            # and returns an image_id; this filter then runs CLIP image-to-
            # video similarity). Requires a separate upload step (server
            # endpoint TBD).
            "visual_search_image_id": visual_search_image_id,
            # classifier_embed_type selects which embedding backend powers
            # the classifier scores ("cosmos" or "caption", server default
            # "cosmos"). Use list_classifier_names(embed_type=...) /
            # get_classifier_embed_type() to discover.
            "classifier_embed_type": classifier_embed_type,
            # rank_mode='rrf' triggers server-side Reciprocal Rank Fusion
            # of multiple search modes (added 2026-04-23 commit 64f6544c).
            # Default 'priority' uses the legacy priority-based scoring.
            "rank_mode": rank_mode,
        }
        for k, v in _maybe.items():
            if v is not None:
                params[k] = v
        _reserved = set(_maybe.keys()) | {
            "page", "n",
            # These are caller-facing kwargs that have already been consumed
            # into _maybe via translation; don't re-add via extra_params.
            "probability_threshold",
            "extra_queries",
            "caption_embed_search_text",
            "probability_expression",
        }
        for k, v in extra_params.items():
            if k not in _reserved:
                params[k] = v

        resp = self._get("videos", params=params)
        if resp.status_code != 200:
            with self._search_error_lock:
                self.last_search_error = f"HTTP {resp.status_code}"
            return 0, []

        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError):
            with self._search_error_lock:
                self.last_search_error = "Invalid JSON response"
            return 0, []
        with self._search_error_lock:
            self.last_search_error = None
        try:
            total = int(data.get("num_videos") or 0)
        except (ValueError, TypeError):
            total = 0
        videos = data.get("videos") or []
        results = [SearchResult.from_video_dict(v) for v in videos if isinstance(v, dict)]

        if total == 0:
            active_filters = sum(1 for v in [
                search, classifier_select, semantic_search_text, semantic_search_clipid,
                visual_search_text, trajectory_shape_clipid, trajectory_pattern,
                wm_class_name, search_comments, label_filter, numeric_filter,
                search_clipid,
            ] if v is not None)
            if active_filters >= 2:
                msg = (
                    f"Composed search with {active_filters} active filters returned 0 results. "
                    "Try running each filter separately and combining with "
                    "intersect_clip_id_lists(). For a structured diagnosis call "
                    "client.diagnose_zero_results(**search_kwargs)."
                )
                if _strict_mode_enabled():
                    raise WheelZeroResultError(msg, kind="composed_search")
                warnings.warn(msg, UserWarning, stacklevel=2)
            # Independent of composed-search warning: if the agent passed an
            # exact clip key as ``search_clipid``, recommend ``lookup_clip()``
            # for clearer intent (same endpoint, cleaner return type). Suppressed
            # when called from ``lookup_clip()`` itself to avoid self-recursion
            # noise (esp. in ``lookup_clips_batch`` parallel scans).
            if (
                search_clipid
                and _EXACT_CLIP_ID_RE.match(search_clipid)
                and not getattr(_lookup_clip_in_progress, "value", False)
            ):
                msg = (
                    f"search(search_clipid={search_clipid!r}) is an exact clip "
                    "key; prefer lookup_clip(clip_id) -> SearchResult | None "
                    "for clearer intent. Note: both routes hit the same /videos "
                    "endpoint — 0 results means the clip is not in the current "
                    "filter universe (try data_source=None to widen). "
                    "See knowledge/anti-patterns.md § 1."
                )
                if _strict_mode_enabled():
                    raise WheelZeroResultError(msg, kind="exact_clip_id_lookup")
                warnings.warn(msg, UserWarning, stacklevel=2)

        return total, results

    # ── Lightweight clip-ID pagination ────────────────────────────────

    # Client kwarg → server URL param mapping. Renames here cover silent
    # server-side renames (the historical "broken classifier" bug B1 and the
    # "broken caption_search_any OR" bug B2 were both caused by the bare
    # client names no longer matching the server's parsed fields). See
    # knowledge/anti-patterns.md.
    _SEARCH_PARAM_RENAME: dict[str, str] = {
        "label_filter": "filter",
        "extra_queries": "caption_extra_queries",  # FIX-B2
        "caption_embed_search_text": "caption_embed_search",
    }
    _SEARCH_BOOL_PARAMS = frozenset({
        "with_ego_data", "query_rewrite", "left_hand_driving",
        "with_metrics", "with_bev", "without_ann", "times",
    })

    _SEARCH_RESERVED_PARAMS = frozenset({"page", "n"})

    @staticmethod
    def _probability_threshold_to_expression(threshold: float) -> str:
        """Convert numeric threshold to server expression grammar.

        Server expects ``probability_expression="p > 0.5"``, not
        ``probability_threshold=0.5``. The numeric form is client-friendly
        sugar; the expression form supports advanced predicates like
        ``"0.3 < p < 0.7"`` (chained comparisons are server-supported).
        """
        return f"p > {threshold}"

    def _build_search_params(self, page: int = 0, n: int = 20, **kwargs) -> dict[str, Any]:
        """Build URL params dict for /videos from search kwargs.

        Replicates the parameter mapping in search() so lightweight endpoints
        can hit /videos without constructing full SearchResult objects.

        Performs the same probability_threshold → probability_expression and
        extra_queries → caption_extra_queries translations as search() so the
        two paths can never silently diverge.
        """
        params: dict[str, Any] = {"page": max(page, 0), "n": max(1, min(n, 20))}
        # Pre-process classifier probability: prefer explicit expression,
        # fall back to numeric threshold sugar.
        prob_expr = kwargs.get("probability_expression")
        if prob_expr is None and kwargs.get("probability_threshold") is not None:
            prob_expr = self._probability_threshold_to_expression(
                kwargs["probability_threshold"]
            )
        # Skip both source kwargs in the loop below; we'll write the URL key
        # explicitly at the end.
        _consumed = {"probability_threshold", "probability_expression"}
        for key, val in kwargs.items():
            if val is None or key in self._SEARCH_RESERVED_PARAMS or key in _consumed:
                continue
            url_key = self._SEARCH_PARAM_RENAME.get(key, key)
            if key in self._SEARCH_BOOL_PARAMS:
                if val:
                    params[url_key] = "true"
            else:
                params[url_key] = val
        if prob_expr is not None:
            params["probability_expression"] = prob_expr
        return params

    def _fetch_clip_ids_page(
        self, page: int = 0, n: int = 20, **kwargs,
    ) -> tuple[int, list[str]]:
        """Fetch one page from /videos and extract only clip IDs.

        Like search() but skips SearchResult construction (~5KB per result),
        returning only (total_count, clip_id_list).
        """
        params = self._build_search_params(page=page, n=n, **kwargs)
        resp = self._get("videos", params=params)
        if resp.status_code != 200:
            return 0, []
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError):
            return 0, []
        try:
            total = int(data.get("num_videos") or 0)
        except (ValueError, TypeError):
            total = 0
        videos = data.get("videos") or []
        ids: list[str] = []
        for v in videos:
            if not isinstance(v, dict):
                continue
            ann = v.get("annotations")
            cid = (ann.get("clip_id") if isinstance(ann, dict) else None) or v.get("clip_id")
            if cid:
                ids.append(str(cid))
        return total, ids

    def _paginate_clip_ids(
        self,
        max_pages: int = 2000,
        n: int = 20,
        max_workers: int = 6,
        **kwargs,
    ) -> list[str]:
        """Paginate through search results extracting only clip IDs.

        Like search_all_pages() but avoids constructing full SearchResult
        objects, reducing memory ~5KB per result. Thread-safe: uses only
        the /videos endpoint with explicit params (no session-scoped state).
        """
        n = max(n, 1)
        if kwargs.get("semantic_search_text") or kwargs.get("visual_search_text"):
            raise ValueError(
                "_paginate_clip_ids with text-to-vector search re-triggers model "
                "inference on every page. Use export_search_clip_ids() instead."
            )
        total, first_ids = self._fetch_clip_ids_page(page=0, n=n, **kwargs)
        if not first_ids or total <= n:
            return first_ids

        remaining_pages = min(max_pages - 1, (total - 1) // n)
        page_ids: dict[int, list[str]] = {0: first_ids}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_page = {
                pool.submit(self._fetch_clip_ids_page, page=p, n=n, **kwargs): p
                for p in range(1, remaining_pages + 1)
            }
            for future in as_completed(future_to_page):
                p = future_to_page[future]
                try:
                    _, ids = future.result()
                    if ids:
                        page_ids[p] = ids
                except Exception as e:
                    warnings.warn(
                        f"_paginate_clip_ids: page {p} failed: {e}", stacklevel=2,
                    )

        all_ids: list[str] = []
        seen: set[str] = set()
        for p in sorted(page_ids.keys()):
            for cid in page_ids[p]:
                if cid not in seen:
                    seen.add(cid)
                    all_ids.append(cid)
        return all_ids[:total]

    def search_all_pages(
        self,
        max_pages: int = 100,
        n: int = 20,
        max_workers: int = 6,
        max_results: int | None = None,
        **kwargs,
    ) -> list[SearchResult]:
        """Paginate through results for a search query, in parallel.

        First page is fetched sequentially to learn ``total`` count, then
        remaining pages are fetched in parallel (``max_workers`` threads).

        Args:
            max_pages: hard ceiling on number of pages to fetch (default 100).
            n: page size (clamped to ≤20 server-side).
            max_workers: parallel page-fetcher threads (default 6).
            max_results: optional hard cap on the number of clips returned.
                When set, ``search_all_pages`` only fetches ``ceil(max_results / n)``
                pages and truncates the returned list. This honours the
                kwarg explicitly — previously ``max_results`` was silently
                ignored.

        WARNING:
            Do not use with semantic_search_text or visual_search_text —
            each page re-triggers model inference. Use export_search_clip_ids()
            instead.
        """
        if kwargs.get("semantic_search_text") or kwargs.get("visual_search_text"):
            raise ValueError(
                "search_all_pages with text-to-vector search re-triggers model inference "
                "on every page (~120s each on dev). Use export_search_clip_ids() instead."
            )

        # Honor max_results by clipping max_pages BEFORE we know `total`.
        if max_results is not None:
            if max_results <= 0:
                return []
            pages_needed = (max_results + n - 1) // n
            max_pages = max(1, min(max_pages, pages_needed))

        total, first_results = self.search(page=0, n=n, **kwargs)
        if not first_results or total <= n:
            return first_results[:max_results] if max_results else first_results

        remaining_pages = min(max_pages - 1, (total - 1) // n)
        if max_results is not None:
            # We may need to clip remaining_pages even after seeing `total`
            # (since `total` could be much smaller than `max_results`).
            remaining_pages = min(
                remaining_pages,
                ((max_results + n - 1) // n) - 1,
            )
            remaining_pages = max(0, remaining_pages)
        page_results: dict[int, list[SearchResult]] = {0: first_results}

        if remaining_pages > 0:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_to_page = {
                    pool.submit(self.search, page=p, n=n, **kwargs): p
                    for p in range(1, remaining_pages + 1)
                }
                for future in as_completed(future_to_page):
                    p = future_to_page[future]
                    try:
                        _, results = future.result()
                        if results:
                            page_results[p] = results
                    except Exception as e:
                        warnings.warn(
                            f"search_all_pages: page {p} failed: {e}", stacklevel=2,
                        )

        all_results: list[SearchResult] = []
        seen: set[str] = set()
        for p in sorted(page_results.keys()):
            for r in page_results[p]:
                if r.clip_id not in seen:
                    seen.add(r.clip_id)
                    all_results.append(r)
        if max_results is not None:
            return all_results[: max_results]
        return all_results[:total]

    # ── Convenience Search Methods ───────────────────────────────────

    def semantic_search_by_clip(
        self, clip_id: str, data_source: str | None = None, n: int = 20
    ) -> list[SearchResult]:
        """Find clips with similar Cosmos embeddings to the given clip."""
        if not clip_id or not clip_id.strip():
            raise ValueError("clip_id must be a non-empty string")
        _, results = self.search(
            semantic_search_clipid=clip_id, data_source=data_source, n=n
        )
        return results

    def semantic_search_by_text(
        self, text: str, data_source: str | None = None, n: int = 20
    ) -> tuple[int, list[SearchResult]]:
        """Find clips matching a text description via Cosmos embeddings.

        Note: SLOW on dev server (CPU-only, ~120s). Fast on production (GPU).
        MADS and MADS-1M clips have Cosmos embeddings indexed, but text-to-video
        queries may return 0 if the query doesn't match the visual distribution.
        Use caption_search() as a fast fallback for keyword-based queries.

        Returns (total_count, results) like search().
        """
        if not text or not str(text).strip():
            raise ValueError("text must be a non-empty string")
        total, results = self.search(
            semantic_search_text=text, data_source=data_source, n=n
        )
        if total == 0 and data_source and "MADS" in data_source.upper():
            warnings.warn(
                f"semantic_search_by_text returned 0 on {data_source}. "
                "The query may not match the visual distribution. "
                "Try caption_search() as a keyword fallback, or broaden "
                "the query. For training clip IDs, use data_source='MADS-1M'.",
                UserWarning,
                stacklevel=2,
            )
        return total, results

    def visual_search_by_text(
        self, text: str, data_source: str | None = None, n: int = 20
    ) -> tuple[int, list[SearchResult]]:
        """Find clips matching a text description via CLIP embeddings.

        Returns (total_count, results) like search().
        """
        if not text or not str(text).strip():
            raise ValueError("text must be a non-empty string")
        return self.search(
            visual_search_text=text, data_source=data_source, n=n
        )

    def trajectory_search_by_clip(
        self, clip_id: str, start_t: float | None = None, end_t: float | None = None,
        data_source: str | None = None, n: int = 20,
    ) -> tuple[int, list[SearchResult]]:
        """Find clips with similar driving trajectory shape.

        Returns (total_count, results) like search().
        """
        if not clip_id or not clip_id.strip():
            raise ValueError("clip_id must be a non-empty string")
        return self.search(
            trajectory_shape_clipid=clip_id,
            trajectory_shape_start_t=start_t,
            trajectory_shape_end_t=end_t,
            data_source=data_source,
            n=n,
        )

    #: Data sources known to NOT have trajectory predicates indexed.
    #: Empirically verified: predicates on these sources return 0 silently
    #: (or after a long ~80s wait). The trajectory store only ingests
    #: AV V1/V2 / Waymo / celsius — see knowledge/feature-compat.md.
    _TRAJECTORY_UNINDEXED_SOURCES = frozenset({
        "MADS",
        "MADS-1M",
        "OpenDV-YouTube",
        "MultiCountry-800K",
        "Physical AI",
        "OGameData",
        "NVIQ",
    })

    def trajectory_predicate_search(
        self, pattern: str | None = None, speed_expr: str | None = None,
        data_source: str | None = None, n: int = 20,
    ) -> tuple[int, list[SearchResult]]:
        """Search by trajectory predicate expression.

        Returns ``(total_count, results)`` like :meth:`search`.

        Args:
            pattern: Predefined name — ``'high_curvature'``, ``'stop_go'``,
                ``'hard_braking'``, ``'prolonged_stop'``, ``'idle_to_cruise'``,
                ``'high_speed_swerve'``, ``'moving_ego'``.
            speed_expr: Python-like expression on trajectory variables.
                Available vars: ``speed``, ``acceleration``, ``jerk``,
                ``curvature``, ``speed_kph``. Available funcs: ``mean``,
                ``min``, ``max``, ``sum``, ``all``, ``any``, ``len``,
                ``abs``. Examples: ``'max(speed_kph) > 120'``,
                ``'max(abs(curvature)) > 0.05'``.

        SILENT FAILURE (v1.8.2 preflight added):
            Trajectory predicates are NOT indexed on MADS-1M, MADS,
            OpenDV-YouTube, MultiCountry-800K, Physical AI, OGameData, or
            NVIQ — they return ``0`` silently (and slowly, ~50-150 s).
            We now emit a ``UserWarning`` before the request when one of
            these is the chosen ``data_source``. Use AV V1/V2/V2.2,
            Waymo, or celsius for trajectory predicates instead.
        """
        if data_source in self._TRAJECTORY_UNINDEXED_SOURCES:
            warnings.warn(
                f"trajectory_predicate_search(data_source={data_source!r}) is known "
                "to silently return 0 — trajectory predicates are not indexed on "
                "this source. Use 'AV V2 train', 'AV V1 train', 'AV V2.2. train', "
                "'Waymo train', or a celsius dataset instead. See "
                "knowledge/feature-compat.md.",
                UserWarning,
                stacklevel=2,
            )
        return self.search(
            trajectory_pattern=pattern,
            search_speed=speed_expr,
            data_source=data_source,
            n=n,
        )

    def caption_search(
        self,
        query: str,
        data_source: str | None = None,
        n: int = 20,
        mode: str = "all",
    ) -> tuple[int, list[SearchResult]]:
        """FTS5 full-text search on clip captions (Qwen2.5-7B generated).

        Fast on both dev and production servers (~1-5s).

        Args:
            query: Search query (whitespace-separated words).
            data_source: Optional data source filter (e.g. ``"MADS-1M"``).
            n: Page size, clamped to [1, 20] server-side.
            mode: ``'all'`` (default — FTS5 AND-of-words; legacy semantics)
                  or ``'any'`` (OR — delegates to ``caption_search_any``).
                  When ``mode='all'`` is used with a query of 4+ words a
                  ``UserWarning`` fires **before** the request because
                  natural-language phrases almost always return 0 from FTS5
                  AND. Switch to ``mode='any'`` for natural language.

        Returns:
            ``(total_count, results)`` like ``search()``.

        Notes:
            FTS5 with ``mode='all'`` requires every word to appear in a
            single caption. For multi-concept queries like
            ``"stroller front car"`` use ``mode='any'`` (OR) — or use
            ``export_search_clip_ids`` per concept + ``intersect_clip_id_lists``
            for true AND across concepts.
        """
        if mode not in ("all", "any"):
            raise ValueError(f"mode must be 'all' or 'any', got {mode!r}")

        if mode == "any":
            terms = [t for t in (query or "").split() if t]
            if not terms:
                raise ValueError("query must contain at least one non-empty token")
            return self.caption_search_any(terms, data_source=data_source, n=n)

        # mode == "all" (FTS5 AND): emit a guidance warning BEFORE the
        # request when the query is long enough that AND-of-words is unlikely
        # to match a single caption. The HTTP call still goes out — the warning
        # tells the agent how to recover if total comes back 0.
        word_count = len((query or "").split())
        preflight_fired = False
        if query and word_count >= 4:
            preflight_msg = (
                f"caption_search(mode='all', words={word_count}) on "
                f"{query!r} rarely matches — FTS5 AND-of-words requires every "
                "word in a single caption. Retry as: "
                f"client.caption_search({query!r}, mode='any')  "
                "(or split into per-concept queries + intersect_clip_id_lists). "
                "See knowledge/anti-patterns.md § 3."
            )
            if _strict_mode_enabled():
                raise WheelZeroResultError(preflight_msg, kind="caption_fts5_and")
            warnings.warn(preflight_msg, UserWarning, stacklevel=2)
            preflight_fired = True

        total, results = self.search(search=query, data_source=data_source, n=n)
        # Post-call warning fires when total==0 for word counts the pre-flight
        # didn't already cover (3 words; 2-word and 1-word zero results aren't
        # strong-enough signal to be worth a warning).
        if (
            total == 0
            and query
            and word_count == 3
            and not preflight_fired
        ):
            postcall_msg = (
                f"caption_search returned 0 for {query!r}. FTS5 requires ALL "
                "words in one caption. Retry as: "
                f"client.caption_search({query!r}, mode='any')  "
                "(or per-concept exports + intersect_clip_id_lists). "
                "See knowledge/anti-patterns.md § 3."
            )
            if _strict_mode_enabled():
                raise WheelZeroResultError(postcall_msg, kind="caption_fts5_and")
            warnings.warn(postcall_msg, UserWarning, stacklevel=2)
        return total, results

    def caption_embedding_search(
        self,
        query: str,
        data_source: str | None = None,
        n: int = 20,
    ) -> tuple[int, list[SearchResult]]:
        """Semantic caption search via Qwen3-Embedding-8B + FAISS.

        Distinct from ``caption_search`` (FTS5 keyword) and from
        ``semantic_search_by_text`` (Cosmos video-text). Uses the server's
        caption-embedding store to find clips whose generated captions are
        semantically close to ``query`` even when no exact words overlap.

        Server-side store: ``caption_embeddings_store.py``. URL param:
        ``caption_embed_search``. Server feature added 2026-03-25 (commit
        ``f94a5ff8``).

        Args:
            query: Natural-language query (no special syntax).
            data_source: Optional data source filter.
            n: Page size, clamped to [1, 20] server-side.

        Returns:
            ``(total_count, results)`` like ``search()``.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        return self.search(
            caption_embed_search_text=query.strip(),
            data_source=data_source,
            n=n,
        )

    def caption_search_any(
        self,
        terms: list[str],
        data_source: str | None = None,
        n: int = 20,
    ) -> tuple[int, list[SearchResult]]:
        """FTS5 caption search matching ANY of the given terms (OR logic).

        Standard caption search requires ALL words to co-occur in a caption,
        which fails for multi-concept queries like 'rain night pedestrian'.
        This method uses the server's extra_queries parameter to combine
        terms with OR logic, returning clips matching any single term.

        Args:
            terms: List of search terms/phrases. At least one required.
            data_source: Optional data source filter.
            n: Max results per page (server caps at 20).

        Returns (total_count, results) like search().

        Example:
            total, results = client.caption_search_any(
                ["rainy night", "pedestrian", "walking person"],
                data_source="MADS",
            )
        """
        if not terms:
            raise ValueError("terms must be a non-empty list")
        cleaned = [t.strip() for t in terms if isinstance(t, str) and t.strip()]
        if not cleaned:
            raise ValueError("terms must contain at least one non-empty string")
        primary = cleaned[0]
        extra = "||".join(cleaned[1:]) if len(cleaned) > 1 else None
        return self.search(
            search=primary, extra_queries=extra, data_source=data_source, n=n,
        )

    def resolve_classifier_name(
        self,
        query: str,
        *,
        min_confidence: float = 0.5,
    ) -> str | None:
        """Find the best classifier name matching a query.

        Uses tiered matching: exact (case-insensitive) > prefix > substring
        of-query-in-label > token-overlap > fuzzy (difflib). Returns the
        single best match if its confidence ≥ ``min_confidence``, else
        ``None``.

        For inspection / picking among ambiguous matches, prefer
        :meth:`resolve_classifier_candidates` which returns the top-K with
        confidence scores.

        Classifier search is case-sensitive on the server — use this to
        resolve the correct casing before calling
        :meth:`classifier_search`.

        v1.8.2: switched to read both ``trained_by_type`` (new schema, all
        sub-types) and ``untrained`` (back-compat). The previous version
        only read ``classifiers['trained']`` which is absent on current
        servers — so 109 ``trained_by_type.cosmos`` labels (including
        ``"Change lane to the right"``) were invisible and the function
        returned wrong fuzzy matches like ``"fence on the right"``.
        """
        candidates = self.resolve_classifier_candidates(query)
        if not candidates:
            return None
        best, score = candidates[0]
        if score < min_confidence:
            return None
        return best

    def resolve_classifier_candidates(
        self, query: str, max_results: int = 5,
    ) -> list[tuple[str, float]]:
        """Return ranked classifier name matches with confidence scores.

        Tiers (per-label highest score wins):
          - exact (case-insensitive)              -> 1.00
          - prefix (label starts with query)      -> 0.90
          - substring (query is in label)         -> 0.80
          - reverse substring (label is in query) -> 0.70
          - token overlap (≥1 multi-char token)   -> 0.55
          - difflib fuzzy fallback                -> 0.50

        v1.8.2 changes:
          - Reads ``trained_by_type`` sub-buckets (cosmos / caption /
            visual) AND ``untrained``, so trained classifiers are no
            longer invisible.
          - Adds reverse-substring + token-overlap tiers so natural-language
            queries like ``"change lane to the right"`` always beat
            unrelated substring matches like ``"fence on the right"``.
        """
        if not query:
            return []
        all_labels = self.list_classifier_names(include_untrained=True)
        if not all_labels:
            return []

        # Normalize: lowercase + collapse {_, -} to spaces so labels like
        # "right_lane_change" tokenize as {"right", "lane", "change"} and
        # match natural-language queries like "right lane change".
        def _norm(s: str) -> str:
            return (s.lower().replace("_", " ").replace("-", " ").strip())

        query_norm = _norm(query)
        query_tokens = {t for t in query_norm.split() if len(t) > 2}
        best_by_label: dict[str, float] = {}

        for label in all_labels:
            ll = _norm(label)
            score = 0.0
            if ll == query_norm:
                score = 1.0
            elif ll.startswith(query_norm):
                score = 0.9
            elif query_norm in ll:
                # The label fully contains the query (e.g. "lanes merging" in
                # "Lanes merging during construction").
                score = 0.8
            elif ll in query_norm:
                # The query fully contains the label (e.g. user typed
                # "find me clips with change lane to the right behaviour"
                # and the label is "Change lane to the right").
                score = 0.7
            elif query_tokens:
                label_tokens = {t for t in ll.split() if len(t) > 2}
                overlap = query_tokens & label_tokens
                if overlap and label_tokens:
                    # Combined coverage: how much of the query AND the label
                    # the overlap explains. This penalises matching a broad
                    # query against a narrow label (e.g. query has 5 words,
                    # label has 1 word, overlap=1 → high recall on label
                    # but low recall on query).
                    q_cov = len(overlap) / max(len(query_tokens), 1)
                    l_cov = len(overlap) / max(len(label_tokens), 1)
                    score = 0.55 * (q_cov + l_cov) / 2
            if score > best_by_label.get(label, 0.0):
                best_by_label[label] = score

        if not best_by_label:
            close = difflib.get_close_matches(
                query, all_labels, n=max_results, cutoff=0.4,
            )
            for m in close:
                best_by_label[m] = 0.5

        scored = sorted(best_by_label.items(), key=lambda x: -x[1])
        return scored[:max_results]

    def classifier_search(
        self, label: str, threshold: float = 0.5,
        data_source: str | None = None, n: int = 20,
    ) -> tuple[int, list[SearchResult]]:
        """Filter clips by trained classifier score.

        Production has 100+ trained classifiers. Use
        :meth:`list_classifier_names` to list them. **Classifier search is
        case-sensitive on the server.** Use :meth:`resolve_classifier_name`
        for case-insensitive lookup before searching.

        The server caps results at 20 per page. For full result sets, use
        :meth:`export_search_clip_ids` which paginates properly.

        Note on calibration:
            Classifier scores are NOT well-calibrated. ``threshold=0.5`` is
            high-recall but noisy (often 30-60% precision). For
            precision-critical work use ``0.7-0.85``. For rare-event
            mining use ``0.5`` to widen the candidate pool then VLM-validate.
            See ``knowledge/search-tool-calibration.md``.

        Note on coverage:
            Classifier inferences are NOT uniformly applied across data
            sources. ``"Change lane to the right"`` returns 1.4M on
            ``AV V2 train`` but 0 on ``MADS-1M``. Use
            :meth:`get_classifier_coverage` to map per-source counts
            BEFORE assuming a missing classifier means missing data.

        Returns ``(total_count, results)`` like :meth:`search`.
        """
        return self.search(
            classifier_select=label,
            probability_threshold=threshold,
            data_source=data_source,
            n=n,
        )

    def get_classifier_coverage(
        self,
        label: str,
        threshold: float = 0.5,
        *,
        data_sources: list[str] | None = None,
        max_workers: int = 4,
    ) -> dict[str, int]:
        """Probe per-data-source clip counts at a given classifier threshold.

        Useful before composing ``classifier_select=X data_source=Y`` —
        avoids the silent-0 trap where the classifier IS trained but has
        no inferences on the chosen source. ::

            cov = client.get_classifier_coverage("Change lane to the right")
            # {"MADS": 0, "MADS-1M": 0, "AV V1 train": 1500454, ...}

        Args:
            label: classifier name (case-sensitive — see
                :meth:`resolve_classifier_name`).
            threshold: probability cutoff (default 0.5).
            data_sources: list of sources to probe. Defaults to the most
                commonly-used set: MADS-1M, MADS, AV V1/V2 train and
                validation, Waymo train, celsius2_l3_55k, OpenDV-YouTube.
            max_workers: parallel probe threads.

        Returns:
            ``{data_source: int_count}``. Sources that erred are mapped to
            ``-1`` (so the caller can distinguish "0 matches" from "probe
            failed"). The counts are ``total`` from a single-page search,
            so they reflect the server's full-pool count.
        """
        if data_sources is None:
            data_sources = [
                "MADS-1M",
                "MADS",
                "AV V1 train",
                "AV V1 validation",
                "AV V2 train",
                "AV V2 validation",
                "AV V2.2. train",
                "Waymo train",
                "celsius2_l3_55k",
                "OpenDV-YouTube",
            ]

        def _probe(ds: str) -> tuple[str, int]:
            try:
                # Suppress the preflight warning we'd otherwise trigger
                # for KNOWN_MISSING entries — that warning is for
                # callers, not for this introspection helper.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    total, _ = self.search(
                        classifier_select=label,
                        probability_threshold=threshold,
                        data_source=ds,
                        n=1,
                    )
                return ds, int(total or 0)
            except Exception:
                return ds, -1

        out: dict[str, int] = {}
        workers = max(1, min(len(data_sources), max_workers))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_probe, ds): ds for ds in data_sources}
            for fut in as_completed(futs):
                ds, n = fut.result()
                out[ds] = n
        return out

    def find_clips_matching_classifiers(
        self,
        classifiers: list[tuple[str, float]],
        *,
        op: str = "and",
        data_source: str | None = None,
        max_per_classifier: int | None = None,
        warn_on_zero: bool = True,
    ) -> list[str]:
        """Compose multi-classifier filters via per-classifier export + set ops.

        The server's ``classifier_select`` URL param accepts only a single
        classifier. To find clips matching MULTIPLE classifiers (with
        per-classifier thresholds), this helper:

        1. exports clip IDs for each ``(label, threshold)`` separately,
        2. unions or intersects them client-side.

        Args:
            classifiers: list of ``(label, threshold)`` tuples. Labels
                must be exact case-sensitive matches — pre-resolve with
                :meth:`resolve_classifier_name` if needed.
            op: ``"and"`` (intersection — clips passing ALL classifiers)
                or ``"or"`` (union — clips passing ANY).
            data_source: pass through to per-classifier export.
            max_per_classifier: cap each export's result count. ``None``
                = no cap (subject to the server's hard 1M cap on exports).
            warn_on_zero: emit a ``UserWarning`` if any individual
                classifier returns 0 — a strong signal that the chosen
                ``data_source`` doesn't have inferences for it.

        Returns:
            sorted list of clip IDs.

        Example::

            both = client.find_clips_matching_classifiers(
                [("Change lane to the right", 0.7),
                 ("Change lane to the left",  0.7)],
                op="and",
                data_source="AV V2 train",
            )
        """
        if not classifiers:
            return []
        if op not in ("and", "or"):
            raise ValueError(f"op must be 'and' or 'or', got {op!r}")

        sets: list[set[str]] = []
        for label, threshold in classifiers:
            ids = self.export_search_clip_ids(
                classifier_select=label,
                probability_threshold=threshold,
                data_source=data_source,
            )
            if max_per_classifier is not None:
                ids = ids[:max_per_classifier]
            if not ids and warn_on_zero:
                warnings.warn(
                    f"find_clips_matching_classifiers: classifier {label!r} "
                    f"returned 0 ids (data_source={data_source!r}). The 'and' "
                    "case will collapse to 0; the 'or' case will simply skip "
                    "this term. If this is unexpected, run "
                    f"client.get_classifier_coverage({label!r}) to find a "
                    "data_source where it has inferences.",
                    UserWarning,
                    stacklevel=2,
                )
            sets.append(set(ids))

        if op == "and":
            result = sets[0]
            for s in sets[1:]:
                result &= s
        else:
            result = set()
            for s in sets:
                result |= s
        return sorted(result)

    def world_model_search(
        self, class_name: str, min_count: int = 1, max_count: int | None = None,
        max_dist: float | None = None, min_time: float | None = None,
        angle_range: str | None = None, data_source: str | None = None, n: int = 20,
    ) -> tuple[int, list[SearchResult]]:
        """Search by world model detected objects.

        Args:
            class_name: Object class (e.g. 'PEDESTRIAN_UNKNOWN', 'CAR', 'TRUCK')
            min_count/max_count: Object count range
            max_dist: Max distance in meters
            min_time: Min presence time in seconds
            angle_range: Comma-separated sectors (FRONT, FRONT_LEFT, FRONT_RIGHT, etc.)
        """
        return self.search(
            wm_class_name=class_name,
            wm_min_count=min_count,
            wm_max_count=max_count,
            wm_max_dist=max_dist,
            wm_min_time=min_time,
            wm_angle_range=angle_range,
            data_source=data_source,
            n=n,
        )

    def cluster_search(
        self, run_id: str, cluster_id: str | int,
        data_source: str | None = None, n: int = 20,
    ) -> tuple[int, list[SearchResult]]:
        """Filter clips by cluster membership from a K-means clustering run.

        Use get_clustering_status() to list available run IDs, then
        get_clustering_results() to see cluster IDs within a run.
        Returns (total_count, results) like search().

        Args:
            run_id: Clustering run identifier.
            cluster_id: Specific cluster to filter (0-indexed).
        """
        return self.search(
            cluster_run_id=run_id,
            cluster_id=str(cluster_id),
            data_source=data_source,
            n=n,
        )

    def annotation_search(
        self, labels: str, mode: str = "any",
        data_source: str | None = None, n: int = 20,
        exclude: str | None = None,
        label_types: str | None = None,
        without_annotations: bool = False,
        with_times: bool = False,
    ) -> tuple[int, list[SearchResult]]:
        """Filter clips by annotation labels (AND/OR logic).

        Returns (total_count, results) like search().

        Args:
            labels: Labels separated by '||' for OR, '&&' for AND.
                    Example: "Snow||Rain" or "Snow&&Rain"
            mode: 'any' (OR — clip has at least one label) or
                  'all' (AND — clip has all labels).
            exclude: Labels to exclude, separated by '||'.
            label_types: Filter by label type ('manual', 'auto', or both).
            without_annotations: If True, return only unannotated clips.
            with_times: If True, return only clips with timed annotations.
        """
        return self.search(
            label_filter=labels,
            filter_mode=mode,
            labels_to_exclude=exclude,
            label_types=label_types,
            without_ann=without_annotations or None,
            times=with_times or None,
            data_source=data_source,
            n=n,
        )

    def country_search(
        self, country: str | None = None,
        left_hand_driving: bool = False,
        data_source: str | None = None, n: int = 20,
    ) -> tuple[int, list[SearchResult]]:
        """Filter clips by country or driving side.

        Returns (total_count, results) like search().

        Args:
            country: Country code (e.g. 'US', 'JP', 'DE').
            left_hand_driving: If True, only left-hand traffic countries.
        """
        return self.search(
            search_country=country,
            left_hand_driving=left_hand_driving or None,
            data_source=data_source,
            n=n,
        )

    def numeric_filter_search(
        self, filter_expr: str,
        data_source: str | None = None, n: int = 20,
    ) -> tuple[int, list[SearchResult]]:
        """Filter clips by numeric eval metrics.

        Returns (total_count, results) like search().

        Args:
            filter_expr: Metric filter expression, e.g.
                         "gws_psnr>25" or "gws_lpips<0.3"
        """
        return self.search(
            numeric_filter=filter_expr,
            with_metrics=True,
            data_source=data_source,
            n=n,
        )

    def comment_search(
        self, query: str,
        data_source: str | None = None, n: int = 20,
    ) -> tuple[int, list[SearchResult]]:
        """Search clips by user comment text.

        Returns (total_count, results) like search().
        """
        return self.search(
            search_comments=query,
            data_source=data_source,
            n=n,
        )

    def sil_api_search(
        self, apis: str,
        data_source: str | None = None, n: int = 20,
    ) -> tuple[int, list[SearchResult]]:
        """Filter clips by SIL API applicability.

        Returns (total_count, results) like search().

        Args:
            apis: Comma-separated SIL API names to filter by.
        """
        return self.search(
            sil_apis=apis,
            data_source=data_source,
            n=n,
        )

    # ── VLM Judge ─────────────────────────────────────────────────────

    def vlm_judge_status(self) -> dict:
        """Check whether VLM Judge is enabled and healthy on the server.

        Returns dict with 'enabled' (bool), 'healthy' (bool), and
        'in_process' (bool). Returns {'enabled': False} if the server
        doesn't support VLM Judge or the endpoint is unavailable.
        """
        resp = self._get("/api/vlm_judge/status")
        if resp.status_code == 200:
            try:
                return resp.json()
            except (json.JSONDecodeError, ValueError):
                return {"enabled": False, "error": "invalid response"}
        if resp.status_code == 503:
            return {"enabled": False, "error": "VLM Judge disabled on server"}
        if resp.status_code == 404:
            return {"enabled": False, "error": "VLM Judge not available (server too old)"}
        return {"enabled": False, "error": f"HTTP {resp.status_code}"}

    def vlm_judge_caption_score(
        self,
        clip_id: str,
        caption: str,
        uid: str | int | None = None,
    ) -> dict:
        """Score a caption against its clip's video using VLM Judge.

        The VLM samples video frames, sends them with the caption to a
        vision-language model, and returns a quality rubric with five
        dimensions (each scored 1-10): scene, action, road_entities,
        temporal, overall — plus free-text reasoning.

        Args:
            clip_id: The clip to score against.
            caption: The caption text to evaluate.
            uid: Caption UID from the captions dict (e.g.,
                 ``result.captions["Qwen2.5-7B"][0]["uid"]``).
                 Required by the server for cache alignment. If omitted,
                 the method auto-generates a hash-based uid, but cached
                 scores won't align with the UI.

        Returns:
            Dict with 'scores' (dict of dimension→int), 'reasoning' (str),
            and 'prompt_tokens'/'response_tokens' (int) on success.
            Dict with 'error' key on failure.
        """
        if uid is None:
            uid = str(abs(hash((clip_id, caption))) % (10**9))
        params: dict[str, str] = {"clip_id": clip_id, "caption": caption,
                                  "uid": str(uid)}
        resp = self._get("/api/vlm_judge/caption_score", params=params)
        if resp.status_code == 200:
            try:
                return resp.json()
            except (json.JSONDecodeError, ValueError):
                return {"error": "invalid JSON response"}
        if resp.status_code == 503:
            return {"error": "VLM Judge disabled on server (NV_INFERENCE_API_KEY not set)"}
        try:
            return resp.json()
        except (json.JSONDecodeError, ValueError):
            return {"error": f"HTTP {resp.status_code}"}

    def vlm_judge_score_clip(self, clip_id: str) -> dict:
        """Score a clip's existing caption using VLM Judge.

        Convenience method: looks up the clip, extracts its first caption
        and uid, then scores it. Requires the clip to have captions in the
        Wheel database.

        Returns the same dict as vlm_judge_caption_score(), with an
        additional 'caption' key showing what was scored.
        """
        result = self.lookup_clip(clip_id)
        if result is None:
            return {"error": f"Clip {clip_id} not found"}
        if not result.captions:
            return {"error": f"Clip {clip_id} has no captions"}
        for model_caps in result.captions.values():
            if isinstance(model_caps, list):
                for entry in model_caps:
                    if isinstance(entry, dict) and entry.get("caption"):
                        score = self.vlm_judge_caption_score(
                            clip_id, entry["caption"],
                            uid=entry.get("uid"),
                        )
                        score["caption"] = entry["caption"][:200]
                        return score
        return {"error": f"Clip {clip_id} has captions but no extractable text"}

    #: Conservative chunk size for vlm_judge_validate_search.
    #: Each clip ID is ~37 chars (UUID-formatted) and the URL also carries
    #: the ``query`` and standard query overhead. ~30 IDs keeps the GET
    #: URL well below the typical ~8 KB server limit. The previous
    #: behaviour silently truncated to 1 result at ~50+ clips.
    _VLM_JUDGE_CHUNK = 30

    #: HTTP statuses worth retrying for VLM Judge endpoints.
    _VLM_JUDGE_RETRY_STATUSES = frozenset({502, 503, 504})

    def vlm_judge_validate_search(
        self,
        query: str,
        clip_ids: list[str],
        *,
        chunk_size: int | None = None,
        max_attempts: int = 3,
        backoff_seconds: float = 30.0,
        on_chunk_complete: Any | None = None,
        progress_fn: Any | None = None,
    ) -> list[dict]:
        """Validate whether search results actually match the query using VLM.

        For each clip, the VLM watches the video and determines whether it
        truly matches the search query. Useful for filtering false positives
        or measuring search precision.

        v1.8.2 changes (fix two critical bugs flagged in the 2026-05-11
        Li Auto session):

        1. **Auto-chunked**: 1000 UUID-formatted clip IDs produce a ~37 KB
           query string. Most servers cap GETs at ~8 KB and silently return
           only the first result with no warning. We now chunk into
           ``chunk_size`` (default 30) and concatenate per-chunk results.
        2. **Retry on transient failures**: HTTP 502/503/504 are common
           under sustained VLM load. We now retry up to ``max_attempts``
           per chunk with exponential backoff (``backoff_seconds *
           2**attempt``). Final-failure chunks are returned with an
           ``error`` field so the caller can drop or re-run them.

        Args:
            query: The search query to validate against.
            clip_ids: clip IDs to validate. No upper limit — chunked.
            chunk_size: clips per HTTP call. Default ~30; only override
                if you know the server's URL limit is different.
            max_attempts: per-chunk retries on 502/503/504 (default 3).
            backoff_seconds: base sleep between retries; scales as
                ``backoff_seconds * 2**attempt`` (default 30s).
            on_chunk_complete: optional callback ``fn(chunk_idx, results)``
                invoked after each successful chunk — useful for
                incremental save (kill-safe long runs).
            progress_fn: optional ``fn(completed, total)`` for UI progress.

        Returns:
            List of dicts (length ≤ ``len(clip_ids)``), each with
            ``clip_id`` (str), ``match`` (bool), ``reasoning`` (str), and
            ``analysis`` (str). On per-chunk failure after all retries,
            the chunk's clip IDs are returned as ``[{"clip_id": cid,
            "error": "..."}, ...]``. The empty list is returned only when
            ``clip_ids`` is empty.
        """
        if not clip_ids:
            return []

        if chunk_size is None:
            chunk_size = self._VLM_JUDGE_CHUNK
        chunk_size = max(1, int(chunk_size))

        results: list[dict] = []
        total_chunks = (len(clip_ids) + chunk_size - 1) // chunk_size
        for chunk_idx in range(total_chunks):
            start = chunk_idx * chunk_size
            chunk = clip_ids[start:start + chunk_size]
            chunk_results = self._vlm_judge_validate_chunk(
                query, chunk,
                max_attempts=max_attempts,
                backoff_seconds=backoff_seconds,
            )
            results.extend(chunk_results)
            if on_chunk_complete is not None:
                try:
                    on_chunk_complete(chunk_idx, chunk_results)
                except Exception as e:
                    warnings.warn(
                        f"on_chunk_complete callback raised at chunk {chunk_idx}: {e}",
                        stacklevel=2,
                    )
            if progress_fn is not None:
                try:
                    progress_fn(min(start + len(chunk), len(clip_ids)), len(clip_ids))
                except Exception:
                    pass

        return results

    def _vlm_judge_validate_chunk(
        self,
        query: str,
        chunk: list[str],
        *,
        max_attempts: int,
        backoff_seconds: float,
    ) -> list[dict]:
        """Single chunk of vlm_judge_validate_search with retry/backoff.

        On final failure, returns one error dict per clip_id so the caller
        can re-run that chunk or drop it without losing the alignment with
        the input clip_ids.
        """
        last_status: int | None = None
        last_error: str | None = None
        for attempt in range(max_attempts):
            params = {"query": query, "clip_ids": ",".join(chunk)}
            try:
                resp = self._get("/api/vlm_judge/validate_search", params=params)
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt < max_attempts - 1:
                    _time_mod.sleep(backoff_seconds * (2 ** attempt))
                    continue
                break
            last_status = resp.status_code
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except (json.JSONDecodeError, ValueError):
                    last_error = "invalid JSON response"
                    break
                results = data.get("results", data) if isinstance(data, dict) else data
                if isinstance(results, list):
                    # Sanity: warn if the server silently truncated the chunk.
                    if len(chunk) >= 2 and len(results) <= 1:
                        warnings.warn(
                            f"vlm_judge_validate_search chunk of {len(chunk)} clips "
                            f"returned only {len(results)} result(s). The server's "
                            f"GET URL limit may have truncated; reduce chunk_size "
                            f"(current default {self._VLM_JUDGE_CHUNK}).",
                            stacklevel=3,
                        )
                    return results
                last_error = f"unexpected response shape: {type(results).__name__}"
                break
            if resp.status_code == 503:
                # Server-disabled — no point retrying
                return [
                    {"clip_id": cid, "error": "VLM Judge disabled on server"}
                    for cid in chunk
                ]
            if resp.status_code in self._VLM_JUDGE_RETRY_STATUSES:
                last_error = f"HTTP {resp.status_code}"
                if attempt < max_attempts - 1:
                    sleep_s = backoff_seconds * (2 ** attempt)
                    warnings.warn(
                        f"vlm_judge_validate_search HTTP {resp.status_code} "
                        f"(attempt {attempt + 1}/{max_attempts}); retrying in {sleep_s:.0f}s",
                        stacklevel=3,
                    )
                    _time_mod.sleep(sleep_s)
                    continue
                break
            # Non-retryable HTTP failure
            last_error = f"HTTP {resp.status_code}"
            break

        return [
            {"clip_id": cid, "error": last_error or f"HTTP {last_status}"}
            for cid in chunk
        ]

    def vlm_judge_search_and_validate(
        self,
        query: str,
        data_source: str | None = None,
        n: int = 20,
    ) -> tuple[list[SearchResult], list[dict]]:
        """Search for clips and validate the results using VLM Judge.

        Convenience method: runs a caption search, then feeds the result
        clip IDs through VLM Judge validation to identify true matches.

        Returns:
            Tuple of (search_results, validation_results). Each validation
            entry has 'clip_id', 'match' (bool), 'reasoning', 'analysis'.
        """
        _, results = self.caption_search(query, data_source=data_source, n=n)
        if not results:
            return results, []
        clip_ids = [r.clip_id for r in results]
        validations = self.vlm_judge_validate_search(query, clip_ids)
        return results, validations

    # ── Version Check ────────────────────────────────────────────────

    @staticmethod
    def check_sdk_version(timeout: int = 10) -> dict:
        """Check whether this SDK is up to date with the deployed version.

        Fetches the remote skill.md frontmatter and compares its version
        against SDK_VERSION. No authentication required.

        Returns:
            Dict with 'local' (str), 'remote' (str), 'up_to_date' (bool),
            and optionally 'update_command' (str) if an update is available.
            On network error, returns 'error' key instead.
        """
        try:
            resp = requests.get(_SKILL_URL, timeout=timeout)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            return {"local": SDK_VERSION, "remote": None,
                    "error": f"Cannot reach update server: {e}"}

        if resp.status_code != 200:
            return {"local": SDK_VERSION, "remote": None,
                    "error": f"HTTP {resp.status_code} fetching skill.md"}

        remote_version = None
        for line in resp.text.splitlines()[:10]:
            line = line.strip()
            if line.startswith("version:"):
                remote_version = line.split(":", 1)[1].strip()
                break

        if remote_version is None:
            return {"local": SDK_VERSION, "remote": None,
                    "error": "Could not parse version from remote skill.md"}

        def _parse_semver(v: str) -> tuple[int, ...]:
            try:
                return tuple(int(x) for x in v.split("."))
            except (ValueError, AttributeError):
                return (0,)

        local_t = _parse_semver(SDK_VERSION)
        remote_t = _parse_semver(remote_version)
        up_to_date = local_t >= remote_t
        result: dict[str, Any] = {
            "local": SDK_VERSION,
            "remote": remote_version,
            "up_to_date": up_to_date,
        }
        if local_t > remote_t:
            result["note"] = "Local version is newer than published"
        elif not up_to_date:
            result["update_command"] = (
                "Re-run the curl install from skill.md to update:\n"
                f"  curl -sf {_SKILL_URL} | head -3  # check new version\n"
                "  # Then re-run the full curl install block from the skill file"
            )
        return result

    @staticmethod
    def update_sdk(timeout: int = 30) -> dict:
        """Download the latest wheel_client.py from the CDN, replacing this file.

        Resolves the target path from ``__file__`` so it works correctly in
        both curl-install (``sil_wheel/``) and git-clone (``src/``) layouts.
        Uses atomic write (temp file + rename) to avoid partial overwrites.

        Checks version first; skips download if already up to date.
        Returns dict with 'updated' (bool), 'from_version', 'to_version',
        and 'files_updated' (list of paths written).
        """
        version_info = WheelClient.check_sdk_version(timeout=timeout)
        if version_info.get("error"):
            return {"updated": False, "error": version_info["error"]}
        if version_info["up_to_date"]:
            return {"updated": False, "reason": "already up to date",
                    "version": version_info["local"]}

        base = _SKILL_URL.rsplit("/", 1)[0]
        url = f"{base}/sil_wheel_agent/wheel_client.py"
        target = Path(__file__).resolve()

        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code != 200:
                return {"updated": False,
                        "error": f"HTTP {resp.status_code} fetching {url}"}
            if len(resp.text) < 1000 or "class WheelClient" not in resp.text:
                return {"updated": False,
                        "error": "Downloaded file appears corrupt or truncated"}
            tmp = target.with_suffix(".py.tmp")
            tmp.write_text(resp.text, encoding="utf-8")
            tmp.replace(target)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            return {"updated": False, "error": f"Download failed: {e}"}
        except OSError as e:
            return {"updated": False, "error": f"File write failed: {e}"}

        return {
            "updated": True,
            "from_version": version_info["local"],
            "to_version": version_info["remote"],
            "files_updated": [str(target)],
            "note": "Restart your Python process to use the new version",
        }

    def lookup_clip(self, clip_id: str) -> SearchResult | None:
        """Look up a single clip by its exact ID.

        Uses the same ``/videos`` endpoint as ``search(search_clipid=clip_id)``
        but with cleaner intent and a singleton return type. Recommended over
        ``search`` for exact-key retrieval — see ``knowledge/anti-patterns.md``.
        """
        if not clip_id or not clip_id.strip():
            return None
        _lookup_clip_in_progress.value = True
        try:
            _, results = self.search(search_clipid=clip_id)
        finally:
            _lookup_clip_in_progress.value = False
        return results[0] if results else None

    def lookup_clips_batch(
        self,
        clip_ids: list[str],
        max_workers: int = 8,
        progress_fn: Any | None = None,
    ) -> dict[str, SearchResult | None]:
        """Look up metadata for multiple clips in parallel.

        Returns {clip_id: SearchResult} for each found clip (None if not found).
        Each clip ID requires one HTTP request — for 500+ clips, consider using
        export_search_clip_ids() or search_all_pages() instead.

        Args:
            max_workers: Max parallel threads (default 8).
            progress_fn: Optional callback(completed: int, total: int).
        """
        if not clip_ids:
            return {}
        if len(clip_ids) > 100:
            # Real-world cliff during a May-2026 server-load incident was
            # ~30s/clip even at max_workers=8 — well before the previous 500
            # threshold. Drop the warning to 100 so callers see it sooner.
            warnings.warn(
                f"lookup_clips_batch with {len(clip_ids)} clips makes {len(clip_ids)} "
                "HTTP requests (one /videos call per clip). Under server load this "
                "can be very slow (>10s/clip even at max_workers=8). Prefer:\n"
                "  - search_all_pages(..., max_results=...) for clips matching a query,\n"
                "  - export_search_clip_ids() for full-dataset id-only exports, or\n"
                "  - splitting into smaller batches with progress_fn for visibility.",
                stacklevel=2,
            )
        results: dict[str, SearchResult | None] = {}
        workers = min(len(clip_ids), max_workers)
        completed = 0

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_cid = {
                pool.submit(self.lookup_clip, cid): cid for cid in clip_ids
            }
            for future in as_completed(future_to_cid):
                cid = future_to_cid[future]
                try:
                    results[cid] = future.result()
                except Exception:
                    results[cid] = None
                completed += 1
                if progress_fn:
                    progress_fn(completed, len(clip_ids))
        return results

    # ── Metrics & Leaderboard ────────────────────────────────────────

    def get_metrics(
        self,
        data_source: str | None = None,
        *,
        reduction: str = "mean",
        with_same_clips: bool = False,
        **search_kwargs: Any,
    ) -> dict:
        """Get aggregate metrics over the search-filtered clip set.

        The server's ``/metrics`` endpoint accepts the full ``SearchFilters``
        plus ``reduction`` and ``with_same_clips``. Previously the client
        only forwarded ``data_source``, so any narrowing by classifier /
        caption / annotation / etc. was silently dropped.

        Args:
            data_source: Optional data source filter (alias for
                ``search_kwargs["data_source"]``; both are accepted).
            reduction: Aggregation across clips. Server default ``"mean"``;
                other valid values depend on server config.
            with_same_clips: If ``True``, server intersects all model
                evaluations to the same clip set before aggregating.
            **search_kwargs: Any kwarg accepted by ``search()`` (e.g.
                ``classifier_select``, ``probability_threshold``, ``search``).

        Returns:
            Dict with ``metrics`` keyed by model name. Cached for 5 min keyed
            by the full param set (so different filters get different entries).
        """
        if data_source is not None and "data_source" not in search_kwargs:
            search_kwargs["data_source"] = data_source
        # Cache key includes all params so filtered/unfiltered don't collide.
        cache_key = f"metrics:{reduction}:{with_same_clips}:" + json.dumps(
            sorted(search_kwargs.items()), default=str
        )
        return self._get_cached(
            cache_key,
            lambda: self._fetch_metrics(
                reduction=reduction,
                with_same_clips=with_same_clips,
                **search_kwargs,
            ),
        )

    def _fetch_metrics(
        self,
        *,
        reduction: str = "mean",
        with_same_clips: bool = False,
        **search_kwargs: Any,
    ) -> dict:
        # Reuse _build_search_params so probability_threshold etc. translations
        # are applied identically to /videos and /metrics.
        params = self._build_search_params(page=0, n=20, **search_kwargs)
        params.pop("page", None)
        params.pop("n", None)
        params.setdefault("project_source", "")
        params["reduction"] = reduction
        if with_same_clips:
            params["with_same_clips"] = "true"
        resp = self._get("metrics", params=params)
        if resp.status_code == 200:
            try:
                return resp.json()
            except json.JSONDecodeError:
                return {"raw": resp.text[:500]}
        return {"error": resp.status_code}

    def get_per_clip_metrics(
        self,
        model_name: str,
        *,
        with_same_clips: bool = False,
        **search_kwargs: Any,
    ) -> dict:
        """Get per-clip metric matrix for a specific model.

        Server route ``/per_clip_metrics`` accepts the full ``SearchFilters``
        in addition to ``model_name`` and ``with_same_clips``. The previous
        client signature dropped all filters silently.
        """
        params = self._build_search_params(page=0, n=20, **search_kwargs)
        params.pop("page", None)
        params.pop("n", None)
        params["model_name"] = model_name
        if with_same_clips:
            params["with_same_clips"] = "true"
        resp = self._get("per_clip_metrics", params=params)
        if resp.status_code == 200:
            try:
                return resp.json()
            except json.JSONDecodeError:
                return {"raw": resp.text[:500]}
        return {"error": resp.status_code}

    # ── Export ────────────────────────────────────────────────────────

    def export_search_clip_ids(
        self,
        max_clips: int = 1_000_000,
        progress_fn: Any | None = None,
        **search_kwargs,
    ) -> list[str]:
        """Run a search and export all matching clip IDs.

        Tries the fast /current_search.csv endpoint first (passing search
        params directly — the server builds filters from query params, not
        session state). Falls back to parallel page fetching if CSV is empty.

        Server-side limits: the CSV endpoint silently returns empty for >500k
        matching clips (server hard cap). For such cases, use multiple narrower
        searches and merge, or use list_data_source_clip_ids() for S3-based
        full-dataset export.

        Args:
            max_clips: Maximum clip IDs to return (default 1M). If <= 0, returns [].
            progress_fn: Optional callback(count: int) called every 10k clips.
        """
        if max_clips <= 0:
            return []
        total, _ = self.search(n=1, **search_kwargs)
        if total == 0 and search_kwargs:
            return []
        csv_params = self._build_search_params(page=0, n=1, **search_kwargs)
        csv_params.pop("page", None)
        csv_params.pop("n", None)
        resp = self._get("current_search.csv", params=csv_params)
        clip_ids: list[str] = []
        seen_ids: set[str] = set()
        if resp.status_code == 200 and resp.content:
            content = resp.content
            try:
                if content[:2] == b"\x1f\x8b":
                    content = gzip.decompress(content)
                text = content.decode("utf-8-sig")
            except (gzip.BadGzipFile, OSError, UnicodeDecodeError, _zlib_error):
                text = ""
            if text:
                reader = csv.reader(io.StringIO(text))
                header_skipped = False
                for row in reader:
                    if not header_skipped:
                        header_skipped = True
                        continue
                    if row:
                        cid = row[0]
                        if cid not in seen_ids:
                            seen_ids.add(cid)
                            clip_ids.append(cid)
                            if progress_fn and len(clip_ids) % 10_000 == 0:
                                progress_fn(len(clip_ids))
                            if len(clip_ids) >= max_clips:
                                break

        if total > 0 and not clip_ids:
            can_paginate = not (
                search_kwargs.get("semantic_search_text")
                or search_kwargs.get("visual_search_text")
            )
            if can_paginate:
                n_per_page = min(max(search_kwargs.get("n", 20), 1), 20)
                max_pages = min((max_clips - 1) // n_per_page + 1, 2000)
                page_cap = max_pages * n_per_page
                if total > page_cap:
                    warnings.warn(
                        f"export_search_clip_ids: pagination fallback limited to "
                        f"{page_cap:,} of {total:,} matching clips (2000 page cap). "
                        f"Use multiple narrower searches to export more.",
                        stacklevel=2,
                    )
                clip_ids = self._paginate_clip_ids(
                    max_pages=max_pages, n=n_per_page, **search_kwargs,
                )[:max_clips]

        # Detect server-side hard cap: when ``total`` exceeds 1M and the
        # CSV path returned exactly 1M, the server has truncated. Surface
        # this so callers don't silently lose 30%+ of their result set.
        # See knowledge/server-caps.md.
        SERVER_EXPORT_CAP = 1_000_000
        if (
            total > SERVER_EXPORT_CAP
            and len(clip_ids) >= SERVER_EXPORT_CAP
            and len(clip_ids) >= max_clips
        ):
            warnings.warn(
                f"export_search_clip_ids: server returned {len(clip_ids):,} ids "
                f"out of {total:,} matching clips — that's the server's hard "
                "cap. To export the full set, split your filter into smaller "
                "buckets (e.g. by sub-data-source, by classifier threshold "
                "tier, or by time window) and union the results with "
                "WheelClient.merge_clip_id_lists(). See knowledge/server-caps.md.",
                UserWarning,
                stacklevel=2,
            )

        return clip_ids

    def export_annotations_csv(self) -> bytes:
        """Export all annotations as gzipped CSV."""
        resp = self._get("annotations.csv")
        return resp.content if resp.status_code == 200 else b""

    # ── Upload / Labeling (dev server only) ──────────────────────────

    @staticmethod
    def _validate_no_protocol_chars(*values: str) -> None:
        """Raise ValueError if any value contains protocol-breaking characters.

        Blocks '::' (protocol separator) and control characters that could
        inject additional commands via newline splitting.
        """
        for v in values:
            s = str(v)
            if "::" in s:
                raise ValueError(f"Value contains '::' which is the protocol separator: {v!r}")
            if any(c in s for c in ("\n", "\r", "\0")):
                raise ValueError(f"Value contains control characters: {v!r}")

    @staticmethod
    def _validate_non_empty(*values: str, names: tuple[str, ...] = ()) -> None:
        """Raise ValueError if any value is empty or whitespace-only."""
        for i, v in enumerate(values):
            if not v or not str(v).strip():
                label = names[i] if i < len(names) else f"argument {i}"
                raise ValueError(f"{label} must be non-empty")

    def upload_labels(
        self,
        clip_ids: list[str],
        label: str,
        project: str = "GWS Curation",
        batch_size: int = 500,
        max_workers: int = 4,
    ) -> list[dict]:
        """Mass-label clips. Batches to avoid URL length limits.

        Batches are POSTed in parallel when there are multiple batches.
        WARNING: Only use on dev server. This server is read-only (WHEEL_READONLY).
        """
        if self.is_production:
            print("WARNING: upload_labels called on a read-only server — skipping. "
                  "Use dev server for write operations.", file=sys.stderr)
            return [{"error": "server is read-only (set via WHEEL_READONLY)"}]
        self._validate_non_empty(label, names=("label",))
        self._validate_no_protocol_chars(label, project)
        for cid in clip_ids:
            self._validate_no_protocol_chars(cid)

        batches = [
            (i, clip_ids[i : i + batch_size])
            for i in range(0, len(clip_ids), batch_size)
        ]
        if not batches:
            return []

        def _post_batch(item: tuple[int, list[str]]) -> dict:
            i, batch = item
            body = f"mass_label::{label}::{','.join(batch)}::{project}"
            resp = self._post("", data=body)
            return {"batch": i, "status": resp.status_code, "count": len(batch)}

        if len(batches) == 1:
            return [_post_batch(batches[0])]

        results: list[dict | None] = [None] * len(batches)
        with ThreadPoolExecutor(max_workers=min(len(batches), max_workers)) as pool:
            future_to_idx = {
                pool.submit(_post_batch, b): idx for idx, b in enumerate(batches)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    i, batch = batches[idx]
                    results[idx] = {"batch": i, "status": -1, "error": str(e), "count": len(batch)}
        return [r for r in results if r is not None]

    def upload_annotations(
        self,
        clip_ids: list[str],
        annotation_name: str,
        values: list[float | str] | None = None,
        project: str = "GWS Curation",
        batch_size: int = 500,
        max_workers: int = 4,
    ) -> list[dict]:
        """Upload annotations with optional numeric values per clip.

        Batches are POSTed in parallel when there are multiple batches.
        WARNING: Only use on dev server. This server is read-only (WHEEL_READONLY).
        """
        if self.is_production:
            print("WARNING: upload_annotations called on a read-only server — skipping.", file=sys.stderr)
            return [{"error": "server is read-only (set via WHEEL_READONLY)"}]
        self._validate_non_empty(annotation_name, names=("annotation_name",))
        if values is not None:
            if len(values) != len(clip_ids):
                raise ValueError(
                    f"values length ({len(values)}) != clip_ids length ({len(clip_ids)})"
                )
            for i, v in enumerate(values):
                if v is None:
                    raise ValueError(f"values[{i}] is None — use a placeholder or omit values")
        self._validate_no_protocol_chars(annotation_name, project)
        for cid in clip_ids:
            self._validate_no_protocol_chars(cid)
        if values is not None:
            for v in values:
                if isinstance(v, str):
                    self._validate_no_protocol_chars(v)

        if not clip_ids:
            return []

        batches: list[tuple[int, list[str], list[float | str] | None]] = []
        for i in range(0, len(clip_ids), batch_size):
            batch_ids = clip_ids[i : i + batch_size]
            batch_vals = values[i : i + batch_size] if values is not None else None
            batches.append((i, batch_ids, batch_vals))

        def _post_batch(item: tuple[int, list[str], list[float | str] | None]) -> dict:
            i, batch_ids, batch_vals = item
            clip_ids_str = ",".join(batch_ids)
            values_str = ",".join(str(v) for v in batch_vals) if batch_vals is not None else ""
            # Protocol: upload_annotations::<clip_ids>::<annotation>::<start_time (unused)>::<end_time (unused)>::<project>::<values>
            body = f"upload_annotations::{clip_ids_str}::{annotation_name}::::::{project}::{values_str}"
            resp = self._post("", data=body)
            return {"batch": i, "status": resp.status_code, "count": len(batch_ids)}

        if len(batches) == 1:
            return [_post_batch(batches[0])]

        results: list[dict | None] = [None] * len(batches)
        with ThreadPoolExecutor(max_workers=min(len(batches), max_workers)) as pool:
            future_to_idx = {
                pool.submit(_post_batch, b): idx for idx, b in enumerate(batches)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    i, batch_ids, _ = batches[idx]
                    results[idx] = {"batch": i, "status": -1, "error": str(e), "count": len(batch_ids)}
        return [r for r in results if r is not None]

    def train_classifier(
        self,
        label: str,
        n_negative_samples: int = 100,
        negative_labels: str = "",
        use_autolabels: bool = True,
    ) -> dict:
        """Trigger classifier training on the server.

        Trains logistic regression on Cosmos embeddings (768-dim).
        The label must exist in annotations first (upload_labels).
        Requires admin role. WARNING: Only use on dev server.
        """
        if self.is_production:
            print("WARNING: train_classifier called on a read-only server — skipping.", file=sys.stderr)
            return {"error": "server is read-only (set via WHEEL_READONLY)"}
        if not label or not label.strip():
            raise ValueError("label must be a non-empty string")
        if n_negative_samples < 1:
            raise ValueError(f"n_negative_samples must be >= 1, got {n_negative_samples}")
        self._validate_no_protocol_chars(label, negative_labels)

        use_auto = "true" if use_autolabels else "false"
        body = f"train_classifier::{label}::{n_negative_samples}::{negative_labels}::{use_auto}"
        resp = self._post("", data=body)
        return {"status": resp.status_code, "text": resp.text[:200]}

    def auto_label_search(
        self,
        label: str,
        project: str = "GWS Curation",
        label_type: str = "",
        n_pages: int | None = None,
        n_clips: int | None = None,
    ) -> dict:
        """Label ALL clips in the current search result set. Dev server only.

        First run a search(), then call this to apply a label to every matching clip.

        Args:
            label_type: Annotation type (e.g. 'manual', 'auto'). Empty for default.
            n_pages: Limit labeling to first N pages of results.
            n_clips: Limit labeling to first N clips of results.
        """
        if self.is_production:
            print("WARNING: auto_label called on a read-only server — skipping.", file=sys.stderr)
            return {"error": "server is read-only (set via WHEEL_READONLY)"}
        self._validate_non_empty(label, names=("label",))
        self._validate_no_protocol_chars(label, project)
        if label_type:
            self._validate_no_protocol_chars(label_type)
        pages_str = str(n_pages) if n_pages is not None else ""
        clips_str = str(n_clips) if n_clips is not None else ""
        body = f"auto_label::::{label}::{label_type}::{pages_str}::{clips_str}::{project}"
        resp = self._post("", data=body)
        return {"status": resp.status_code, "text": resp.text[:200]}

    def rename_label(self, old_name: str, new_name: str, project: str = "Alpamayo") -> dict:
        """Rename a label. Dev server only.

        Args:
            project: Project scope for the rename (server requires this).
        """
        if self.is_production:
            print("WARNING: rename_label called on a read-only server — skipping.", file=sys.stderr)
            return {"error": "server is read-only (set via WHEEL_READONLY)"}
        self._validate_non_empty(old_name, new_name, names=("old_name", "new_name"))
        self._validate_no_protocol_chars(old_name, new_name, project)
        body = f"rename_label::{old_name}::{new_name}::{project}"
        resp = self._post("", data=body)
        return {"status": resp.status_code, "text": resp.text[:200]}

    def delete_label(self, label: str, project: str = "Alpamayo") -> dict:
        """Delete a label entirely. Dev server only. USE WITH CAUTION.

        Args:
            project: Project scope for the deletion (server requires this).
        """
        if self.is_production:
            print("WARNING: delete_label called on a read-only server — skipping.", file=sys.stderr)
            return {"error": "server is read-only (set via WHEEL_READONLY)"}
        self._validate_non_empty(label, names=("label",))
        self._validate_no_protocol_chars(label, project)
        body = f"delete_label::{label}::_::{project}"
        resp = self._post("", data=body)
        return {"status": resp.status_code, "text": resp.text[:200]}

    _VALID_CLUSTERING_METRICS = frozenset({"cosine", "euclidean", "manhattan"})

    def merge_label(
        self, old_labels: list[str], new_label: str, project: str = "Alpamayo",
    ) -> dict:
        """Merge multiple labels into a single new label. Dev server only.

        All clips annotated with any of the old labels will get the new label.
        Old labels are removed.

        Args:
            old_labels: Labels to merge (comma-separated in protocol).
            new_label: Target label name.
            project: Project scope.
        """
        if self.is_production:
            print("WARNING: merge_label called on a read-only server — skipping.", file=sys.stderr)
            return {"error": "server is read-only (set via WHEEL_READONLY)"}
        if not old_labels:
            raise ValueError("old_labels must not be empty")
        self._validate_non_empty(new_label, names=("new_label",))
        self._validate_no_protocol_chars(new_label, project)
        for lbl in old_labels:
            self._validate_non_empty(lbl, names=("old_label",))
            self._validate_no_protocol_chars(lbl)
        old_str = ",".join(old_labels)
        self._validate_no_protocol_chars(old_str)
        body = f"merge_label::{old_str}::{new_label}::{project}"
        resp = self._post("", data=body)
        return {"status": resp.status_code, "text": resp.text[:200]}

    def add_annotation(
        self, clip_id: str, label: str, project: str = "Alpamayo",
    ) -> dict:
        """Add a single annotation to a clip. Dev server only."""
        if self.is_production:
            print("WARNING: add_annotation called on a read-only server — skipping.", file=sys.stderr)
            return {"error": "server is read-only (set via WHEEL_READONLY)"}
        self._validate_non_empty(clip_id, label, names=("clip_id", "label"))
        self._validate_no_protocol_chars(clip_id, label, project)
        body = f"add::{clip_id}::{label}::{project}"
        resp = self._post("", data=body)
        return {"status": resp.status_code, "text": resp.text[:200]}

    def remove_annotation(
        self, clip_id: str, label: str, project: str = "Alpamayo",
    ) -> dict:
        """Remove a single annotation from a clip. Dev server only."""
        if self.is_production:
            print("WARNING: remove_annotation called on a read-only server — skipping.", file=sys.stderr)
            return {"error": "server is read-only (set via WHEEL_READONLY)"}
        self._validate_non_empty(clip_id, label, names=("clip_id", "label"))
        self._validate_no_protocol_chars(clip_id, label, project)
        body = f"remove::{clip_id}::{label}::{project}"
        resp = self._post("", data=body)
        return {"status": resp.status_code, "text": resp.text[:200]}

    def update_annotation_times(
        self, clip_id: str, label: str,
        start_time: float | None = None, end_time: float | None = None,
        project: str = "Alpamayo",
    ) -> dict:
        """Update time range on an annotation. Dev server only.

        Args:
            start_time: Start time in seconds within the clip.
            end_time: End time in seconds within the clip.
        """
        if self.is_production:
            print("WARNING: update_annotation_times called on a read-only server — skipping.", file=sys.stderr)
            return {"error": "server is read-only (set via WHEEL_READONLY)"}
        self._validate_non_empty(clip_id, label, names=("clip_id", "label"))
        self._validate_no_protocol_chars(clip_id, label, project)
        st = str(start_time) if start_time is not None else ""
        et = str(end_time) if end_time is not None else ""
        if st:
            self._validate_no_protocol_chars(st)
        if et:
            self._validate_no_protocol_chars(et)
        body = f"update_times::{clip_id}::{label}::{st}::{et}::{project}"
        resp = self._post("", data=body)
        return {"status": resp.status_code, "text": resp.text[:200]}

    def verify_annotation(
        self, clip_id: str, label: str, project: str = "Alpamayo",
    ) -> dict:
        """Mark an annotation as verified. Dev server only."""
        if self.is_production:
            print("WARNING: verify_annotation called on a read-only server — skipping.", file=sys.stderr)
            return {"error": "server is read-only (set via WHEEL_READONLY)"}
        self._validate_non_empty(clip_id, label, names=("clip_id", "label"))
        self._validate_no_protocol_chars(clip_id, label, project)
        body = f"verify::{clip_id}::{label}::{project}"
        resp = self._post("", data=body)
        return {"status": resp.status_code, "text": resp.text[:200]}

    def delete_clustering(self, run_id: str) -> dict:
        """Delete a clustering run. Dev server only."""
        if self.is_production:
            print("WARNING: delete_clustering called on a read-only server — skipping.", file=sys.stderr)
            return {"error": "server is read-only (set via WHEEL_READONLY)"}
        self._validate_no_protocol_chars(run_id)
        body = f"delete_clustering::{run_id}"
        resp = self._post("", data=body)
        return {"status": resp.status_code, "text": resp.text[:200]}

    def run_clustering(
        self, n_clusters: int = 100, metric: str = "cosine",
        max_points: int | None = None,
    ) -> dict:
        """Trigger a clustering run on the current search results. Dev server only.

        Runs K-means on Cosmos embeddings for the current search result set.
        When the server is configured with ``--captions_db``, the clustering
        job also extracts per-cluster TF-IDF topic keywords and (if an LLM key
        is configured) one-phrase LLM descriptions — both available via
        :py:meth:`get_clustering_results` / :py:meth:`get_cluster_topics` /
        :py:meth:`summarize_clustering_run`.

        Args:
            n_clusters: Number of clusters (2-10000).
            metric: Distance metric — 'cosine', 'euclidean', or 'manhattan'.
                    Maps to the server's 'spherical' boolean: cosine→true, others→false.
            max_points: Max embeddings to cluster (None for all).
        """
        if self.is_production:
            print("WARNING: run_clustering called on a read-only server — skipping.", file=sys.stderr)
            return {"error": "server is read-only (set via WHEEL_READONLY)"}
        if not 2 <= n_clusters <= 10000:
            raise ValueError(f"n_clusters must be 2-10000, got {n_clusters}")
        if metric not in self._VALID_CLUSTERING_METRICS:
            raise ValueError(
                f"metric must be one of {sorted(self._VALID_CLUSTERING_METRICS)}, got {metric!r}"
            )
        spherical = "true" if metric == "cosine" else "false"
        mp = str(max_points) if max_points is not None else ""
        # Protocol: run_clustering::<data_source (empty=current)>::<n_clusters>::<spherical>::<max_points>
        body = f"run_clustering::::{n_clusters}::{spherical}::{mp}"
        resp = self._post("", data=body)
        return {"status": resp.status_code, "text": resp.text[:200]}

    # ── 3D Reconstruction (NuRec) ────────────────────────────────────

    NUREC_NCORE_PATH = "ncore-lidar-model-static-full/{clip_id}/"
    NUREC_NCORE_JSON = "/media/data1/datasets/ncore/lidar-model-static-full-nrm/{clip_id}/{clip_id}.json"

    def reconstruct(
        self, clip_id: str, method: str = "InstantNuRec",
    ) -> dict:
        """Trigger 3D reconstruction of a clip via the SIL Wheel server.

        The server broadcasts the request to a connected NuRec viewer process
        (a contributor Instant NuRec) via WebSocket. The viewer reconstructs
        the scene and updates its interactive 3D HTML page.

        Architecture:
          1. Agent POSTs reconstruction::{clip_id}::{method} to SIL Wheel
          2. SIL Wheel broadcasts the ncore data path via WebSocket (port 7000)
          3. NuRec viewer (connected to WebSocket) receives the request
          4. NuRec reconstructs the clip and updates its viewer page
          5. User views the result at the NuRec viewer URL

        Args:
            clip_id: UUID of the clip to reconstruct.
            method: "InstantNuRec" (pre-computed ncore data, fast) or
                    "NuRec" (on-demand reconstruction, launches subprocess).

        Returns:
            dict with status, ncore_path, and viewer metadata.
        """
        valid_methods = ("InstantNuRec", "NuRec")
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {method!r}")
        if not clip_id or not clip_id.strip():
            raise ValueError("clip_id must be a non-empty string")
        self._validate_no_protocol_chars(clip_id)
        body = f"reconstruction::{clip_id}::{method}"
        resp = self._post("", data=body)
        ncore_path = self.NUREC_NCORE_JSON.format(clip_id=clip_id)
        return {
            "status": resp.status_code,
            "clip_id": clip_id,
            "method": method,
            "ncore_path": ncore_path,
            "ncore_dir": self.NUREC_NCORE_PATH.format(clip_id=clip_id),
            "submitted": resp.status_code == 200,
            "text": resp.text[:300] if resp.text else "",
        }

    def reconstruction_ncore_path(self, clip_id: str) -> str:
        """Get the ncore data directory path for a clip (used by NuRec)."""
        if not clip_id or not clip_id.strip():
            raise ValueError("clip_id must be a non-empty string")
        return self.NUREC_NCORE_PATH.format(clip_id=clip_id)

    # ── Browser URLs ──────────────────────────────────────────────────

    def clip_url(self, clip_id: str, project_source: str | None = None) -> str:
        """Generate a browser URL to view a clip in the SIL Wheel UI.

        Args:
            project_source: Project source for the URL. Defaults to "Alpamayo".
                           Pass a SearchResult's data_source for accurate linking.
        """
        cid = self._require_clip_id(clip_id)
        ps = project_source or "Alpamayo"
        return f"{self.base_url}/#&page=0&search_clipid={quote(cid, safe='')}&project_source={quote(ps, safe='')}"

    def search_url(self, **search_params) -> str:
        """Generate a browser URL that reproduces a search in the SIL Wheel UI.

        Accepts the same kwargs as search() (minus page/n). The UI shares the
        same query-param vocabulary as the ``/videos`` endpoint, so we reuse
        the same client→server name translations (``probability_threshold``
        → ``probability_expression``, ``extra_queries`` →
        ``caption_extra_queries``). See knowledge/anti-patterns.md.
        """
        # Pre-process classifier probability identically to search().
        prob_expr = search_params.get("probability_expression")
        if (
            prob_expr is None
            and search_params.get("probability_threshold") is not None
        ):
            prob_expr = self._probability_threshold_to_expression(
                search_params["probability_threshold"]
            )
        param_map = {
            "data_source": "data_source",
            "search": "search",
            "label_filter": "filter",
            "classifier_select": "classifier_select",
            "semantic_search_text": "semantic_search_text",
            "semantic_search_clipid": "semantic_search_clipid",
            "visual_search_text": "visual_search_text",
            "trajectory_shape_clipid": "trajectory_shape_clipid",
            "trajectory_pattern": "trajectory_pattern",
            "search_speed": "search_speed",
            "search_country": "search_country",
            "wm_class_name": "wm_class_name",
            "wm_min_count": "wm_min_count",
            "wm_max_count": "wm_max_count",
            "wm_max_dist": "wm_max_dist",
            "cluster_run_id": "cluster_run_id",
            "cluster_id": "cluster_id",
            "project_source": "project_source",
            "filter_mode": "filter_mode",
            "sil_apis": "sil_apis",
            "labels_to_exclude": "labels_to_exclude",
            "label_types": "label_types",
            "search_comments": "search_comments",
            "numeric_filter": "numeric_filter",
            "left_hand_driving": "left_hand_driving",
            "trajectory_shape_start_t": "trajectory_shape_start_t",
            "trajectory_shape_end_t": "trajectory_shape_end_t",
            "wm_min_time": "wm_min_time",
            "wm_angle_range": "wm_angle_range",
            "search_clipid": "search_clipid",
            "query_rewrite": "query_rewrite",
            "with_ego_data": "with_ego_data",
            "with_metrics": "with_metrics",
            "with_bev": "with_bev",
            "without_ann": "without_ann",
            # Server renamed bare extra_queries → caption_extra_queries.
            "extra_queries": "caption_extra_queries",
            "caption_embed_search_text": "caption_embed_search",
            "rank_mode": "rank_mode",
            "times": "times",
        }
        parts = ["page=0"]
        for kwarg, param in param_map.items():
            val = search_params.get(kwarg)
            if val is None:
                continue
            if isinstance(val, bool):
                if not val:
                    continue
                val = "true"
            parts.append(f"{param}={quote(str(val), safe='')}")
        if prob_expr is not None:
            parts.append(f"probability_expression={quote(str(prob_expr), safe='')}")
        return f"{self.base_url}/#&{'&'.join(parts)}"

    def _require_clip_id(self, clip_id: str) -> str:
        """Validate clip_id is a non-empty string, return stripped value."""
        if not clip_id or not str(clip_id).strip():
            raise ValueError("clip_id must be a non-empty string")
        return str(clip_id).strip()

    def video_url(self, clip_id: str) -> str:
        """Direct URL to stream clip video (S3-backed, range-request enabled)."""
        cid = self._require_clip_id(clip_id)
        return f"{self.base_url}/video/{quote(cid, safe='-_.')}.mp4"

    def depth_video_url(self, clip_id: str) -> str:
        """URL for the autolabel depth visualization video."""
        cid = self._require_clip_id(clip_id)
        return f"{self.base_url}/depth_video/{quote(cid, safe='-_.')}.mp4"

    def boxes_video_url(self, clip_id: str) -> str:
        """URL for the autolabel bounding box visualization video."""
        cid = self._require_clip_id(clip_id)
        return f"{self.base_url}/boxes_video/{quote(cid, safe='-_.')}.mp4"

    def point_video_url(self, clip_id: str) -> str:
        """URL for the autolabel point map visualization video."""
        cid = self._require_clip_id(clip_id)
        return f"{self.base_url}/point_video/{quote(cid, safe='-_.')}.mp4"

    def mfmrh_video_url(self, clip_id: str) -> str:
        """URL for the MFMRH visualization video."""
        cid = self._require_clip_id(clip_id)
        return f"{self.base_url}/mfmrh_video/{quote(cid, safe='-_.')}.mp4"

    def vipe_video_url(self, clip_id: str) -> str:
        """URL for the VIPE visualization video."""
        cid = self._require_clip_id(clip_id)
        return f"{self.base_url}/vipe_video/{quote(cid, safe='-_.')}.mp4"

    def bev_url(self, clip_id: str) -> str:
        """URL for bird's eye view data (msgpack format)."""
        cid = self._require_clip_id(clip_id)
        return f"{self.base_url}/api/bev/{quote(cid, safe='-_.')}"

    @staticmethod
    def _md_escape(text: str) -> str:
        """Escape chars that break markdown link syntax and inline formatting."""
        return (text.replace("\\", "\\\\").replace("[", "\\[")
                .replace("]", "\\]").replace("(", "\\(").replace(")", "\\)")
                .replace("*", "\\*").replace("_", "\\_").replace("`", "\\`")
                .replace("|", "\\|"))

    def format_clip_id_links(self, clip_ids: list[str], max_links: int = 50) -> str:
        """Generate markdown with a browser link for each clip ID.

        Useful for sharing lists of clips — each gets a clickable deep link.
        """
        if not clip_ids:
            return "No clips to show."
        lines = []
        for i, cid in enumerate(clip_ids[:max_links], 1):
            url = self.clip_url(cid)
            short = f"{cid[:16]}..." if len(cid) > 16 else cid
            lines.append(f"{i}. [{self._md_escape(short)}]({url})")
        if len(clip_ids) > max_links:
            lines.append(f"\n*...and {len(clip_ids) - max_links} more clips*")
        return "\n".join(lines)

    # ── Query Rewrite ────────────────────────────────────────────────

    def get_query_rewrites(self, query: str) -> list[str]:
        """Use the server's LLM query rewriter to expand a caption search query.

        Returns a list of semantically related queries. Requires the server
        to have NV_INFERENCE_API_KEY configured.
        """
        resp = self._get("rewrite", params={"query": query})
        if resp.status_code == 200:
            try:
                data = resp.json()
                return data.get("rewrites", []) if isinstance(data, dict) else []
            except (json.JSONDecodeError, ValueError):
                return []
        return []

    # ── Clustering ───────────────────────────────────────────────────

    def get_clustering_status(self) -> list[dict]:
        """List all clustering runs and their status (done/pending/failed)."""
        resp = self._get("clustering_status")
        if resp.status_code == 200:
            try:
                data = resp.json()
                return data.get("runs", []) if isinstance(data, dict) else []
            except (json.JSONDecodeError, ValueError):
                return []
        return []

    def get_clustering_results(self, run_id: str) -> dict:
        """Get results for a completed clustering run.

        Returns the full server payload::

            {
                "run_id":      str,
                "clusters":    {cid_str: {"cluster_size": int,
                                           "representative_clip_id": str | None},
                                ...},
                "umap":        {                     # UMAP 2D plot data
                    "centroids": {cid_str: [x, y]},
                    "clips":     {cid_str: [[x, y], ...]},
                    "clip_ids":  {cid_str: [str, ...]},
                    "distances": {cid_str: [float, ...]},
                },
                "metadata":    {                     # raw metadata.json
                    "run_id": str,
                    "n_clusters": int,
                    "spherical_kmeans": bool,        # True == cosine metric
                    "max_points_per_centroid": int,
                    "n_input_clips": int,            # NB: NOT "n_clips" —
                                                     # the rename only happens
                                                     # in /clustering_status.
                    "started_at": float,             # epoch seconds
                    "search_params": str,            # query-string used to
                                                     # build the run
                    "embed_type": str,               # e.g. "cosmos"
                },
                "topics":      {cid_str: {"keywords": [str, ...],
                                          "description": str},
                                ...},                # {} for old runs / no captions
                "topics_meta": {                     # {} pre-feature
                    "caption_model": str | None,
                    "captions_found": int,
                    "captions_total": int,
                },
            }

        ``topics`` (TF-IDF keywords per cluster, added 2026-04) is populated
        for clustering runs created after the feature shipped; older runs
        return ``topics: {}``. The per-cluster ``description`` field is only
        present after sil/wheel server MR !1 ("Add llm summary for cluster
        topics") merges and a server with ``NV_INFERENCE_API_KEY`` configured
        re-runs (or runs fresh) clustering — at v1.8.0 client release the
        MR was open and unmerged on prod, so callers MUST tolerate
        ``description`` being absent. Use :py:meth:`get_cluster_topics` for
        a topics-only view, or :py:meth:`summarize_clustering_run` for a
        ready-to-print theme listing.

        **Caveat — race window**: server flips ``status: "done"`` in
        ``/clustering_status`` only when both ``representative_by_cluster.json``
        AND ``umap.json`` are written, but ``/clustering_results`` only checks
        the first. Mid-write you can get a 200 response with ``umap: {}`` and
        possibly ``topics: {}``. To avoid this, poll
        :py:meth:`get_clustering_status` first and only call this method
        once the target run shows ``status == "done"``.
        """
        resp = self._get("clustering_results", params={"run_id": run_id})
        if resp.status_code == 200:
            try:
                return resp.json()
            except (json.JSONDecodeError, ValueError):
                return {"error": "invalid_json"}
        return {"error": resp.status_code}

    # Shared internal helpers for clustering payload defensiveness.
    # All four public clustering helpers funnel through these so that one
    # malformed shape (string-where-list, list-where-dict, etc.) is rejected
    # exactly the same way everywhere — preventing silent data corruption
    # like ``list("abc") -> ["a","b","c"]`` (Rule [T1] #31, AGENTS.md).
    @staticmethod
    def _coerce_topics_dict(value) -> dict:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _coerce_cluster_keywords(value) -> list[str]:
        if not isinstance(value, list):
            return []
        # Drop anything that isn't a string — the server contract is list[str]
        # but a future shape change (e.g. list[tuple[str, float]]) shouldn't
        # silently surface as substring-matchable strings.
        return [t for t in value if isinstance(t, str)]

    @staticmethod
    def _coerce_cluster_description(value) -> str:
        return value if isinstance(value, str) else ""

    def get_cluster_topics(self, run_id: str) -> dict:
        """Return TF-IDF topic keywords (and optional LLM descriptions) per cluster.

        Convenience wrapper around :py:meth:`get_clustering_results` that returns
        only the ``topics`` field (no UMAP, metadata, or sizes).

        Returns ``{cluster_id_str: {"keywords": [str, ...], "description": str}}``,
        where ``description`` is **only present** for clusters where the
        server-side LLM summarization succeeded (sil/wheel MR !1, currently
        unmerged on production). On servers without the MR, ``description``
        is absent — not ``None``, not ``""``, **absent**. The agent helpers
        (``find_clusters_by_keyword``, ``summarize_clustering_run``) handle
        both shapes.

        Returns ``{}`` if the run has no topics (older run that pre-dates the
        TF-IDF feature, no captions found, or clustering still in progress) —
        or if the request fails entirely.

        Topics are produced server-side by sampling 50 clips per cluster
        (random, fixed seed=42), auto-selecting the caption model with the
        highest coverage, and running per-clip TF-IDF (1-2 ngrams) with
        cluster-mean aggregation followed by AV-jargon stopword filtering
        (top-15 surviving terms).
        """
        results = self.get_clustering_results(run_id)
        if not isinstance(results, dict) or "error" in results:
            return {}
        return self._coerce_topics_dict(results.get("topics"))

    def get_cluster_members(
        self, run_id: str, cluster_id: str | int,
    ) -> tuple[list[str], list[float]]:
        """Get the full clip-id list and per-clip distances for one cluster.

        Returns ``(clip_ids, distances)`` where ``distances`` are the per-clip
        distances to the cluster centroid (smaller = more central; ASC
        sorted by the server). Returns ``([], [])`` on any error — including
        404 (run/cluster not found), 5xx (server crash), and malformed
        responses. 5xx errors are additionally logged to stderr so silent
        server crashes don't go unnoticed (Rule [T1] #31, AGENTS.md).

        For a paged ``SearchResult`` view of the same clips (with captions,
        scores, etc.) use :py:meth:`cluster_search`. ``cluster_search`` is
        also the only path that populates the per-result
        ``cluster_distance_score`` field; use this method when you only need
        clip ids + raw distances and want to avoid pagination.
        """
        if not run_id or not str(run_id).strip():
            raise ValueError("run_id must be a non-empty string")
        if (
            cluster_id is None or isinstance(cluster_id, bool)
            or str(cluster_id).strip() == ""
        ):
            raise ValueError("cluster_id must be a non-empty integer or string")
        resp = self._get(
            "cluster_members",
            params={"run_id": str(run_id), "cluster_id": str(cluster_id)},
        )
        if resp.status_code != 200:
            # Surface 5xx loudly so a malformed-parquet server crash isn't
            # silently rendered as "this cluster has no members". 4xx is
            # quietly swallowed because "missing run/cluster" is a normal
            # caller-driven case.
            if 500 <= resp.status_code < 600:
                detail = resp.text[:200] if resp.text else ""
                print(
                    f"[wheel_client] get_cluster_members: server returned "
                    f"{resp.status_code} for run={run_id!r} "
                    f"cluster_id={cluster_id!r} — returning ([], []). "
                    f"Detail: {detail!r}",
                    file=sys.stderr,
                )
            return [], []
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError):
            return [], []
        if not isinstance(data, dict):
            return [], []
        # Strict list-shape validation: a server that emits ``clip_ids: "abc"``
        # would otherwise silently produce ``["a","b","c"]`` from list("abc").
        clip_ids_raw = data.get("clip_ids")
        distances_raw = data.get("distances")
        clip_ids = (
            [str(c) for c in clip_ids_raw]
            if isinstance(clip_ids_raw, list) else []
        )
        distances: list[float] = []
        if isinstance(distances_raw, list):
            for d in distances_raw:
                try:
                    distances.append(float(d))
                except (TypeError, ValueError):
                    distances.append(float("nan"))
        return clip_ids, distances

    def find_clusters_by_keyword(
        self, run_id: str, keyword: str,
        match_description: bool = True, top_k: int | None = None,
    ) -> list[dict]:
        """Find clusters in a run whose TF-IDF keywords (or LLM description)
        contain ``keyword`` (case-insensitive substring match).

        Useful when an agent has a clustering run with hundreds of clusters and
        wants to jump directly to the ones about a specific theme — e.g.
        "find any clusters about pedestrians", "are there construction zone
        clusters?".

        Args:
            run_id: Clustering run identifier.
            keyword: Substring to match (case-insensitive). Whitespace stripped.
            match_description: If True (default), also match the optional
                LLM ``description`` (only present after sil/wheel MR !1
                lands; absent on prod today). If False, only TF-IDF keywords.
            top_k: If set and > 0, return at most ``top_k`` clusters
                (sorted by cluster size descending — biggest matching
                themes first). ``top_k=0`` and ``top_k=None`` both mean
                "no limit"; ``top_k`` < 0 also means "no limit".

        Returns:
            List of dicts ordered by cluster size (descending)::

                [{"cluster_id": str, "cluster_size": int,
                  "representative_clip_id": str | None,
                  "keywords": [str, ...], "description": str | None,
                  "match": "keyword" | "description"}, ...]

            Empty list if no match or the run has no topics.
        """
        if not keyword or not keyword.strip():
            raise ValueError("keyword must be a non-empty string")
        if not run_id or not str(run_id).strip():
            raise ValueError("run_id must be a non-empty string")
        kw = keyword.strip().lower()

        results = self.get_clustering_results(run_id)
        if not isinstance(results, dict) or "error" in results:
            return []
        topics = self._coerce_topics_dict(results.get("topics"))
        clusters = self._coerce_topics_dict(results.get("clusters"))
        if not topics:
            return []

        matches: list[dict] = []
        for cid, topic in topics.items():
            if not isinstance(topic, dict):
                continue
            keywords = self._coerce_cluster_keywords(topic.get("keywords"))
            description = self._coerce_cluster_description(
                topic.get("description")
            )
            kw_hit = any(kw in t.lower() for t in keywords)
            desc_hit = (
                match_description and bool(description)
                and kw in description.lower()
            )
            if not (kw_hit or desc_hit):
                continue
            cluster_meta = (
                clusters.get(str(cid)) or clusters.get(cid) or {}
            )
            if not isinstance(cluster_meta, dict):
                cluster_meta = {}
            matches.append({
                "cluster_id": str(cid),
                "cluster_size": _safe_int(cluster_meta.get("cluster_size")),
                "representative_clip_id": cluster_meta.get(
                    "representative_clip_id"
                ),
                "keywords": list(keywords),
                "description": description or None,
                # Keyword hit takes precedence; that's the more specific signal.
                "match": "keyword" if kw_hit else "description",
            })

        matches.sort(key=lambda m: m["cluster_size"], reverse=True)
        if top_k is not None and top_k > 0:
            matches = matches[:top_k]
        return matches

    def summarize_clustering_run(
        self, run_id: str, top_k: int | None = None,
        min_cluster_size: int = 0,
    ) -> dict:
        """Build a compact, agent-friendly summary of a clustering run's themes.

        Combines metadata, topic keywords, and (when available) LLM
        descriptions into a single dict suitable for logging, displaying to
        the user, or feeding back to a higher-level agent loop.

        Args:
            run_id: Clustering run identifier.
            top_k: If set and > 0, only include the ``top_k`` largest
                clusters. ``top_k=0`` / ``None`` / negative all mean "no limit".
            min_cluster_size: Skip clusters smaller than this (default 0 = all).

        Returns:
            ``{"run_id": str, "n_clusters": int, "n_clips": int,
              "caption_model": str | None, "captions_found": int,
              "captions_total": int, "captions_coverage": float,
              "clusters_with_topics": int,
              "clusters": [{"cluster_id": str, "cluster_size": int,
                            "representative_clip_id": str | None,
                            "keywords": [str, ...], "description": str | None,
                            "theme": str}, ...]}``

            ``theme`` is the LLM ``description`` if present, else the top-3
            keywords joined by ", " — a stable single-line label suitable
            for plots, log messages, or Slack posts. ``"(no caption coverage)"``
            is used as the fallback when both are absent. Returns ``{}`` if
            the run is missing or unfinished.

            ``clusters_with_topics`` counts every cluster whose topic entry
            has either non-empty keywords OR a non-empty description (so a
            description-only cluster, e.g. from a future LLM-direct path,
            is counted).

            ``n_clips`` is sourced from ``metadata.n_clips`` when present,
            else ``metadata.n_input_clips`` — the latter is the actual key
            written by the server today (the ``n_clips`` rename only happens
            in the ``/clustering_status`` projection, NOT in
            ``/clustering_results.metadata``). Without this fallback every
            summary would silently report ``n_clips: 0``.
        """
        if not run_id or not str(run_id).strip():
            raise ValueError("run_id must be a non-empty string")
        results = self.get_clustering_results(run_id)
        if not isinstance(results, dict) or "error" in results:
            return {}

        clusters = self._coerce_topics_dict(results.get("clusters"))
        topics = self._coerce_topics_dict(results.get("topics"))
        topics_meta = self._coerce_topics_dict(results.get("topics_meta"))
        metadata = self._coerce_topics_dict(results.get("metadata"))

        cap_total = _safe_int(topics_meta.get("captions_total"))
        cap_found = _safe_int(topics_meta.get("captions_found"))
        coverage = (cap_found / cap_total) if cap_total else 0.0

        rows: list[dict] = []
        for cid, meta in clusters.items():
            if not isinstance(meta, dict):
                continue
            size = _safe_int(meta.get("cluster_size"))
            if size < min_cluster_size:
                continue
            topic_raw = topics.get(str(cid)) or topics.get(cid) or {}
            if not isinstance(topic_raw, dict):
                topic_raw = {}
            keywords = self._coerce_cluster_keywords(topic_raw.get("keywords"))
            description_str = self._coerce_cluster_description(
                topic_raw.get("description")
            )
            description = description_str if description_str else None
            if description:
                theme = description
            elif keywords:
                theme = ", ".join(keywords[:3])
            else:
                theme = self._NO_TOPIC_THEME_FALLBACK
            rows.append({
                "cluster_id": str(cid),
                "cluster_size": size,
                "representative_clip_id": meta.get("representative_clip_id"),
                "keywords": keywords,
                "description": description,
                "theme": theme,
            })

        rows.sort(key=lambda r: r["cluster_size"], reverse=True)
        if top_k is not None and top_k > 0:
            rows = rows[:top_k]

        # n_clips: server's metadata.json uses ``n_input_clips``; the
        # ``n_clips`` alias only appears in the /clustering_status projection.
        # Reading both keys keeps the summary correct across server versions.
        n_clips = _safe_int(
            metadata.get("n_clips")
            if metadata.get("n_clips") is not None
            else metadata.get("n_input_clips")
        )
        n_clusters = _safe_int(
            metadata.get("n_clusters")
            if metadata.get("n_clusters") is not None
            else len(clusters)
        )

        # Count clusters with EITHER non-empty keywords or a description —
        # otherwise a description-only cluster (from a future LLM-direct
        # path) would appear in `clusters[].theme` but be excluded from
        # the count, which is internally inconsistent.
        with_topics = 0
        for t in topics.values():
            if not isinstance(t, dict):
                continue
            kws = self._coerce_cluster_keywords(t.get("keywords"))
            desc = self._coerce_cluster_description(t.get("description"))
            if kws or desc:
                with_topics += 1

        return {
            "run_id": str(run_id),
            "n_clusters": n_clusters,
            "n_clips": n_clips,
            "caption_model": topics_meta.get("caption_model"),
            "captions_found": cap_found,
            "captions_total": cap_total,
            "captions_coverage": round(coverage, 4),
            "clusters_with_topics": with_topics,
            "clusters": rows,
        }

    # Theme string used when a cluster has neither keywords nor a description.
    # Exposed as a class constant so tests / CLI / external consumers can
    # check for it without hardcoding the literal.
    _NO_TOPIC_THEME_FALLBACK = "(no caption coverage)"

    # ── Data Stats ───────────────────────────────────────────────────

    def get_data_stats(self) -> list[dict]:
        """List per-dataset statistics. Cached for 5 min."""
        return self._get_cached("data_stats", self._fetch_data_stats)

    def _fetch_data_stats(self) -> list[dict]:
        resp = self._get("data_stats_list")
        if resp.status_code == 200:
            try:
                data = resp.json()
            except (json.JSONDecodeError, ValueError):
                return []
            if isinstance(data, dict):
                return data.get("datasets", [])
            if isinstance(data, list):
                return data
            return []
        return []

    # ── Annotations Summary ──────────────────────────────────────────

    def get_annotations_summary(self) -> list[dict]:
        """Get per-label annotation counts (manual, timed, autolabel)."""
        resp = self._get("annotations_summary.csv")
        if resp.status_code != 200:
            return []
        rows = []
        reader = csv.DictReader(io.StringIO(resp.text))
        for row in reader:
            rows.append(row)
        return rows

    # ── Predictions & Full Metrics ───────────────────────────────────

    def get_predictions(self, clip_id: str) -> dict:
        """Get per-model predictions for a clip (predicted/GT positions, captions)."""
        if not clip_id or not str(clip_id).strip():
            raise ValueError("clip_id must be a non-empty string")
        resp = self._get("predictions", params={"clip_id": clip_id})
        if resp.status_code == 200:
            try:
                return resp.json()
            except json.JSONDecodeError:
                return {"error": "invalid_json"}
        return {"error": resp.status_code}

    def get_full_clip_metrics(self, clip_id: str, model_name: str = "ground_truth") -> dict:
        """Get full per-metric breakdown for a single clip and model."""
        if not clip_id or not str(clip_id).strip():
            raise ValueError("clip_id must be a non-empty string")
        resp = self._get("full_metrics", params={
            "clip_id": clip_id, "model_name": model_name,
        })
        if resp.status_code == 200:
            try:
                return resp.json()
            except json.JSONDecodeError:
                return {"error": "invalid_json"}
        return {"error": resp.status_code}

    # ── Classifier Export ────────────────────────────────────────────

    def export_classifier_weights(
        self, label: str, embed_type: str | None = None,
    ) -> dict:
        """Download classifier weights as JSON (coefficients + intercept).

        Args:
            label: Classifier label (case-sensitive — use
                :meth:`resolve_classifier_name` for fuzzy lookup).
            embed_type: Embedding backend the classifier was trained on
                (``"cosmos"`` or ``"caption"``). If omitted, auto-detected
                via :meth:`get_classifier_embed_type`. Required by the
                server URL since 2026-Q2 (the route is now
                ``/classifier/export/{embed_type}/{label}``).

        Returns:
            ``{'label', 'version', 'coefficients', 'intercept'}`` on success,
            or ``{'error': code, 'reason': str}`` on failure with an
            actionable explanation.
        """
        if not label or not str(label).strip():
            raise ValueError("label must be a non-empty string")
        label = label.strip()
        if embed_type is None:
            embed_type = self.get_classifier_embed_type(label)
            if embed_type is None:
                return {
                    "error": 404,
                    "reason": (
                        f"Classifier {label!r} is not trained on either "
                        f"'cosmos' or 'caption'. Use list_classifier_names() "
                        f"to discover trained classifiers, or pass embed_type "
                        f"explicitly."
                    ),
                }
        resp = self._get(
            f"classifier/export/{quote(embed_type, safe='')}/"
            f"{quote(label, safe='')}"
        )
        if resp.status_code == 200:
            try:
                return resp.json()
            except json.JSONDecodeError:
                return {"error": "invalid_json", "reason": "Server returned non-JSON body"}
        return {
            "error": resp.status_code,
            "reason": (
                f"GET /classifier/export/{embed_type}/{label} returned "
                f"HTTP {resp.status_code}"
            ),
        }

    # ── High-Level Helpers ───────────────────────────────────────────

    def find_clips_for_scenario(
        self,
        description: str,
        data_source: str | None = "MADS",
        use_classifiers: bool = True,
        use_query_rewrite: bool = False,
        use_semantic: bool = True,
    ) -> dict[str, list[SearchResult]]:
        """Find clips matching a scenario description using multiple strategies.

        Searches across caption, semantic, and optionally classifier modes,
        returning results grouped by strategy with scores for ranking.
        This is the primary "idea -> clip_ids" workflow.

        Args:
            description: Natural language scenario description.
            data_source: Restrict to a single Wheel data source (default
                ``"MADS"``). Pass ``None`` to search across **ALL** sources
                (Waymo, AV V1/V2, OpenDV-YouTube, Physical AI, etc.).
                For ``"MADS-1M"`` callers must pass it **explicitly** —
                the default is ``"MADS"`` (the smaller human-curated set),
                not ``"MADS-1M"``. See ``knowledge/anti-patterns.md``.
            use_classifiers: If True, checks if a matching classifier exists.
            use_query_rewrite: If True, uses LLM query expansion for captions.
            use_semantic: If True, runs Cosmos text-to-video search (~3-5s prod,
                         ~120s dev). Set False for fast-only mode.

        Returns dict with keys like 'caption', 'semantic', 'classifier',
        each mapping to a list of SearchResult with relevance scores.
        """
        if not description or not description.strip():
            raise ValueError("description must be a non-empty string")
        results: dict[str, list[SearchResult]] = {}
        futures: dict[str, Any] = {}

        with ThreadPoolExecutor(max_workers=6) as pool:
            # Caption branch routes through caption_search() so the FTS5
            # AND-of-words pre-flight warning fires for long natural-language
            # descriptions (otherwise the warning is bypassed — silent bug
            # B5 found in audit-pass-C). When query_rewrite is on, fall back
            # to the raw search() path because caption_search doesn't pass
            # query_rewrite through.
            if use_query_rewrite:
                futures["caption"] = pool.submit(
                    self.search,
                    search=description, data_source=data_source, n=20,
                    query_rewrite=True,
                )
            else:
                # mode='all' is the default and matches historical behaviour
                # (FTS5 AND); the pre-flight warning will fire for 4+ word
                # descriptions, telling the agent to retry with mode='any'.
                futures["caption"] = pool.submit(
                    self.caption_search,
                    description, data_source=data_source, n=20, mode="all",
                )
            if use_semantic:
                futures["semantic"] = pool.submit(
                    self.search, semantic_search_text=description,
                    data_source=data_source, n=20,
                )

            if use_classifiers:
                try:
                    classifiers = self.get_classifiers()
                except Exception as e:
                    warnings.warn(f"get_classifiers() failed, skipping classifier strategies: {e}", stacklevel=2)
                    classifiers = {"trained": []}
                _stopwords = {"a", "an", "the", "in", "on", "at", "of", "to", "for",
                              "and", "or", "with", "is", "are", "was", "were", "it",
                              "its", "by", "as", "be", "no", "not", "but", "from"}
                desc_words = {w.strip(_string_mod.punctuation) for w in description.lower().split()} - _stopwords - {""}
                for label in classifiers.get("trained", []):
                    label_words = {w.strip(_string_mod.punctuation) for w in label.lower().replace("_", " ").replace("-", " ").split()} - _stopwords - {""}
                    if label_words and desc_words and label_words <= desc_words:
                        futures[f"classifier:{label}"] = pool.submit(
                        self.search, classifier_select=label,
                        probability_threshold=0.5, data_source=data_source, n=20,
                    )

            for key, future in futures.items():
                try:
                    _, search_results = future.result()
                    if search_results:
                        results[key] = search_results
                except Exception as e:
                    warnings.warn(f"Search strategy {key!r} failed: {e}", stacklevel=2)

        if futures and not results:
            sub_queries = self._decompose_scenario(description)
            if len(sub_queries) >= 2:
                try:
                    id_sets = []
                    for sq in sub_queries:
                        try:
                            ids = self.export_search_clip_ids(
                                search=sq, data_source=data_source,
                            )
                        except Exception:
                            ids = []
                        if ids:
                            id_sets.append(set(ids))
                    if len(id_sets) >= 2:
                        intersected = list(set.intersection(*id_sets))
                        if intersected:
                            batch = self.lookup_clips_batch(
                                intersected[:50], max_workers=8,
                            )
                            found = [r for r in batch.values() if r is not None]
                            if found:
                                results["decomposed"] = found
                except Exception as e:
                    warnings.warn(
                        f"Decomposition fallback failed: {e}", stacklevel=2,
                    )
            if not results:
                warnings.warn(
                    f"find_clips_for_scenario: ALL strategies failed for "
                    f"{description!r}",
                    stacklevel=2,
                )
        return results

    @staticmethod
    def _decompose_scenario(description: str) -> list[str]:
        """Split a composite scenario into shorter sub-queries for FTS5 fallback.

        When the full description returns 0 from caption search (because FTS5
        requires ALL words to co-occur), this splits on connecting phrases and
        returns individual concept chunks that can be searched separately and
        intersected.

        Splits on connector words (with, and, in front of, behind, ...):
        "rainy night with pedestrian walking" → ["rainy night", "pedestrian walking"]

        Queries without connectors (e.g., "stroller front car") are NOT decomposed
        because individual words lose semantic structure. For those, use
        caption_search_any() for OR matching or manual search + intersect.

        Returns a list of sub-queries if decomposition is useful (>= 2 chunks),
        or an empty list if the description is already simple enough.
        """
        import re
        _splitters = (
            r'\b(?:with|and|while|near|in\s+front\s+of|behind|beside'
            r'|next\s+to|between|around|containing|showing)\b'
        )
        chunks = re.split(_splitters, description, flags=re.IGNORECASE)

        _stops = {
            "a", "an", "the", "in", "on", "at", "of", "to", "for", "or",
            "is", "are", "was", "were", "it", "its", "by", "as", "be",
            "no", "not", "but", "from", "that", "this", "someone",
            "something",
        }
        queries: list[str] = []
        for chunk in chunks:
            words = [
                w for w in chunk.strip().split()
                if w.lower().strip(_string_mod.punctuation) not in _stops
                and len(w.strip(_string_mod.punctuation)) > 1
            ]
            phrase = " ".join(words).strip()
            if phrase:
                queries.append(phrase)

        return queries if len(queries) >= 2 else []

    def diagnose_zero_results(
        self,
        query: str | None = None,
        *,
        data_source: str | None = "MADS",
        clip_id: str | None = None,
        **search_kwargs: Any,
    ) -> dict:
        """Diagnose WHY a search returned 0 results.

        The single most common agent failure mode is concluding "no clips
        match" when the actual cause is a slightly-wrong filter. This
        method runs a series of cheap checks and returns a structured
        explanation + concrete next-step suggestions.

        Args:
            query: Optional natural-language scenario / caption query.
            data_source: Filter the agent was using (defaults to ``"MADS"``).
            clip_id: If the agent was looking up a specific clip, pass it
                here for targeted diagnosis (key shape, source).
            **search_kwargs: Any other ``search()`` kwargs the agent passed.

        Returns:
            ``{
                "summary": str,
                "checks": [{name, status, detail}, ...],
                "suggestions": [str, ...],
                "last_search_error": str | None,
            }``

        This method **never raises** even in strict mode — it's the
        recovery path. Cost: 1-3 cheap queries.
        """
        checks: list[dict] = []
        suggestions: list[str] = []

        # 0. Surface any prior transport-level error first. (Defensive
        # getattr because mocked clients in tests may not initialize the
        # attribute; the runtime client always does in __init__.)
        last_err = getattr(self, "last_search_error", None)

        # 1. Auth / connectivity sanity.
        try:
            who = self.whoami()
            checks.append({
                "name": "auth",
                "status": "ok" if who.get("authenticated") else "fail",
                "detail": f"whoami={who.get('user', who)}",
            })
            if not who.get("authenticated"):
                suggestions.append(
                    "Re-authenticate: client.login() or check WHEEL_USERNAME/WHEEL_PASSWORD in .env."
                )
        except Exception as e:
            checks.append({"name": "auth", "status": "error", "detail": repr(e)})
            suggestions.append(f"Connectivity error: {e!r}. Verify VPN + base_url={self.base_url}.")

        # 2. Clip-key footgun.
        if clip_id:
            checks.append({
                "name": "clip_id_shape",
                "status": "ok" if _EXACT_CLIP_ID_RE.match(clip_id) else "warn",
                "detail": (
                    "looks like exact UUID/MADS-1M key" if _EXACT_CLIP_ID_RE.match(clip_id)
                    else "doesn't match exact-key shape — server treats search_clipid "
                          "as exact intersection so a substring will return 0"
                ),
            })
            r = self.lookup_clip(clip_id)
            if r is not None:
                checks.append({"name": "lookup_clip_widened", "status": "ok",
                               "detail": f"lookup_clip(...) FOUND the clip in source={r.data_source!r}"})
                if data_source and r.data_source != data_source:
                    suggestions.append(
                        f"The clip lives in data_source={r.data_source!r} but you passed "
                        f"data_source={data_source!r}. Pass data_source=None or "
                        f"data_source={r.data_source!r}."
                    )
            else:
                checks.append({"name": "lookup_clip_widened", "status": "fail",
                               "detail": "lookup_clip(clip_id) returned None across all sources"})
                suggestions.append(
                    f"clip_id={clip_id!r} not in the Wheel index. Verify the ID matches a "
                    "known data source via client.get_data_sources()."
                )

        # 3. Caption query footguns.
        if query:
            wc = len(query.split())
            if wc >= 4:
                checks.append({
                    "name": "caption_length",
                    "status": "warn",
                    "detail": f"{wc} words; FTS5 AND-of-words rarely matches long natural-language queries",
                })
                suggestions.append(
                    f"Retry with mode='any': client.caption_search({query!r}, mode='any', data_source={data_source!r})"
                )
            # Try a permissive ANY-mode search to see if results exist at all.
            try:
                t_any, _ = self.caption_search(query, data_source=data_source, n=5, mode='any')
                checks.append({"name": "caption_search_any_probe",
                               "status": "ok" if t_any > 0 else "fail",
                               "detail": f"mode='any' returned total={t_any}"})
                if t_any > 0:
                    suggestions.append(
                        f"mode='any' returns {t_any} clips. Use that mode for natural-language queries."
                    )
            except Exception as e:
                checks.append({"name": "caption_search_any_probe", "status": "error", "detail": repr(e)})

            # Probe semantic embedding too — different recall surface.
            try:
                t_emb, _ = self.caption_embedding_search(query, data_source=data_source, n=5)
                checks.append({"name": "caption_embedding_probe",
                               "status": "ok" if t_emb > 0 else "fail",
                               "detail": f"caption_embed total={t_emb}"})
                if t_emb > 0:
                    suggestions.append(
                        f"caption_embedding_search returns {t_emb} clips — semantic match worked. "
                        f"Try: client.caption_embedding_search({query!r}, data_source={data_source!r})"
                    )
            except Exception as e:
                checks.append({"name": "caption_embedding_probe", "status": "error", "detail": repr(e)})

        # 4. Composed-filter trap.
        composed_keys = [
            k for k, v in search_kwargs.items()
            if v is not None and k in (
                "classifier_select", "semantic_search_text", "visual_search_text",
                "trajectory_shape_clipid", "trajectory_pattern", "wm_class_name",
                "search_comments", "label_filter", "numeric_filter", "search_clipid",
            )
        ]
        if (query is not None) + len(composed_keys) >= 2:
            checks.append({
                "name": "composed_filters",
                "status": "warn",
                "detail": f"{len(composed_keys) + (1 if query else 0)} active filters — server intersects all",
            })
            suggestions.append(
                "Composed search server-intersects every filter; many clips fail at least one. "
                "Run each filter via client.export_search_clip_ids() and combine with "
                "WheelClient.intersect_clip_id_lists()."
            )

        # 5. Data source narrowness.
        if data_source is not None:
            suggestions.append(
                f"Try widening: pass data_source=None to search across all sources "
                f"(currently restricted to {data_source!r}; default 'MADS' is small)."
            )

        summary_bits = []
        if last_err:
            summary_bits.append(f"transport error: {last_err}")
        bad = [c for c in checks if c["status"] in ("fail", "warn", "error")]
        if bad:
            summary_bits.append(f"{len(bad)} suspicious checks")
        else:
            summary_bits.append("all checks ok")

        return {
            "summary": "; ".join(summary_bits) if summary_bits else "no diagnostic signal",
            "checks": checks,
            "suggestions": suggestions,
            "last_search_error": last_err,
        }

    def find_clips_for_scenario_ids(
        self,
        description: str,
        data_source: str | None = "MADS",
        max_clips: int = 1000,
        rrf_k: int = 60,
    ) -> list[str]:
        """Find clips for a scenario and return deduplicated clip IDs ranked by RRF.

        Uses Reciprocal Rank Fusion to combine results across strategies
        (caption, semantic, classifier) which have incomparable score scales.
        RRF score = sum(1 / (k + rank_in_list)) across all lists containing the clip.

        Args:
            description: Natural-language scenario description.
            data_source: Restrict to a single Wheel data source (default
                ``"MADS"``). Pass ``None`` for clips from **ALL** sources
                (Waymo / AV V1+V2 / OpenDV-YouTube / Physical AI / ...).
                For ``"MADS-1M"`` (the larger 1M-clip set) you MUST pass
                ``data_source="MADS-1M"`` explicitly — the default is the
                smaller human-curated MADS set, NOT MADS-1M. Forgetting
                this is a recurring footgun (Agent Hub post 161a7f6c).
            max_clips: Maximum number of ranked clip IDs to return.
            rrf_k: RRF smoothing constant (default 60, standard in literature).
                   Higher values reduce the influence of top-ranked results.
                   Must be > 0 to avoid division by zero.
        """
        if rrf_k <= 0:
            raise ValueError(f"rrf_k must be positive, got {rrf_k}")
        grouped = self.find_clips_for_scenario(description, data_source=data_source)
        rrf_scores: dict[str, float] = {}
        for results in grouped.values():
            for rank, r in enumerate(results, start=1):
                rrf_scores[r.clip_id] = rrf_scores.get(r.clip_id, 0.0) + 1.0 / (rrf_k + rank)
        ranked = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)
        return ranked[:max_clips]

    def find_similar_to_clip(
        self,
        clip_id: str,
        modes: list[str] | None = None,
        data_source: str | None = None,
        n: int = 20,
    ) -> dict[str, list[SearchResult]]:
        """Find clips similar to a given clip across multiple similarity modes.

        Runs cosmos and trajectory searches in parallel for ~2x speedup.
        Graceful degradation: if one mode fails, the other's results are still returned.

        Args:
            clip_id: Non-empty clip UUID to find similar clips for.
            modes: Subset of ['cosmos', 'trajectory']. Default: both.
                   CLIP visual search is text-only (no clip-to-clip mode).

        Raises:
            ValueError: If clip_id is empty or whitespace-only.
        """
        if not clip_id or not clip_id.strip():
            raise ValueError("clip_id must be non-empty")
        _valid_modes = {"cosmos", "trajectory"}
        if modes is None:
            modes = ["cosmos", "trajectory"]
        else:
            invalid = set(modes) - _valid_modes
            if invalid:
                raise ValueError(f"Invalid modes: {invalid}. Valid: {_valid_modes}")
        results: dict[str, list[SearchResult]] = {}
        futures: dict[str, Any] = {}

        with ThreadPoolExecutor(max_workers=2) as pool:
            if "cosmos" in modes:
                futures["cosmos"] = pool.submit(
                    self.semantic_search_by_clip, clip_id, data_source, n)
            if "trajectory" in modes:
                futures["trajectory"] = pool.submit(
                    self.trajectory_search_by_clip, clip_id,
                    data_source=data_source, n=n)
            for key, future in futures.items():
                try:
                    raw = future.result()
                    results[key] = raw[1] if isinstance(raw, tuple) and len(raw) >= 2 else raw
                except Exception as e:
                    warnings.warn(
                        f"Similarity mode {key!r} failed for {clip_id!r}: {e}",
                        stacklevel=2,
                    )

        return results

    def find_similar_to_clips(
        self,
        clip_ids: list[str],
        data_source: str | None = None,
        n_per_clip: int = 20,
        max_workers: int = 8,
        progress_fn: Any | None = None,
    ) -> dict[str, dict[str, list[SearchResult]]]:
        """Find clips similar to each clip in a list (batch similarity search).

        Uses a single flat thread pool for all searches (cosmos + trajectory
        per clip) instead of nesting through find_similar_to_clip(), which
        would create max_workers × 2 nested pools.

        Args:
            max_workers: Max parallel threads (default 8).
            progress_fn: Optional callback(completed: int, total: int) for progress.
        """
        if not clip_ids:
            return {}
        clip_ids = list(dict.fromkeys(clip_ids))
        results: dict[str, dict[str, list[SearchResult]]] = {}
        _MODES = 2  # cosmos + trajectory

        workers = min(len(clip_ids) * _MODES, max_workers)
        futures: dict[Any, tuple[str, str]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for cid in clip_ids:
                futures[pool.submit(
                    self.semantic_search_by_clip, cid,
                    data_source=data_source, n=n_per_clip,
                )] = ("cosmos", cid)
                futures[pool.submit(
                    self.trajectory_search_by_clip, cid,
                    data_source=data_source, n=n_per_clip,
                )] = ("trajectory", cid)

            clip_done: dict[str, int] = {}
            n_complete = 0
            for future in as_completed(futures):
                mode_name, cid = futures[future]
                try:
                    raw = future.result()
                    results.setdefault(cid, {})[mode_name] = raw[1] if isinstance(raw, tuple) and len(raw) >= 2 else raw
                except Exception as e:
                    warnings.warn(
                        f"Similarity mode {mode_name!r} failed for {cid!r}: {e}",
                        stacklevel=2,
                    )
                clip_done[cid] = clip_done.get(cid, 0) + 1
                if clip_done[cid] == _MODES:
                    n_complete += 1
                    if progress_fn:
                        progress_fn(n_complete, len(clip_ids))
        return results

    def expand_clip_set(
        self,
        seed_clip_ids: list[str],
        data_source: str | None = None,
        n_similar_per_clip: int = 20,
        max_total: int = 1000,
        max_workers: int = 8,
        progress_fn: Any | None = None,
    ) -> list[tuple[str, float, str]]:
        """Expand a seed set of clip_ids by finding similar clips for each.

        Returns list of (clip_id, similarity_score, source_clip_id) tuples,
        deduplicated and sorted by score. Useful for growing a small curated
        set into a large training-ready dataset.

        Args:
            max_workers: Max parallel threads (default 8, capped at len(seed_clip_ids)).
            progress_fn: Optional callback(completed_seeds: int, total_seeds: int,
                         candidates_found: int) for progress monitoring.
        """
        if not seed_clip_ids:
            return []

        n_seeds = len(seed_clip_ids)
        reqs_per_seed = 2  # cosmos + trajectory
        est_seconds = n_seeds * reqs_per_seed * 6 / min(n_seeds, max_workers)
        if est_seconds > 300:
            warnings.warn(
                f"expand_clip_set: {n_seeds} seeds × ~12s each / "
                f"{min(n_seeds, max_workers)} workers ≈ "
                f"{est_seconds / 60:.0f} min estimated runtime. "
                f"Early stop triggers at {int(max_total * 1.2):,} candidates.",
                stacklevel=2,
            )

        scored: dict[str, tuple[float, str]] = {}
        seed_set = set(seed_clip_ids)
        lock = threading.Lock()
        stop = threading.Event()
        completed_count = [0]

        for seed_id in seed_clip_ids:
            scored[seed_id] = (1.0, seed_id)

        def _process_seed(sid: str) -> None:
            if stop.is_set():
                return
            similar = self.find_similar_to_clip(
                sid, data_source=data_source, n=n_similar_per_clip,
            )
            with lock:
                for mode, results in similar.items():
                    for r in results:
                        if not r.clip_id or r.clip_id in seed_set:
                            continue
                        s = r.best_score if r.best_score is not None else 0.0
                        if r.clip_id not in scored or s > scored[r.clip_id][0]:
                            scored[r.clip_id] = (s, sid)
                completed_count[0] += 1
                if progress_fn:
                    progress_fn(completed_count[0], len(seed_clip_ids), len(scored))
                if len(scored) >= int(max_total * 1.2):
                    stop.set()

        unique_seeds = list(dict.fromkeys(seed_clip_ids))
        workers = min(len(unique_seeds), max_workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_seed = {pool.submit(_process_seed, sid): sid for sid in unique_seeds}
            for f in as_completed(future_to_seed):
                try:
                    f.result()
                except Exception as e:
                    warnings.warn(
                        f"expand_clip_set: seed {future_to_seed[f]!r} failed: {e}",
                        stacklevel=2,
                    )

        ranked = sorted(scored.items(), key=lambda x: x[1][0], reverse=True)
        return [(cid, score, source) for cid, (score, source) in ranked[:max_total]]

    @staticmethod
    def parse_clip_id(clip_id: str) -> tuple[str, str | None, str | None]:
        """Parse a MADS-1M clip ID into (session_id, start_timestamp, end_timestamp).

        MADS-1M clip IDs use the format: {session_id}_{start_timestamp}_{end_timestamp}
        where session_id is a UUID and timestamps are microsecond strings.

        For plain MADS clip IDs (UUID-only), returns (clip_id, None, None).

        The session_id can be used with the NDAS image-retrieval-serve clip-GT
        API to download full ground truth packages (multi-camera video, object
        detections, HD map, calibration, planning data).

        Example:
            sid, start, end = WheelClient.parse_clip_id(
                "955a526c-a388-11ec-a932-00044bf65dfd_15834978771_15844978771"
            )
            # sid   = "955a526c-a388-11ec-a932-00044bf65dfd"
            # start = "15834978771"
            # end   = "15844978771"
        """
        if not clip_id or not clip_id.strip():
            raise ValueError("clip_id must be a non-empty string")
        parts = clip_id.rsplit("_", 2)
        if len(parts) == 3 and len(parts[0]) >= 36:
            return parts[0], parts[1], parts[2]
        return clip_id, None, None

    @staticmethod
    def merge_clip_id_lists(
        *clip_id_lists: list[str],
    ) -> list[str]:
        """Merge multiple clip ID lists into a single deduplicated list.

        Preserves order (first occurrence wins). Use this for combining clips
        from different searches into a single training set.

        Example:
            urban = client.export_search_clip_ids(search="urban", data_source="MADS-1M")
            weather = client.export_search_clip_ids(search="rain OR snow", data_source="MADS-1M")
            combined = WheelClient.merge_clip_id_lists(urban, weather)
            client.save_clip_ids(combined, "urban_and_weather.txt")
        """
        seen: set[str] = set()
        result: list[str] = []
        for lst in clip_id_lists:
            for cid in lst:
                if cid not in seen:
                    seen.add(cid)
                    result.append(cid)
        return result

    @staticmethod
    def intersect_clip_id_lists(
        *clip_id_lists: list[str],
    ) -> list[str]:
        """Return clip IDs present in ALL lists. Preserves order from the first list.

        Use this to find clips matching multiple criteria simultaneously.

        Example:
            urban = client.export_search_clip_ids(search="urban")
            snowy = client.export_search_clip_ids(classifier_select="Snow", probability_threshold=0.7)
            urban_snow = WheelClient.intersect_clip_id_lists(urban, snowy)
        """
        if not clip_id_lists:
            return []
        for i, lst in enumerate(clip_id_lists):
            if lst is None:
                raise ValueError(f"intersect_clip_id_lists() received None for list at index {i}")
        sets = [set(lst) for lst in clip_id_lists]
        common = sets[0]
        for s in sets[1:]:
            common &= s
        first_list = clip_id_lists[0] if clip_id_lists[0] else []
        seen: set[str] = set()
        result = []
        for cid in first_list:
            if cid in common and cid not in seen:
                seen.add(cid)
                result.append(cid)
        return result

    @staticmethod
    def subtract_clip_id_lists(
        base: list[str], *exclude_lists: list[str],
    ) -> list[str]:
        """Remove clip IDs found in any exclude list from the base list.

        Use this to filter out unwanted clips (e.g., remove already-used
        clips, remove specific scenarios).

        Example:
            all_clips = client.export_search_clip_ids(data_source="MADS-1M")
            boring = client.export_search_clip_ids(search="highway straight road")
            interesting = WheelClient.subtract_clip_id_lists(all_clips, boring)
        """
        if not base:
            return []
        exclude: set[str] = set()
        for lst in exclude_lists:
            if lst:
                exclude.update(lst)
        seen: set[str] = set()
        result = []
        for cid in base:
            if cid not in exclude and cid not in seen:
                seen.add(cid)
                result.append(cid)
        return result

    _VALID_MERGE_MODES = frozenset({"union", "intersection", "subtract"})

    def multi_search_export(
        self,
        search_specs: list[dict],
        max_clips_per_search: int = 100_000,
        merge_mode: str = "union",
        max_workers: int = 4,
    ) -> list[str]:
        """Run multiple searches and combine results into a single clip ID list.

        Paginatable searches (all except text-to-vector) run in parallel via
        _paginate_clip_ids(), which bypasses the session-scoped CSV endpoint.
        Text-to-vector specs run sequentially via export_search_clip_ids().

        Args:
            search_specs: List of search kwarg dicts (each passed to export_search_clip_ids).
            max_clips_per_search: Max clip IDs per search.
            merge_mode: How to combine results:
                       'union' (default) — all clips from any search (deduplicated)
                       'intersection' — only clips in ALL searches
                       'subtract' — first search minus all subsequent searches
            max_workers: Max parallel threads for paginatable specs (default 4).

        Returns:
            Deduplicated list of clip IDs.

        Example:
            clip_ids = client.multi_search_export([
                {"search": "urban intersection", "data_source": "MADS-1M"},
                {"classifier_select": "Snow", "probability_threshold": 0.7},
                {"search": "rain", "data_source": "MADS-1M"},
            ])
            client.save_clip_ids(clip_ids, "urban_or_weather.txt")
        """
        if merge_mode not in self._VALID_MERGE_MODES:
            raise ValueError(
                f"merge_mode must be one of {sorted(self._VALID_MERGE_MODES)}, got {merge_mode!r}"
            )

        lists: list[list[str]] = [[] for _ in search_specs]

        paginatable: list[int] = []
        sequential: list[int] = []
        for i, spec in enumerate(search_specs):
            if spec.get("semantic_search_text") or spec.get("visual_search_text"):
                sequential.append(i)
            else:
                paginatable.append(i)

        if len(paginatable) > 1:
            n_per_page = 20
            _PAGINATE_RESERVED = {"n", "max_pages", "max_workers"}

            def _paginate_spec(idx: int) -> list[str]:
                spec = {k: v for k, v in search_specs[idx].items() if k not in _PAGINATE_RESERVED}
                max_pages = min((max_clips_per_search - 1) // n_per_page + 1, 2000)
                return self._paginate_clip_ids(
                    max_pages=max_pages, n=n_per_page, **spec,
                )[:max_clips_per_search]

            with ThreadPoolExecutor(max_workers=min(len(paginatable), max_workers)) as pool:
                future_to_idx = {
                    pool.submit(_paginate_spec, idx): idx for idx in paginatable
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        lists[idx] = future.result()
                    except Exception as e:
                        warnings.warn(
                            f"multi_search_export: search {idx} failed: {e}",
                            stacklevel=2,
                        )
        elif paginatable:
            idx = paginatable[0]
            lists[idx] = self.export_search_clip_ids(
                max_clips=max_clips_per_search, **search_specs[idx],
            )

        for idx in sequential:
            lists[idx] = self.export_search_clip_ids(
                max_clips=max_clips_per_search, **search_specs[idx],
            )

        if merge_mode == "intersection":
            return self.intersect_clip_id_lists(*lists)
        if merge_mode == "subtract":
            if len(lists) < 2:
                return lists[0] if lists else []
            return self.subtract_clip_id_lists(lists[0], *lists[1:])
        return self.merge_clip_id_lists(*lists)

    def score_clips(
        self,
        search_dimensions: list[tuple[str, dict]],
        n_per_dimension: int = 20,
    ) -> list[ScoredClipID]:
        """Score clips across multiple search dimensions (parallelized).

        Each dimension is a (name, search_kwargs) pair. Each clip found by any
        dimension gets a score vector showing how it ranked in each search.
        Dimensions are searched in parallel (up to 6 threads) since each
        search() call is stateless.

        Use this to produce multi-dimensional relevance scores that downstream
        pipelines can use for smooth data weighting (e.g., how "urban", how
        "rainy", how "interesting" each clip is).

        Args:
            search_dimensions: List of (dimension_name, search_kwargs) tuples.
                              search_kwargs are passed to search().
            n_per_dimension: Results per dimension (server caps at 20).

        Returns:
            List of ScoredClipID, one per unique clip found, with scores from
            each dimension that found it. Sorted by aggregate_score descending.

        Example:
            scored = client.score_clips([
                ("urban", {"search": "urban intersection", "data_source": "MADS"}),
                ("weather", {"classifier_select": "Rain", "probability_threshold": 0.5}),
                ("interesting", {"classifier_select": "interesting", "probability_threshold": 0.5}),
            ])
            client.save_scored_clip_ids(scored, "multi_scored.tsv")
        """
        clip_scores: dict[str, dict[str, float]] = {}
        _succeeded = 0
        lock = threading.Lock()

        def _score_dimension(dim_name: str, search_kwargs: dict) -> None:
            nonlocal _succeeded
            try:
                _, results = self.search(n=min(n_per_dimension, 20), **search_kwargs)
            except Exception as e:
                warnings.warn(f"score_clips: dimension {dim_name!r} failed: {e}", stacklevel=2)
                return
            with lock:
                _succeeded += 1
                for r in results:
                    if r.clip_id not in clip_scores:
                        clip_scores[r.clip_id] = {}
                    scored = ScoredClipID.from_search_result(r, dimension=dim_name)
                    clip_scores[r.clip_id].update(scored.scores)

        workers = min(len(search_dimensions), 6)
        if workers <= 1:
            for dim_name, search_kwargs in search_dimensions:
                _score_dimension(dim_name, search_kwargs)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_score_dimension, dim_name, search_kwargs): dim_name
                    for dim_name, search_kwargs in search_dimensions
                }
                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception as e:
                        warnings.warn(
                            f"score_clips: dimension {futures[f]!r} raised: {e}",
                            stacklevel=2,
                        )

        if search_dimensions and _succeeded == 0:
            warnings.warn("score_clips: ALL dimensions failed — returning empty results", stacklevel=2)

        out = [ScoredClipID(clip_id=cid, scores=scores) for cid, scores in clip_scores.items()]
        out.sort(key=lambda s: s.aggregate_score, reverse=True)
        return out

    def score_clips_large(
        self,
        search_dimensions: list[tuple[str, dict]],
        max_clips_per_dimension: int = 10_000,
        progress_fn: Any | None = None,
    ) -> list[ScoredClipID]:
        """Score clips at scale across multiple search dimensions.

        Like score_clips() but uses export_search_clip_ids() for each dimension
        to handle large result sets, then enriches the top 20 with real scores
        via ``search(n=20)``.

        For dimensions that produce scores (classifier, similarity), the top-50
        clips get fine-grained scores from the search results. All remaining
        clips get a binary presence score (1.0 = present in dimension).
        Enriched clips get BOTH an aggregate score AND sub-dimension
        detail (e.g., ``"urban": 0.85, "urban:classifier": 0.9``).

        Dimensions are processed **sequentially** because export_search_clip_ids()
        relies on session-scoped server state (current_search.csv). Parallel
        dimension exports would clobber each other's session, corrupting results.
        The pagination fallback (_paginate_clip_ids) IS stateless and safe to
        parallelize, but the primary CSV path is not.

        Args:
            search_dimensions: List of (dimension_name, search_kwargs).
            max_clips_per_dimension: Max clips exported per dimension.
            progress_fn: Optional callback(dimension_name: str, clip_count: int).

        Returns:
            List of ScoredClipID sorted by aggregate_score descending.
        """
        clip_scores: dict[str, dict[str, float]] = {}

        _large_succeeded = 0
        for dim_name, search_kwargs in search_dimensions:
            try:
                clip_ids = self.export_search_clip_ids(
                    max_clips=max_clips_per_dimension, **search_kwargs,
                )
                _large_succeeded += 1
            except Exception as e:
                warnings.warn(f"score_clips_large: dimension {dim_name!r} export failed: {e}", stacklevel=2)
                continue
            exported_set = set(clip_ids)
            for cid in clip_ids:
                if cid not in clip_scores:
                    clip_scores[cid] = {}
            if progress_fn:
                progress_fn(dim_name, len(clip_ids))

            try:
                _, results = self.search(n=20, **search_kwargs)
            except Exception:
                results = []
            enriched_ids: set[str] = set()
            for r in results:
                if r.clip_id in exported_set:
                    scored = ScoredClipID.from_search_result(r, dimension=dim_name)
                    clip_scores[r.clip_id].update(scored.scores)
                    best = scored.aggregate_score if scored.scores else 1.0
                    existing = clip_scores[r.clip_id].get(dim_name, 0.0)
                    clip_scores[r.clip_id][dim_name] = max(best, existing)
                    enriched_ids.add(r.clip_id)

            for cid in clip_ids:
                if cid not in enriched_ids:
                    clip_scores[cid][dim_name] = 1.0

        if search_dimensions and _large_succeeded == 0:
            warnings.warn("score_clips_large: ALL dimensions failed — returning empty results", stacklevel=2)

        out = [ScoredClipID(clip_id=cid, scores=scores) for cid, scores in clip_scores.items()]
        out.sort(key=lambda s: s.aggregate_score, reverse=True)
        return out

    def score_clips_by_similarity(
        self,
        target_clip_ids: list[str],
        base_clip_ids: list[str] | None = None,
        modes: list[str] | None = None,
        data_source: str | None = None,
        n_per_target: int = 50,
    ) -> list[ScoredClipID]:
        """Score clips by their similarity to one or more target clips.

        For each target clip, runs similarity search (cosmos/trajectory) and
        assigns the similarity score to every clip found. Clips similar to
        multiple targets accumulate scores across all of them.

        This is the core method for "find me clips LIKE this one" with smooth
        scores — downstream pipelines can use these to weight training data
        by proximity to exemplar clips.

        Args:
            target_clip_ids: Clip IDs to measure similarity against.
            base_clip_ids: If provided, only score these clips (filter results
                          to this set). If None, score all clips found.
            modes: Similarity modes — subset of ['cosmos', 'trajectory'].
                   Default: both.
            data_source: Filter results to this data source.
            n_per_target: Number of similar clips per target per mode (server caps at 20).

        Returns:
            List of ScoredClipID sorted by aggregate_score descending.
            Scores are named like "similar_to:<target_clip_id>:<mode>".

        Example:
            # Score clips by similarity to a known good example
            scored = client.score_clips_by_similarity(
                target_clip_ids=["dd87da72-...", "a1b2c3d4-..."],
                data_source="MADS-1M",
            )
            client.save_scored_clip_ids(scored, "similar_to_exemplars.tsv")

            # Combine similarity scoring with search-based scoring
            search_scored = client.score_clips([
                ("urban", {"search": "urban", "data_source": "MADS"}),
            ])
            sim_scored = client.score_clips_by_similarity(["dd87da72-..."])
            combined = WheelClient.merge_scored_clips(search_scored, sim_scored)
        """
        if modes is None:
            modes = ["cosmos", "trajectory"]
        base_set = set(base_clip_ids) if base_clip_ids else None
        clip_scores: dict[str, dict[str, float]] = {}
        n = min(n_per_target, 50)

        # Flat thread pool over (target, mode) pairs — avoids nested pools
        # from find_similar_to_clip() and parallelizes across all targets.
        _mode_fn = {
            "cosmos": lambda cid: self.semantic_search_by_clip(
                cid, data_source=data_source, n=n),
            "trajectory": lambda cid: self.trajectory_search_by_clip(
                cid, data_source=data_source, n=n),
        }
        tasks = [(tid, m) for tid in target_clip_ids for m in modes if m in _mode_fn]
        workers = min(len(tasks), 8) if tasks else 1

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_mode_fn[m], tid): (tid, m) for tid, m in tasks
            }
            for future in as_completed(futures):
                tid, mode = futures[future]
                try:
                    raw = future.result()
                    result_list = raw[1] if isinstance(raw, tuple) and len(raw) >= 2 else raw
                except Exception as e:
                    warnings.warn(
                        f"score_clips_by_similarity: {mode!r} for "
                        f"{tid[:12]!r} failed: {e}",
                        stacklevel=2,
                    )
                    continue
                short_id = tid[:12]
                for r in result_list:
                    if base_set is not None and r.clip_id not in base_set:
                        continue
                    if r.clip_id not in clip_scores:
                        clip_scores[r.clip_id] = {}
                    score = r.best_score if r.best_score is not None else 0.0
                    dim_key = f"similar_to:{short_id}:{mode}"
                    existing = clip_scores[r.clip_id].get(dim_key, 0.0)
                    clip_scores[r.clip_id][dim_key] = max(existing, score)

        out = [ScoredClipID(clip_id=cid, scores=scores) for cid, scores in clip_scores.items()]
        out.sort(key=lambda s: s.aggregate_score, reverse=True)
        return out

    @staticmethod
    def merge_scored_clips(*scored_lists: list[ScoredClipID]) -> list[ScoredClipID]:
        """Merge multiple ScoredClipID lists, combining scores per clip.

        Clips appearing in multiple lists get the union of all their scores.
        For duplicate dimension names, the maximum score is kept.

        Use this to combine search-based scores with similarity-based scores,
        or to merge scored exports from different runs.

        Returns:
            Merged list sorted by aggregate_score descending.

        Example:
            search_scores = client.score_clips([("urban", {...})])
            sim_scores = client.score_clips_by_similarity(["dd87da72-..."])
            combined = WheelClient.merge_scored_clips(search_scores, sim_scores)
            client.save_scored_clip_ids(combined, "combined_scores.tsv")
        """
        merged: dict[str, dict[str, float]] = {}
        for lst in scored_lists:
            if not lst:
                continue
            for s in lst:
                if s.clip_id not in merged:
                    merged[s.clip_id] = {}
                for dim, val in s.scores.items():
                    existing = merged[s.clip_id].get(dim, 0.0)
                    merged[s.clip_id][dim] = max(existing, val)
        out = [ScoredClipID(clip_id=cid, scores=scores) for cid, scores in merged.items()]
        out.sort(key=lambda s: s.aggregate_score, reverse=True)
        return out

    def save_scored_clip_ids(
        self,
        scored: list[ScoredClipID],
        filename: str,
        output_dir: Path | str | None = None,
    ) -> Path:
        """Save scored clip IDs as TSV with one column per score dimension.

        Format: clip_id \\t dim1 \\t dim2 \\t ... \\t aggregate
        Downstream pipelines can read this to get per-clip relevance signals
        for smooth data weighting without needing to re-query the SIL Wheel.

        Args:
            scored: List of ScoredClipID objects.
            filename: Output filename.
            output_dir: Override the default output directory. Also respects
                        the WHEEL_OUTPUT_DIR environment variable.

        Returns:
            Path to the saved TSV file.
        """
        out_dir = self._resolve_output_dir(output_dir) or self._output_dir()
        out_path = (out_dir / filename).resolve()
        try:
            out_path.relative_to(out_dir.resolve())
        except ValueError:
            raise ValueError(f"Filename escapes output directory: {filename!r}") from None

        def _sanitize_dim(name: str) -> str:
            return (name.replace("\t", "_").replace("\n", "_")
                    .replace("\r", "_").replace("\0", "_"))

        all_dims: list[str] = []
        seen_dims: set[str] = set()
        for s in scored:
            for d in s.scores:
                if d not in seen_dims:
                    seen_dims.add(d)
                    all_dims.append(d)

        safe_dims = [_sanitize_dim(d) for d in all_dims]
        with open(out_path, "w", encoding="utf-8", buffering=1 << 16) as f:
            if safe_dims:
                header = "clip_id\t" + "\t".join(safe_dims) + "\taggregate\n"
            else:
                header = "clip_id\taggregate\n"
            f.write(header)
            chunk: list[str] = []
            for s in scored:
                cid = s.clip_id.replace("\t", "").replace("\n", "").replace("\r", "").replace("\0", "")
                if not cid:
                    continue
                vals = [f"{s.scores.get(d, 0.0):.6f}" for d in all_dims]
                agg = f"{s.aggregate_score:.6f}"
                chunk.append(f"{cid}\t" + "\t".join(vals) + f"\t{agg}")
                if len(chunk) >= 10_000:
                    f.write("\n".join(chunk) + "\n")
                    chunk.clear()
            if chunk:
                f.write("\n".join(chunk) + "\n")
        return out_path

    def scenario_inventory(self, data_source: str = "MADS") -> dict:
        """Get a summary of what's available for a data source.

        Runs classifiers, leaderboard, and clip count in parallel for ~2-3x speedup.
        """
        with ThreadPoolExecutor(max_workers=3) as pool:
            f_cls = pool.submit(self.get_classifiers)
            f_lb = pool.submit(self.get_leaderboard)
            f_total = pool.submit(self.search, data_source=data_source, n=1)
            try:
                classifiers = f_cls.result()
            except Exception:
                classifiers = {}
            try:
                leaderboard = f_lb.result()
            except Exception:
                leaderboard = {}
            try:
                total, _ = f_total.result()
            except Exception:
                total = 0

        return {
            "data_source": data_source,
            "total_clips": total,
            "trained_classifiers": classifiers.get("trained") or [],
            "num_untrained_labels": len(classifiers.get("untrained") or []),
            "leaderboard_models": leaderboard.get("models_by_leaderboard") or {},
            "annotation_counts": classifiers.get("number_of_annotations") or {},
        }

    # ── Local Output Helpers ────────────────────────────────────────

    @staticmethod
    def _output_dir() -> Path:
        env_dir = os.environ.get("WHEEL_OUTPUT_DIR")
        if env_dir:
            d = Path(env_dir)
        else:
            d = Path.cwd()
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _resolve_output_dir(output_dir: Path | str | None) -> Path | None:
        """Resolve the output directory from explicit param, env var, or default.

        Returns None when no override is configured — callers fall through to
        _output_dir() for the package-relative default.
        """
        if output_dir is not None and str(output_dir):
            d = Path(output_dir)
            d.mkdir(parents=True, exist_ok=True)
            return d
        env_dir = os.environ.get("WHEEL_OUTPUT_DIR", "").strip()
        if env_dir:
            d = Path(env_dir)
            d.mkdir(parents=True, exist_ok=True)
            return d
        return None

    def save_results(
        self,
        results: list[SearchResult],
        filename: str,
        include_urls: bool = True,
        output_dir: Path | str | None = None,
    ) -> Path:
        """Save search results as TSV.

        Returns the path to the saved file. Includes clip IDs, scores,
        captions, and optionally browser URLs.

        Args:
            output_dir: Override the default output directory. Also respects
                        the WHEEL_OUTPUT_DIR environment variable.
        """
        out_dir = self._resolve_output_dir(output_dir) or self._output_dir()
        out_path = (out_dir / filename).resolve()
        try:
            out_path.relative_to(out_dir.resolve())
        except ValueError:
            raise ValueError(f"Filename escapes output directory: {filename!r}") from None
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("clip_id\tscore\tdata_source\tcaption\turl\n")
            for r in results:
                s = r.best_score
                if s is not None and not (math.isnan(s) or math.isinf(s)):
                    score = f"{s:.4f}"
                else:
                    score = ""
                cap = r.caption_text[:200].replace("\t", " ").replace("\n", " ").replace("\0", "")
                url = self.clip_url(r.clip_id) if include_urls else ""
                ds = r.data_source.replace("\t", " ").replace("\n", " ")
                f.write(f"{r.clip_id}\t{score}\t{ds}\t{cap}\t{url}\n")
        return out_path

    def save_clip_ids(
        self, clip_ids: list[str], filename: str,
        output_dir: Path | str | None = None,
    ) -> Path:
        """Save a list of clip IDs (one per line).

        Uses buffered writes for efficiency with large lists (100k+ IDs).

        Args:
            output_dir: Override the default output directory. Also respects
                        the WHEEL_OUTPUT_DIR environment variable.
        """
        out_dir = self._resolve_output_dir(output_dir) or self._output_dir()
        out_path = (out_dir / filename).resolve()
        try:
            out_path.relative_to(out_dir.resolve())
        except ValueError:
            raise ValueError(f"Filename escapes output directory: {filename!r}") from None
        with open(out_path, "w", encoding="utf-8", buffering=1 << 16) as f:
            chunk = []
            for cid in clip_ids:
                clean = cid.replace("\n", "").replace("\r", "").replace("\0", "")
                if clean:
                    chunk.append(clean)
                if len(chunk) >= 10_000:
                    f.write("\n".join(chunk) + "\n")
                    chunk.clear()
            if chunk:
                f.write("\n".join(chunk) + "\n")
        return out_path

    def request_access(
        self,
        username: str,
        password: str,
        email: str,
        reason: str = "API access via sil-wheel-agent",
    ) -> dict:
        """Submit an access request to the SIL Wheel server.

        New users must request access before they can log in. An admin
        (typically your SIL Wheel administrator) must approve the request.
        The server sends a Slack notification on submission.

        After approval, the user can log in normally. Speed up approval
        by pinging in the project's GitHub Issues or DMing the server admin.

        Args:
            username: Desired NVIDIA username.
            password: Desired password.
            email: NVIDIA email address.
            reason: Why you need access (shown to admin).
        """
        self._validate_non_empty(username, password, email, names=("username", "password", "email"))
        self._validate_no_protocol_chars(username, password, email, reason)
        body = f"request_access::{username}::{password}::{email}::{reason}"
        resp = self._post("", data=body)
        if resp.status_code == 200:
            return {"status": "submitted", "message": "Access request submitted. An admin must approve it. Ping the project's GitHub Issues for faster approval."}
        return {"status": resp.status_code, "text": resp.text[:200]}

    # ── Result Formatting Helpers ────────────────────────────────────

    @staticmethod
    def _md_table_escape(text: str) -> str:
        """Escape characters that break markdown table cells."""
        return (text.replace("\\", "\\\\").replace("|", "\\|")
                .replace("*", "\\*").replace("_", "\\_")
                .replace("`", "\\`").replace("\n", " ")
                .replace("\r", " ").replace("\t", " ").replace("\0", ""))

    def format_results_table(
        self, results: list[SearchResult], max_rows: int = 20,
    ) -> str:
        """Format search results as a readable markdown table."""
        if not results:
            return "No results found."
        lines = ["| # | Clip ID | Source | Score | Caption |",
                  "|---|---------|--------|-------|---------|"]
        for i, r in enumerate(results[:max_rows], 1):
            bs = r.best_score
            score = f"{bs:.3f}" if bs is not None else "—"
            cap = r.caption_text[:60] + "..." if len(r.caption_text) > 60 else r.caption_text
            cap = self._md_table_escape(cap)
            cid = f"`{r.clip_id[:12]}...`" if len(r.clip_id) > 12 else (f"`{r.clip_id}`" if r.clip_id else "—")
            ds = self._md_table_escape(r.data_source) if r.data_source else "—"
            lines.append(f"| {i} | {cid} | {ds} | {score} | {cap} |")
        if len(results) > max_rows:
            lines.append(f"| ... | *{len(results) - max_rows} more results* | | | |")
        return "\n".join(lines)

    def format_results_with_urls(
        self, results: list[SearchResult], max_rows: int = 10,
    ) -> str:
        """Format results as markdown with clickable browser links."""
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results[:max_rows], 1):
            url = self.clip_url(r.clip_id)
            bs = r.best_score
            score = f" (score: {bs:.3f})" if bs is not None else ""
            cap = r.caption_text[:80] if r.caption_text else "No caption"
            cap = self._md_escape(cap)
            cid = f"{r.clip_id[:16]}..." if len(r.clip_id) > 16 else (r.clip_id or "—")
            lines.append(f"{i}. [{self._md_escape(cid)}]({url}){score}")
            lines.append(f"   {cap}")
        if len(results) > max_rows:
            lines.append(f"\n*...and {len(results) - max_rows} more results*")
        return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────


_BLOCKED_OUTPUT_DIRS = frozenset({
    "etc", "usr", "bin", "sbin", "lib", "boot", "proc", "sys", "dev",
})

def _safe_open_output(filepath: str, mode: str = "w"):
    """Open a file for writing, blocking path traversal and system directories."""
    raw = Path(filepath).expanduser()
    if ".." in raw.parts:
        print(f"Error: path contains '..': {filepath!r}", file=sys.stderr)
        sys.exit(1)
    resolved = raw.resolve()
    for path_obj in (raw, resolved):
        if path_obj.is_absolute() and len(path_obj.parts) >= 2:
            top_dir = path_obj.parts[1]
            if top_dir == "private" and len(path_obj.parts) >= 3:
                top_dir = path_obj.parts[2]
            if top_dir in _BLOCKED_OUTPUT_DIRS:
                print(f"Error: writing to system directory not allowed: {filepath!r}", file=sys.stderr)
                sys.exit(1)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return open(resolved, mode, encoding="utf-8")


def _format_result(r: SearchResult, client: WheelClient | None = None) -> str:
    scores = []
    for label, val in [("cosmos", r.semantic_clip_score), ("text", r.semantic_text_score),
                        ("clip", r.visual_score), ("traj", r.trajectory_score),
                        ("cls", r.classifier_score), ("cluster", r.cluster_distance_score)]:
        if val is not None and math.isfinite(val):
            scores.append(f"{label}={val:.3f}")
    score_str = " ".join(scores) if scores else ""
    cap_preview = r.caption_text[:80]
    line = f"  {r.clip_id} [{r.data_source}] {score_str}"
    if cap_preview:
        ellipsis = "..." if len(r.caption_text) > 80 else ""
        line += f"\n    caption: {cap_preview}{ellipsis}"
    if client:
        line += f"\n    view: {client.clip_url(r.clip_id)}"
    return line


def main():
    parser = argparse.ArgumentParser(
        description="SIL Wheel API Client — search, export, and manage AV clip data"
    )
    parser.add_argument("--server", default=None, help=f"Server URL (default: $WHEEL_SERVER_URL or {PROD_SERVER})")
    parser.add_argument("--user", default=None, help="Username (default: $WHEEL_USERNAME)")
    parser.add_argument("--password", default=None, help="Password (default: $WHEEL_PASSWORD)")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="Show server info, data sources, classifiers")
    sub.add_parser("check", help="Quick connectivity check (no auth needed)")

    def _add_search_filters(p: argparse.ArgumentParser) -> None:
        """Add the shared search-filter arguments used by both search and export."""
        p.add_argument("--data-source", default=None, help="Filter by data source (e.g. MADS, MADS-1M)")
        p.add_argument("--semantic-text", default=None, help="Cosmos text-to-video search (SLOW on dev)")
        p.add_argument("--semantic-clip", default=None, help="Cosmos clip-to-clip similarity")
        p.add_argument("--visual-text", default=None, help="CLIP text-to-video search")
        p.add_argument("--trajectory-clip", default=None, help="Trajectory shape similarity")
        p.add_argument("--caption", default=None, help="Caption FTS5 search (fast)")
        p.add_argument("--classifier", default=None, help="Classifier label filter")
        p.add_argument("--threshold", type=float, default=0.5, help="Classifier probability threshold")
        p.add_argument("--label", default=None, help="Annotation label filter")
        p.add_argument("--speed-expr", default=None, help="Trajectory speed expression")
        p.add_argument("--pattern", default=None, help="Named trajectory pattern")
        p.add_argument("--wm-class", default=None, help="World model class name")
        p.add_argument("--wm-min-count", type=int, default=None, help="Min object count for world model filter")
        p.add_argument("--wm-max-count", type=int, default=None, help="Max object count for world model filter")
        p.add_argument("--wm-max-dist", type=float, default=None, help="Max distance (meters) for world model filter")
        p.add_argument("--wm-min-time", type=float, default=None, help="Min presence time (seconds) for world model filter")
        p.add_argument("--wm-angle-range", default=None, help="Angle sector filter (FRONT, FRONT_LEFT, etc.)")
        p.add_argument("--trajectory-shape-start", type=float, default=None, help="Trajectory window start time")
        p.add_argument("--trajectory-shape-end", type=float, default=None, help="Trajectory window end time")
        p.add_argument("--country", default=None, help="Country filter")
        p.add_argument("--left-hand-driving", action="store_true", help="Left-hand traffic filter")
        p.add_argument("--query-rewrite", action="store_true", help="Enable LLM query expansion")
        p.add_argument("--comment", default=None, help="Search by comment text")
        p.add_argument("--filter-mode", default=None, choices=["any", "all"], help="AND/OR for labels")
        p.add_argument("--numeric-filter", default=None, help="Numeric metric filter expression")
        p.add_argument("--cluster-run-id", default=None, help="Cluster run ID filter")
        p.add_argument("--cluster-id", default=None, help="Specific cluster ID filter")
        p.add_argument("--sil-apis", default=None, help="SIL API applicability filter")
        p.add_argument("--labels-to-exclude", default=None, help="Labels to exclude")
        p.add_argument("--label-types", default=None, help="Annotation type filter (manual, autolabel)")
        p.add_argument("--without-ann", action="store_true", help="Only return un-annotated clips")
        p.add_argument("--with-times", action="store_true", help="Only return clips with timed annotations")
        p.add_argument("--extra-queries", default=None, help="Additional caption queries (|| separated)")
        p.add_argument("--project-source", default=None, help="Project source filter")
        p.add_argument("--search-clipid", default=None, help="Look up by exact clip ID")

    p_search = sub.add_parser("search", help="Search clips with any combination of filters")
    _add_search_filters(p_search)
    p_search.add_argument("-n", type=int, default=10, help="Number of results (server max 20)")
    p_search.add_argument("--output", "-o", default=None, help="Save clip IDs to file")

    p_export = sub.add_parser("export", help="Export all matching clip IDs to file")
    _add_search_filters(p_export)
    p_export.add_argument("--output", "-o", required=True, help="Output file path")

    p_metrics = sub.add_parser("metrics", help="Show leaderboard or per-clip metrics")
    p_metrics.add_argument("--model", default=None, help="Per-clip metrics for this model")

    p_label = sub.add_parser("label", help="Upload labels for clips (dev server only)")
    p_label.add_argument("--clips", required=True, help="File with clip IDs (one per line, or .json)")
    p_label.add_argument("--label", required=True, help="Label name")
    p_label.add_argument("--project", default="GWS Curation", help="Project name")

    p_scenario = sub.add_parser("scenario", help="Find clips matching a scenario description")
    p_scenario.add_argument("description", help="Text description of the scenario")
    p_scenario.add_argument("--data-source", default="MADS")
    p_scenario.add_argument("--output", "-o", default=None)

    p_inv = sub.add_parser("inventory", help="Show available data for a source")
    p_inv.add_argument("--data-source", default="MADS")

    p_similar = sub.add_parser("similar", help="Find clips similar to a given clip")
    p_similar.add_argument("clip_id", help="Source clip ID")
    p_similar.add_argument("--data-source", default=None)
    p_similar.add_argument("-n", type=int, default=10)

    p_expand = sub.add_parser("expand", help="Expand a set of clip IDs by similarity")
    p_expand.add_argument("--clips", required=True, help="File with seed clip IDs")
    p_expand.add_argument("--data-source", default=None)
    p_expand.add_argument("--max-total", type=int, default=500)
    p_expand.add_argument("--output", "-o", required=True)

    p_merge = sub.add_parser("merge-clips", help="Merge/intersect/subtract clip ID files")
    p_merge.add_argument("files", nargs="+", help="Clip ID files to combine")
    p_merge.add_argument("--output", "-o", required=True, help="Output clip ID file")
    p_merge.add_argument("--mode", default="union", choices=["union", "intersection", "subtract"],
                         help="union: all clips from any file; intersection: only clips in all files; subtract: base minus others")

    p_lookup = sub.add_parser("lookup", help="Look up metadata for clips by ID")
    p_lookup.add_argument("--clips", required=True, help="File with clip IDs (one per line)")
    p_lookup.add_argument("-n", type=int, default=None, help="Limit number of lookups")

    p_vlm = sub.add_parser("vlm-judge", help="VLM Judge: score captions or validate search results")
    p_vlm.add_argument("--status", action="store_true", help="Check VLM Judge availability")
    p_vlm.add_argument("--score-caption", metavar="CLIP_ID", help="Score a specific caption for a clip")
    p_vlm.add_argument("--caption", help="Caption text to score (required with --score-caption)")
    p_vlm.add_argument("--score-clip", metavar="CLIP_ID", help="Auto-score a clip's existing caption")
    p_vlm.add_argument("--validate", metavar="QUERY", help="Validate search results against query")
    p_vlm.add_argument("--clips", help="File with clip IDs for --validate")

    p_clusters = sub.add_parser(
        "clusters",
        help="List clustering runs and inspect per-cluster TF-IDF topics",
    )
    p_clusters.add_argument(
        "--run-id", default=None,
        help="Show topics for this run. Omit to list all runs and exit.",
    )
    p_clusters.add_argument(
        "--top-k", type=int, default=20,
        help="Show only the top-K largest clusters (default: 20).",
    )
    p_clusters.add_argument(
        "--keyword", default=None,
        help="Filter to clusters whose keywords/description match this substring.",
    )
    p_clusters.add_argument(
        "--min-size", type=int, default=0,
        help="Skip clusters smaller than this (default: 0).",
    )

    p_version = sub.add_parser("version", help="Check SDK version and update if needed")
    p_version.add_argument("--update", action="store_true", help="Download latest SDK if newer version available")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(2)

    client = WheelClient(args.server)

    if args.command == "check":
        status = client.check_connection()
        print(json.dumps(status, indent=2))
        sys.exit(0 if status.get("reachable") else 1)

    if args.command == "version":
        if args.update:
            result = WheelClient.update_sdk()
            if result.get("updated"):
                print(f"Updated: {result['from_version']} → {result['to_version']}")
                for f in result.get("files_updated", []):
                    print(f"  Updated: {f}")
                print(result.get("note", ""))
            elif result.get("error"):
                print(f"Update failed: {result['error']}", file=sys.stderr)
                sys.exit(1)
            else:
                print(f"Already up to date (v{result.get('version', SDK_VERSION)})")
        else:
            result = WheelClient.check_sdk_version()
            print(f"Local:  v{result['local']}")
            if result.get("error"):
                print(f"Remote: unavailable ({result['error']})")
                sys.exit(1)
            else:
                print(f"Remote: v{result['remote']}")
                if result["up_to_date"]:
                    note = result.get("note", "up to date")
                    print(f"Status: {note}")
                else:
                    print("Status: UPDATE AVAILABLE")
                    print(f"\n{result.get('update_command', '')}")
        sys.exit(0)

    if not client.login(args.user, args.password):
        print("Authentication failed. Check WHEEL_USERNAME and WHEEL_PASSWORD in .env",
              file=sys.stderr)
        sys.exit(1)

    def _build_search_kwargs(a) -> dict[str, Any]:
        """Map CLI filter args to search() keyword arguments."""
        kw: dict[str, Any] = {}
        if a.data_source:
            kw["data_source"] = a.data_source
        if a.semantic_text:
            kw["semantic_search_text"] = a.semantic_text
        if a.semantic_clip:
            kw["semantic_search_clipid"] = a.semantic_clip
        if a.visual_text:
            kw["visual_search_text"] = a.visual_text
        if a.trajectory_clip:
            kw["trajectory_shape_clipid"] = a.trajectory_clip
        if a.caption:
            kw["search"] = a.caption
        if a.classifier:
            kw["classifier_select"] = a.classifier
            kw["probability_threshold"] = a.threshold
        if a.label:
            kw["label_filter"] = a.label
        if a.speed_expr:
            kw["search_speed"] = a.speed_expr
        if a.pattern:
            kw["trajectory_pattern"] = a.pattern
        if a.wm_class:
            kw["wm_class_name"] = a.wm_class
            kw["wm_min_count"] = a.wm_min_count if a.wm_min_count is not None else 1
            if a.wm_max_count is not None:
                kw["wm_max_count"] = a.wm_max_count
            if a.wm_max_dist is not None:
                kw["wm_max_dist"] = a.wm_max_dist
            if a.wm_min_time is not None:
                kw["wm_min_time"] = a.wm_min_time
            if a.wm_angle_range:
                kw["wm_angle_range"] = a.wm_angle_range
        if a.trajectory_shape_start is not None:
            kw["trajectory_shape_start_t"] = a.trajectory_shape_start
        if a.trajectory_shape_end is not None:
            kw["trajectory_shape_end_t"] = a.trajectory_shape_end
        if a.country:
            kw["search_country"] = a.country
        if a.left_hand_driving:
            kw["left_hand_driving"] = True
        if a.query_rewrite:
            kw["query_rewrite"] = True
        if a.comment:
            kw["search_comments"] = a.comment
        if a.filter_mode:
            kw["filter_mode"] = a.filter_mode
        if a.numeric_filter:
            kw["numeric_filter"] = a.numeric_filter
        if a.cluster_run_id:
            kw["cluster_run_id"] = a.cluster_run_id
        if a.cluster_id:
            kw["cluster_id"] = a.cluster_id
        if a.sil_apis:
            kw["sil_apis"] = a.sil_apis
        if a.labels_to_exclude:
            kw["labels_to_exclude"] = a.labels_to_exclude
        if a.label_types:
            kw["label_types"] = a.label_types
        if a.project_source:
            kw["project_source"] = a.project_source
        if a.without_ann:
            kw["without_ann"] = True
        if a.with_times:
            kw["times"] = True
        if a.extra_queries:
            kw["extra_queries"] = a.extra_queries
        if a.search_clipid:
            kw["search_clipid"] = a.search_clipid
        return kw

    if args.command == "info":
        print(json.dumps(client.whoami(), indent=2))
        sources = client.get_data_sources()
        print(f"\nData sources: {sources}")
        classifiers = client.get_classifiers()
        trained = classifiers.get("trained", [])
        print(f"\nTrained classifiers ({len(trained)}):")
        for c in trained:
            print(f"  {c}")

    elif args.command == "search":
        kwargs = _build_search_kwargs(args)
        kwargs["n"] = args.n

        total, results = client.search(**kwargs)
        print(f"Total: {total:,}")
        for r in results:
            print(_format_result(r, client))

        if args.output:
            with _safe_open_output(args.output) as f:
                for r in results:
                    f.write(r.clip_id + "\n")
            print(f"\nSaved {len(results)} clip IDs to {args.output}")

    elif args.command == "export":
        kwargs = _build_search_kwargs(args)
        clip_ids = client.export_search_clip_ids(**kwargs)
        print(f"Exported {len(clip_ids):,} clip IDs")
        with _safe_open_output(args.output) as f:
            for c in clip_ids:
                f.write(c + "\n")
        print(f"Saved to {args.output}")

    elif args.command == "metrics":
        if args.model:
            data = client.get_per_clip_metrics(args.model)
            output = json.dumps(data, indent=2, default=str)
            print(output[:2000] + ("\n... (truncated)" if len(output) > 2000 else ""))
        else:
            data = client.get_leaderboard()
            mbl = data.get("models_by_leaderboard", {})
            for lb, models in mbl.items():
                print(f"\n{lb}:")
                for m in models:
                    metrics = data.get("metrics", {}).get(m, {})
                    metric_strs = [
                        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in list(metrics.items())[:5]
                    ]
                    print(f"  {m}: {', '.join(metric_strs)}")

    elif args.command == "label":
        try:
            clip_ids = WheelClient.load_clip_ids(args.clips)
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading clip IDs from {args.clips}: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Labeling {len(clip_ids)} clips as '{args.label}' in project '{args.project}'")
        results = client.upload_labels(clip_ids, args.label, args.project)
        for r in results:
            print(f"  batch {r.get('batch', '?')}: {r.get('status', 'unknown')} ({r.get('count', '?')} clips)")

    elif args.command == "scenario":
        grouped = client.find_clips_for_scenario(
            args.description, data_source=args.data_source,
        )
        for strategy, strategy_results in grouped.items():
            print(f"\n{strategy}: {len(strategy_results)} results")
            for r in strategy_results[:3]:
                print(_format_result(r, client))
        rrf_scores: dict[str, float] = {}
        for results in grouped.values():
            for rank, r in enumerate(results, start=1):
                rrf_scores[r.clip_id] = rrf_scores.get(r.clip_id, 0.0) + 1.0 / (60 + rank)
        clip_ids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)
        print(f"\nTotal unique clip IDs: {len(clip_ids)} (ranked by Reciprocal Rank Fusion)")
        if args.output:
            with _safe_open_output(args.output) as f:
                for c in clip_ids:
                    f.write(c + "\n")
            print(f"Saved to {args.output}")

    elif args.command == "inventory":
        inv = client.scenario_inventory(args.data_source)
        print(f"Data source: {inv['data_source']}")
        print(f"Total clips: {inv['total_clips']:,}")
        print(f"\nTrained classifiers ({len(inv['trained_classifiers'])}):")
        for c in inv["trained_classifiers"]:
            print(f"  {c}")
        print(f"Untrained labels: {inv['num_untrained_labels']}")
        print(f"\nLeaderboard models:")
        for lb, models in inv["leaderboard_models"].items():
            print(f"  {lb}: {', '.join(models)}")

    elif args.command == "similar":
        print(f"Finding clips similar to {args.clip_id}...")
        results = client.find_similar_to_clip(
            args.clip_id, data_source=args.data_source, n=args.n,
        )
        for mode, mode_results in results.items():
            print(f"\n{mode.upper()} similarity ({len(mode_results)} results):")
            for r in mode_results:
                print(_format_result(r, client))

    elif args.command == "expand":
        try:
            seed_ids = WheelClient.load_clip_ids(args.clips)
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading seed clip IDs from {args.clips}: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Expanding {len(seed_ids)} seed clips...")
        expanded = client.expand_clip_set(
            seed_ids, data_source=args.data_source, max_total=args.max_total,
        )
        print(f"Expanded to {len(expanded)} total clips")
        with _safe_open_output(args.output) as f:
            for cid, score, source in expanded:
                s = f"{score:.4f}" if math.isfinite(score) else "0.0000"
                f.write(f"{cid}\t{s}\t{source}\n")
        print(f"Saved to {args.output} (clip_id\\tscore\\tsource_clip)")

    elif args.command == "merge-clips":
        lists: list[list[str]] = []
        for filepath in args.files:
            try:
                ids = WheelClient.load_clip_ids(filepath)
            except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error loading clip IDs from {filepath}: {e}", file=sys.stderr)
                sys.exit(1)
            lists.append(ids)
            print(f"  Loaded {filepath}: {len(ids):,} clips")
        if args.mode == "union":
            result = WheelClient.merge_clip_id_lists(*lists)
        elif args.mode == "intersection":
            result = WheelClient.intersect_clip_id_lists(*lists)
        elif args.mode == "subtract":
            if len(lists) < 2:
                print("Error: subtract mode requires at least 2 files (base + exclude)", file=sys.stderr)
                sys.exit(1)
            result = WheelClient.subtract_clip_id_lists(lists[0], *lists[1:])
        else:
            result = WheelClient.merge_clip_id_lists(*lists)
        with _safe_open_output(args.output) as f:
            for cid in result:
                f.write(cid + "\n")
        print(f"Result: {len(result):,} clips ({args.mode})")
        print(f"Saved to {args.output}")

    elif args.command == "lookup":
        try:
            clip_ids = WheelClient.load_clip_ids(args.clips)
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading clip IDs from {args.clips}: {e}", file=sys.stderr)
            sys.exit(1)
        if args.n is not None:
            clip_ids = clip_ids[:args.n]
        print(f"Looking up {len(clip_ids)} clips...")
        results = client.lookup_clips_batch(clip_ids)
        found = sum(1 for v in results.values() if v is not None)
        print(f"Found {found}/{len(clip_ids)} clips")
        for cid, r in results.items():
            if r is not None:
                print(_format_result(r, client))
            else:
                print(f"  {cid} — NOT FOUND")


    elif args.command == "clusters":
        if not args.run_id:
            runs = client.get_clustering_status()
            if not runs:
                print("No clustering runs found.")
                sys.exit(0)
            print(f"{'run_id':<14} {'status':<10} {'n_clusters':>10} {'n_clips':>10}")
            for r in runs:
                rid = str(r.get('run_id', ''))
                rid_disp = (rid[:13] + '…') if len(rid) > 14 else rid
                status = str(r.get('status', '?'))
                status_disp = (status[:9] + '…') if len(status) > 10 else status
                print(
                    f"{rid_disp:<14} "
                    f"{status_disp:<10} "
                    f"{_safe_int(r.get('n_clusters')):>10} "
                    f"{_safe_int(r.get('n_clips')):>10}"
                )
            print(f"\n{len(runs)} runs total. Use --run-id <id> to inspect topics.")
            sys.exit(0)

        # When --run-id is supplied (with or without --keyword), probe the run
        # FIRST so we can give a single coherent "run not found" exit code (1).
        # Without this probe, the --keyword path would conflate "no matches"
        # (legitimate exit 0) with "run doesn't exist" (should be exit 1).
        probe = client.get_clustering_results(args.run_id)
        if not isinstance(probe, dict) or "error" in probe:
            print(
                f"Run {args.run_id!r} not found or not finished. "
                "Try `clusters` (no args) to list runs.",
                file=sys.stderr,
            )
            sys.exit(1)

        if args.keyword:
            matches = client.find_clusters_by_keyword(
                args.run_id, args.keyword, top_k=args.top_k,
            )
            if not matches:
                print(
                    f"No clusters in run {args.run_id} matched "
                    f"keyword {args.keyword!r}."
                )
                sys.exit(0)
            print(
                f"\n{len(matches)} clusters matched {args.keyword!r} "
                f"(by cluster size desc):\n"
            )
            for m in matches:
                rep = m.get("representative_clip_id") or ""
                desc = m.get("description") or ", ".join(m["keywords"][:3])
                print(
                    f"  cluster {m['cluster_id']:>4} "
                    f"({m['cluster_size']:>5} clips, match={m['match']}): "
                    f"{desc}"
                )
                if rep:
                    print(f"      rep clip: {rep}")
            sys.exit(0)

        summary = client.summarize_clustering_run(
            args.run_id, top_k=args.top_k, min_cluster_size=args.min_size,
        )
        if not summary:
            # Should be unreachable now (probe above already exited 1 on
            # missing run) but keep the guard so future code changes can't
            # silently regress to "summary is empty -> exit 0".
            print(
                f"Run {args.run_id} not found or not finished. "
                "Try `clusters` (no args) to list runs.",
                file=sys.stderr,
            )
            sys.exit(1)
        n_with_topics = summary["clusters_with_topics"]
        cap_pct = summary["captions_coverage"] * 100
        print(
            f"Run {summary['run_id']}: {summary['n_clusters']} clusters, "
            f"{summary['n_clips']:,} clips. "
            f"{n_with_topics} clusters have topics; "
            f"caption coverage {summary['captions_found']:,}/"
            f"{summary['captions_total']:,} ({cap_pct:.1f}%)."
        )
        if summary.get("caption_model"):
            print(f"Caption model: {summary['caption_model']}")
        rows = summary["clusters"]
        if not rows:
            print("(no clusters to display)")
            sys.exit(0)
        print(f"\nTop {len(rows)} clusters by size:\n")
        for row in rows:
            theme = row["theme"]
            print(
                f"  cluster {row['cluster_id']:>4} "
                f"({row['cluster_size']:>5} clips): {theme}"
            )
        sys.exit(0)

    elif args.command == "vlm-judge":
        if args.status:
            print(json.dumps(client.vlm_judge_status(), indent=2))
        elif args.score_caption:
            if not args.caption:
                print("Error: --caption required with --score-caption", file=sys.stderr)
                sys.exit(1)
            result = client.vlm_judge_caption_score(args.score_caption, args.caption)
            print(json.dumps(result, indent=2))
        elif args.score_clip:
            result = client.vlm_judge_score_clip(args.score_clip)
            if "error" in result:
                print(f"Error: {result['error']}", file=sys.stderr)
                sys.exit(1)
            print(json.dumps(result, indent=2))
        elif args.validate:
            if not args.clips:
                print("Error: --clips required with --validate", file=sys.stderr)
                sys.exit(1)
            try:
                clip_ids = WheelClient.load_clip_ids(args.clips)
            except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error loading clip IDs: {e}", file=sys.stderr)
                sys.exit(1)
            results = client.vlm_judge_validate_search(args.validate, clip_ids)
            matched = sum(1 for r in results if r.get("match"))
            print(f"VLM Judge: {matched}/{len(results)} clips match '{args.validate}'")
            for r in results:
                mark = "+" if r.get("match") else "-"
                print(f"  [{mark}] {r.get('clip_id', '?')}: {r.get('reasoning', '')[:80]}")
        else:
            print("Use --status, --score-caption, or --validate. See --help.", file=sys.stderr)
            sys.exit(2)


if __name__ == "__main__":
    main()
