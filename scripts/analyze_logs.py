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
Server log analyzer

Parses server logs produced by launch_server.py (format: "%(asctime)s %(message)s")
where messages are prefixed by "[user=<username>]" via RequestHandler.log_message.

It computes useful usage metrics, such as:
- Unique users per day and overall, with daily averages
- Request counts by endpoint/path (with and without query), by method, and by hour
- HTTP status distribution (overall and per top endpoints)
- Anonymous vs authenticated request ratios
- S3 error counts and codes (e.g., missing videos)
- Video/autolabel streaming endpoint usage
 - Cache effectiveness: hit/miss counts and average durations (by parsing
   lines like "from cache 276374 took 0.000113" and
   "from search 333 took 0.271533")

Usage:
  python analyze_logs.py /path/to/logs /path/to/plots \
      [--json-out summary.json] [--show] [-r] [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--top-n N]

Notes:
- Reads plain text log files; use -r to scan subdirectories.
- Supports date filtering on parsed timestamps.

This script reads files without modifying them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from typing import Dict, Iterable, Optional
from urllib.parse import parse_qs, urlsplit

# Regexes for parsing lines
TS_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}),\d{3}\s+(?P<rest>.*)$"
)
USER_RE = re.compile(r"^\[user=(?P<user>[^\]]+)\]\s+(?P<msg>.*)$")
REQ_RE = re.compile(
    r'"(?P<method>[A-Z]+)\s+(?P<path>[^\s]+)\s+HTTP/(?P<httpver>[\d\.]+)"\s+(?P<code>\d{3})'
)
S3_ERR_RE = re.compile(r"S3 error fetching\s+(?P<key>[^:]+):\s+(?P<err>\w+)")
# Capture optional filter key prefix before cache/search timing
CACHE_HIT_RE = re.compile(
    r"^(?P<prefix>.*?)\s*from\s+cache\s+(?P<id>\d+)\s+took\s+(?P<sec>[0-9]*\.?[0-9]+)\s*$"
)
CACHE_MISS_RE = re.compile(
    r"^(?P<prefix>.*?)\s*from\s+search\s+(?P<id>\d+)\s+took\s+(?P<sec>[0-9]*\.?[0-9]+)\s*$"
)
# Current server format: "<key> → <count> results in <sec> s"
CACHE_MISS_RESULT_RE = re.compile(
    r"^(?P<prefix>.*?)\s*→\s*(?P<id>\d+)\s+results\s+in\s+(?P<sec>[0-9]*\.?[0-9]+)\s+s\s*$"
)
ADD_RE = re.compile(
    r"Adding\s+label\s+(?P<label>.+?)\s+for\s+project\s+(?P<project>.+?)\s+took\s+(?P<sec>[0-9]*\.?[0-9]+)"
)
REMOVE_RE = re.compile(
    r"Removing\s+label\s+(?P<label>.+?)\s+for\s+project\s+(?P<project>.+?)\s+took\s+(?P<sec>[0-9]*\.?[0-9]+)"
)
VERIFY_RE = re.compile(
    r"Verifying\s+label\s+(?P<label>.+?)\s+for\s+project\s+(?P<project>.+?)\s+took\s+(?P<sec>[0-9]*\.?[0-9]+)"
)
MASS_LABEL_RE = re.compile(
    r"Labelling\s+(?P<count>\d+)\s+took\s+(?P<sec>[0-9]*\.?[0-9]+)"
)
UPLOAD_ANN_RE = re.compile(
    r"Uploading\s+(?P<count>\d+)\s+took\s+(?P<sec>[0-9]*\.?[0-9]+)"
)
AUTO_LABEL_RE = re.compile(
    r"Autolabel\s+(?P<pages>\d+)\s+pages\s+with\s+label\s+(?P<label>.+?)\s+for\s+project\s+(?P<project>.+?)\s+took\s+(?P<sec>[0-9]*\.?[0-9]+)"
)
REWRITE_RE = re.compile(r"Query rewrite: query=(?P<query>.+)")
CLASSIFIER_TRAIN_RE = re.compile(
    r"Classifier training started: label=(?P<label>.+?) embed_type=(?P<embed_type>\w+) n_pos=(?P<n_pos>\S*) n_neg=(?P<n_neg>\S*)"
)


def strip_query(path: str) -> str:
    q = path.find("?")
    return path if q == -1 else path[:q]


def first_segment(path: str) -> str:
    # returns like '/videos' or '/video' from '/video/abc.mp4'
    if not path.startswith("/"):
        return path
    parts = path.split("/")
    return "/" + (parts[1] if len(parts) > 1 and parts[1] else "")


def parse_line(line: str) -> Optional[dict]:
    m = TS_RE.match(line)
    if not m:
        return None
    ts_str = m.group("ts")
    rest = m.group("rest")
    try:
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        # Fallback for alternate separator
        ts = datetime.fromisoformat(ts_str.replace("T", " "))

    user = "unknown"
    msg = rest
    mu = USER_RE.match(rest)
    if mu:
        user = mu.group("user") or "unknown"
        msg = mu.group("msg")

    # Try HTTP request
    req = REQ_RE.search(msg)
    if req:
        method = req.group("method")
        path = req.group("path")
        code = int(req.group("code"))
        return {
            "type": "http",
            "ts": ts,
            "user": user,
            "method": method,
            "path": path,
            "path_no_q": strip_query(path),
            "first_seg": first_segment(path),
            "status": code,
            "raw": line.rstrip("\n"),
        }

    # Try S3 error message
    s3 = S3_ERR_RE.search(msg)
    if s3:
        return {
            "type": "s3_error",
            "ts": ts,
            "user": user,
            "key": s3.group("key"),
            "error": s3.group("err"),
            "raw": line.rstrip("\n"),
        }

    # Try cache hit/miss message
    mh = CACHE_HIT_RE.search(msg)
    if mh:
        try:
            sec = float(mh.group("sec"))
        except Exception:
            sec = 0.0
        return {
            "type": "cache",
            "ts": ts,
            "user": user,
            "hit": True,
            "duration_s": sec,
            "key": mh.group("prefix").strip() if mh.group("prefix") else None,
            "count": int(mh.group("id")) if mh.group("id") else None,
            "raw": line.rstrip("\n"),
        }
    mm = CACHE_MISS_RE.search(msg)
    if mm:
        try:
            sec = float(mm.group("sec"))
        except Exception:
            sec = 0.0
        return {
            "type": "cache",
            "ts": ts,
            "user": user,
            "hit": False,
            "duration_s": sec,
            "key": mm.group("prefix").strip() if mm.group("prefix") else None,
            "count": int(mm.group("id")) if mm.group("id") else None,
            "raw": line.rstrip("\n"),
        }
    mm2 = CACHE_MISS_RESULT_RE.search(msg)
    if mm2:
        try:
            sec = float(mm2.group("sec"))
        except Exception:
            sec = 0.0
        return {
            "type": "cache",
            "ts": ts,
            "user": user,
            "hit": False,
            "duration_s": sec,
            "key": mm2.group("prefix").strip() if mm2.group("prefix") else None,
            "count": int(mm2.group("id")) if mm2.group("id") else None,
            "raw": line.rstrip("\n"),
        }

    # Try annotation events
    ma = ADD_RE.search(msg)
    if ma:
        return {
            "type": "annotation_event",
            "ts": ts,
            "user": user,
            "event": "added",
            "count": 1,
            "raw": line.rstrip("\n"),
        }
    mr = REMOVE_RE.search(msg)
    if mr:
        return {
            "type": "annotation_event",
            "ts": ts,
            "user": user,
            "event": "deleted",
            "count": 1,
            "raw": line.rstrip("\n"),
        }
    mv = VERIFY_RE.search(msg)
    if mv:
        return {
            "type": "annotation_event",
            "ts": ts,
            "user": user,
            "event": "verified",
            "count": 1,
            "raw": line.rstrip("\n"),
        }
    mmass = MASS_LABEL_RE.search(msg)
    if mmass:
        try:
            cnt = int(mmass.group("count"))
        except Exception:
            cnt = 0
        return {
            "type": "annotation_event",
            "ts": ts,
            "user": user,
            "event": "added",
            "count": cnt,
            "raw": line.rstrip("\n"),
        }
    mupl = UPLOAD_ANN_RE.search(msg)
    if mupl:
        try:
            cnt = int(mupl.group("count"))
        except Exception:
            cnt = 0
        return {
            "type": "annotation_event",
            "ts": ts,
            "user": user,
            "event": "added",
            "count": cnt,
            "raw": line.rstrip("\n"),
        }
    maut = AUTO_LABEL_RE.search(msg)
    if maut:
        try:
            pages = int(maut.group("pages"))
        except Exception:
            pages = 0
        return {
            "type": "annotation_event",
            "ts": ts,
            "user": user,
            "event": "autolabeled",
            "count": pages * 6,
            "raw": line.rstrip("\n"),
        }

    mrw = REWRITE_RE.search(msg)
    if mrw:
        return {
            "type": "rewrite",
            "ts": ts,
            "user": user,
            "query": mrw.group("query").strip(),
            "raw": line.rstrip("\n"),
        }

    mct = CLASSIFIER_TRAIN_RE.search(msg)
    if mct:
        return {
            "type": "classifier_train",
            "ts": ts,
            "user": user,
            "label": mct.group("label"),
            "embed_type": mct.group("embed_type"),
            "raw": line.rstrip("\n"),
        }

    # Generic log
    return {
        "type": "other",
        "ts": ts,
        "user": user,
        "raw": line.rstrip("\n"),
        "message": msg,
    }


def aggregate_records(records: Iterable[dict], top_n: int = 15) -> dict:
    # internal cap for top lists used for display/plots
    TOP_N = max(1, int(top_n))
    users_by_day: Dict[str, set] = defaultdict(set)
    user_request_counts = Counter()
    req_total = 0
    anon_reqs = 0
    http_methods = Counter()
    http_status = Counter()
    http_status_by_path = defaultdict(Counter)  # path_no_q -> Counter
    req_by_path = Counter()
    req_by_first_seg = Counter()
    req_by_hour = Counter()  # hour string HH
    s3_errors = Counter()
    s3_error_keys = Counter()
    # Search usage and data sources
    search_usage = Counter()
    data_source_counts = Counter()
    visual_search_texts = Counter()
    semantic_text_searches = Counter()
    caption_embed_search_texts: Counter = Counter()
    # Annotation label filters usage (from `filter=` query param)
    label_filter_counts = Counter()
    # Caption search text usage (from `search=` query param)
    caption_search_texts = Counter()
    cache_hits = 0
    cache_misses = 0
    cache_hit_time = 0.0
    cache_miss_time = 0.0
    cache_hits_by_day: Dict[str, int] = defaultdict(int)
    cache_misses_by_day: Dict[str, int] = defaultdict(int)
    # Track search durations by type (from cache misses)
    search_time_by_type: Dict[str, list[float]] = defaultdict(list)
    # Zero-result tracking
    total_searches = 0
    zero_searches = 0
    searches_by_type = Counter()
    zero_by_type = Counter()
    # Track zero-result queries by text per search type
    zero_semantic_texts = Counter()
    zero_caption_texts = Counter()
    zero_visual_texts = Counter()
    zero_caption_embed_texts = Counter()
    # Query rewrite tracking
    rewrite_total = 0
    rewrite_users: Counter = Counter()
    rewrite_query_texts: Counter = Counter()
    rewrite_by_type: Dict[str, int] = defaultdict(int)
    rewrite_texts_by_type: Dict[str, Counter] = defaultdict(Counter)
    # Classifier training tracking
    classifier_trains_by_label: Counter = Counter()
    classifier_trains_by_embed_type: Counter = Counter()
    classifier_trains_by_user: Counter = Counter()
    # Annotation events per day
    annotations_daily: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"added": 0, "deleted": 0, "verified": 0, "autolabeled": 0}
    )

    for r in records:
        day = r["ts"].strftime("%Y-%m-%d")
        hour = r["ts"].strftime("%H")
        user = r.get("user", "unknown")
        if user and user != "anonymous":
            users_by_day[day].add(user)

        if r["type"] == "http":
            req_total += 1
            method = r["method"]
            path_no_q = r["path_no_q"]
            first_seg = r["first_seg"]
            code = r["status"]

            http_methods[method] += 1
            http_status[code] += 1
            http_status_by_path[path_no_q][code] += 1
            req_by_path[path_no_q] += 1
            req_by_first_seg[first_seg] += 1
            req_by_hour[hour] += 1
            if user == "anonymous":
                anon_reqs += 1
            else:
                user_request_counts[user] += 1
            # Inspect query params for search and data_source usage
            try:
                qs = parse_qs(urlsplit(r["path"]).query)
                if qs.get("semantic_search_clipid", [None])[0]:
                    search_usage["semantic_video_to_video"] += 1
                sem_txt = qs.get("semantic_search_text", [None])[0]
                if sem_txt:
                    search_usage["semantic_text_to_video"] += 1
                    try:
                        sem_norm = (sem_txt or "").strip().lower()
                        if sem_norm:
                            semantic_text_searches[sem_norm] += 1
                    except Exception:
                        pass
                if qs.get("trajectory_shape_clipid", [None])[0]:
                    search_usage["trajectory_shape"] += 1
                # Caption-based search (caption text)
                cap_txt = qs.get("search", [None])[0]
                if cap_txt:
                    search_usage["caption_search"] += 1
                    try:
                        cap_norm = (cap_txt or "").strip().lower()
                        if cap_norm:
                            caption_search_texts[cap_norm] += 1
                    except Exception:
                        pass
                # Classifier search
                if qs.get("classifier_run_id", [None])[0] or qs.get("classifier_select", [None])[0]:
                    search_usage["classifier_search"] += 1
                # World-model search
                if qs.get("wm_class_name", [None])[0]:
                    search_usage["wm_search"] += 1
                # Visual search by text
                vtxt = qs.get("visual_search_text", [None])[0]
                if vtxt:
                    search_usage["visual_search_text"] += 1
                    try:
                        # Normalize by trimming and lowercasing to avoid splitting
                        # counts on case-only differences (e.g., "Dog" vs "dog").
                        vtxt_norm = (vtxt or "").strip().lower()
                        if vtxt_norm:
                            visual_search_texts[vtxt_norm] += 1
                    except Exception:
                        pass
                # Visual search by image
                if qs.get("visual_search_image_id", [None])[0]:
                    search_usage["visual_search_image"] += 1
                # Cluster search
                if qs.get("cluster_run_id", [None])[0]:
                    search_usage["cluster_search"] += 1
                # VLM Judge usage
                if path_no_q == "/api/vlm_judge/validate_search":
                    search_usage["vlm_judge_validate_search"] += 1
                elif path_no_q == "/api/vlm_judge/caption_score":
                    search_usage["vlm_judge_caption_score"] += 1
                # Caption embedding search
                cap_emb_txt = qs.get("caption_embed_search", [None])[0]
                if cap_emb_txt:
                    search_usage["caption_embed_search"] += 1
                    try:
                        cap_emb_norm = (cap_emb_txt or "").strip().lower()
                        if cap_emb_norm:
                            caption_embed_search_texts[cap_emb_norm] += 1
                    except Exception:
                        pass
                for stype, param in (
                    ("caption", "caption_extra_queries"),
                    ("caption-embed", "caption_embed_extra_queries"),
                    ("semantic", "semantic_extra_queries"),
                    ("visual", "visual_extra_queries"),
                ):
                    raw = qs.get(param, [None])[0]
                    if raw:
                        rewrite_by_type[stype] += 1
                        for token in raw.split("||"):
                            token = token.strip()
                            if token:
                                rewrite_texts_by_type[stype][token] += 1
                for raw in qs.get("data_source", []):
                    for token in raw.split("||") if raw else []:
                        token = token.strip()
                        if token:
                            data_source_counts[token] += 1
                # Annotation labels via `filter` (pipe-separated)
                for raw in qs.get("filter", []):
                    for token in raw.split("||") if raw else []:
                        token = token.strip()
                        if token:
                            label_filter_counts[token] += 1
            except Exception:
                pass
        elif r["type"] == "s3_error":
            s3_errors[r["error"]] += 1
            s3_error_keys[r["key"]] += 1
        elif r["type"] == "cache":
            if r.get("hit"):
                cache_hits += 1
                cache_hit_time += float(r.get("duration_s", 0.0))
                cache_hits_by_day[day] += 1
            else:
                cache_misses += 1
                cache_miss_time += float(r.get("duration_s", 0.0))
                cache_misses_by_day[day] += 1
                # Attribute miss duration to search type via the filter key
                key = (r.get("key") or "").strip()
                if key:
                    try:
                        qs = parse_qs(key)
                        # keep durations in seconds
                        dur_s = float(r.get("duration_s", 0.0))
                        if qs.get("semantic_search_clipid", [None])[0]:
                            search_time_by_type[
                                "semantic_video_to_video"
                            ].append(dur_s)
                        if qs.get("semantic_search_text", [None])[0]:
                            search_time_by_type[
                                "semantic_text_to_video"
                            ].append(dur_s)
                        if qs.get("trajectory_shape_clipid", [None])[0]:
                            search_time_by_type["trajectory_shape"].append(
                                dur_s
                            )
                        if qs.get("search", [None])[0]:
                            search_time_by_type["caption_search"].append(dur_s)
                        if qs.get("classifier_run_id", [None])[0] or qs.get("classifier_select", [None])[0]:
                            search_time_by_type["classifier_search"].append(
                                dur_s
                            )
                        if qs.get("wm_class_name", [None])[0]:
                            search_time_by_type["wm_search"].append(dur_s)
                        if qs.get("visual_search_text", [None])[0]:
                            search_time_by_type["visual_search_text"].append(
                                dur_s
                            )
                        if qs.get("visual_search_image_id", [None])[0]:
                            search_time_by_type["visual_search_image"].append(
                                dur_s
                            )
                        if qs.get("cluster_run_id", [None])[0]:
                            search_time_by_type["cluster_search"].append(dur_s)
                        if qs.get("caption_embed_search", [None])[0]:
                            search_time_by_type["caption_embed_search"].append(
                                dur_s
                            )
                    except Exception:
                        pass
            # For zero-result tracking, count both hits and misses
            total_searches += 1
            try:
                cnt = int(r.get("count") or 0)
            except Exception:
                cnt = 0
            if (r.get("key") or "").strip():
                try:
                    qs2 = parse_qs(r.get("key") or "")
                    types = []
                    if qs2.get("semantic_search_clipid", [None])[0]:
                        types.append("semantic_video_to_video")
                    if qs2.get("semantic_search_text", [None])[0]:
                        types.append("semantic_text_to_video")
                        # If zero results, capture the semantic text query
                        if cnt == 0:
                            try:
                                sem_txt = (
                                    qs2.get("semantic_search_text", [None])[0]
                                    or ""
                                ).strip()
                                if sem_txt:
                                    zero_semantic_texts[sem_txt.lower()] += 1
                            except Exception:
                                pass
                    if qs2.get("trajectory_shape_clipid", [None])[0]:
                        types.append("trajectory_shape")
                    if qs2.get("search", [None])[0]:
                        types.append("caption_search")
                        # If zero results, capture the caption search text
                        if cnt == 0:
                            try:
                                cap_txt = (
                                    qs2.get("search", [None])[0] or ""
                                ).strip()
                                if cap_txt:
                                    zero_caption_texts[cap_txt.lower()] += 1
                            except Exception:
                                pass
                    if qs2.get("classifier_run_id", [None])[0] or qs2.get("classifier_select", [None])[0]:
                        types.append("classifier_search")
                    if qs2.get("wm_class_name", [None])[0]:
                        types.append("wm_search")
                    if qs2.get("visual_search_text", [None])[0]:
                        types.append("visual_search_text")
                        # If zero results, capture the visual search text
                        if cnt == 0:
                            try:
                                vtxt = (
                                    qs2.get("visual_search_text", [None])[0]
                                    or ""
                                ).strip()
                                if vtxt:
                                    zero_visual_texts[vtxt.lower()] += 1
                            except Exception:
                                pass
                    if qs2.get("visual_search_image_id", [None])[0]:
                        types.append("visual_search_image")
                    if qs2.get("cluster_run_id", [None])[0]:
                        types.append("cluster_search")
                    if qs2.get("caption_embed_search", [None])[0]:
                        types.append("caption_embed_search")
                        if cnt == 0:
                            try:
                                cap_emb_txt = (
                                    qs2.get("caption_embed_search", [None])[0]
                                    or ""
                                ).strip()
                                if cap_emb_txt:
                                    zero_caption_embed_texts[
                                        cap_emb_txt.lower()
                                    ] += 1
                            except Exception:
                                pass
                    for t in types:
                        searches_by_type[t] += 1
                        if cnt == 0:
                            zero_by_type[t] += 1
                except Exception:
                    pass
            if cnt == 0:
                zero_searches += 1

        elif r["type"] == "rewrite":
            rewrite_total += 1
            rewrite_users[user] += 1
            q = (r.get("query") or "").strip()
            if q:
                rewrite_query_texts[q] += 1

        elif r["type"] == "classifier_train":
            classifier_trains_by_label[r["label"]] += 1
            classifier_trains_by_embed_type[r["embed_type"]] += 1
            classifier_trains_by_user[user] += 1

        elif r["type"] == "annotation_event":
            ev = r.get("event")
            c = int(r.get("count") or 0)
            if ev in annotations_daily[day]:
                annotations_daily[day][ev] += c

    # Daily user counts
    daily_counts = {d: len(u) for d, u in users_by_day.items()}
    avg_users_per_day = (
        (sum(daily_counts.values()) / len(daily_counts))
        if daily_counts
        else 0.0
    )

    # Compute average search time by type (seconds)
    avg_search_s_by_type = [
        (stype, (sum(durs) / len(durs)) if durs else 0.0)
        for stype, durs in search_time_by_type.items()
    ]
    avg_search_s_by_type.sort(key=lambda x: x[1], reverse=True)

    # Zero-result counts (overall and per type).
    # Use searches_by_type as the source of truth so all active search types
    # appear, even those (like visual_search_image) that never produce 0 results.
    zero_count_by_type = [
        (t, int(zero_by_type[t])) for t in sorted(searches_by_type.keys())
    ]
    zero_count_by_type.sort(key=lambda x: x[1], reverse=True)

    # Summaries
    # Specific endpoint usage requested for admin dashboard panels
    specific_endpoints = [
        "/admin_stats",
        "/data_stats",
        "/videos",
        "/policy_predictions",
        "/leaderboard",
        "/arena",
    ]
    endpoint_usage = [
        (p, int(req_by_path.get(p, 0))) for p in specific_endpoints
    ]

    summary = {
        "days_observed": sorted(users_by_day.keys()),
        "unique_users_overall": (
            len(set().union(*users_by_day.values())) if users_by_day else 0
        ),
        "daily_unique_users": daily_counts,
        "avg_users_per_day": avg_users_per_day,
        "top_n": TOP_N,
        "top_users": user_request_counts.most_common(TOP_N),
        "requests_per_user": dict(user_request_counts),
        "total_http_requests": req_total,
        "anonymous_request_ratio": (
            (anon_reqs / req_total) if req_total else 0.0
        ),
        "http_methods": http_methods.most_common(),
        "http_status": sorted(http_status.items()),
        "top_paths": req_by_path.most_common(TOP_N),
        "endpoint_usage": endpoint_usage,
        "top_first_segments": req_by_first_seg.most_common(TOP_N),
        "requests_by_hour": [
            (h, round(c / max(1, len(users_by_day)), 1))
            for h, c in sorted(req_by_hour.items())
        ],
        "s3_errors_by_code": s3_errors.most_common(),
        "top_s3_error_keys": s3_error_keys.most_common(TOP_N),
        "status_by_top_paths": {
            p: sorted(http_status_by_path[p].items())
            for p, _ in req_by_path.most_common(TOP_N)
        },
        "video_streaming_counts": {
            "video": sum(
                c for p, c in req_by_first_seg.items() if p == "/video"
            ),
            "depth_video": sum(
                c for p, c in req_by_first_seg.items() if p == "/depth_video"
            ),
            "boxes_video": sum(
                c for p, c in req_by_first_seg.items() if p == "/boxes_video"
            ),
            "point_video": sum(
                c for p, c in req_by_first_seg.items() if p == "/point_video"
            ),
            "mfmrh_video": sum(
                c for p, c in req_by_first_seg.items() if p == "/mfmrh_video"
            ),
        },
        "search_usage": search_usage.most_common(),
        "top_data_sources": data_source_counts.most_common(TOP_N),
        "top_label_filters": label_filter_counts.most_common(TOP_N),
        "top_caption_searches": caption_search_texts.most_common(TOP_N),
        "top_visual_search_texts": visual_search_texts.most_common(TOP_N),
        "top_semantic_search_texts": semantic_text_searches.most_common(TOP_N),
        "top_caption_embed_search_texts": caption_embed_search_texts.most_common(
            TOP_N
        ),
        "cache": {
            "hits": cache_hits,
            "misses": cache_misses,
            "hit_rate": (
                (cache_hits / (cache_hits + cache_misses))
                if (cache_hits + cache_misses)
                else 0.0
            ),
            "hit_mean_ms": (
                (cache_hit_time / cache_hits * 1000.0) if cache_hits else 0.0
            ),
            "miss_mean_ms": (
                (cache_miss_time / cache_misses * 1000.0)
                if cache_misses
                else 0.0
            ),
            "daily": {
                d: {
                    "hits": cache_hits_by_day.get(d, 0),
                    "misses": cache_misses_by_day.get(d, 0),
                    "hit_rate": (
                        cache_hits_by_day.get(d, 0)
                        / max(
                            1,
                            cache_hits_by_day.get(d, 0)
                            + cache_misses_by_day.get(d, 0),
                        )
                    ),
                }
                for d in sorted(
                    set(cache_hits_by_day.keys())
                    | set(cache_misses_by_day.keys())
                )
            },
        },
        "avg_search_s_by_type": avg_search_s_by_type,
        "zero_result_overall_count": int(zero_searches),
        "zero_result_count_by_type": zero_count_by_type,
        # New: top zero-result queries per type (limited to top N)
        "searches_by_type": searches_by_type.most_common(),
        "zero_semantic_search_texts": zero_semantic_texts.most_common(TOP_N),
        "zero_caption_search_texts": zero_caption_texts.most_common(TOP_N),
        "zero_visual_search_texts": zero_visual_texts.most_common(TOP_N),
        "zero_caption_embed_search_texts": zero_caption_embed_texts.most_common(
            TOP_N
        ),
        "query_rewrite": {
            "total_calls": rewrite_total,
            "unique_users": len(rewrite_users),
            "top_queries": rewrite_query_texts.most_common(TOP_N),
            "top_users": rewrite_users.most_common(TOP_N),
            "by_type": {
                stype: {
                    "searches_with_rewrites": rewrite_by_type[stype],
                    "top_queries": rewrite_texts_by_type[stype].most_common(
                        TOP_N
                    ),
                }
                for stype in ("caption", "caption-embed", "semantic", "visual")
                if rewrite_by_type.get(stype)
            },
        },
        "classifier_trains_by_label": classifier_trains_by_label.most_common(
            TOP_N
        ),
        "classifier_trains_by_embed_type": classifier_trains_by_embed_type.most_common(),
        "classifier_trains_by_user": classifier_trains_by_user.most_common(
            TOP_N
        ),
        "annotations_daily": annotations_daily,
        "total_parsed_records": (
            req_total
            + sum(s3_errors.values())
            + cache_hits
            + cache_misses
            + 0  # "other" lines are not counted precisely without storing; omit from total
        ),
    }
    return summary


def iter_log_lines(logs_dir: str, recursive: bool = False) -> Iterable[str]:
    # Read plain text log files in directory (optionally recursive)
    def file_iter(root: str) -> Iterable[str]:
        if recursive:
            for dirpath, _dirnames, filenames in os.walk(root):
                for name in sorted(filenames):
                    yield os.path.join(dirpath, name)
        else:
            for name in sorted(os.listdir(root)):
                path = os.path.join(root, name)
                if os.path.isfile(path):
                    yield path

    for path in file_iter(logs_dir):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip():
                        yield line
        except Exception:
            continue


def _parse_date(d: Optional[str]) -> Optional[date]:
    if not d:
        return None
    return datetime.strptime(d, "%Y-%m-%d").date()


def iter_records(
    lines: Iterable[str],
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> Iterable[dict]:
    for ln in lines:
        rec = parse_line(ln)
        if not rec:
            continue
        if start or end:
            d = rec["ts"].date()
            if start and d < start:
                continue
            if end and d > end:
                continue
        yield rec


def _ensure_matplotlib():
    try:
        import importlib.util

        return (
            importlib.util.find_spec("matplotlib") is not None
            and importlib.util.find_spec("matplotlib.pyplot") is not None
        )
    except Exception:
        return False


def _save_bar(ax, labels, values, title, xlabel, ylabel, rotate=False):
    ax.bar(range(len(labels)), values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        labels, rotation=45 if rotate else 0, ha="right" if rotate else "center"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def generate_plots(summary: dict, out_dir: str, show: bool = False) -> None:
    if not _ensure_matplotlib():
        print("matplotlib is not available; skipping plot generation")
        return

    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # 1) Daily unique users
    days = sorted(summary["daily_unique_users"].keys())
    day_vals = [summary["daily_unique_users"][d] for d in days]
    if days:
        fig, ax = plt.subplots(figsize=(8, 3))
        _save_bar(
            ax,
            days,
            day_vals,
            "Daily Unique Users",
            "Day",
            "Users",
            rotate=True,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "daily_unique_users.png"))
        if show:
            plt.show()
        plt.close(fig)

    # 2) Requests by hour
    hours = [h for h, _ in summary["requests_by_hour"]]
    hour_vals = [v for _, v in summary["requests_by_hour"]]
    if hours:
        fig, ax = plt.subplots(figsize=(8, 3))
        _save_bar(ax, hours, hour_vals, "Requests by Hour", "Hour", "Requests")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "requests_by_hour.png"))
        if show:
            plt.show()
        plt.close(fig)

    # 3) HTTP status distribution
    status_labels = [str(s) for s, _ in summary["http_status"]]
    status_vals = [n for _, n in summary["http_status"]]
    if status_labels:
        fig, ax = plt.subplots(figsize=(6, 3))
        _save_bar(
            ax,
            status_labels,
            status_vals,
            "HTTP Status Codes",
            "Status",
            "Count",
        )
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "http_status.png"))
        if show:
            plt.show()
        plt.close(fig)

    # 4) Method distribution
    methods = [m for m, _ in summary["http_methods"]]
    method_vals = [n for _, n in summary["http_methods"]]
    if methods:
        fig, ax = plt.subplots(figsize=(5, 3))
        _save_bar(ax, methods, method_vals, "HTTP Methods", "Method", "Count")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "http_methods.png"))
        if show:
            plt.show()
        plt.close(fig)

    # 5) Top paths
    top_paths = summary["top_paths"]
    if top_paths:
        labels = [p for p, _ in top_paths]
        vals = [n for _, n in top_paths]
        fig, ax = plt.subplots(figsize=(10, max(3, 0.3 * len(labels))))
        title = "Top Paths (no query)"
        if summary.get("top_n"):
            title += f" [top {summary['top_n']}]"
        _save_bar(ax, labels, vals, title, "Path", "Requests", rotate=True)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "top_paths.png"))
        if show:
            plt.show()
        plt.close(fig)

    # 6) Streaming endpoint usage
    stream_labels = list(summary["video_streaming_counts"].keys())
    stream_vals = [summary["video_streaming_counts"][k] for k in stream_labels]
    if any(stream_vals):
        fig, ax = plt.subplots(figsize=(7, 3))
        _save_bar(
            ax,
            stream_labels,
            stream_vals,
            "Streaming Endpoint Requests",
            "Endpoint",
            "Requests",
            rotate=True,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "streaming_requests.png"))
        if show:
            plt.show()
        plt.close(fig)

    # 6b) Search usage (new)
    search_usage = summary.get("search_usage") or []
    if search_usage:
        labels = [k for k, _ in search_usage]
        vals = [v for _, v in search_usage]
        fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(labels))))
        _save_bar(
            ax,
            labels,
            vals,
            "Search Usage",
            "Search Type",
            "Requests",
            rotate=True,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "search_usage.png"))
        if show:
            plt.show()
        plt.close(fig)

    # 6c) Top visual search texts
    top_vtxt = summary.get("top_visual_search_texts") or []
    if top_vtxt:
        labels = [q for q, _ in top_vtxt]
        vals = [v for _, v in top_vtxt]
        fig, ax = plt.subplots(figsize=(10, max(3, 0.3 * len(labels))))
        title = "Top Visual Search Texts"
        if summary.get("top_n"):
            title += f" [top {summary['top_n']}]"
        _save_bar(ax, labels, vals, title, "Query", "Count", rotate=True)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "top_visual_search_texts.png"))
        if show:
            plt.show()
        plt.close(fig)

    # 6d) Average search latency by type
    avg_search = summary.get("avg_search_s_by_type") or []
    if avg_search:
        labels = [k for k, _ in avg_search]
        vals = [v * 1000 for _, v in avg_search]
        fig, ax = plt.subplots(figsize=(10, max(3, 0.3 * len(labels))))
        _save_bar(
            ax,
            labels,
            vals,
            "Average Search Latency by Type",
            "Search Type",
            "ms",
            rotate=True,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "avg_search_latency_by_type.png"))
        if show:
            plt.show()
        plt.close(fig)

    # 6e) Zero-result rate by type
    zero_by_type = summary.get("zero_result_count_by_type") or []
    if zero_by_type:
        by_type_total = dict(summary.get("searches_by_type") or [])
        labels = [k for k, _ in zero_by_type]
        vals = [
            (cnt / by_type_total[k]) if by_type_total.get(k) else 0.0
            for k, cnt in zero_by_type
        ]
        fig, ax = plt.subplots(figsize=(10, max(3, 0.3 * len(labels))))
        _save_bar(
            ax,
            labels,
            vals,
            "Zero-Result Rate by Search Type",
            "Search Type",
            "Rate",
            rotate=True,
        )
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "zero_result_rate_by_type.png"))
        if show:
            plt.show()
        plt.close(fig)

    # 7) S3 errors by code
    s3_err = summary["s3_errors_by_code"]
    if s3_err:
        labels = [k for k, _ in s3_err]
        vals = [v for _, v in s3_err]
        fig, ax = plt.subplots(figsize=(6, 3))
        _save_bar(ax, labels, vals, "S3 Errors by Code", "Error", "Count")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "s3_errors.png"))
        if show:
            plt.show()
        plt.close(fig)

    # 7b) Top data sources (new)
    top_ds = summary.get("top_data_sources") or []
    if top_ds:
        labels = [ds for ds, _ in top_ds]
        vals = [v for _, v in top_ds]
        fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(labels))))
        title = "Top Data Sources"
        if summary.get("top_n"):
            title += f" [top {summary['top_n']}]"
        _save_bar(
            ax, labels, vals, title, "Data Source", "Requests", rotate=True
        )
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "top_data_sources.png"))
        if show:
            plt.show()
        plt.close(fig)

    # 7c) Top label filters (new)
    top_labels = summary.get("top_label_filters") or []
    if top_labels:
        labels = [lb for lb, _ in top_labels]
        vals = [v for _, v in top_labels]
        fig, ax = plt.subplots(figsize=(10, max(3, 0.3 * len(labels))))
        title = "Top Label Filters"
        if summary.get("top_n"):
            title += f" [top {summary['top_n']}]"
        _save_bar(ax, labels, vals, title, "Label", "Requests", rotate=True)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "top_label_filters.png"))
        if show:
            plt.show()
        plt.close(fig)

    # 8) Cache hit/miss counts
    cache = summary.get("cache", {})
    if cache and (cache.get("hits", 0) + cache.get("misses", 0) > 0):
        fig, ax = plt.subplots(figsize=(5, 3))
        _save_bar(
            ax,
            ["hit", "miss"],
            [cache.get("hits", 0), cache.get("misses", 0)],
            "Cache Results",
            "Result",
            "Count",
        )
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "cache_counts.png"))
        if show:
            plt.show()
        plt.close(fig)

        # Mean duration
        fig, ax = plt.subplots(figsize=(6, 3))
        _save_bar(
            ax,
            ["hit_mean_ms", "miss_mean_ms"],
            [cache.get("hit_mean_ms", 0.0), cache.get("miss_mean_ms", 0.0)],
            "Cache Mean Duration",
            "Metric",
            "ms",
        )
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "cache_mean_duration.png"))
        if show:
            plt.show()
        plt.close(fig)

        # Daily hit rate
        daily = cache.get("daily", {})
        if daily:
            days = sorted(daily.keys())
            rates = [daily[d]["hit_rate"] for d in days]
            fig, ax = plt.subplots(figsize=(8, 3))
            _save_bar(
                ax,
                days,
                rates,
                "Daily Cache Hit Rate",
                "Day",
                "Hit Rate",
                rotate=True,
            )
            ax.set_ylim(0, 1)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "cache_hit_rate_by_day.png"))
            if show:
                plt.show()
            plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze server logs for usage metrics and produce plots"
    )
    ap.add_argument("logs_dir", help="Directory containing server logs")
    ap.add_argument("out_dir", help="Directory to write plot PNGs")
    ap.add_argument(
        "--json-out", default=None, help="Optional path to write JSON summary"
    )
    ap.add_argument(
        "--show", action="store_true", help="Show plots interactively"
    )
    ap.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively read logs_dir",
    )
    ap.add_argument("--start", help="Start date (YYYY-MM-DD) inclusive")
    ap.add_argument("--end", help="End date (YYYY-MM-DD) inclusive")
    ap.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Limit for top lists (default: 15)",
    )
    args = ap.parse_args()

    start_date = _parse_date(args.start)
    end_date = _parse_date(args.end)
    lines = iter_log_lines(args.logs_dir, recursive=args.recursive)
    records_iter = iter_records(lines, start=start_date, end=end_date)
    summary = aggregate_records(records_iter, top_n=args.top_n)

    # Pretty print to stdout
    def hdr(t: str):
        print("\n" + t)
        print("-" * len(t))

    parsed_total = summary.get("total_parsed_records")
    if parsed_total is not None:
        print(f"Parsed ~{parsed_total} structured entries from {args.logs_dir}")
    else:
        print(f"Parsed logs from {args.logs_dir}")
    hdr("Users")
    print(f"Days observed: {len(summary['days_observed'])}")
    print(f"Unique users overall: {summary['unique_users_overall']}")
    print(f"Average users/day: {summary['avg_users_per_day']:.2f}")
    for d, c in sorted(summary["daily_unique_users"].items()):
        print(f"  {d}: {c}")

    hdr("Requests")
    print(f"Total HTTP requests: {summary['total_http_requests']}")
    print(f"Anonymous request ratio: {summary['anonymous_request_ratio']:.2%}")
    print("Methods:", ", ".join(f"{m}:{n}" for m, n in summary["http_methods"]))
    print("Status:", ", ".join(f"{s}:{n}" for s, n in summary["http_status"]))
    # Clarify the hour bucket range in the printout (HH:00–HH:59)
    print(
        "Requests by hour (HH:00–HH:59):",
        ", ".join(
            f"{h}:00–{h}:59: {n}" for h, n in summary["requests_by_hour"]
        ),
    )

    hdr(f"Top Paths (no query) [top {summary.get('top_n', 'N')}]")
    for p, n in summary["top_paths"]:
        print(
            f"  {p}  ({n})  statuses: {dict(summary['status_by_top_paths'].get(p, []))}"
        )

    hdr(f"Top First Segments [top {summary.get('top_n', 'N')}]")
    for p, n in summary["top_first_segments"]:
        print(f"  {p}: {n}")
    print("Video streaming counts:", summary["video_streaming_counts"])

    # Search usage and data sources
    if summary.get("search_usage"):
        hdr("Search Usage")
        print(", ".join(f"{k}:{v}" for k, v in summary["search_usage"]))
    if summary.get("top_visual_search_texts"):
        hdr(f"Top Visual Search Texts [top {summary.get('top_n', 'N')}]")
        for q, cnt in summary["top_visual_search_texts"]:
            print(f"  {q}: {cnt}")
    if summary.get("avg_search_s_by_type"):
        hdr("Average Search Latency by Type")
        for stype, avg_s in summary["avg_search_s_by_type"]:
            print(f"  {stype}: {avg_s * 1000:.1f} ms")
    if summary.get("zero_result_count_by_type"):
        hdr("Zero-Result Searches by Type")
        by_type_total = dict(summary.get("searches_by_type") or [])
        for stype, cnt in summary["zero_result_count_by_type"]:
            total = by_type_total.get(stype)
            pct = f" ({cnt / total:.1%})" if total else ""
            print(f"  {stype}: {cnt}{pct}")
    if summary.get("top_data_sources"):
        hdr(f"Top Data Sources [top {summary.get('top_n', 'N')}]")
        for ds, cnt in summary["top_data_sources"]:
            print(f"  {ds}: {cnt}")
    if summary.get("top_label_filters"):
        hdr(f"Top Label Filters [top {summary.get('top_n', 'N')}]")
        for label, cnt in summary["top_label_filters"]:
            print(f"  {label}: {cnt}")

    hdr("S3 Errors")
    if summary["s3_errors_by_code"]:
        print(
            "By code:",
            ", ".join(f"{k}:{v}" for k, v in summary["s3_errors_by_code"]),
        )
    if summary["top_s3_error_keys"]:
        print(f"Top keys [top {summary.get('top_n', 'N')}]:")
        for k, v in summary["top_s3_error_keys"]:
            print(f"  {k}: {v}")

    # Cache stats
    cache = summary.get("cache", {})
    if cache:
        hdr("Cache")
        print(
            f"Hits: {cache.get('hits', 0)}  Misses: {cache.get('misses', 0)}  Hit rate: {cache.get('hit_rate', 0.0):.2%}"
        )
        print(
            f"Mean durations: hit={cache.get('hit_mean_ms', 0.0):.3f} ms  miss={cache.get('miss_mean_ms', 0.0):.3f} ms"
        )

    # Top users
    if summary.get("top_users"):
        hdr(f"Top Users (by HTTP requests) [top {summary.get('top_n', 'N')}]")
        for u, n in summary["top_users"]:
            print(f"  {u}: {n}")

    rw = summary.get("query_rewrite", {})
    if rw.get("total_calls") or rw.get("by_type"):
        hdr("Query Rewrite")
        print(
            f"Total calls: {rw.get('total_calls', 0)}  "
            f"Unique users: {rw.get('unique_users', 0)}"
        )
        by_type = rw.get("by_type") or {}
        for stype in ("caption", "caption-embed", "semantic", "visual"):
            td = by_type.get(stype)
            if td:
                print(
                    f"  {stype}: {td['searches_with_rewrites']} searches with rewrites"
                )
        for stype in ("caption", "caption-embed", "semantic", "visual"):
            td = by_type.get(stype)
            if td and td.get("top_queries"):
                print(f"  [{stype}] top rewritten queries:")
                for q, cnt in td["top_queries"]:
                    print(f"    {q}: {cnt}")
        if rw.get("top_queries"):
            print(f"Top queries (all types) [top {summary.get('top_n', 'N')}]:")
            for q, cnt in rw["top_queries"]:
                print(f"  {q}: {cnt}")
        if rw.get("top_users"):
            print("Top users:")
            for u, cnt in rw["top_users"]:
                print(f"  {u}: {cnt}")

    if args.json_out:
        try:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"\nWrote JSON summary to {args.json_out}")
        except Exception as e:
            print(f"Failed to write JSON output: {e}")

    # Generate plots
    generate_plots(summary, args.out_dir, show=args.show)


if __name__ == "__main__":
    main()
