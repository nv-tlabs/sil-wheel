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

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Optional
from urllib import parse


def project_dict(data, keys):
    if isinstance(keys, (set, dict)):
        # If we have way fewer keys than data then do the scan O(len(keys))
        # instead of O(len(data))
        if 2 * len(keys) < len(data):
            return {k: data[k] for k in keys if k in data}
        else:
            return {k: v for k, v in data.items() if k in keys}
    else:
        return {k: data[k] for k in keys if k in data}


def project_starmap(fn, data, *keys_and_args):
    for kai in keys_and_args:
        data = {k: fn(data[k], *rest) for k, *rest in kai if k in data}
    return data


def project_dict_any(data, *keys):
    if all(isinstance(ki, (set, dict)) for ki in keys):
        num_keys = sum(len(ki) for ki in keys)
        if 2 * num_keys < len(data):
            return {k: data[k] for ki in keys for k in ki if k in data}
        else:
            return {k: data[k] for k in data if any(k in ki for ki in keys)}
    else:
        return {k: data[k] for ki in keys for k in ki if k in data}


def project_dict_all(data, *keys):
    """Return entries from data that appear in ALL of the provided key sets (AND)."""
    if not keys:
        return dict(data)
    sorted_keys = sorted(keys, key=len)
    result = project_dict(data, sorted_keys[0])
    for ki in sorted_keys[1:]:
        result = {k: result[k] for k in result if k in ki}
    return result


def exclude_dict_any(data, *keys):
    if all(isinstance(ki, (set, dict)) for ki in keys):
        return {k: data[k] for k in data if not any(k in ki for ki in keys)}
    else:
        raise NotImplementedError(
            "Exclude is not implemented for lists/tuples for now"
        )


def exclude_dict_all(data, *keys):
    if not keys:
        return data
    if all(isinstance(ki, (set, dict)) for ki in keys):
        return {k: data[k] for k in data if not all(k in ki for ki in keys)}
    else:
        raise NotImplementedError(
            "Exclude is not implemented for lists/tuples for now"
        )


def select_key_source(a, b):
    sa = a.keys() if isinstance(a, dict) else a
    sb = b.keys() if isinstance(b, dict) else b
    return sa if len(a) < len(b) else sb


@dataclass
class SearchFilters:
    # Annotation search
    annotation_filter: list[str]
    annotation_filter_mode: str  # "any" (OR) | "all" (AND)
    without_ann: Optional[bool]
    times_filter: Optional[bool]
    with_metrics: Optional[bool]
    with_bev: Optional[bool]
    raw_data_source: list[str]
    data_source_filter_mode: str  # "any" (OR) | "all" (AND)
    labels_to_exclude: list[str]
    labels_to_exclude_filter_mode: str  # "any" (OR) | "all" (AND)
    label_types: list[str]
    search_clipid: Optional[str]
    # The label and its range (min, max)
    numeric_filter: list[tuple[str, float, float]]

    # Project
    project_source: list[str]

    # Caption search
    search: Optional[str]
    query_rewrite: Optional[bool]

    # Country search
    left_hand_driving: Optional[bool]
    search_country: Optional[str]

    # Trajectory search
    with_ego_data: Optional[bool]
    trajectory_pattern: Optional[str]
    trajectory_shape_clipid: Optional[str]
    trajectory_shape_start_t: Optional[float]
    trajectory_shape_end_t: Optional[float]
    search_speed: Optional[str]

    # Semantic search
    semantic_search_clipid: Optional[str]
    semantic_search_text: Optional[str]
    visual_search_text: Optional[str]
    visual_search_image_id: Optional[str]

    # Classifier search
    classifier_run_id: Optional[str]
    probability_expression: Optional[str]

    # Cluster search
    cluster_run_id: Optional[str]
    cluster_ids: list[str]
    cluster_distance_min: Optional[float]
    cluster_distance_max: Optional[float]

    # Clip-id-list search.
    clip_id_list_hash: Optional[str]

    # Comments search
    search_comments: Optional[str]

    # Caption embedding search
    caption_embed_search_text: Optional[str]

    # World model search
    wm_class_name: Optional[str]
    wm_min_count: Optional[int]
    wm_max_count: Optional[int]
    wm_max_dist: Optional[float]
    wm_min_time: Optional[float]
    wm_angle_range: list[str]

    # SIL APIs
    sil_apis: list[str]

    # User options
    user_options: dict

    # Fields with defaults must be last
    caption_extra_queries: list[str] = field(default_factory=list)
    caption_embed_extra_queries: list[str] = field(default_factory=list)
    semantic_extra_queries: list[str] = field(default_factory=list)
    visual_extra_queries: list[str] = field(default_factory=list)
    rank_mode: str = "priority"

    def has_prior_filters(self, query_filter):
        """Returns True if any search filter other than query_filter is active."""
        active = {
            "search": self.search,
            "trajectory_pattern": self.trajectory_pattern,
            "trajectory_shape_clipid": self.trajectory_shape_clipid,
            "with_ego_data": self.with_ego_data,
            "search_speed": self.search_speed,
            "semantic_search_text": self.semantic_search_text,
            "semantic_search_clipid": self.semantic_search_clipid,
            "visual_search_text": self.visual_search_text,
            "visual_search_image_id": self.visual_search_image_id,
            "caption_embed_search_text": self.caption_embed_search_text,
        }
        return any(v for k, v in active.items() if k != query_filter)

    @staticmethod
    def _extract_list(query, key, delim, default=[]):
        value = query.get(key, None)
        if value is None:
            return default
        return value[0].split(delim)

    @staticmethod
    def _extract_optional(query, key, fn):
        value = query.get(key, [None])[0]
        if value is None:
            return None
        return fn(value)

    @staticmethod
    def _extract_optional_bool(query, key):
        return SearchFilters._extract_optional(
            query, key, lambda v: v == "true"
        )

    @staticmethod
    def _extract_optional_float(query, key):
        return SearchFilters._extract_optional(query, key, float)

    @staticmethod
    def _extract_optional_int(query, key):
        return SearchFilters._extract_optional(query, key, int)

    @staticmethod
    def _extract_numeric_filters(query, key, delim, default=[]):
        value = query.get(key, None)
        if value is None:
            return default

        # The expected format is "key,min,max||other_key,min,max"
        filters = []
        for token in value[0].split(delim):
            token = token.strip()
            parts = token.split(",")
            if len(parts) != 4:
                continue
            key, vmin, vmax, ordering = [p.strip() for p in parts]
            filters.append((key, float(vmin), float(vmax), ordering))

        return filters

    @classmethod
    def from_query(cls, query, user_options={}):
        return cls(
            # Annotation
            annotation_filter=cls._extract_list(query, "filter", "||"),
            annotation_filter_mode=query.get("filter_mode", ["any"])[0],
            without_ann=cls._extract_optional_bool(query, "without_ann"),
            times_filter=cls._extract_optional_bool(query, "times"),
            with_metrics=cls._extract_optional_bool(query, "with_metrics"),
            with_bev=cls._extract_optional_bool(query, "with_bev"),
            raw_data_source=cls._extract_list(query, "data_source", "||"),
            data_source_filter_mode=query.get("data_source_mode", ["any"])[0],
            labels_to_exclude=cls._extract_list(
                query, "labels_to_exclude", "||"
            ),
            labels_to_exclude_filter_mode=query.get("labels_to_exclude_mode", ["any"])[0],
            label_types=cls._extract_list(query, "label_types", "||"),
            search_clipid=query.get("search_clipid", [None])[0],
            numeric_filter=cls._extract_numeric_filters(
                query, "numeric_filter", "||"
            ),
            # Project
            project_source=cls._extract_list(
                query, "project_source", "||", ["Alpamayo"]
            ),
            # Caption
            search=query.get("search", [None])[0],
            query_rewrite=cls._extract_optional_bool(query, "query_rewrite"),
            caption_extra_queries=cls._extract_list(query, "caption_extra_queries", "||"),
            caption_embed_extra_queries=cls._extract_list(query, "caption_embed_extra_queries", "||"),
            semantic_extra_queries=cls._extract_list(query, "semantic_extra_queries", "||"),
            visual_extra_queries=cls._extract_list(query, "visual_extra_queries", "||"),
            # Country
            left_hand_driving=cls._extract_optional_bool(
                query, "left_hand_driving"
            ),
            search_country=query.get("search_country", [None])[0],
            # Trajectory
            with_ego_data=cls._extract_optional_bool(query, "with_ego_data"),
            trajectory_pattern=query.get("trajectory_pattern", [None])[0],
            trajectory_shape_clipid=query.get(
                "trajectory_shape_clipid", [None]
            )[0],
            trajectory_shape_start_t=cls._extract_optional_float(
                query, "trajectory_shape_start_t"
            ),
            trajectory_shape_end_t=cls._extract_optional_float(
                query, "trajectory_shape_end_t"
            ),
            search_speed=query.get("search_speed", [None])[0],
            # Semantic
            semantic_search_clipid=query.get("semantic_search_clipid", [None])[
                0
            ],
            semantic_search_text=query.get("semantic_search_text", [None])[0],
            visual_search_text=query.get("visual_search_text", [None])[0],
            visual_search_image_id=query.get("visual_search_image_id", [None])[0],
            # Classifier
            classifier_run_id=query.get("classifier_run_id", [None])[0],
            probability_expression=query.get(
                "probability_expression", [None]
            )[0],
            # Cluster
            cluster_run_id=query.get("cluster_run_id", [None])[0],
            cluster_ids=cls._extract_list(query, "cluster_ids", ","),
            cluster_distance_min=cls._extract_optional_float(
                query, "cluster_distance_min",
            ),
            cluster_distance_max=cls._extract_optional_float(
                query, "cluster_distance_max",
            ),
            # Clip-id-list
            clip_id_list_hash=query.get("clip_id_list_hash", [None])[0],
            # Comments
            search_comments=query.get("search_comments", [None])[0],
            # World model
            caption_embed_search_text=query.get("caption_embed_search", [None])[0],
            wm_class_name=query.get("wm_class_name", [None])[0],
            wm_min_count=cls._extract_optional_int(query, "wm_min_count"),
            wm_max_count=cls._extract_optional_int(query, "wm_max_count"),
            wm_max_dist=cls._extract_optional_float(query, "wm_max_dist"),
            wm_min_time=cls._extract_optional_float(query, "wm_min_time"),
            wm_angle_range=cls._extract_list(query, "wm_angle_range", ","),
            sil_apis=cls._extract_list(query, "sil_apis", "||"),
            user_options=user_options,
            rank_mode=query.get("rank_mode", ["priority"])[0],
        )

    @property
    def data_source(self):
        if "data_source" in self.user_options:
            all_options = set(self.user_options["data_source"])
            if self.raw_data_source:
                all_options &= set(self.raw_data_source)
            return list(all_options) if all_options else ["Non-existent-source"]
        else:
            return self.raw_data_source

    def has_semantic_search(self):
        return (
            self.semantic_search_clipid is not None
            or self.semantic_search_text is not None
            or self.visual_search_text is not None
            or self.visual_search_image_id is not None
        )

    def has_trajectory_search(self):
        return (
            self.with_ego_data is not None
            or self.trajectory_pattern is not None
            or self.trajectory_shape_clipid is not None
            or self.search_speed is not None
        )

    def has_classifier_search(self):
        return self.classifier_run_id is not None

    def has_comments_search(self):
        return self.search_comments is not None

    def has_caption_embed_search(self):
        return self.caption_embed_search_text is not None

    def has_wm_search(self):
        return self.wm_class_name is not None

    def has_caption_search(self):
        return self.search is not None

    def has_country_search(self):
        return self.left_hand_driving or self.search_country

    def has_external_search(self):
        return (
            self.has_semantic_search()
            or self.has_caption_search()
            or self.has_caption_embed_search()
            or self.has_country_search()
            or self.has_trajectory_search()
            or self.has_classifier_search()
            or self.has_comments_search()
            or self.has_wm_search()
        )

    @staticmethod
    def calculate_key(args, default="ALL"):
        query_parts = []
        for k, v in sorted(args.items()):
            if v is None:
                continue

            k = parse.quote(k)
            if isinstance(v, list):
                for vi in sorted(v):
                    vi = parse.quote(str(vi))
                    query_parts.append(f"{k}={vi}")
            else:
                if isinstance(v, bool):
                    v = "true" if v else "false"
                else:
                    v = str(v)
                v = parse.quote(v)
                query_parts.append(f"{k}={v}")
        return "&".join(query_parts) if query_parts else default

    @property
    def key(self):
        if not hasattr(self, "_key"):
            filters = self.to_dict()
            filters["data_source"] = self.data_source
            self._key = SearchFilters.calculate_key(filters, "ALL")
        return self._key

    @property
    def external_search_key(self):
        return SearchFilters.calculate_key(
            self._external_search_filters(),
            "",
        )

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return self.key == other.key

    def _external_search_filters(self):
        return {
            "search": self.search,
            "query_rewrite": self.query_rewrite,
            "caption_extra_queries": self.caption_extra_queries or None,
            "caption_embed_extra_queries": self.caption_embed_extra_queries or None,
            "semantic_extra_queries": self.semantic_extra_queries or None,
            "visual_extra_queries": self.visual_extra_queries or None,
            "left_hand_driving": self.left_hand_driving,
            "search_country": self.search_country,
            "with_ego_data": self.with_ego_data,
            "trajectory_pattern": self.trajectory_pattern,
            "trajectory_shape_clipid": (self.trajectory_shape_clipid),
            "trajectory_shape_start_t": self.trajectory_shape_start_t,
            "trajectory_shape_end_t": self.trajectory_shape_end_t,
            "search_speed": self.search_speed,
            "semantic_search_clipid": (self.semantic_search_clipid),
            "semantic_search_text": self.semantic_search_text,
            "visual_search_text": self.visual_search_text,
            "visual_search_image_id": self.visual_search_image_id,
            "classifier_run_id": self.classifier_run_id,
            "probability_expression": (
                self.probability_expression if self.classifier_run_id else None
            ),
            "cluster_run_id": self.cluster_run_id,
            "cluster_ids": ",".join(self.cluster_ids) if self.cluster_ids else None,
            "cluster_distance_min": self.cluster_distance_min,
            "cluster_distance_max": self.cluster_distance_max,
            "search_comments": self.search_comments,
            "caption_embed_search": self.caption_embed_search_text,
            "wm_class_name": self.wm_class_name,
            "wm_min_count": self.wm_min_count,
            "wm_max_count": self.wm_max_count,
            "wm_max_dist": self.wm_max_dist,
            "wm_min_time": self.wm_min_time,
            "wm_angle_range": self.wm_angle_range or None,
            "with_metrics": self.with_metrics,
            "with_bev": self.with_bev,
            "rank_mode": self.rank_mode,
        }

    def to_dict(self):
        return {
            "filter": self.annotation_filter or None,
            "filter_mode": self.annotation_filter_mode if self.annotation_filter else None,
            "numeric_filter": self.numeric_filter or None,
            "without_ann": self.without_ann,
            "times": self.times_filter,
            "data_source": self.raw_data_source or None,
            "data_source_mode": self.data_source_filter_mode if self.raw_data_source else None,
            "labels_to_exclude": self.labels_to_exclude or None,
            "labels_to_exclude_mode": self.labels_to_exclude_filter_mode if self.labels_to_exclude else None,
            "label_types": self.label_types or None,
            "search_clipid": self.search_clipid,
            "clip_id_list_hash": self.clip_id_list_hash,
            "project_source": self.project_source or None,
            "sil_apis": self.sil_apis or None,
            **self._external_search_filters(),
        }


@dataclass
class SearchResults:
    semantic_search_text_score: Optional[float] = None
    semantic_search_clip_score: Optional[float] = None
    visual_search_score: Optional[float] = None
    classifier_score: Optional[float] = None
    trajectory_shape_score: Optional[float] = None
    numeric_scores: Optional[dict[str, float]] = None
    cluster_distance_score: Optional[float] = None
    caption_embed_score: Optional[float] = None
    visual_image_score: Optional[float] = None
    rrf_score: Optional[float] = None

    @property
    def has_scores(self):
        return (
            self.semantic_search_text_score is not None
            or self.semantic_search_clip_score is not None
            or self.visual_search_score is not None
            or self.visual_image_score is not None
            or self.classifier_score is not None
            or self.trajectory_shape_score is not None
            or self.numeric_scores is not None
            or self.cluster_distance_score is not None
            or self.caption_embed_score is not None
            or self.rrf_score is not None
        )

    def primary_score(self, filters) -> float:
        if self.numeric_scores is not None:
            # Parse numeric filters from last (most recent) to first
            for fi, _, _, order in reversed(filters.numeric_filter):
                v = self.numeric_scores[fi]
                if order == "asc":
                    # small to large
                    return -v
                else:
                    # large to small
                    return v

        if self.classifier_score is not None:
            return self.classifier_score
        if self.caption_embed_score is not None:
            return self.caption_embed_score
        if self.semantic_search_text_score is not None:
            return self.semantic_search_text_score
        if self.semantic_search_clip_score is not None:
            return self.semantic_search_clip_score
        if self.visual_search_score is not None:
            return self.visual_search_score
        if self.visual_image_score is not None:
            return self.visual_image_score
        if self.trajectory_shape_score is not None:
            return -self.trajectory_shape_score
        if self.cluster_distance_score is not None:
            # smaller distance = closer to centroid = ranked first
            return -self.cluster_distance_score
        return 0.0

    def with_trajectory_score(self, s):
        if self is self.default:
            return replace(self, trajectory_shape_score=s)
        self.trajectory_shape_score = s
        return self

    def with_semantic_search_text_score(self, s):
        if self is self.default:
            return replace(self, semantic_search_text_score=s)
        self.semantic_search_text_score = s
        return self

    def with_semantic_search_clip_score(self, s):
        if self is self.default:
            return replace(self, semantic_search_clip_score=s)
        self.semantic_search_clip_score = s
        return self

    def with_visual_search_score(self, s):
        if self is self.default:
            return replace(self, visual_search_score=s)
        self.visual_search_score = s
        return self

    def with_visual_image_score(self, s):
        if self is self.default:
            return replace(self, visual_image_score=s)
        self.visual_image_score = s
        return self

    def with_classifier_score(self, s):
        if self is self.default:
            return replace(self, classifier_score=s)
        self.classifier_score = s
        return self

    def with_numeric_score(self, key, value):
        if self is self.default:
            return replace(self, numeric_scores={key: value})
        if self.numeric_scores is None:
            self.numeric_scores = {}
        self.numeric_scores[key] = value
        return self

    def with_cluster_distance_score(self, s):
        if self is self.default:
            return replace(self, cluster_distance_score=s)
        self.cluster_distance_score = s
        return self

    def with_caption_embed_score(self, s):
        if self is self.default:
            return replace(self, caption_embed_score=s)
        self.caption_embed_score = s
        return self

    def with_rrf_score(self, s):
        if self is self.default:
            return replace(self, rrf_score=s)
        self.rrf_score = s
        return self

    def active_scores(self, filters=None):
        scores = {}
        if self.numeric_scores is not None and filters is not None:
            for fi, _, _, order in filters.numeric_filter:
                v = self.numeric_scores.get(fi)
                if v is not None:
                    scores[f"numeric:{fi}"] = -v if order == "asc" else v
        if self.classifier_score is not None:
            scores["classifier"] = self.classifier_score
        if self.caption_embed_score is not None:
            scores["caption_embed"] = self.caption_embed_score
        if self.semantic_search_text_score is not None:
            scores["semantic_text"] = self.semantic_search_text_score
        if self.semantic_search_clip_score is not None:
            scores["semantic_clip"] = self.semantic_search_clip_score
        if self.visual_search_score is not None:
            scores["visual_text"] = self.visual_search_score
        if self.visual_image_score is not None:
            scores["visual_image"] = self.visual_image_score
        if self.trajectory_shape_score is not None:
            scores["trajectory"] = -self.trajectory_shape_score
        if self.cluster_distance_score is not None:
            scores["cluster"] = -self.cluster_distance_score
        return scores


def rrf_rank(results, filters, k=60):
    """Combine multiple search results into a single list uusing Reciprocal Rank Fusion (RRF).
    """
    # Group the clip_ids and their scores by search modality
    modality_scores = defaultdict(dict)
    for cid, r in results.items():
        for name, score in r.active_scores(filters).items():
            modality_scores[name][cid] = score

    fused_score = defaultdict(float)
    for name, scores in modality_scores.items():
        # Sort clips by their modality-specific score to find their rank (1st, 2nd, etc.)
        ranked = sorted(scores, key=scores.get, reverse=True)
        for rank, cid in enumerate(ranked):
            fused_score[cid] += 1.0 / (k + rank + 1)

    # Attach the fused score so clients can display it alongside the
    # per-modality scores that contributed to the ranking.
    for cid, s in fused_score.items():
        if cid in results:
            results[cid] = results[cid].with_rrf_score(s)

    return sorted(
        results.keys(),
        key=lambda cid: fused_score.get(cid, 0.0),
        reverse=True,
    )


SearchResults.default = SearchResults()
