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
"""Programmatic Wheel client.

`WheelClient` exposes the same search modalities as the HTTP UI but returns
plain Python / pandas objects instead of HTTP responses. It wraps the same
`SearchPipeline` the server uses, so search composition and ranking are
identical. No UI-only dependencies (video fetcher, static pages, favicon,
session handling).

Typical usage::

    from sil_wheel.client import WheelClient

    client = WheelClient.from_config("config/wheel_launch_dev_server_config.yaml")
    result = client.search_caption("hard braking at intersection")
    df = result.as_dataframe()
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs

import pandas as pd
import yaml

try:
    from yaml import CLoader as _Loader
except ImportError:
    from yaml import Loader as _Loader

from sil_wheel.search.search_pipeline import SearchPipeline
from sil_wheel.stores.autolabels_store import AutolabelsDataStore
from sil_wheel.stores.caption_embeddings_store import CaptionEmbeddingsStore
from sil_wheel.stores.classifier_search import ClassifierSearch
from sil_wheel.stores.clip_list_search import ClipListSearch
from sil_wheel.stores.cluster_search import ClusterSearch
from sil_wheel.stores.cosmos_embeddings_store import CosmosEmbeddingsStore
from sil_wheel.stores.models_with_metrics_store import ModelsWithMetricsDataStore
from sil_wheel.stores.search_utils import SearchFilters
from sil_wheel.stores.sqlite_caption_store import FTSCaptionStore
from sil_wheel.stores.sqlite_data_store import SQLiteDataStore
from sil_wheel.stores.trajectory_store import TrajectoryStore
from sil_wheel.stores.visual_embeddings_store import Florence2SigCLIPEmbeddingStore
from sil_wheel.stores.wm_store import WMStore


@dataclass
class WheelSearchResult:
    """Outcome of a `WheelClient.search` call.

    Attributes:
        clip_ids: clip IDs in ranked order (best match first).
        scores: clip_id -> `SearchResults` (per-clip score breakdown).
        filters: the `SearchFilters` that produced this result.
    """
    clip_ids: list[str]
    scores: dict
    filters: SearchFilters

    def __len__(self) -> int:
        return len(self.clip_ids)

    def head(self, n: int = 10) -> list[str]:
        return self.clip_ids[:n]

    def as_dataframe(self):
        """Return a pandas DataFrame with one row per clip and per-modality scores."""
        rows = []
        for rank, cid in enumerate(self.clip_ids):
            r = self.scores.get(cid)
            row = {"rank": rank, "clip_id": cid}
            if r is not None:
                row.update({
                    "caption_score": getattr(r, "caption_score", None),
                    "caption_embed_score": getattr(r, "caption_embed_score", None),
                    "semantic_text_score": getattr(r, "semantic_search_text_score", None),
                    "semantic_clip_score": getattr(r, "semantic_search_clip_score", None),
                    "visual_text_score": getattr(r, "visual_search_score", None),
                    "visual_image_score": getattr(r, "visual_image_score", None),
                    "trajectory_shape_score": getattr(r, "trajectory_shape_score", None),
                    "cluster_distance_score": getattr(r, "cluster_distance_score", None),
                    "classifier_score": getattr(r, "classifier_score", None),
                    "rrf_score": getattr(r, "rrf_score", None),
                })
            rows.append(row)
        return pd.DataFrame(rows)


class WheelClientBase:
    """Shared behavior for local and remote Wheel clients.

    Subclasses implement ``_search_with_query(query)`` against their
    transport (in-process pipeline or HTTP). This base supplies every
    user-facing entry point — ``search(**kwargs)``, ``search_from_url``,
    and the per-modality helpers — by funneling them through that one
    method.
    """

    def _search_with_query(self, query: dict) -> "WheelSearchResult":
        raise NotImplementedError

    def _kwargs_to_query(self, kwargs: dict) -> dict:
        """Coerce a kwargs dict into ``SearchFilters.from_query`` shape.

        Scalars become single-element string lists; lists/tuples become
        string lists; bools render as ``"true"``/``"false"``; ``None`` is
        dropped. Same encoding the server sees off the URL query string.
        """
        query: dict[str, list[str]] = {}
        for key, value in kwargs.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                query[key] = [str(v) for v in value]
            elif isinstance(value, bool):
                query[key] = [str(value).lower()]
            else:
                query[key] = [str(value)]
        return query

    def search(self, **kwargs: Any) -> "WheelSearchResult":
        """Run a search with the given filter kwargs.

        Any field accepted by ``SearchFilters.from_query`` can be passed.
        Scalars are wrapped to single-element lists; lists pass through;
        ``None`` values are dropped.
        """
        return self._search_with_query(self._kwargs_to_query(kwargs))

    @staticmethod
    def query_from_url(url_or_query: str) -> str:
        """Extract the wheel query-string portion of a UI URL.

        Wheel UI URLs put filters in the hash fragment for SPA routing
        (``http://host:port/#&search=foo&...``); server-rendered URLs use
        a normal query string (``?search=foo&...``). Both forms — plus a
        bare ``search=foo`` query string — are normalized to the same
        ``key=value&key2=value2`` shape (matching what the server stamps
        into ``metadata.json:search_params``).
        """
        for sep in ("?", "#"):
            idx = url_or_query.find(sep)
            if idx != -1:
                url_or_query = url_or_query[idx + 1:]
                break
        return url_or_query.lstrip("&?")

    def search_from_url(self, url_or_query: str) -> "WheelSearchResult":
        """Run a search by copying a URL out of the wheel UI."""
        return self._search_with_query(parse_qs(self.query_from_url(url_or_query)))

    def search_caption(self, text: str, **kwargs) -> "WheelSearchResult":
        return self.search(search=text, **kwargs)

    def search_caption_embed(self, text: str, **kwargs) -> "WheelSearchResult":
        return self.search(caption_embed_search=text, **kwargs)

    def search_semantic_text(self, text: str, **kwargs) -> "WheelSearchResult":
        return self.search(semantic_search_text=text, **kwargs)

    def search_clip(self, clip_id: str, **kwargs) -> "WheelSearchResult":
        """Semantic clip-to-video search."""
        return self.search(semantic_search_clipid=clip_id, **kwargs)

    def search_visual_text(self, text: str, **kwargs) -> "WheelSearchResult":
        return self.search(visual_search_text=text, **kwargs)

    def search_visual_image(self, upload_id: str, **kwargs) -> "WheelSearchResult":
        return self.search(visual_search_image_id=upload_id, **kwargs)

    def search_trajectory_pattern(self, pattern: str, **kwargs) -> "WheelSearchResult":
        return self.search(trajectory_pattern=pattern, **kwargs)

    def search_trajectory_shape(
        self, clip_id: str, start_t: float | None = None,
        end_t: float | None = None, **kwargs,
    ) -> "WheelSearchResult":
        return self.search(
            trajectory_shape_clipid=clip_id,
            trajectory_shape_start_t=start_t,
            trajectory_shape_end_t=end_t,
            **kwargs,
        )

    def search_classifier(
        self, run_id: str, expression: str, **kwargs,
    ) -> "WheelSearchResult":
        """Search clips by a trained classifier's score expression.

        ``run_id`` selects which classifier run to apply (one classifier
        run = one trained model). ``expression`` is a predicate on the
        probability variable ``p`` (e.g. ``"p > 0.95"``).
        """
        return self.search(
            classifier_run_id=run_id,
            probability_expression=expression,
            **kwargs,
        )

    def search_cluster(
        self,
        run_id: str,
        cluster_ids: str | int | list[str | int],
        **kwargs,
    ) -> "WheelSearchResult":
        """Search clips inside one or more clusters of a given run."""
        if isinstance(cluster_ids, (list, tuple)):
            joined = ",".join(str(c) for c in cluster_ids)
        else:
            joined = str(cluster_ids)
        return self.search(
            cluster_run_id=run_id, cluster_ids=joined, **kwargs,
        )

    def search_country(self, country: str, **kwargs) -> "WheelSearchResult":
        return self.search(search_country=country, **kwargs)

    def search_clip_list(self, hash: str, **kwargs) -> "WheelSearchResult":
        """Search clips against a previously uploaded clip-id list.

        ``hash`` is the 16-hex content-addressed identifier returned by
        :meth:`upload_clip_list` (or the ``hash`` field of
        ``POST /upload_clip_list``). Combines cleanly with other filter
        kwargs (e.g. ``search_clip_list(h, search="hard braking")``).
        """
        return self.search(clip_id_list_hash=hash, **kwargs)

    def search_world_model(
        self, class_name: str,
        min_count: int | None = None, max_count: int | None = None,
        max_dist: float | None = None, min_time: float | None = None,
        angle_range: list[str] | None = None, **kwargs,
    ) -> "WheelSearchResult":
        return self.search(
            wm_class_name=class_name,
            wm_min_count=min_count,
            wm_max_count=max_count,
            wm_max_dist=max_dist,
            wm_min_time=min_time,
            wm_angle_range=angle_range,
            **kwargs,
        )


class WheelClient(WheelClientBase):
    """Local Wheel API: runs the same ``SearchPipeline`` the server uses,
    in your Python process. Requires read access to the wheel data files
    (FAISS indices, parquet shards, SQLite DBs).
    """

    def __init__(
        self,
        pipeline: SearchPipeline,
        clips_to_apis: Optional[dict] = None,
    ):
        self.pipeline = pipeline
        self.clips_to_apis = clips_to_apis or {}

    @classmethod
    def from_config(cls, config_path: str | Path) -> "WheelClient":
        """Build a `WheelClient` from the same YAML the server reads.

        Loads only the data stores. Skips server-only setup (websocket,
        slack, VLM judge, static pages, S3 video fetcher). ``bev_fetcher``
        is left unset on the pipeline because it is a UI-only streaming
        dependency.
        """
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=_Loader)

        ds_cfg = config["datastores"]

        with open(config["clips_to_sil_apis"], "r") as f:
            clips_to_apis = json.load(f)
        AutolabelsDataStore(clips_to_apis)

        datastore = SQLiteDataStore(
            ds_cfg["annotations_db"], clips_to_apis=clips_to_apis
        )

        cosmos_cfg = ds_cfg["cosmos_embed_store"]
        embeddingsstore = CosmosEmbeddingsStore(
            cosmos_cfg["embeddings_dir"],
            index_spec=cosmos_cfg["index_spec"],
            mmap=cosmos_cfg.get("mmap", False),
        )

        visual_cfg = ds_cfg["visual_embed_store"]
        clipembeddingsstore = Florence2SigCLIPEmbeddingStore(
            visual_cfg["embeddings_dir"],
            index_spec=visual_cfg.get("index_spec", "IVF4096,PQ64x8"),
            mmap=visual_cfg.get("mmap", False),
            siglip_model=visual_cfg.get("siglip_model", "google/siglip2-base-patch16-224"),
        )

        caption_embed_cfg = ds_cfg["caption_embed_store"]
        captionembeddingsstore = CaptionEmbeddingsStore(
            caption_embed_cfg["embeddings_dir"],
            index_spec=caption_embed_cfg.get("index_spec", "IVF4096,PQ128x8"),
            mmap=caption_embed_cfg.get("mmap", False),
        )

        traj_cfg = ds_cfg["trajectory_store"]
        trajectorystore = TrajectoryStore(
            traj_cfg["trajectory_dir"],
            config.get("server", {}).get("debug", False),
        )

        wm_store = WMStore(ds_cfg["wm_store"]["data_file"])
        captionstore = FTSCaptionStore(ds_cfg["captions_db"])

        bev_index_dir = ds_cfg["bev_store"]["metrics_index_dir"]
        metricstore = ModelsWithMetricsDataStore(
            ds_cfg["predictions_store"]["predictions_dir"],
            index_dir=bev_index_dir,
        )

        embeddingsstore.warmup()
        captionembeddingsstore.warmup()
        clipembeddingsstore.warmup()

        classifiersearch = ClassifierSearch(
            ds_cfg["classifier_search"]["classifier_dir"]
        )
        clustersearch = ClusterSearch(
            ds_cfg["cluster_search"]["clustering_dir"]
        )
        cliplistsearch = ClipListSearch(
            ds_cfg["clip_list_search"]["clip_lists_dir"]
        )

        pipeline = SearchPipeline(
            datastore=datastore,
            captionstore=captionstore,
            captionembeddingsstore=captionembeddingsstore,
            embeddingsstore=embeddingsstore,
            clipembeddingsstore=clipembeddingsstore,
            classifiersearch=classifiersearch,
            clustersearch=clustersearch,
            cliplistsearch=cliplistsearch,
            trajectorystore=trajectorystore,
            metricstore=metricstore,
            bev_fetcher=None,
            wm_store=wm_store,
        )

        return cls(pipeline=pipeline, clips_to_apis=clips_to_apis)

    def _search_with_query(self, query: dict) -> WheelSearchResult:
        """Run a search in-process via the local ``SearchPipeline``.

        Returns the full ranked clip list with per-clip score breakdowns
        in ``scores`` (all in memory).
        """
        filters = SearchFilters.from_query(query)
        clip_ids, scores = self.pipeline.search(filters)
        return WheelSearchResult(
            clip_ids=list(clip_ids), scores=dict(scores), filters=filters,
        )

    def upload_clip_list(self, clip_ids) -> dict:
        """Register a list of clip_ids with the local clip-list store.

        Skips the HTTP round-trip — writes the file directly to the
        configured ``clip_lists_dir``. Returns
        ``{"hash": str, "count": int, "created": bool}`` so the caller
        can pair the hash with :meth:`search_clip_list`.
        """
        hash, created = self.pipeline.cliplistsearch.save(clip_ids)
        return {
            "hash": hash,
            "count": len(set(clip_ids)),
            "created": created,
        }
