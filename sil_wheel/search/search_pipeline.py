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

"""Cross-store search aggregation, shared by the HTTP server and `WheelClient`."""
import logging
from urllib import parse

from sil_wheel.stores.search_utils import rrf_rank
from sil_wheel.stores.time_utils import Timer
from sil_wheel.stores.utils import LRUDict

logging.getLogger(__name__).addHandler(logging.StreamHandler())


class SearchPipeline:
    def __init__(
        self,
        datastore,
        captionstore,
        captionembeddingsstore,
        embeddingsstore,
        clipembeddingsstore,
        classifiersearch,
        clustersearch,
        cliplistsearch,
        trajectorystore,
        metricstore,
        bev_fetcher,
        wm_store,
        cache_size: int = 50,
        logger: logging.Logger | None = None,
    ):
        self.datastore = datastore
        self.captionstore = captionstore
        self.captionembeddingsstore = captionembeddingsstore
        self.embeddingsstore = embeddingsstore
        self.clipembeddingsstore = clipembeddingsstore
        self.classifiersearch = classifiersearch
        self.clustersearch = clustersearch
        self.cliplistsearch = cliplistsearch
        self.trajectorystore = trajectorystore
        self.metricstore = metricstore
        self.bev_fetcher = bev_fetcher
        self.wm_store = wm_store

        self.searches = LRUDict(size=cache_size)
        self.timers = Timer()
        self.log = logger or logging.getLogger(__name__)

    def search(self, filters):
        self.timers.tic()

        if filters in self.searches:
            clip_ids, results = self.searches[filters]
            self.log.info(
                "%s from cache %d took %f",
                filters.key, len(results), self.timers.toc(),
            )
            return clip_ids, results

        self.timers.tic()
        results = self.datastore.default_results
        self.log.info(
            "Results dictionary creation took %f", self.timers.toc()
        )

        self.timers.tic()
        if self.metricstore is not None:
            results = self.metricstore.search(filters, results)
        if self.bev_fetcher is not None:
            results = self.bev_fetcher.search(filters, results)
        if self.wm_store is not None:
            results = self.wm_store.search(filters, results)
        results = self.trajectorystore.search(filters, results)
        results = self.clipembeddingsstore.search(filters, results)
        results = self.captionstore.search(filters, results)
        results = self.captionembeddingsstore.search(filters, results)
        results = self.embeddingsstore.search(filters, results)
        results = self.datastore.search(filters, results)
        results = self.cliplistsearch.search(filters, results)
        results = self.clustersearch.search(filters, results)
        results = self.classifiersearch.search(filters, results)
        self.log.info("Search took %f", self.timers.toc())

        self.timers.tic()
        if results and next(iter(results.values())).has_scores:
            if filters.rank_mode == "rrf":
                clip_ids = rrf_rank(results, filters)
            else:
                clip_ids = sorted(
                    results.keys(),
                    key=lambda cid: results[cid].primary_score(filters),
                    reverse=True,
                )
        else:
            clip_ids = list(results.keys())
        self.log.info("Sorting took %f", self.timers.toc())

        self.searches[filters] = (clip_ids, results)
        self.log.info(
            "%s → %d results in %f s",
            filters.key, len(results), self.timers.toc(),
        )
        return clip_ids, results

    def invalidate_with_terms(self, *terms):
        for_deletion = [
            f for f in self.searches if any(t in f.key for t in terms)
        ]
        for f in for_deletion:
            del self.searches[f]

    def invalidate_annotation(self, *labels):
        self.invalidate_with_terms(
            "without_ann=",
            "label_types=manual",
            "label_types=autolabel",
            *[f"labels_to_exclude={parse.quote(label)}" for label in labels],
            *[f"filter={parse.quote(label)}" for label in labels],
        )

    def invalidate_times(self):
        self.invalidate_with_terms("times=")

    def invalidate_comments(self):
        self.invalidate_with_terms("search_comments=")

    def invalidate_classifier(self, run_id):
        self.classifiersearch.invalidate(run_id)
        self.invalidate_with_terms(
            f"classifier_run_id={parse.quote(run_id)}"
        )
