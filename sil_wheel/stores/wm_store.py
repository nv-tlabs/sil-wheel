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

from threading import Lock

import pandas as pd
from sil_wheel.stores.search_utils import project_dict
from sil_wheel.stores.utils import LRUDict
from sil_wheel.stores.wm_search import WMSearch


class WMStore:
    def __init__(self, wm_data_file):
        self.lock = Lock()

        if wm_data_file is not None:
            self.wm_data = pd.read_parquet(wm_data_file)

            self.searches_count = LRUDict(size=10)
            self.searches_range = LRUDict(size=10)
            self.wm_searcher = WMSearch(self.wm_data)

    def search_count(self, class_name, min_count=1, max_count=None):
        key = (class_name, min_count, max_count)
        with self.lock:
            if key in self.searches_count:
                return self.searches_count[key]
            video_ids = self.wm_searcher.search_count(
                class_name, min_count, max_count
            )
            self.searches_count[key] = video_ids

            return video_ids

    def search_range(self, class_name, angle_range, max_dist=10, min_time=0):
        key = (class_name, tuple(angle_range), max_dist, min_time)
        with self.lock:
            if key in self.searches_range:
                return self.searches_range[key]
            video_ids = self.wm_searcher.search_range(
                class_name, angle_range, max_dist, min_time
            )
            self.searches_range[key] = video_ids
            return video_ids

    def search(self, filters, current_results):
        if (
            filters.wm_class_name is not None
            and filters.wm_min_count is not None
        ):
            current_results = project_dict(
                current_results,
                self.search_count(
                    filters.wm_class_name,
                    filters.wm_min_count,
                    filters.wm_max_count,
                ),
            )

        if (
            filters.wm_class_name is not None
            and filters.wm_max_dist is not None
            and filters.wm_min_time is not None
            and filters.wm_angle_range
        ):
            current_results = project_dict(
                current_results,
                self.search_range(
                    filters.wm_class_name,
                    filters.wm_angle_range,
                    filters.wm_max_dist,
                    filters.wm_min_time,
                ),
            )

        return current_results
