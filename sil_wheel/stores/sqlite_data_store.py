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

import bisect
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property, lru_cache
from itertools import chain
from threading import RLock

import pycountry
from sil_wheel.stores.search_utils import (
    SearchResults,
    exclude_dict_all,
    exclude_dict_any,
    project_dict,
    project_dict_all,
    project_dict_any,
    project_starmap,
)
from tqdm import tqdm

LABEL_TYPES = ["manual", "autolabel"]


def unique_id(length=10):
    x = [random.randint(0, 35) for _ in range(length)]
    return "".join(
        [chr((xi < 10) * (48 + xi) + (xi >= 10) * (87 + xi)) for xi in x]
    )


@dataclass
class Annotation:
    uid: str
    project: str
    key: str
    start_time: float
    end_time: float
    label_type: str
    value: float

    def to_dict(self):
        return {
            "uid": self.uid,
            "project": self.project,
            "key": self.key,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "label_type": self.label_type,
            "value": self.value,
        }


@dataclass
class Clip:
    clip_id: str
    annotations: list[Annotation]
    data_source: str

    def to_dict(self):
        return {
            "clip_id": self.clip_id,
            "annotations": [a.to_dict() for a in self.annotations],
            "data_source": self.data_source,
        }


class SQLiteDataStore:
    def __init__(
        self, db_path: str, json_path: str = None, clips_to_apis: str = None
    ):

        self.lock = RLock()
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

        self._build_reverse_indexes()
        self._build_default_results()
        self._get_annotations_state()

        if clips_to_apis is not None:
            self.apis_to_clips = defaultdict(list)
            for clip, apis in clips_to_apis.items():
                for api in apis:
                    if api:
                        self.apis_to_clips[api].append(clip)

    def _create_tables(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS clips (
                clip_id TEXT PRIMARY KEY,
                data_source TEXT,
                country TEXT,
                has_time INTEGER DEFAULT 0,
                has_manual_annotations INTEGER DEFAULT 0,
                has_autolabels INTEGER DEFAULT 0
            )
        """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                uid TEXT PRIMARY KEY,
                project TEXT,
                clip_id TEXT,
                key TEXT,
                value REAL,
                start_time REAL,
                end_time REAL,
                label_type TEXT,
                FOREIGN KEY (clip_id) REFERENCES clips (clip_id)
            )
        """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS video_paths (
                clip_id TEXT PRIMARY KEY,
                path TEXT
            )
        """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                name     TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                license  TEXT NOT NULL
            )
        """
        )

        # Add indexes
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_clips_data_source ON clips(data_source)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_clips_country ON clips(country)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_annotations_clip ON annotations(clip_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_annotations_project ON annotations(project)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ann_label_key_clip ON annotations(label_type, key, clip_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_annotations_clip_project ON annotations(clip_id, project)"
        )

    def dataset_metadata(self, allowed_names=None):
        """Return {name: {"category": str, "license": str}}.

        When allowed_names is given, every name in it is guaranteed to be
        a key in the returned dict; names without a row in the datasets
        table get a generic default so callers (including the frontend)
        don't need their own guards.
        """
        rows = self.conn.execute(
            "SELECT name, category, license FROM datasets"
        ).fetchall()
        by_name = {
            r["name"]: {
                "category": r["category"],
                "license": r["license"],
            }
            for r in rows
        }
        if allowed_names is None:
            return by_name
        default = {"category": "Other", "license": "unknown"}
        return {name: by_name.get(name, default) for name in allowed_names}

    def _build_reverse_indexes(self):
        """Precompute reverse indexes from annotations and countries (clips)."""
        self.key_to_clip_ids = defaultdict(lambda: defaultdict(Counter))
        self.labeltype_to_clip_ids = defaultdict(lambda: defaultdict(Counter))
        self.data_source_to_clip_ids = defaultdict(set)
        self.clip_to_data_source = {}

        # For a project and a key a sorted list of tuples
        # [(clip1, val1), (clip2, val2)]
        self.numeric_index = defaultdict(lambda: defaultdict(list))

        # From annotations -> project/key/labeltype to clip_ids
        query = (
            "SELECT clip_id, key, label_type, project, value FROM annotations"
        )
        with self.lock:
            cur = self.conn.execute(query)
            for row in tqdm(cur):
                project = row["project"]
                clip_id = sys.intern(row["clip_id"])

                # If a row has value i.e it's label_type is numeric it should
                # be treated differently
                if row["value"] is not None:
                    self.numeric_index[project][row["key"]].append(
                        (clip_id, row["value"])
                    )
                else:
                    self.key_to_clip_ids[project][row["key"]][clip_id] += 1
                    self.labeltype_to_clip_ids[project][row["label_type"]][
                        clip_id
                    ] += 1

            # Now that we have the reverse index for the numeric values, we
            # need to sort it by the value
            for project, indices in tqdm(self.numeric_index.items()):
                for key, index in indices.items():
                    index.sort(key=lambda x: x[1])

        self.clip_to_country = {}
        self.country_to_clip_ids = defaultdict(set)
        # From clips -> country and data_source to clip_ids
        query_countries = "SELECT clip_id, country, data_source FROM clips"
        with self.lock:
            cur = self.conn.execute(query_countries)
            for row in tqdm(cur):
                clip_id = sys.intern(row["clip_id"])
                ctry = row["country"] or ""
                self.clip_to_country[clip_id] = ctry
                self.country_to_clip_ids[ctry].add(clip_id)

                ds_val = row["data_source"]
                if ds_val:
                    for item in ds_val.split(","):
                        item = item.strip()
                        self.data_source_to_clip_ids[item].add(clip_id)

                self.clip_to_data_source[clip_id] = ds_val or ""

        def _get_deep_size(obj, seen=None):
            """Calculate deep size of object including all nested objects."""
            size = sys.getsizeof(obj)
            if seen is None:
                seen = set()

            obj_id = id(obj)
            if obj_id in seen:
                return 0

            seen.add(obj_id)

            if isinstance(obj, dict):
                size += sum(
                    _get_deep_size(k, seen) + _get_deep_size(v, seen)
                    for k, v in obj.items()
                )
            elif isinstance(obj, (list, tuple, set, frozenset)):
                size += sum(_get_deep_size(item, seen) for item in obj)
            elif hasattr(obj, "__dict__"):
                size += _get_deep_size(obj.__dict__, seen)

            return size

        def _format_size(bytes_size):
            """Format bytes into human-readable string (GB or MB)."""
            gb = bytes_size / (1024**3)
            if gb >= 1:
                return f"{gb:.2f} GB"
            mb = bytes_size / (1024**2)
            return f"{mb:.2f} MB"

        print("\n=== Memory Usage of Indexes ===")
        print(
            f"key_to_clip_ids: {_format_size(_get_deep_size(self.key_to_clip_ids))}"
        )
        print(
            f"labeltype_to_clip_ids: {_format_size(_get_deep_size(self.labeltype_to_clip_ids))}"
        )
        print(
            f"data_source_to_clip_ids: {_format_size(_get_deep_size(self.data_source_to_clip_ids))}"
        )
        print(
            f"clip_to_data_source: {_format_size(_get_deep_size(self.clip_to_data_source))}"
        )
        print(
            f"clip_to_country: {_format_size(_get_deep_size(self.clip_to_country))}"
        )
        print(
            f"country_to_clip_ids: {_format_size(_get_deep_size(self.country_to_clip_ids))}"
        )
        print(
            f"numeric_index: {_format_size(_get_deep_size(self.numeric_index))}"
        )
        print("=" * 35)

    def _build_default_results(self):
        """Collect all the clips that have associated videos for the case we
        have no filtering."""
        # Build from scratch
        query = "SELECT clip_id FROM video_paths"
        with self.lock:
            self._default_results = {
                sys.intern(row["clip_id"]): SearchResults.default
                for row in tqdm(
                    self.conn.execute(query), desc="Building default results"
                )
            }

    def _get_table_stats(self):
        stats = {}
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row["name"] for row in cur]

        for table in tables:
            cur = self.conn.execute(f"SELECT COUNT(*) AS cnt FROM {table}")
            count = cur.fetchone()["cnt"]
            stats[table] = count

        # database file size in bytes
        cur = self.conn.execute("PRAGMA page_count")
        page_count = cur.fetchone()[0]
        cur = self.conn.execute("PRAGMA page_size")
        page_size = cur.fetchone()[0]
        db_size = page_count * page_size
        print("\nTable Stats:")
        for table, count in stats.items():
            print(f"  {table}: {count:,} rows")
        print(f"Database Size: {db_size / (1024*1024):.2f} MB")

    def _get_annotations_state(self):
        # Build from scratch
        query = """
            SELECT key, project,
                   SUM(label_type = 'manual') AS manual_count,
                   SUM(label_type = 'autolabel') AS autolabel_count,
                   SUM(start_time != -1) AS with_time_count
            FROM annotations
            GROUP BY project, key
        """
        with self.lock:
            cur = self.conn.execute(query)
            self.stats = {}
            for row in cur:
                project, key = row["project"], row["key"]
                self.stats.setdefault(project, {})[key] = {
                    "manual": row["manual_count"],
                    "autolabel": row["autolabel_count"],
                    "with_time": row["with_time_count"],
                }

    def _add_annotation_to_index(self, clip_id, key, label_type, project):
        """Update the reverse index when adding a new annotation."""
        self.key_to_clip_ids[project][key][clip_id] += 1
        self.labeltype_to_clip_ids[project][label_type][clip_id] += 1

    def _remove_annotation_from_index(self, clip_id, key, label_type, project):
        """Update the reverse index when removing an annotation."""
        self.key_to_clip_ids[project][key][clip_id] -= 1
        if self.key_to_clip_ids[project][key][clip_id] <= 0:
            self.key_to_clip_ids[project][key].pop(clip_id)

        self.labeltype_to_clip_ids[project][label_type][clip_id] -= 1
        if self.labeltype_to_clip_ids[project][label_type][clip_id] <= 0:
            self.labeltype_to_clip_ids[project][label_type].pop(clip_id)

    @property
    def default_results(self):
        return self._default_results

    @property
    def project_options(self):
        return sorted(
            set(self.key_to_clip_ids.keys()) | set(self.numeric_index.keys())
        )

    @cached_property
    def data_source_options(self):
        with self.lock, self.conn:
            cur = self.conn.execute("SELECT data_source FROM clips")
            data_sources = set()
            for row in cur:
                ds = row["data_source"]
                if not ds:
                    continue
                for item in ds.split(","):
                    item = item.strip()
                    if item:
                        data_sources.add(item)
        return sorted(data_sources)

    def num_manual_annotations(self, project_source):
        return sum(
            self.stats[proj][key]["manual"]
            for proj in project_source
            if proj in self.stats
            for key in self.stats[proj]
        )

    def num_autolabel_annotations(self, project_source):
        return sum(
            self.stats[proj][key]["autolabel"]
            for proj in project_source
            if proj in self.stats
            for key in self.stats[proj]
        )

    def num_annotations(self, project_source):
        return sum(
            self.stats[proj][key]["manual"] + self.stats[proj][key]["autolabel"]
            for proj in project_source
            if proj in self.stats
            for key in self.stats[proj]
        )

    def option_set_manual(self, project_source):
        agg = {}
        for proj in project_source:
            if proj not in self.stats:
                continue
            for key, counts in self.stats[proj].items():
                if counts["manual"] > 0:
                    agg[key] = agg.get(key, 0) + counts["manual"]
        # Ensure only keys with positive totals are returned
        return {k: v for k, v in agg.items() if v > 0}

    def option_set_autolabelled(self, project_source):
        agg = {}
        for proj in project_source:
            if proj not in self.stats:
                continue
            for key, counts in self.stats[proj].items():
                if counts["autolabel"] > 0:
                    agg[key] = agg.get(key, 0) + counts["autolabel"]
        # Ensure only keys with positive totals are returned
        return {k: v for k, v in agg.items() if v > 0}

    def option_set(self, project_source):
        manual = self.option_set_manual(project_source)
        autolabel = self.option_set_autolabelled(project_source)
        # Only include keys whose combined count is > 0
        combined = {}
        for k in manual.keys() | autolabel.keys():
            total = manual.get(k, 0) + autolabel.get(k, 0)
            if total > 0:
                combined[k] = total
        return combined

    def options(self, project_source):
        keys = set()
        for proj in project_source:
            if proj in self.key_to_clip_ids:
                keys.update(
                    k
                    for k, counter in self.key_to_clip_ids[proj].items()
                    if counter
                )
        return sorted(keys)

    def metric_names(self, project_source):
        keys = set()
        for proj in project_source:
            if proj in self.numeric_index:
                keys.update(
                    k
                    for k, values in self.numeric_index[proj].items()
                    if len(values) > 0
                )
        return sorted(keys)

    def get_clips_dict(self, clip_ids, project_source):
        """Returns a dictionary of clip_id -> Clip dataclass for the given list of clip_ids."""
        if not clip_ids:
            return {}

        if len(clip_ids) > 10000:
            raise NotImplementedError("Too many clips to get all at once")

        if len(clip_ids) > 1000:
            # Call our selves recursively
            # TODO: Maybe implement with a temporary table if it is too slow
            clips = {}
            batch_size = 100
            print(f"Getting {len(clip_ids)} clips")
            for start in range(0, len(clip_ids), batch_size):
                end = start + batch_size
                clips.update(
                    self.get_clips_dict(clip_ids[start:end], project_source)
                )
                print(f"    Processing {start} - {end}")
            return clips

        project_holders = ", ".join(["?"] * len(project_source))
        if len(project_source) > 0:
            project_clause = f"AND a.project IN ({project_holders})"
        else:
            project_clause = f"AND 1 = 0"

        clip_holders = ", ".join(["?"] * len(clip_ids))
        query = f"""
            SELECT c.clip_id, c.data_source, a.uid, a.project, a.key, a.start_time, a.end_time, a.label_type, a.value
            FROM clips c
            LEFT JOIN annotations a ON c.clip_id = a.clip_id {project_clause}
            WHERE c.clip_id IN ({clip_holders})
        """

        clips = {}
        with self.lock:
            cur = self.conn.execute(query, project_source + clip_ids)

            for row in cur:
                if row["clip_id"] not in clips:
                    clips[row["clip_id"]] = Clip(
                        row["clip_id"], [], row["data_source"]
                    )

                if row["uid"] is not None:
                    clips[row["clip_id"]].annotations.append(
                        Annotation(
                            uid=row["uid"],
                            project=row["project"],
                            key=row["key"],
                            start_time=row["start_time"],
                            end_time=row["end_time"],
                            label_type=row["label_type"],
                            value=row["value"],
                        )
                    )
        return clips

    def get_clip_data_sources(self, clip_ids):
        return {cid: self.clip_to_data_source.get(cid, "") for cid in clip_ids}

    def get_clip_ids_without_annotations(self):
        query = """
            SELECT c.clip_id
            FROM clips c LEFT JOIN annotations a ON c.clip_id = a.clip_id WHERE
            a.uid IS NULL
        """
        with self.lock:
            cur = self.conn.execute(query)
            return set(r["clip_id"] for r in cur)

    @lru_cache
    def get_clip_ids_for_data_sources(self, data_sources):
        """Return union of clip_ids that belong to any of the given data_sources.

        Uses in-memory reverse index built at init for speed.
        """
        if not data_sources:
            return set()
        result = set()
        for ds in data_sources:
            if not ds:
                continue
            result |= self.data_source_to_clip_ids.get(ds, set())
        return result

    @lru_cache
    def get_clip_ids_for_data_sources_all(self, data_sources):
        """Return clip_ids that belong to ALL of the given data_sources."""
        if not data_sources:
            return set()
        sets = [self.data_source_to_clip_ids.get(ds, set()) for ds in data_sources if ds]
        if not sets:
            return set()
        return set.intersection(*sets)

    def get_clip_ids_with_times(self, project_source: list[str] | None = None):
        """Returns clip_ids that have at least one timed annotation within the
        provided projects.

        If no project list is provided, falls back to the global flag on clips.
        """
        if project_source:
            placeholders = ", ".join(["?"] * len(project_source))
            query = f"""
                SELECT DISTINCT a.clip_id
                FROM annotations a
                WHERE a.project IN ({placeholders}) AND a.start_time != -1
            """
            with self.lock:
                cur = self.conn.execute(query, project_source)
                return set(row["clip_id"] for row in cur)

    def get_clip_ids_without_times(
        self, project_source: list[str] | None = None
    ):
        """Returns clip_ids where, within the provided projects, all annotations
        have start_time == -1. Requires at least one annotation in those
        projects to avoid vacuous matches.

        If no project list is provided, falls back to the global flag on clips.
        """
        if project_source:
            placeholders = ", ".join(["?"] * len(project_source))
            query = f"""
                SELECT a.clip_id
                FROM annotations a
                WHERE a.project IN ({placeholders})
                GROUP BY a.clip_id
                HAVING SUM(CASE WHEN a.start_time != -1 THEN 1 ELSE 0 END) = 0
            """
            with self.lock:
                cur = self.conn.execute(query, project_source)
                return set(row["clip_id"] for row in cur)

    def get_video_path(self, clip_id):
        query = "SELECT path FROM video_paths WHERE clip_id=?"
        with self.lock:
            cur = self.conn.execute(query, (clip_id,))
            return cur.fetchone()["path"]

    def get_clip_ids_with_sil_apis(self, sil_apis):
        clip_ids = set()
        for api in sil_apis:
            clip_ids |= set(self.apis_to_clips[api])
        return clip_ids

    def get_clips_with_key_in_numeric_range(
        self,
        project_source: str = None,
        key: str = None,
        vmin: float = None,
        vmax: float = None,
    ):
        """
        Return a generator of tuples (clip_id, value, label) for values in [vmin, vmax].
        """
        if key is None or project_source is None:
            return {}

        values = self.numeric_index[project_source][key]
        lo = (
            bisect.bisect_left(values, vmin, key=lambda x: x[1])
            if vmin is not None
            else 0
        )
        hi = (
            bisect.bisect_right(values, vmax, key=lambda x: x[1])
            if vmax is not None
            else len(values)
        )

        return (
            self.numeric_index[project_source][key][i] + (key,)
            for i in range(lo, hi)
        )

    def search(self, filters, current_results):
        if filters.search_clipid:
            current_results = project_dict(
                current_results, [filters.search_clipid]
            )

        if filters.sil_apis:
            clip_ids = self.get_clip_ids_with_sil_apis(filters.sil_apis)
            current_results = project_dict(current_results, clip_ids)

        if filters.search_country:
            country_clip_ids = self.country_to_clip_ids.get(
                filters.search_country, set()
            )
            current_results = project_dict(current_results, country_clip_ids)
        elif filters.left_hand_driving:
            lhd_countries = {"MT", "GB", "JP"}
            country_clip_ids = set().union(
                *(self.country_to_clip_ids.get(c, set()) for c in lhd_countries)
            )
            current_results = project_dict(current_results, country_clip_ids)

        if filters.without_ann is None and filters.annotation_filter:
            label_sets = [
                self.key_to_clip_ids[p][a]
                for p in filters.project_source
                for a in filters.annotation_filter
            ]
            if filters.annotation_filter_mode == "all":
                current_results = project_dict_all(current_results, *label_sets)
            else:
                current_results = project_dict_any(current_results, *label_sets)

        if filters.numeric_filter:
            # Group by each numeric filter and union results across projects
            groups = [
                chain.from_iterable(
                    self.get_clips_with_key_in_numeric_range(p, a, vmin, vmax)
                    for p in filters.project_source
                )
                for a, vmin, vmax, _ in filters.numeric_filter
            ]

            current_results = project_starmap(
                lambda r, v, f: r.with_numeric_score(f, v),
                current_results,
                *groups,
            )

        if filters.data_source:
            ds = tuple(filters.data_source)
            if filters.data_source_filter_mode == "all":
                clip_ids = self.get_clip_ids_for_data_sources_all(ds)
            else:
                clip_ids = self.get_clip_ids_for_data_sources(ds)
            current_results = project_dict(current_results, clip_ids)

        if filters.without_ann is None and filters.times_filter is not None:
            if filters.times_filter:
                clip_ids = self.get_clip_ids_with_times(filters.project_source)
            else:
                clip_ids = self.get_clip_ids_without_times(
                    filters.project_source
                )
            current_results = project_dict(current_results, clip_ids)

        if filters.label_types:
            current_results = project_dict_any(
                current_results,
                *[
                    self.labeltype_to_clip_ids[p][l]
                    for p in filters.project_source
                    for l in filters.label_types
                ],
            )

        if filters.labels_to_exclude:
            label_sets = [
                self.key_to_clip_ids[p][a]
                for p in filters.project_source
                for a in filters.labels_to_exclude
            ]
            if filters.labels_to_exclude_filter_mode == "all":
                current_results = exclude_dict_all(current_results, *label_sets)
            else:
                current_results = exclude_dict_any(current_results, *label_sets)

        if filters.without_ann:
            current_results = exclude_dict_any(
                current_results,
                *[
                    clip_ids
                    for p in filters.project_source
                    for clip_ids in self.key_to_clip_ids[p].values()
                ],
            )

        return current_results

    def get(self, clip_id, project_source):
        return self.get_clips_dict([clip_id], project_source)[clip_id]

    def get_country(self, clip_id):
        return self.clip_to_country.get(clip_id, "")

    def get_country_name(self, clip_id):
        code = self.get_country(clip_id)
        t = pycountry.countries.get(alpha_2=code)
        return code if t is None else t.name

    def add(
        self,
        clip_id,
        uid,
        annotation,
        label_type,
        project,
        start_time=-1,
        end_time=-1,
    ):
        annotation = re.sub(r"\s+", " ", annotation).strip()
        query = """
            INSERT INTO annotations (uid, project, clip_id, key, start_time, end_time, label_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self.lock, self.conn:
            self.conn.execute(
                query,
                (
                    uid,
                    project,
                    clip_id,
                    annotation,
                    start_time,
                    end_time,
                    label_type,
                ),
            )

            # Update the state of the annotations
            ann_key = (
                annotation.key if hasattr(annotation, "key") else annotation
            )
            if project not in self.stats:
                self.stats[project] = {}
            if ann_key not in self.stats[project]:
                self.stats[project][ann_key] = {
                    "manual": 0,
                    "autolabel": 0,
                    "with_time": 0,
                }

            self.stats[project][ann_key][label_type] += 1

            # Update the reverse indexes
            self._add_annotation_to_index(
                clip_id, annotation, label_type, project
            )

    def update_times(
        self,
        clip_id,
        uid,
        annotation,
        start_time,
        end_time,
        project,
        label_type="manual",
    ):
        """Update the start/end times for an annotation by uid.
        If the annotation is not manual, convert it to manual (verify) first.
        Properly maintains stats and clip flags.
        """
        # Only update with_time if we transition from -1 -> non -1
        sel = """
            SELECT clip_id, key AS ann_key, project AS ann_project,
                   start_time AS prev_start_time, label_type AS prev_label_type
            FROM annotations
            WHERE uid = ?
        """
        upd_times = """
            UPDATE annotations SET start_time = ?, end_time = ?, label_type = 'manual'
            WHERE uid = ?
        """

        with self.lock, self.conn:
            row = self.conn.execute(sel, (uid,)).fetchone()
            if not row:
                return

            # Use canonical values from DB to avoid key/project mismatches
            ann_key = row["ann_key"]
            ann_project = row["ann_project"]
            prev_start = row["prev_start_time"]

            # Ensure annotation is marked as manual if not already (in-place)
            if row["prev_label_type"] != "manual":
                # Flip to manual with proper counters/flags/indexes
                self.verify(uid, ann_project)
            # Apply DB update by uid
            self.conn.execute(upd_times, (start_time, end_time, uid))

            # Increment with_time only on transition: -1 -> non -1
            if start_time != -1 and prev_start == -1:
                if (
                    ann_project in self.stats
                    and ann_key in self.stats[ann_project]
                ):
                    d = self.stats[ann_project][ann_key]
                    d["with_time"] = d.get("with_time", 0) + 1

    def remove(self, clip_id, uid, annotation, project):
        select_query = """
            SELECT label_type, start_time FROM annotations
            WHERE clip_id = ? AND uid = ? AND key = ? AND project = ?
        """
        delete_query = """
            DELETE FROM annotations
            WHERE clip_id = ? AND uid = ? AND key = ? AND project = ?
        """
        with self.lock, self.conn:
            cur = self.conn.execute(
                select_query, (clip_id, uid, annotation, project)
            )
            # Get the label type of the clip to be removed
            row = cur.fetchone()
            if not row:
                return
            label_type = row["label_type"]
            self.conn.execute(delete_query, (clip_id, uid, annotation, project))

            # Update the state of the annotations
            if project in self.stats and annotation in self.stats[project]:
                d = self.stats[project][annotation]
                d[label_type] = max(0, (d.get(label_type, 0) - 1))
                if row["start_time"] != -1:
                    d["with_time"] = max(0, (d.get("with_time", 0) - 1))

            # Update the reverse indexes
            self._remove_annotation_from_index(
                clip_id, annotation, label_type, project
            )

    def verify(self, uid, project):
        """
        Convert an existing annotation's label_type to 'manual' while preserving times.
        Only applies if current label_type is 'autolabel'.
        Updates stats, clip flags, and reverse indexes accordingly.
        """
        select_query = "SELECT * FROM annotations WHERE uid = ?"
        update_query = """
            UPDATE annotations SET label_type = 'manual', project=?
            WHERE uid = ?
        """
        with self.lock, self.conn:
            row = self.conn.execute(select_query, (uid,)).fetchone()
            if not row:
                return

            # Extract info in local variables
            prev_label_type = row["label_type"]
            prev_project = row["project"]
            clip_id = row["clip_id"]
            annotation = row["key"]
            start_time = row["start_time"]
            has_times = start_time != -1

            if prev_label_type == "manual":
                return

            # Update DB row
            self.conn.execute(update_query, (project, uid))

            # Update stats
            stats_from = self.stats[prev_project]
            stats_to = self.stats.get(project, {})
            if annotation not in stats_to:
                stats_to[annotation] = {
                    "manual": 0,
                    "autolabel": 0,
                    "with_time": 0,
                }
            cnts_from = stats_from[annotation]
            cnts_to = stats_to[annotation]
            if prev_label_type in cnts_from:
                cnts_from[prev_label_type] -= 1
            cnts_to["manual"] += 1
            cnts_from["with_time"] -= int(has_times)
            cnts_to["with_time"] += int(has_times)

            # Update reverse indexes
            self._remove_annotation_from_index(
                clip_id, annotation, prev_label_type, prev_project
            )
            self._add_annotation_to_index(
                clip_id, annotation, "manual", project
            )

    def remove_autolabel(self, annotation, project):
        query = "DELETE FROM annotations WHERE key=? AND label_type='autolabel' AND project=?"
        with self.lock, self.conn:
            # Find affected clips first, within this project
            cur = self.conn.execute(
                "SELECT DISTINCT clip_id FROM annotations WHERE key=? AND label_type='autolabel' AND project=?",
                (annotation, project),
            )
            affected_clips = [row["clip_id"] for row in cur.fetchall()]
            if not affected_clips:
                return

            self.conn.execute(query, (annotation, project))

            # Update the state of the annotations
            if project in self.stats and annotation in self.stats[project]:
                self.stats[project][annotation]["autolabel"] = 0

            # Update the reverse index
            for clip_id in affected_clips:
                self._remove_annotation_from_index(
                    clip_id, annotation, "autolabel", project
                )

    def remove_label(self, annotation, project):
        delete_query = "DELETE FROM annotations WHERE key=? AND project=?"
        with self.lock, self.conn:
            # Find affected clips before deleting
            cur = self.conn.execute(
                "SELECT DISTINCT clip_id FROM annotations WHERE key=? AND project=?",
                (annotation, project),
            )
            affected_clips = [row["clip_id"] for row in cur.fetchall()]
            if len(affected_clips) == 0:
                return

            # Delete
            self.conn.execute(delete_query, (annotation, project))

            # Update stats
            if project in self.stats and annotation in self.stats[project]:
                for label_type in LABEL_TYPES:
                    self.stats[project][annotation][label_type] = 0
                self.stats[project][annotation]["with_time"] = 0

            # Update the reverse index
            for clip_id in affected_clips:
                for label_type in LABEL_TYPES:
                    self._remove_annotation_from_index(
                        clip_id, annotation, label_type, project
                    )

    def rename(self, old_name, new_name, project):
        if project not in self.stats:
            self.stats[project] = {}
        if new_name not in self.stats[project]:
            self.stats[project][new_name] = {
                "manual": 0,
                "autolabel": 0,
                "with_time": 0,
            }

        placeholders = ", ".join(["?"] * len(old_name))
        query = f"UPDATE annotations SET key=? WHERE key IN ({placeholders}) AND project=?"
        with self.lock, self.conn:
            _ = self.conn.execute(query, (new_name, *old_name, project))
            for annotation in old_name:
                # Update the reverse index
                clip_ids = self.key_to_clip_ids[project].pop(
                    annotation, Counter()
                )
                for cid, cnt in clip_ids.items():
                    self.key_to_clip_ids[project][new_name][cid] += cnt

                for label_type in LABEL_TYPES:
                    num = (
                        self.stats[project]
                        .get(annotation, {})
                        .get(label_type, 0)
                    )
                    self.stats[project][new_name][label_type] += num
                    self.stats[project][annotation][label_type] = 0

                self.stats[project][new_name]["with_time"] += self.stats[
                    project
                ][annotation].get("with_time", 0)
                self.stats[project][annotation]["with_time"] = 0

    def add_many(
        self,
        clip_ids,
        annotation,
        project,
        start_time=-1,
        end_time=-1,
        value=None,
        label_type="manual",
    ):
        # Normalize annotations -> list of keys
        if isinstance(annotation, (list, tuple)) and len(annotation) == len(
            clip_ids
        ):
            ann_keys = [re.sub(r"\s+", " ", str(a)).strip() for a in annotation]
        else:
            annotation_key = re.sub(r"\s+", " ", str(annotation)).strip()
            ann_keys = [annotation_key] * len(clip_ids)

        # Normalize times -> lists
        if isinstance(start_time, (list, tuple)):
            start_time = list(start_time)
        else:
            start_time = [start_time] * len(clip_ids)
        if isinstance(end_time, (list, tuple)):
            end_time = list(end_time)
        else:
            end_time = [end_time] * len(clip_ids)

        # Normalize value -> lists
        if isinstance(value, (list, tuple)):
            values = list(value)
        elif value is None:
            values = [None] * len(clip_ids)
        else:
            values = [None] * len(clip_ids)

        # Fallback if time lengths mismatch
        if len(start_time) != len(clip_ids) or len(end_time) != len(clip_ids):
            print(
                f"Length mismatch: got start_time={len(start_time)}, end_time={len(end_time)}, but clip_ids={len(clip_ids)}"
            )
            start_time = [-1] * len(clip_ids)
            end_time = [-1] * len(clip_ids)
        if len(values) != len(clip_ids):
            print(
                f"Length mismatch: got values={len(values)}, but clip_ids={len(clip_ids)}"
            )
            values = [None] * len(clip_ids)

        # Pair inputs per row
        targets = list(zip(clip_ids, ann_keys, start_time, end_time, values))

        with self.lock, self.conn:
            # Temp table with a unique constraint to deduplicate per (clip_id, key)
            self.conn.execute("DROP TABLE IF EXISTS temp_targets")
            self.conn.execute(
                """
                CREATE TEMP TABLE temp_targets (
                    tid INTEGER PRIMARY KEY AUTOINCREMENT,
                    clip_id TEXT,
                    key TEXT,
                    start_time REAL,
                    end_time REAL,
                    value REAL
                )
                """
            )
            self.conn.executemany(
                "INSERT INTO temp_targets (clip_id, key, start_time, end_time, value) VALUES (?, ?, ?, ?, ?)",
                targets,
            )

            # Eligible rows: no existing (project, clip_id, key)
            cur = self.conn.execute(
                """
                SELECT t.clip_id, t.key, t.start_time, t.end_time, t.value
                FROM temp_targets AS t
                LEFT JOIN annotations AS a
                       ON a.clip_id = t.clip_id
                      AND a.project = ?
                      AND a.key = t.key
                WHERE a.uid IS NULL
                """,
                (project,),
            )
            eligible = cur.fetchall()

            # Prepare rows for insertion
            rows = [
                (
                    unique_id(),
                    project,
                    row["clip_id"],
                    row["key"],
                    row["value"],
                    row["start_time"],
                    row["end_time"],
                    label_type if row["value"] is None else "numeric",
                )
                for row in eligible
            ]

            # Insert eligible rows
            self.conn.executemany(
                """
                INSERT OR IGNORE INTO annotations
                    (uid, project, clip_id, key, value, start_time, end_time, label_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

            # Update in-memory stats and reverse index
            if project not in self.stats:
                self.stats[project] = {}

            needs_sorting = set()

            for _, _, cid, key, val, st, _et, _lt in rows:
                if val is None:
                    if key not in self.stats[project]:
                        self.stats[project][key] = {
                            "manual": 0,
                            "autolabel": 0,
                            "with_time": 0,
                        }
                    self.stats[project][key][label_type] += 1
                    if st != -1:
                        self.stats[project][key]["with_time"] += 1
                    self._add_annotation_to_index(cid, key, label_type, project)
                else:
                    self.numeric_index[project][key].append((cid, val))
                    needs_sorting.add((project, key))

            for proj, key in needs_sorting:
                self.numeric_index[proj][key].sort(key=lambda x: x[1])

    def export_to_json(self, fp, keys=None):
        if keys is None:
            query = """
                SELECT c.clip_id, c.data_source, a.uid, a.key, a.label_type, a.start_time, a.end_time
                FROM clips c JOIN annotations a ON a.clip_id = c.clip_id
                WHERE a.label_type IN ('manual', 'autolabel')
                ORDER BY a.clip_id
            """
        else:
            placeholders = ", ".join(["?"] * len(keys))
            query = f"""
                SELECT c.clip_id, c.data_source, a.uid, a.key, a.label_type, a.start_time, a.end_time
                FROM clips c JOIN annotations a ON a.clip_id = c.clip_id
                WHERE a.label_type IN ('manual', 'autolabel')
                  AND a.key IN ({placeholders})
                ORDER BY a.clip_id
            """
            params = tuple(keys)

            # Fast encoder (uses orjson if available)
            try:
                import orjson

                def dumps(obj):
                    return orjson.dumps(obj).decode("utf-8")

            except Exception:
                import json

                def dumps(obj):
                    return json.dumps(obj, separators=(",", ":"))

            write = fp.write
            first_obj = True
            current_clip = None
            current_ds = None
            anns = []

            with self.lock:
                cur = self.conn.execute(query, params)
                write("[")
                for row in cur:
                    # Access by index is a bit faster than dict rows
                    clip_id, data_source, uid, key, label_type, st, et = (
                        row[0],
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                        row[5],
                        row[6],
                    )

                    if current_clip is None:
                        current_clip, current_ds = clip_id, data_source

                    if clip_id != current_clip:
                        # flush previous clip
                        if not first_obj:
                            write(",")
                        write(
                            dumps(
                                {
                                    "clip_id": current_clip,
                                    "data_source": current_ds,
                                    "annotations": anns,
                                }
                            )
                        )
                        first_obj = False
                        # start new bucket
                        current_clip, current_ds, anns = (
                            clip_id,
                            data_source,
                            [],
                        )

                    anns.append(
                        {
                            "uid": uid,
                            "key": key,
                            "label_type": label_type,
                            "start_time": st,
                            "end_time": et,
                        }
                    )

                # flush last clip
                if current_clip is not None:
                    if not first_obj:
                        write(",")
                    write(
                        dumps(
                            {
                                "clip_id": current_clip,
                                "data_source": current_ds,
                                "annotations": anns,
                            }
                        )
                    )

                write("]")

    def summarize_annotations(self):
        query = """
            SELECT key,
                   SUM(label_type = 'manual') as manual_count,
                   SUM(label_type = 'autolabel') as autolabel_count,
                   SUM(start_time != -1) as with_time_count
            FROM annotations
            GROUP BY key;
        """
        with self.lock:
            cur = self.conn.execute(query)
            return cur.fetchall()
