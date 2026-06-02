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
Caption store — schema

Tables
------
captions
    Source-of-truth table.  One row per caption (a clip may have multiple
    captions from different models or covering different time windows).

    Columns: uid, clip_id, model_name, caption, data_source, start_time,
             end_time.

clip_fts  (FTS5 virtual table)
    One row per *clip* (not per caption).  Aggregates all captions for a clip
    into a single searchable document so that FTS5 BM25 ranking and the LIMIT
    heap optimisation work without any JOIN.

    Columns:
      clip_id   UNINDEXED — not tokenized; used only for retrieval.
      sources             — space-separated sanitized data_source tokens,
                            e.g. "nexar araani avv21train".
      captions            — all caption text for the clip, space-separated.

    Tokenizer: unicode61 (splits on any non-letter/non-digit character).
    Data-source values are sanitized to alphanumeric-only before insertion so
    that a value like "AV-V2.1_train" becomes the single token "avv21train"
    and can be targeted with a column-filter query (sources:avv21train).

clip_fts_index
    Maps clip_id → clip_fts rowid.  Needed because clip_id is UNINDEXED in
    clip_fts, so a direct lookup by clip_id would require a full FTS table
    scan.  Used by _update_clip_fts for O(1) delete-then-reinsert updates.

"""

import re
import sqlite3
from threading import Lock

from sil_wheel.stores.search_utils import project_dict
from tqdm import tqdm
from sil_wheel.stores.utils import LRUDict

_DS_RE = re.compile(r"[^a-zA-Z0-9]")
_QUERY_RE = re.compile(r'"([^"]+)"|(\S+)')


def _sanitize_ds(ds: str) -> str:
    """Strip non-alphanumeric chars so the value survives the unicode61 tokenizer."""
    return _DS_RE.sub("", ds).lower()


class FTSCaptionStore:
    """SQLite-backed caption store with FTS5 full-text search.

    Wraps a SQLite database that holds raw captions (captions table) and a
    clip-level FTS5 index (clip_fts) for fast text and data-source filtered
    search.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    """

    def __init__(self, db_path: str):
        self.lock = Lock()
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

        with self.conn:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.execute("PRAGMA cache_size=-131072;")  # 128 MB

        # Warm clip_fts pages into the OS page cache so the first real search
        # does not pay a cold-read penalty.
        self.conn.execute(
            "SELECT clip_id FROM clip_fts WHERE captions MATCH 'turn' LIMIT 1"
        ).fetchone()

        # LRU cache keyed by (query, data_sources) to short-circuit repeated
        # identical searches within a single server session.
        self.searches = LRUDict(size=24)

    def _create_tables(self):
        """Create the v4 schema tables and indexes if they do not yet exist."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS captions (
                uid          INTEGER PRIMARY KEY,
                clip_id      TEXT    NOT NULL,
                model_name   TEXT    NOT NULL,
                caption      TEXT    NOT NULL,
                data_source  TEXT,
                start_time   REAL,
                end_time     REAL
            )
            """
        )

        # One FTS row per clip.
        # sources  — sanitized data_source tokens, space-separated
        # captions — all caption text for the clip, space-separated
        self.conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS clip_fts USING fts5(
                clip_id  UNINDEXED,
                sources,
                captions,
                tokenize='unicode61'
            )
            """
        )

        # Maps clip_id to clip_fts rowid for O(1) incremental deletes.
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS clip_fts_index (
                clip_id   TEXT    PRIMARY KEY,
                fts_rowid INTEGER NOT NULL
            )
            """
        )

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_captions_clip_id"
            " ON captions(clip_id)",
            "CREATE INDEX IF NOT EXISTS idx_captions_model_name"
            " ON captions(model_name)",
            "CREATE INDEX IF NOT EXISTS idx_captions_model_start"
            " ON captions(clip_id, model_name, start_time)",
            "CREATE INDEX IF NOT EXISTS idx_captions_data_source"
            " ON captions(data_source)",
        ]
        for idx in indexes:
            self.conn.execute(idx)
        self.conn.commit()

    def _update_clip_fts(self, clip_ids):
        """Rebuild clip_fts rows for the given clip_ids.

        Aggregates all captions and data_source values for each clip from the
        captions table, then performs a delete-then-reinsert in clip_fts.
        clip_fts_index is updated to reflect the new rowid.

        Must be called inside an active transaction (with self.conn).
        """
        for clip_id in clip_ids:
            rows = self.conn.execute(
                "SELECT caption, data_source FROM captions WHERE clip_id = ?",
                (clip_id,),
            ).fetchall()

            all_captions = " ".join(r["caption"] for r in rows)
            sources = " ".join(
                {
                    _sanitize_ds(r["data_source"])
                    for r in rows
                    if r["data_source"]
                }
            )

            idx = self.conn.execute(
                "SELECT fts_rowid FROM clip_fts_index WHERE clip_id = ?",
                (clip_id,),
            ).fetchone()
            if idx:
                self.conn.execute(
                    "DELETE FROM clip_fts WHERE rowid = ?",
                    (idx["fts_rowid"],),
                )

            self.conn.execute(
                "INSERT INTO clip_fts(clip_id, sources, captions)"
                " VALUES (?, ?, ?)",
                (clip_id, sources, all_captions),
            )
            new_rowid = self.conn.execute(
                "SELECT last_insert_rowid()"
            ).fetchone()[0]
            self.conn.execute(
                "INSERT OR REPLACE INTO clip_fts_index(clip_id, fts_rowid)"
                " VALUES (?, ?)",
                (clip_id, new_rowid),
            )

    def insert_from_dataframe(self, df, model_name, data_source):
        """
        Insert captions from a pandas DataFrame with columns:
          - clip_id (str)
          - summary (str)  -> stored in captions.caption
          - start_time (int/float)
          - end_time   (int/float)

        Assumes a single model_name and data_source for all rows.
        """
        required = {"clip_id", "summary", "start_time", "end_time"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        tuples = [
            (
                str(row["clip_id"]),
                model_name,
                str(row["summary"]),
                data_source,
                (
                    float(row["start_time"])
                    if row["start_time"] is not None
                    else None
                ),
                float(row["end_time"]) if row["end_time"] is not None else None,
            )
            for _, row in df.iterrows()
        ]

        caption_sql = """
            INSERT OR REPLACE INTO captions
                (clip_id, model_name, caption, data_source, start_time, end_time)
            VALUES (?, ?, ?, ?, ?, ?)
        """

        for i in tqdm(range(0, len(tuples), 2000)):
            chunk = tuples[i : i + 2000]
            affected = list({t[0] for t in chunk})
            with self.lock, self.conn:
                self.conn.executemany(caption_sql, chunk)
                self._update_clip_fts(affected)

        with self.lock:
            self.searches.clear()

    def get(self, clip_id, model_name=None):
        """
        Return captions for a clip (optionally scoped to a model).

          { model_name: [
              {"caption": str, "start_time": float|None, "end_time": float|None},
              ...
            ],
            ...
          }
        """
        if model_name:
            query = """
                SELECT uid, caption, start_time, end_time FROM captions
                WHERE clip_id = ? AND model_name = ?
                ORDER BY start_time ASC
            """
            params = (clip_id, model_name)
            with self.lock:
                rows = self.conn.execute(query, params).fetchall()
            return {
                model_name: [
                    {
                        "uid": row["uid"],
                        "caption": row["caption"],
                        "start_time": row["start_time"],
                        "end_time": row["end_time"],
                    }
                    for row in rows
                ]
            }
        else:
            query = """
                SELECT uid, model_name, caption, start_time, end_time FROM captions
                WHERE clip_id = ?
                ORDER BY model_name ASC, start_time ASC
            """
            with self.lock:
                rows = self.conn.execute(query, (clip_id,)).fetchall()

            captions = {}
            for row in rows:
                captions.setdefault(row["model_name"], []).append(
                    {
                        "uid": row["uid"],
                        "caption": row["caption"],
                        "start_time": row["start_time"],
                        "end_time": row["end_time"],
                    }
                )
            return captions

    _BOOL_OPS = frozenset({"AND", "OR", "NOT"})

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        parts = []
        pending = []
        for phrase, token in _QUERY_RE.findall(query.replace("-", " ")):
            if phrase:
                if pending:
                    parts.append(f'"{" ".join(pending)}"')
                    pending = []
                phrase = " ".join(phrase.strip().split())
                if phrase:
                    parts.append(f'"{phrase}"')
            elif token:
                if token.upper() in FTSCaptionStore._BOOL_OPS:
                    if pending:
                        parts.append(f'"{" ".join(pending)}"')
                        pending = []
                    parts.append(token.upper())
                else:
                    pending.append(token.strip())
        if pending:
            parts.append(f'"{" ".join(pending)}"')
        return " ".join(parts) if parts else ""

    def _inner_search(self, queries, limit: int = 100000, data_sources=None):
        """Run an FTS5 search and return a list of matching clip_ids.

        Parameters
        ----------
        queries:
            A single query string or a list of query strings.  Multiple queries
            are combined into a single FTS5 OR expression so the engine unions
            them in one inverted-index pass with unified BM25 ranking.
        limit:
            Maximum number of results to return.
        data_sources:
            Optional list of raw data_source strings to restrict the search to.
            Each value is sanitized and embedded as a ``sources:<token>`` filter
            in the MATCH expression so no JOIN is needed.

        Returns
        -------
        list[str]
            Clip IDs ordered by BM25 rank (best match first), up to ``limit``.
        """
        if isinstance(queries, str):
            queries = [queries]
        sanitized = [s for q in queries if q.strip() for s in [self._sanitize_fts5_query(q)] if s]
        if not sanitized:
            return []
        search = (
            " OR ".join(f"({s})" for s in sanitized)
            if len(sanitized) > 1
            else sanitized[0]
        )
        # Sort both queries and data_sources so the cache key is order-independent.
        cache_key = (
            tuple(sorted(sanitized)),
            tuple(sorted(data_sources)) if data_sources else None,
        )

        with self.lock:
            if cache_key in self.searches:
                return self.searches[cache_key]

        if data_sources:
            # Embed data_source filter directly in the FTS MATCH expression.
            # The sources column only contains sanitized alphanumeric tokens,
            # so FTS5 resolves (sources:ds1 OR sources:ds2) cheaply, and the
            # BM25 heap on the combined query fires normally.
            ds_tokens = [_sanitize_ds(ds) for ds in data_sources if ds]
            ds_tokens = [t for t in ds_tokens if t]
            if ds_tokens:
                ds_clause = " OR ".join(f"sources:{tok}" for tok in ds_tokens)
                match_expr = f"({ds_clause}) AND ({search})"
            else:
                match_expr = search
            sql = """
                SELECT clip_id FROM clip_fts
                WHERE clip_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """
            params = [match_expr, limit]
        else:
            # Single-column MATCH triggers FTS5's BM25 heap — stops after
            # LIMIT results without scanning all matching rows.
            sql = """
                SELECT clip_id FROM clip_fts
                WHERE captions MATCH ?
                ORDER BY rank
                LIMIT ?
            """
            params = [search, limit]

        # Execute the FTS5 query without holding the lock.  WAL mode supports
        # concurrent readers, so releasing here lets other server threads
        # proceed with their own searches in parallel.  Two threads racing on
        # the same cache-miss will both run the query (wasteful but correct);
        # last writer wins when storing the result.
        rows = self.conn.execute(sql, params).fetchall()
        clip_subset = [r["clip_id"] for r in rows]

        with self.lock:
            self.searches[cache_key] = clip_subset
        return clip_subset

    def search(self, filters, current_results):
        """Filter current_results to clips matching the caption search query.

        If ``filters.search`` is set, performs an FTS5 search (optionally
        restricted to ``filters.data_source``) and intersects the results with
        ``current_results``.  Returns ``current_results`` unchanged when no
        caption search is active.

        Parameters
        ----------
        filters:
            A SearchFilters instance.  ``filters.search`` is the query string;
            ``filters.data_source`` is an optional list of data_source values.
        current_results:
            Dict mapping clip_id → clip metadata, as maintained by the server.

        Returns
        -------
        dict
            Subset of ``current_results`` whose clip_ids matched the search.
        """
        if filters.search is not None:
            ds = getattr(filters, "data_source", None)
            extra = getattr(filters, "caption_extra_queries", None)
            queries = list(dict.fromkeys([filters.search] + list(extra or [])))
            if filters.has_prior_filters("search") and current_results:
                # Scale limit to the candidate pool so clips in current_results
                # that rank beyond the default limit are not silently dropped.
                limit = min(max(len(current_results) * 3, 100_000), 200_000)
                clips = self._inner_search(queries, data_sources=ds, limit=limit)
            else:
                clips = self._inner_search(queries, data_sources=ds)
            current_results = project_dict(current_results, clips)

        return current_results
