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

import csv
import io
import json
import logging
import math
import os
import random
import shutil
import sqlite3
import time
import uuid
from itertools import combinations
from threading import RLock

# Debug logging gated by ARENA_DEBUG=1
log = logging.getLogger("arena")
if os.environ.get("ARENA_DEBUG"):
    logging.basicConfig()
    log.setLevel(logging.DEBUG)
else:
    log.setLevel(logging.WARNING)

EXT = {"video": ".mp4", "text": ".txt", "image": ".jpg", "json": ".json"}
CONTENT_TYPE = {".mp4": "video/mp4", ".jpg": "image/jpeg", ".png": "image/png", ".txt": "text/plain", ".json": "application/json"}
# Types that are fetched as text content (vs served as asset URLs)
TEXT_TYPES = {"text", "json"}
ELO_SCORE_MAP = {"a_strong": 1.0, "a": 0.75, "tie": 0.5, "b": 0.25, "b_strong": 0.0}


def _elo_update(r_a, r_b, winner, k):
    """Compute new ELO ratings for a single vote. Returns (new_a, new_b)."""
    e_a = 1.0 / (1 + 10 ** ((r_b - r_a) / 400))
    if winner == "both_bad":
        return r_a - k * e_a, r_b - k * (1 - e_a)
    s_a = ELO_SCORE_MAP.get(winner, 0.5)
    return r_a + k * (s_a - e_a), r_b + k * ((1 - s_a) - (1 - e_a))


class ArenaStore:
    def __init__(self, db_path, s3_client, bucket):
        self.lock = RLock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.s3 = s3_client
        self.bucket = bucket
        self._last_s3_sync = 0

        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS arena_votes (
                    id INTEGER PRIMARY KEY,
                    match_id TEXT UNIQUE,
                    arena_name TEXT NOT NULL,
                    input_id TEXT NOT NULL,
                    model_a TEXT NOT NULL,
                    model_b TEXT NOT NULL,
                    winner TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    username TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    reasoning TEXT
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS arena_elo (
                    arena_name TEXT NOT NULL,
                    model TEXT NOT NULL,
                    rating REAL NOT NULL DEFAULT 1500,
                    matches INTEGER NOT NULL DEFAULT 0,
                    wins INTEGER NOT NULL DEFAULT 0,
                    losses INTEGER NOT NULL DEFAULT 0,
                    ties INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (arena_name, model)
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS arena_manifests (
                    arena_name TEXT PRIMARY KEY,
                    manifest_json TEXT NOT NULL,
                    published INTEGER NOT NULL DEFAULT 0,
                    updated_at INTEGER NOT NULL
                )
            """)
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_votes_arena ON arena_votes (arena_name)"
            )
        log.debug("ArenaStore ready: db=%s bucket=%s", db_path, bucket)


    def _s3_key(self, arena_name, item_id, filename):
        return f"arenas/{arena_name}/assets/{item_id}/{filename}"

    def _fetch_text(self, key):
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read().decode("utf-8")
        except Exception as e:
            log.warning("Failed to fetch %s: %s", key, e)
            return None

    def fetch_asset_bytes(self, arena_name, item_id, filename):
        """Fetch raw bytes of an arena asset from S3."""
        key = self._s3_key(arena_name, item_id, filename)
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except Exception as e:
            log.warning("Failed to fetch bytes %s: %s", key, e)
            return None

    def load_match_assets(self, arena_name, manifest, item_id, model_a, model_b):
        """Load all asset content for a match (bytes for video/image, text for text/json).

        Returns:
            {inputs: [{name, type, label, content|bytes, start_time?, end_time?}],
             outputs_a: [...], outputs_b: [...], instructions: str}
        """
        # Load per-item metadata sidecar (same as _load_match_data)
        meta_key = self._s3_key(arena_name, item_id, "meta.json")
        meta_text = self._fetch_text(meta_key)
        meta = json.loads(meta_text) if meta_text else {}

        def _load_entries(defs, prefix_fn):
            entries = []
            for d in defs:
                fname = prefix_fn(d)
                entry = {"name": d["name"], "type": d["type"], "label": d.get("label", d["name"])}
                if d["type"] in TEXT_TYPES:
                    key = self._s3_key(arena_name, item_id, fname)
                    entry["content"] = self._fetch_text(key) or ""
                else:
                    entry["bytes"] = self.fetch_asset_bytes(arena_name, item_id, fname)
                # Attach video time range from meta.json
                if d["type"] == "video":
                    inp_meta = meta.get(f"input_{d['name']}", {})
                    if "start_time" in inp_meta:
                        entry["start_time"] = inp_meta["start_time"]
                    if "end_time" in inp_meta:
                        entry["end_time"] = inp_meta["end_time"]
                entries.append(entry)
            return entries

        inputs = _load_entries(
            manifest["inputs"],
            lambda d: f"input_{d['name']}{EXT[d['type']]}",
        )
        outputs_a = _load_entries(
            manifest["outputs"],
            lambda d: f"{model_a}_{d['name']}{EXT[d['type']]}",
        )
        outputs_b = _load_entries(
            manifest["outputs"],
            lambda d: f"{model_b}_{d['name']}{EXT[d['type']]}",
        )
        return {
            "inputs": inputs,
            "outputs_a": outputs_a,
            "outputs_b": outputs_b,
            "instructions": manifest.get("instructions", ""),
        }

    def _sync_s3_arenas(self):
        """Discover new arena folders on S3 and insert them into SQLite.
        Runs at most once every 60 seconds."""
        now = time.time()
        if now - self._last_s3_sync < 60:
            return
        self._last_s3_sync = now

        log.debug("Syncing arenas from S3")
        try:
            resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix="arenas/", Delimiter="/")
        except Exception as e:
            log.warning("S3 sync failed: %s", e)
            return

        s3_names = set()
        for p in resp.get("CommonPrefixes", []):
            s3_names.add(p["Prefix"].rstrip("/").split("/")[-1])

        with self.lock:
            existing = {r["arena_name"] for r in self.conn.execute(
                "SELECT arena_name FROM arena_manifests"
            ).fetchall()}

        for name in s3_names - existing:
            key = f"arenas/{name}/manifest.json"
            try:
                body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
                manifest = json.loads(body)
                with self.lock, self.conn:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO arena_manifests (arena_name, manifest_json, published, updated_at) VALUES (?,?,?,?)",
                        (name, json.dumps(manifest), 0, int(time.time())),
                    )
                log.debug("Inserted new arena from S3: %s", name)
            except Exception as e:
                log.warning("Failed to fetch manifest for new arena %s: %s", name, e)

    def list_arenas(self, user_id=None, username=None, role=None):
        """List arenas with visibility filtering.
        Returns all published arenas for everyone.
        Also returns unpublished arenas where user is in owners list OR role == 'admin'.
        """
        self._sync_s3_arenas()

        with self.lock:
            rows = self.conn.execute(
                "SELECT arena_name, manifest_json, published FROM arena_manifests"
            ).fetchall()
            count_rows = self.conn.execute(
                "SELECT arena_name, COUNT(*) as cnt FROM arena_votes GROUP BY arena_name"
            ).fetchall()
        vote_counts = {r["arena_name"]: r["cnt"] for r in count_rows}

        arenas = []
        for r in rows:
            manifest = json.loads(r["manifest_json"])
            published = bool(r["published"])
            owners = manifest.get("owners", [])
            is_owner = username is not None and username in owners
            is_admin = role == "admin"

            # Visibility: published arenas for everyone, unpublished only for owners/admins
            if not published and not is_owner and not is_admin:
                continue

            arenas.append({
                "name": r["arena_name"],
                "display_name": manifest.get("display_name", r["arena_name"]),
                "description": manifest.get("description", ""),
                "total_votes": vote_counts.get(r["arena_name"], 0),
                "num_models": len(manifest.get("models", [])),
                "published": published,
                "is_owner": is_owner or is_admin,
            })
        log.debug("Listed %d arenas for user=%s role=%s", len(arenas), username, role)
        return arenas

    def get_manifest(self, arena_name):
        """Read manifest from SQLite. Falls back to S3 if not in DB."""
        with self.lock:
            row = self.conn.execute(
                "SELECT manifest_json FROM arena_manifests WHERE arena_name=?", (arena_name,)
            ).fetchone()
        if row:
            return json.loads(row["manifest_json"])

        # Fallback: fetch from S3 and insert
        key = f"arenas/{arena_name}/manifest.json"
        log.debug("Manifest not in DB, fetching from S3: %s", key)
        try:
            body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
            manifest = json.loads(body)
            with self.lock, self.conn:
                self.conn.execute(
                    "INSERT OR IGNORE INTO arena_manifests (arena_name, manifest_json, published, updated_at) VALUES (?,?,?,?)",
                    (arena_name, json.dumps(manifest), 0, int(time.time())),
                )
            return manifest
        except Exception as e:
            log.warning("Failed to load manifest for %s: %s", arena_name, e)
            return None

    def refresh_manifest(self, arena_name):
        """Re-fetch manifest from S3 and update the DB row."""
        key = f"arenas/{arena_name}/manifest.json"
        log.debug("Refreshing manifest from S3: %s", key)
        try:
            body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
            manifest = json.loads(body)
            with self.lock, self.conn:
                self.conn.execute(
                    "INSERT INTO arena_manifests (arena_name, manifest_json, published, updated_at) VALUES (?,?,0,?) "
                    "ON CONFLICT(arena_name) DO UPDATE SET manifest_json=excluded.manifest_json, updated_at=excluded.updated_at",
                    (arena_name, json.dumps(manifest), int(time.time())),
                )
            log.debug("Manifest refreshed for %s", arena_name)
            return True
        except Exception as e:
            log.warning("Failed to refresh manifest for %s: %s", arena_name, e)
            return False

    def publish_arena(self, arena_name):
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE arena_manifests SET published=1, updated_at=? WHERE arena_name=?",
                (int(time.time()), arena_name),
            )

    def unpublish_arena(self, arena_name):
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE arena_manifests SET published=0, updated_at=? WHERE arena_name=?",
                (int(time.time()), arena_name),
            )

    def _get_arena_row(self, arena_name):
        """Fetch manifest row from DB. Returns (manifest_dict, published) or (None, None)."""
        with self.lock:
            row = self.conn.execute(
                "SELECT manifest_json, published FROM arena_manifests WHERE arena_name=?", (arena_name,)
            ).fetchone()
        if not row:
            return None, None
        return json.loads(row["manifest_json"]), bool(row["published"])

    def can_access(self, arena_name, username, role):
        """Returns True if arena is published, OR user is in owners list, OR role is admin."""
        manifest, published = self._get_arena_row(arena_name)
        if manifest is None:
            return False
        return published or role == "admin" or username in manifest.get("owners", [])

    def is_owner_or_admin(self, arena_name, username, role):
        """For publish/unpublish/refresh actions."""
        if role == "admin":
            return True
        manifest, _ = self._get_arena_row(arena_name)
        if manifest is None:
            return False
        return username in manifest.get("owners", [])


    def _sample_match(self, arena_name, models, items):
        """Pick a (model_a, model_b, item_id) triple with smart weighting.

        Pair weight = coverage × elo_proximity × provisional_boost:
        - Coverage: 1/(1+count)^2 — aggressively prioritizes unseen pairs
        - ELO proximity: Gaussian decay exp(-(diff^2)/(2*400^2)) — prefers informative matches
        - Provisional boost: 3x for pairs involving a model with < num_models matches

        Item weight = 1/(1+count) — gentler, items are independent of model pairs.
        A coin flip randomizes A/B assignment.
        """
        num_models = len(models)
        all_pairs = list(combinations(sorted(models), 2))
        pair_counts = {p: 0 for p in all_pairs}
        item_counts = {it: 0 for it in items}

        with self.lock:
            pair_rows = self.conn.execute(
                "SELECT model_a, model_b, COUNT(*) as cnt FROM arena_votes WHERE arena_name = ? GROUP BY model_a, model_b",
                (arena_name,),
            ).fetchall()
            item_rows = self.conn.execute(
                "SELECT input_id, COUNT(*) as cnt FROM arena_votes WHERE arena_name = ? GROUP BY input_id",
                (arena_name,),
            ).fetchall()
            elo_rows = self.conn.execute(
                "SELECT model, rating, matches FROM arena_elo WHERE arena_name = ?",
                (arena_name,),
            ).fetchall()

        for r in pair_rows:
            pair = tuple(sorted([r["model_a"], r["model_b"]]))
            if pair in pair_counts:
                pair_counts[pair] += r["cnt"]

        for r in item_rows:
            if r["input_id"] in item_counts:
                item_counts[r["input_id"]] = r["cnt"]

        # Build ELO lookup — default to 1500 for models with no matches yet
        elo_rating = {}
        elo_matches = {}
        for r in elo_rows:
            elo_rating[r["model"]] = r["rating"]
            elo_matches[r["model"]] = r["matches"]

        # Compute pair weights: coverage × elo_proximity × provisional_boost
        sigma = 400  # same constant as ELO formula
        pair_weights = []
        for pair in all_pairs:
            count = pair_counts[pair]
            # Aggressive coverage: 1/(1+count)^2
            coverage_w = 1.0 / (1 + count) ** 2

            # ELO proximity: Gaussian decay on rating difference
            r_a = elo_rating.get(pair[0], 1500)
            r_b = elo_rating.get(pair[1], 1500)
            diff = abs(r_a - r_b)
            elo_w = max(0.2, math.exp(-(diff ** 2) / (2 * sigma ** 2)))

            # Provisional boost: 3x if either model has < num_models matches
            m_a = elo_matches.get(pair[0], 0)
            m_b = elo_matches.get(pair[1], 0)
            prov_boost = 3.0 if (m_a < num_models or m_b < num_models) else 1.0

            w = coverage_w * elo_w * prov_boost
            pair_weights.append(w)
            log.debug(
                "Pair %s: count=%d coverage=%.4f elo_diff=%d elo_w=%.4f prov=%.1f final=%.4f",
                pair, count, coverage_w, diff, elo_w, prov_boost, w,
            )

        (m1, m2) = random.choices(all_pairs, weights=pair_weights, k=1)[0]

        # Item weights: gentle 1/(1+count)
        item_weights = [1.0 / (1 + c) for c in item_counts.values()]
        item_id = random.choices(items, weights=item_weights, k=1)[0]

        # Coin-flip A/B assignment
        if random.random() < 0.5:
            m1, m2 = m2, m1

        log.debug("Sampled: %s vs %s on %s", m1, m2, item_id)
        return m1, m2, item_id


    def _load_match_data(self, arena_name, manifest, item_id, m1, m2):
        """Load inputs and outputs for a specific item + model pair."""
        inputs_def = manifest["inputs"]
        outputs_def = manifest["outputs"]

        # Load optional per-item metadata sidecar
        meta_key = self._s3_key(arena_name, item_id, "meta.json")
        meta_text = self._fetch_text(meta_key)
        meta = json.loads(meta_text) if meta_text else {}

        # Build inputs
        inputs_data = []
        for inp in inputs_def:
            fname = f"input_{inp['name']}{EXT[inp['type']]}"
            key = self._s3_key(arena_name, item_id, fname)
            entry = {"name": inp["name"], "type": inp["type"], "label": inp.get("label", inp["name"])}
            if inp["type"] in TEXT_TYPES:
                entry["content"] = self._fetch_text(key)
            else:
                entry["url"] = f"/arena/asset/{arena_name}/{item_id}/{fname}"
                # Video time range from meta.json
                inp_meta = meta.get(f"input_{inp['name']}", {})
                if inp["type"] == "video":
                    if "start_time" in inp_meta:
                        entry["start_time"] = inp_meta["start_time"]
                    if "end_time" in inp_meta:
                        entry["end_time"] = inp_meta["end_time"]
            inputs_data.append(entry)

        # Build outputs — one entry per output definition, per model
        def make_outputs(model):
            result = []
            for out in outputs_def:
                fname = f"{model}_{out['name']}{EXT[out['type']]}"
                key = self._s3_key(arena_name, item_id, fname)
                entry = {"name": out["name"], "type": out["type"], "label": out.get("label", out["name"])}
                if out["type"] in TEXT_TYPES:
                    entry["content"] = self._fetch_text(key)
                else:
                    entry["url"] = f"/arena/asset/{arena_name}/{item_id}/{fname}"
                result.append(entry)
            return result

        return {
            "arena_name": arena_name,
            "item_id": item_id,
            "model_a": m1,
            "model_b": m2,
            "inputs": inputs_data,
            "outputs_a": make_outputs(m1),
            "outputs_b": make_outputs(m2),
            "instructions": manifest.get("instructions", ""),
        }

    def get_next_match(self, arena_name, user_id):
        manifest = self.get_manifest(arena_name)
        if not manifest or len(manifest["models"]) < 2 or not manifest["items"]:
            return None

        m1, m2, item_id = self._sample_match(arena_name, manifest["models"], manifest["items"])
        match_id = str(uuid.uuid4())
        log.debug("Match: %s vs %s on %s (match=%s)", m1, m2, item_id, match_id)

        data = self._load_match_data(arena_name, manifest, item_id, m1, m2)
        data["match_id"] = match_id
        return data

    def get_match(self, arena_name, match_id):
        """Load a specific match by match_id for review."""
        with self.lock:
            row = self.conn.execute(
                "SELECT input_id, model_a, model_b FROM arena_votes WHERE arena_name=? AND match_id=?",
                (arena_name, match_id),
            ).fetchone()
        if not row:
            return None
        manifest = self.get_manifest(arena_name)
        if not manifest:
            return None
        return self._load_match_data(arena_name, manifest, row["input_id"], row["model_a"], row["model_b"])


    def submit_vote(self, arena_name, match_id, item_id, model_a, model_b, winner, user_id, username, reasoning=None):
        manifest = self.get_manifest(arena_name)
        elo_cfg = (manifest or {}).get("elo_config", {})
        k = elo_cfg.get("k_factor", 32)
        initial = elo_cfg.get("initial_rating", 1500)

        # Skip — no record, just move on
        if winner == "skip":
            return {"skipped": True}

        with self.lock, self.conn:
            # INSERT OR IGNORE: match_id has a UNIQUE constraint for dedup
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO arena_votes (match_id, arena_name, input_id, model_a, model_b, winner, user_id, username, created_at, reasoning) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (match_id, arena_name, item_id, model_a, model_b, winner, user_id, username, int(time.time()), reasoning),
            )
            if cur.rowcount == 0:
                return {"duplicate": True}

            # Ensure ELO rows exist
            for m in (model_a, model_b):
                self.conn.execute(
                    "INSERT OR IGNORE INTO arena_elo (arena_name, model, rating) VALUES (?,?,?)",
                    (arena_name, m, initial),
                )

            r_a = self.conn.execute("SELECT rating FROM arena_elo WHERE arena_name=? AND model=?", (arena_name, model_a)).fetchone()["rating"]
            r_b = self.conn.execute("SELECT rating FROM arena_elo WHERE arena_name=? AND model=?", (arena_name, model_b)).fetchone()["rating"]

            new_a, new_b = _elo_update(r_a, r_b, winner, k)

            if winner == "both_bad":
                outcome_a = outcome_b = "losses"
            else:
                outcome_a = "wins" if winner in ("a", "a_strong") else ("losses" if winner in ("b", "b_strong") else "ties")
                outcome_b = "wins" if winner in ("b", "b_strong") else ("losses" if winner in ("a", "a_strong") else "ties")

            log.debug("ELO: %s %.1f->%.1f, %s %.1f->%.1f (winner=%s)", model_a, r_a, new_a, model_b, r_b, new_b, winner)

            # outcome_a/outcome_b are always one of "wins", "losses", "ties" (hardcoded above, not user input)
            self.conn.execute(
                f"UPDATE arena_elo SET rating=?, matches=matches+1, {outcome_a}={outcome_a}+1 WHERE arena_name=? AND model=?",
                (new_a, arena_name, model_a),
            )
            self.conn.execute(
                f"UPDATE arena_elo SET rating=?, matches=matches+1, {outcome_b}={outcome_b}+1 WHERE arena_name=? AND model=?",
                (new_b, arena_name, model_b),
            )

        return {
            "model_a_name": model_a,
            "model_b_name": model_b,
            "elo_a": round(new_a, 1),
            "elo_b": round(new_b, 1),
            "elo_change_a": round(new_a - r_a, 1),
            "elo_change_b": round(new_b - r_b, 1),
        }


    def get_leaderboard(self, arena_name):
        with self.lock:
            rows = self.conn.execute(
                "SELECT model, rating, matches, wins, losses, ties FROM arena_elo WHERE arena_name=? ORDER BY rating DESC",
                (arena_name,),
            ).fetchall()
            total = self.conn.execute(
                "SELECT COUNT(*) as cnt FROM arena_votes WHERE arena_name=?", (arena_name,)
            ).fetchone()["cnt"]

        rankings = [
            {
                "rank": i + 1,
                "model": r["model"],
                "rating": round(r["rating"], 1),
                "matches": r["matches"],
                "wins": r["wins"],
                "losses": r["losses"],
                "ties": r["ties"],
                "win_rate": round(r["wins"] / r["matches"] * 100, 1) if r["matches"] else 0,
            }
            for i, r in enumerate(rows)
        ]

        # Seed from manifest if no votes yet
        if not rankings:
            manifest = self.get_manifest(arena_name)
            if manifest:
                init = manifest.get("elo_config", {}).get("initial_rating", 1500)
                rankings = [
                    {"rank": i + 1, "model": m, "rating": init, "matches": 0, "wins": 0, "losses": 0, "ties": 0, "win_rate": 0}
                    for i, m in enumerate(manifest.get("models", []))
                ]

        log.debug("Leaderboard %s: %d models, %d votes", arena_name, len(rankings), total)
        return {"rankings": rankings, "total_matches": total}

    def export_votes_csv(self, arena_name):
        """Return all votes for an arena as a CSV string."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, input_id, model_a, model_b, winner, username, created_at, reasoning FROM arena_votes WHERE arena_name=? ORDER BY id",
                (arena_name,),
            ).fetchall()
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["id", "input_id", "model_a", "model_b", "winner", "username", "timestamp", "reasoning"])
        for r in rows:
            writer.writerow([r["id"], r["input_id"], r["model_a"], r["model_b"], r["winner"], r["username"],
                             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created_at"])), r["reasoning"] or ""])
        return buf.getvalue()

    def get_history(self, arena_name, limit=50, offset=0):
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, match_id, input_id, model_a, model_b, winner, username, created_at, reasoning FROM arena_votes WHERE arena_name=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (arena_name, limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_elo_history(self, arena_name, max_points=200):
        """Replay all votes to produce ELO rating over time per model.

        Returns {model: [{vote: N, rating: R}, ...], ...} with at most
        max_points entries per model (evenly sampled if there are more votes).
        Pure computation — no DB writes, no lock held during math.
        """
        manifest = self.get_manifest(arena_name)
        elo_cfg = (manifest or {}).get("elo_config", {})
        k = elo_cfg.get("k_factor", 32)
        initial = elo_cfg.get("initial_rating", 1500)

        with self.lock:
            votes = self.conn.execute(
                "SELECT model_a, model_b, winner FROM arena_votes WHERE arena_name=? ORDER BY id",
                (arena_name,),
            ).fetchall()

        if not votes:
            return {}

        ratings = {}    # model -> current rating
        match_num = {}  # model -> per-model match counter
        history = {}    # model -> [(match_num, rating), ...]

        for v in votes:
            ma, mb, winner = v["model_a"], v["model_b"], v["winner"]
            for m in (ma, mb):
                if m not in ratings:
                    ratings[m] = initial
                    match_num[m] = 0
                    history[m] = [(0, initial)]

            if winner == "skip":
                continue

            ratings[ma], ratings[mb] = _elo_update(ratings[ma], ratings[mb], winner, k)

            match_num[ma] += 1
            match_num[mb] += 1
            history[ma].append((match_num[ma], ratings[ma]))
            history[mb].append((match_num[mb], ratings[mb]))

        # Downsample if too many points
        result = {}
        for model, pts in history.items():
            if len(pts) <= max_points:
                result[model] = [{"match": p[0], "rating": round(p[1], 1)} for p in pts]
            else:
                step = len(pts) / max_points
                sampled = [pts[int(i * step)] for i in range(max_points - 1)]
                sampled.append(pts[-1])
                result[model] = [{"match": p[0], "rating": round(p[1], 1)} for p in sampled]

        log.debug("ELO history for %s: %d votes, %d models", arena_name, len(votes), len(result))
        return result

    @staticmethod
    def _replay_elo(votes, k, initial):
        """Replay a list of (model_a, model_b, winner) tuples and return final ratings."""
        ratings = {}
        for ma, mb, winner in votes:
            for m in (ma, mb):
                if m not in ratings:
                    ratings[m] = initial
            if winner == "skip":
                continue
            ratings[ma], ratings[mb] = _elo_update(ratings[ma], ratings[mb], winner, k)
        return ratings

    def get_elo_confidence(self, arena_name, n_bootstrap=1000):
        """Bootstrap 95% confidence intervals for ELO ratings.

        Only computed for non-provisional models (those with >= num_models matches).
        """
        manifest = self.get_manifest(arena_name)
        elo_cfg = (manifest or {}).get("elo_config", {})
        k = elo_cfg.get("k_factor", 32)
        initial = elo_cfg.get("initial_rating", 1500)
        num_models = len((manifest or {}).get("models", []))

        # Get per-model match counts to filter out provisional models
        with self.lock:
            rows = self.conn.execute(
                "SELECT model_a, model_b, winner FROM arena_votes WHERE arena_name=? ORDER BY id",
                (arena_name,),
            ).fetchall()
            match_counts = {}
            for r in self.conn.execute(
                "SELECT model, matches FROM arena_elo WHERE arena_name=?", (arena_name,)
            ).fetchall():
                match_counts[r["model"]] = r["matches"]

        votes = [(r["model_a"], r["model_b"], r["winner"]) for r in rows]
        if not votes:
            return {}

        # Collect bootstrap samples
        samples = {}  # model -> list of final ratings
        n = len(votes)
        for _ in range(n_bootstrap):
            resampled = [votes[random.randint(0, n - 1)] for _ in range(n)]
            ratings = self._replay_elo(resampled, k, initial)
            for model, rating in ratings.items():
                samples.setdefault(model, []).append(rating)

        # Compute percentiles — only for non-provisional models
        result = {}
        for model, vals in samples.items():
            if match_counts.get(model, 0) < num_models:
                continue
            vals.sort()
            lo = vals[int(len(vals) * 0.025)]
            hi = vals[int(len(vals) * 0.975)]
            result[model] = {"ci_low": round(lo, 1), "ci_high": round(hi, 1)}

        return result


    def serve_asset(self, handler, arena_name, item_id, filename):
        key = self._s3_key(arena_name, item_id, filename)
        ext = os.path.splitext(filename)[1].lower()
        ct = CONTENT_TYPE.get(ext, "application/octet-stream")

        try:
            kwargs = {"Bucket": self.bucket, "Key": key}
            if "Range" in handler.headers:
                kwargs["Range"] = handler.headers["Range"]

            resp = self.s3.get_object(**kwargs)

            if "ContentRange" in resp:
                handler.send_response(206)
                handler.send_header("Content-Range", resp["ContentRange"])
            else:
                handler.send_response(200)

            handler.send_header("Content-Type", ct)
            handler.send_header("Content-Length", resp["ContentLength"])
            handler.send_header("Accept-Ranges", "bytes")
            handler.end_headers()
            shutil.copyfileobj(resp["Body"], handler.wfile)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            log.warning("serve_asset %s: %s", key, e)
            handler.send_error(404, "Asset not found")
