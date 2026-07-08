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
from urllib.parse import quote
import threading
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

# ── Rating system: Glicko-2 (one vote = one rating period) ──
# Each (arena, model, criterion) has state (rating, rd, volatility). Display is on the
# ELO-compatible scale (1500 ± hundreds); the internal μ/φ scale is confined to
# _glicko2_update. See http://www.glicko.net/glicko/glicko2.pdf.
#
# Every vote is its own rating period, so the chart shows a smooth per-vote trajectory.
# Volatility σ is injected once per vote — kept low (default 0.02) since arena models are
# static artifacts; the only genuine drift comes from evolving annotator standards.
SCORE_MAP = {"a_strong": 1.0, "a": 0.75, "tie": 0.5, "b": 0.25, "b_strong": 0.0}
GLICKO2_SCALE = 400.0 / math.log(10)          # ≈ 173.7178 — display units per internal unit
DEFAULT_INITIAL_RATING = 1500.0
DEFAULT_INITIAL_RD = 350.0
DEFAULT_INITIAL_VOLATILITY = 0.02              # σ injected per vote; low because models don't drift
DEFAULT_TAU = 0.5                              # system volatility constraint (0.3–1.2 typical)
CI_Z = 1.96                                    # 95% Gaussian CI half-width in RD units


def _glicko_cfg(manifest):
    """Return (initial_rating, initial_rd, initial_volatility, tau) from manifest.rating_config.
    Any field absent from the manifest falls back to the DEFAULT_* constants defined above —
    those are the single source of truth for defaults."""
    cfg = (manifest or {}).get("rating_config", {})
    return (
        cfg.get("initial_rating", DEFAULT_INITIAL_RATING),
        cfg.get("initial_rd", DEFAULT_INITIAL_RD),
        cfg.get("initial_volatility", DEFAULT_INITIAL_VOLATILITY),
        cfg.get("tau", DEFAULT_TAU),
    )


def _glicko2_update(state, opp_state, score, tau):
    """Apply one Glicko-2 single-match update to *state* against *opp_state*.

    Args:
        state: (rating, rd, volatility) on display scale (rating around 1500, rd in same units).
        opp_state: opponent state, same shape.
        score: outcome in [0, 1] for *state*'s player (1 = win, 0.5 = tie, 0 = loss).
        tau: system volatility constraint.
    Returns:
        (new_rating, new_rd, new_volatility) on display scale.
    """
    r, rd, vol = state
    opp_r, opp_rd, _ = opp_state

    mu = (r - DEFAULT_INITIAL_RATING) / GLICKO2_SCALE
    phi = rd / GLICKO2_SCALE
    opp_mu = (opp_r - DEFAULT_INITIAL_RATING) / GLICKO2_SCALE
    opp_phi = opp_rd / GLICKO2_SCALE

    g = 1.0 / math.sqrt(1.0 + 3.0 * opp_phi * opp_phi / (math.pi * math.pi))
    exp_val = math.exp(-g * (mu - opp_mu))
    E = 1.0 / (1.0 + exp_val)
    E_1mE = max(E * (1.0 - E), 1e-12)  # clamp to avoid division blow-up at E→0/1
    v = 1.0 / (g * g * E_1mE)
    delta = v * g * (score - E)

    # New volatility via Illinois-method root-finding (Glickman 2013, Step 5)
    a = math.log(vol * vol)
    def f(x):
        ex = math.exp(x)
        num = ex * (delta * delta - phi * phi - v - ex)
        den = 2.0 * (phi * phi + v + ex) ** 2
        return num / den - (x - a) / (tau * tau)
    A = a
    if delta * delta > phi * phi + v:
        B = math.log(delta * delta - phi * phi - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        B = a - k * tau
    fA, fB = f(A), f(B)
    for _ in range(100):  # bounded loop; converges in ~5-10 iters in practice
        if abs(B - A) <= 1e-6:
            break
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA /= 2.0
        B, fB = C, fC
    new_vol = math.exp(A / 2.0)

    phi_star = math.sqrt(phi * phi + new_vol * new_vol)
    new_phi = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    new_mu = mu + new_phi * new_phi * g * (score - E)

    return (
        DEFAULT_INITIAL_RATING + GLICKO2_SCALE * new_mu,
        GLICKO2_SCALE * new_phi,
        new_vol,
    )


def _apply_match(state_a, state_b, winner, tau):
    """Run one Glicko-2 match update on both players.

    Returns (new_state_a, new_state_b). Handles our vote codes including 'both_bad'
    (both players scored 0). 'skip' is not accepted here — the caller must filter.
    """
    if winner == "both_bad":
        s_a, s_b = 0.0, 0.0
    else:
        s_a = SCORE_MAP.get(winner, 0.5)
        s_b = 1.0 - s_a
    return (
        _glicko2_update(state_a, state_b, s_a, tau),
        _glicko2_update(state_b, state_a, s_b, tau),
    )


def _info_gain(state_a, state_b):
    """Expected variance reduction from one match between *state_a* and *state_b*.

    Approximate closed-form: computes φ'² using pre-update φ* (i.e. ignores the tiny
    outcome-dependent σ update via the Illinois solver). At σ≈0.02 this drift is 5th-decimal
    and doesn't perturb the sampler's ranking of pairs. Sum of (φ*² − φ'²) across both
    players — larger = a match between these two shrinks their combined uncertainty more.
    """
    r_a, rd_a, vol_a = state_a
    r_b, rd_b, vol_b = state_b
    phi_a = rd_a / GLICKO2_SCALE
    phi_b = rd_b / GLICKO2_SCALE
    mu_a = (r_a - DEFAULT_INITIAL_RATING) / GLICKO2_SCALE
    mu_b = (r_b - DEFAULT_INITIAL_RATING) / GLICKO2_SCALE

    def side_gain(mu_x, phi_x, vol_x, mu_y, phi_y):
        g = 1.0 / math.sqrt(1.0 + 3.0 * phi_y * phi_y / (math.pi * math.pi))
        E = 1.0 / (1.0 + math.exp(-g * (mu_x - mu_y)))
        v = 1.0 / (g * g * max(E * (1.0 - E), 1e-12))
        phi_star_sq = phi_x * phi_x + vol_x * vol_x
        post_phi_sq = 1.0 / (1.0 / phi_star_sq + 1.0 / v)
        return phi_star_sq - post_phi_sq

    return side_gain(mu_a, phi_a, vol_a, mu_b, phi_b) + side_gain(mu_b, phi_b, vol_b, mu_a, phi_a)


def _aggregate_ratings_prob_space(ratings, initial=DEFAULT_INITIAL_RATING):
    """Combine per-criterion ratings into one via probability-space averaging.

    Ratings are logit-scaled — arithmetic means over-weight extreme values. Instead,
    convert each rating to its implied win probability against an *initial*-rated
    reference, average those probabilities, then convert back. Reduces to arithmetic
    mean only when all ratings are equal.
    """
    if not ratings:
        return initial
    probs = [1.0 / (1.0 + 10 ** ((initial - r) / 400.0)) for r in ratings]
    p = sum(probs) / len(probs)
    p = max(min(p, 1.0 - 1e-9), 1e-9)
    return initial + 400.0 * math.log10(p / (1.0 - p))


def _get_criteria(manifest):
    """Return list of criteria dicts from a manifest.
    If the manifest has a ``criteria`` list, return it.
    Otherwise fall back to a single 'overall' criterion using the legacy ``instructions`` field.
    """
    if manifest and manifest.get("criteria"):
        return manifest["criteria"]
    return [{"name": "overall", "description": manifest.get("instructions", "") if manifest else ""}]


class ArenaStore:
    def __init__(self, db_path, s3_client, bucket):
        self.lock = RLock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.s3 = s3_client
        self.bucket = bucket
        self._last_s3_sync = 0
        # Sync interval (seconds). The set of arenas changes rarely — only when
        # someone manually uploads a new manifest.json. A background daemon
        # thread does the sync so request handlers never wait on S3.
        self._sync_interval = 300  # 5 minutes
        # Dev server scans arenas_dev/; prod scans arenas_prod/
        self._s3_prefixes = ["arenas_dev"] if os.environ.get("ARENA_DEV") else ["arenas_prod"]
        self._arena_prefix = {}  # arena_name → S3 prefix (populated by _sync_s3_arenas)
        # In-memory mirror of arena_manifests for fast access checks.
        # Source of truth is the DB; this dict is always kept in sync — writes update both.
        # Opening an arena triggers 4-5 access checks; without this they'd all hit SQLite.
        self._arenas = {}  # arena_name → (manifest_dict, published_bool)

        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS arena_votes (
                    id INTEGER PRIMARY KEY,
                    match_id TEXT NOT NULL,
                    arena_name TEXT NOT NULL,
                    input_id TEXT NOT NULL,
                    model_a TEXT NOT NULL,
                    model_b TEXT NOT NULL,
                    winner TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    username TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    reasoning TEXT,
                    criterion TEXT NOT NULL DEFAULT 'overall',
                    duration_ms INTEGER,
                    UNIQUE (match_id, criterion)
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS arena_elo (
                    arena_name TEXT NOT NULL,
                    model TEXT NOT NULL,
                    criterion TEXT NOT NULL DEFAULT 'overall',
                    rating REAL NOT NULL DEFAULT 1500,
                    rating_deviation REAL NOT NULL DEFAULT 350,
                    volatility REAL NOT NULL DEFAULT 0.02,
                    matches INTEGER NOT NULL DEFAULT 0,
                    wins INTEGER NOT NULL DEFAULT 0,
                    losses INTEGER NOT NULL DEFAULT 0,
                    ties INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (arena_name, model, criterion)
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
        self._reload_arena_cache()
        log.debug("ArenaStore ready: db=%s bucket=%s s3_prefixes=%s arenas=%d",
                  db_path, bucket, self._s3_prefixes, len(self._arenas))

        # Initial S3 sync (synchronous, tolerate failure) so /arena/list returns
        # fresh data on the first request. Then start the background sync loop.
        try:
            self._sync_s3_arenas(force=True)
        except Exception as e:
            log.warning("Initial S3 sync failed: %s — will retry in background", e)
        t = threading.Thread(target=self._sync_loop, daemon=True, name="arena-s3-sync")
        t.start()

    def _sync_loop(self):
        """Daemon background loop that re-syncs from S3 on a fixed interval."""
        while True:
            time.sleep(self._sync_interval)
            try:
                self._sync_s3_arenas(force=True)
            except Exception as e:
                log.warning("Background S3 sync failed: %s", e)

    def _reload_arena_cache(self):
        """Rebuild the in-memory arena map from the DB. Called at startup and after S3 sync."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT arena_name, manifest_json, published FROM arena_manifests"
            ).fetchall()
        self._arenas = {
            r["arena_name"]: (json.loads(r["manifest_json"]), bool(r["published"]))
            for r in rows
        }


    def _s3_key(self, arena_name, item_id, filename):
        prefix = self._arena_prefix.get(arena_name, self._s3_prefixes[0])
        return f"{prefix}/{arena_name}/assets/{item_id}/{filename}"

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
            "criteria": _get_criteria(manifest),
        }

    def _discover_arenas_recursive(self, prefix, rel_path="", depth=0):
        """Recursively scan S3 for arena folders (folders containing manifest.json).
        Returns list of arena paths relative to *prefix* (e.g. ``"world-models/arena-x"``)."""
        if depth > 5:
            return []
        found = []
        search = f"{prefix}/{rel_path}" if rel_path else f"{prefix}/"
        try:
            resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=search, Delimiter="/")
        except Exception as e:
            log.warning("S3 recursive scan failed for %s: %s", search, e)
            return found

        for cp in resp.get("CommonPrefixes", []):
            folder = cp["Prefix"]                       # e.g. "arenas/world-models/"
            name = folder[len(prefix) + 1:].rstrip("/") # e.g. "world-models"
            manifest_key = f"{folder}manifest.json"
            try:
                self.s3.head_object(Bucket=self.bucket, Key=manifest_key)
                found.append(name)                       # has manifest → arena
            except Exception:
                # No manifest at this level → recurse into subfolders
                found.extend(self._discover_arenas_recursive(prefix, name + "/", depth + 1))
        return found

    def _sync_s3_arenas(self, force=False):
        """Discover new arena folders on S3 and insert them into SQLite.
        Normally driven by the background daemon thread. ``force=True`` bypasses
        the interval guard (used for the initial sync and the periodic loop).
        Scans all prefixes in ``self._s3_prefixes``; supports nested folders —
        any folder with a ``manifest.json`` is an arena."""
        now = time.time()
        if not force and now - self._last_s3_sync < self._sync_interval:
            return
        self._last_s3_sync = now

        with self.lock:
            existing = set(self._arenas.keys())

        for prefix in self._s3_prefixes:
            log.debug("Syncing arenas from S3 prefix: %s/", prefix)
            arena_names = self._discover_arenas_recursive(prefix)

            for name in arena_names:
                # Track which prefix owns this arena (first prefix wins on conflicts)
                if name not in self._arena_prefix:
                    self._arena_prefix[name] = prefix

                if name in existing:
                    continue
                key = f"{prefix}/{name}/manifest.json"
                try:
                    body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
                    manifest = json.loads(body)
                    # DB + in-memory mirror updated together under the same lock.
                    with self.lock, self.conn:
                        self.conn.execute(
                            "INSERT OR IGNORE INTO arena_manifests (arena_name, manifest_json, published, updated_at) VALUES (?,?,?,?)",
                            (name, json.dumps(manifest), 0, int(time.time())),
                        )
                        self._arenas[name] = (manifest, False)
                    existing.add(name)
                    log.debug("Inserted new arena from S3: %s (prefix=%s)", name, prefix)
                except Exception as e:
                    log.warning("Failed to fetch manifest for new arena %s: %s", name, e)

    def list_arenas(self, user_id=None, username=None, role=None):
        """List arenas with visibility filtering.
        Returns all published arenas for everyone.
        Also returns unpublished arenas where user is in owners list OR role == 'admin'.
        Reads from the in-memory mirror; the background sync thread keeps it fresh.
        """
        with self.lock:
            count_rows = self.conn.execute(
                "SELECT arena_name, COUNT(*) as cnt FROM arena_votes GROUP BY arena_name"
            ).fetchall()
        vote_counts = {r["arena_name"]: r["cnt"] for r in count_rows}

        # Snapshot under lock — bg thread may mutate self._arenas concurrently.
        with self.lock:
            arena_items = list(self._arenas.items())

        arenas = []
        for name, (manifest, published) in arena_items:
            owners = manifest.get("owners", [])
            is_owner = username is not None and username in owners
            is_admin = role == "admin"

            # Visibility: published arenas for everyone, unpublished only for owners/admins
            if not published and not is_owner and not is_admin:
                continue

            arenas.append({
                "name": name,
                "display_name": manifest.get("display_name", name),
                "description": manifest.get("description", ""),
                "total_votes": vote_counts.get(name, 0),
                "num_models": len(manifest.get("models", [])),
                "published": published,
                "is_owner": is_owner or is_admin,
            })
        log.debug("Listed %d arenas for user=%s role=%s", len(arenas), username, role)
        return arenas

    def get_manifest(self, arena_name):
        """Read manifest from the in-memory mirror. Falls back to S3 if missing."""
        cached = self._arenas.get(arena_name)
        if cached:
            return cached[0]

        # Fallback: fetch from S3 and insert (rare — arena not yet synced)
        prefix = self._arena_prefix.get(arena_name, self._s3_prefixes[0])
        key = f"{prefix}/{arena_name}/manifest.json"
        log.debug("Manifest not in cache, fetching from S3: %s", key)
        try:
            body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
            manifest = json.loads(body)
            # DB + in-memory mirror updated together under the same lock.
            with self.lock, self.conn:
                self.conn.execute(
                    "INSERT OR IGNORE INTO arena_manifests (arena_name, manifest_json, published, updated_at) VALUES (?,?,?,?)",
                    (arena_name, json.dumps(manifest), 0, int(time.time())),
                )
                self._arenas[arena_name] = (manifest, False)
            return manifest
        except Exception as e:
            log.warning("Failed to load manifest for %s: %s", arena_name, e)
            return None

    def refresh_manifest(self, arena_name):
        """Re-fetch manifest from S3 and update the DB row."""
        prefix = self._arena_prefix.get(arena_name, self._s3_prefixes[0])
        key = f"{prefix}/{arena_name}/manifest.json"
        log.debug("Refreshing manifest from S3: %s", key)
        try:
            body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
            manifest = json.loads(body)
            # DB + in-memory mirror updated together under the same lock; preserve
            # the existing published flag if the row already exists, else default to False.
            with self.lock, self.conn:
                self.conn.execute(
                    "INSERT INTO arena_manifests (arena_name, manifest_json, published, updated_at) VALUES (?,?,0,?) "
                    "ON CONFLICT(arena_name) DO UPDATE SET manifest_json=excluded.manifest_json, updated_at=excluded.updated_at",
                    (arena_name, json.dumps(manifest), int(time.time())),
                )
                published = self._arenas.get(arena_name, (None, False))[1]
                self._arenas[arena_name] = (manifest, published)
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
            m = self._arenas.get(arena_name)
            if m: self._arenas[arena_name] = (m[0], True)

    def unpublish_arena(self, arena_name):
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE arena_manifests SET published=0, updated_at=? WHERE arena_name=?",
                (int(time.time()), arena_name),
            )
            m = self._arenas.get(arena_name)
            if m: self._arenas[arena_name] = (m[0], False)

    def delete_match(self, arena_name, match_id):
        """Delete all vote rows for a match (every criterion) and rebuild the arena's
        rating state by replaying the remaining votes. Returns the number of deleted rows.
        Intended for admin/owner cleanup of bad votes; the caller is responsible for auth.
        """
        with self.lock, self.conn:
            cur = self.conn.execute(
                "DELETE FROM arena_votes WHERE arena_name=? AND match_id=?",
                (arena_name, match_id),
            )
            deleted = cur.rowcount
            if deleted > 0:
                self._rebuild_elo(arena_name)
        log.debug("Deleted %d vote rows for match %s in %s", deleted, match_id, arena_name)
        return deleted

    def _rebuild_elo(self, arena_name):
        """Truncate ``arena_elo`` for this arena and replay every remaining vote, per criterion.
        One vote = one rating period. Caller must already hold ``self.lock`` and an open
        ``self.conn`` transaction."""
        manifest, _ = self._arenas.get(arena_name, (None, None))
        initial, initial_rd, initial_vol, tau = _glicko_cfg(manifest)

        votes = self.conn.execute(
            "SELECT model_a, model_b, winner, criterion FROM arena_votes WHERE arena_name=? ORDER BY id",
            (arena_name,),
        ).fetchall()

        self.conn.execute("DELETE FROM arena_elo WHERE arena_name=?", (arena_name,))

        by_crit = {}
        for v in votes:
            by_crit.setdefault(v["criterion"], []).append((v["model_a"], v["model_b"], v["winner"]))

        for crit, vlist in by_crit.items():
            state, matches, wins, losses, ties = {}, {}, {}, {}, {}
            for ma, mb, w in vlist:
                if w == "skip":  # Skip rows are recorded but don't move ratings or stats.
                    continue
                for m in (ma, mb):
                    state.setdefault(m, (initial, initial_rd, initial_vol))
                    matches[m] = matches.get(m, 0) + 1
                if w in ("a_strong", "a"): wins[ma] = wins.get(ma, 0) + 1; losses[mb] = losses.get(mb, 0) + 1
                elif w in ("b_strong", "b"): wins[mb] = wins.get(mb, 0) + 1; losses[ma] = losses.get(ma, 0) + 1
                elif w == "tie": ties[ma] = ties.get(ma, 0) + 1; ties[mb] = ties.get(mb, 0) + 1
                else:  # both_bad
                    losses[ma] = losses.get(ma, 0) + 1; losses[mb] = losses.get(mb, 0) + 1
                state[ma], state[mb] = _apply_match(state[ma], state[mb], w, tau)
            for m, (r, rd, vol) in state.items():
                self.conn.execute(
                    "INSERT INTO arena_elo (arena_name, model, criterion, rating, rating_deviation, volatility, matches, wins, losses, ties) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (arena_name, m, crit, r, rd, vol,
                     matches.get(m, 0), wins.get(m, 0), losses.get(m, 0), ties.get(m, 0)),
                )

    def _get_arena_row(self, arena_name):
        """Returns (manifest_dict, published) or (None, None). Reads from the in-memory mirror."""
        return self._arenas.get(arena_name, (None, None))

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
        """Pick a (model_a, model_b, item_id) triple weighted by expected information gain.

        Pair weight = Σ_criterion of expected variance reduction from one Glicko-2 match
        between (A, B) on that criterion. Closed-form: posterior RD² is deterministic
        given priors, so no need to marginalize over outcomes. This one number replaces
        the previous coverage/proximity/provisional heuristic stack — all three signals
        emerge from the info-gain expression:
          * settled pairs → small φ² → small info gain (coverage falls out)
          * competitive pairs → E≈0.5 → 1/v peaks → larger info gain (proximity falls out)
          * new models → high φ² → large info gain (provisional falls out)

        Item weight = 1/(1+count) — items are orthogonal to model uncertainty.
        A coin flip randomizes A/B assignment.
        """
        manifest = self.get_manifest(arena_name)
        initial, initial_rd, initial_vol, _ = _glicko_cfg(manifest)
        criteria = [c["name"] for c in _get_criteria(manifest)]

        item_counts = {it: 0 for it in items}
        with self.lock:
            item_rows = self.conn.execute(
                "SELECT input_id, COUNT(DISTINCT match_id) as cnt FROM arena_votes WHERE arena_name = ? GROUP BY input_id",
                (arena_name,),
            ).fetchall()
            state_rows = self.conn.execute(
                "SELECT model, criterion, rating, rating_deviation, volatility FROM arena_elo WHERE arena_name=?",
                (arena_name,),
            ).fetchall()
        for r in item_rows:
            if r["input_id"] in item_counts:
                item_counts[r["input_id"]] = r["cnt"]

        # {(model, criterion): (rating, rd, volatility)}; missing → seed defaults
        state = {(r["model"], r["criterion"]): (r["rating"], r["rating_deviation"], r["volatility"])
                 for r in state_rows}
        def _s(model, crit):
            return state.get((model, crit), (initial, initial_rd, initial_vol))

        # Pair weights: expected info gain summed over criteria
        all_pairs = list(combinations(sorted(models), 2))
        pair_weights = []
        for a, b in all_pairs:
            gain = sum(_info_gain(_s(a, c), _s(b, c)) for c in criteria)
            # Small floor so pathologically-settled pairs remain sampleable
            pair_weights.append(max(gain, 1e-4))
            log.debug("Pair (%s, %s): expected info gain = %.5f", a, b, gain)

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
                entry["url"] = f"/arena/asset/{quote(arena_name, safe='')}/{item_id}/{fname}"
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
                    entry["url"] = f"/arena/asset/{quote(arena_name, safe='')}/{item_id}/{fname}"
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
        data["criteria"] = _get_criteria(manifest)
        return data

    def get_match(self, arena_name, match_id):
        """Load a specific match by match_id for review, including all criterion votes."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT input_id, model_a, model_b, criterion, winner, reasoning FROM arena_votes WHERE arena_name=? AND match_id=?",
                (arena_name, match_id),
            ).fetchall()
        if not rows:
            return None
        manifest = self.get_manifest(arena_name)
        if not manifest:
            return None
        first = rows[0]
        data = self._load_match_data(arena_name, manifest, first["input_id"], first["model_a"], first["model_b"])
        data["criteria"] = _get_criteria(manifest)
        data["criterion_votes"] = [
            {"criterion": r["criterion"], "winner": r["winner"], "reasoning": r["reasoning"] or ""}
            for r in rows
        ]
        return data


    def submit_vote(self, arena_name, match_id, item_id, model_a, model_b, winner, user_id, username, reasoning=None, criterion="overall", duration_ms=None):
        manifest = self.get_manifest(arena_name)
        initial, initial_rd, initial_vol, tau = _glicko_cfg(manifest)

        with self.lock, self.conn:
            # INSERT OR IGNORE: (match_id, criterion) has a UNIQUE constraint for dedup
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO arena_votes (match_id, arena_name, input_id, model_a, model_b, winner, user_id, username, created_at, reasoning, criterion, duration_ms) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (match_id, arena_name, item_id, model_a, model_b, winner, user_id, username, int(time.time()), reasoning, criterion, duration_ms),
            )
            if cur.rowcount == 0:
                return {"duplicate": True}

            def _current_state(m):
                row = self.conn.execute(
                    "SELECT rating, rating_deviation, volatility FROM arena_elo "
                    "WHERE arena_name=? AND model=? AND criterion=?",
                    (arena_name, m, criterion),
                ).fetchone()
                if row:
                    return (row["rating"], row["rating_deviation"], row["volatility"])
                return (initial, initial_rd, initial_vol)

            # Skip rows are recorded for timing analytics but do not move the rating.
            # Response mirrors a real vote (zero deltas, current effective state) so the
            # display code can format it uniformly with no branching.
            if winner == "skip":
                state_a = _current_state(model_a)
                state_b = _current_state(model_b)
                return {
                    "skipped": True,
                    "model_a_name": model_a,
                    "model_b_name": model_b,
                    "criterion": criterion,
                    "rating_a": round(state_a[0], 1),
                    "rating_b": round(state_b[0], 1),
                    "rd_a": round(state_a[1], 1),
                    "rd_b": round(state_b[1], 1),
                    "rating_change_a": 0,
                    "rating_change_b": 0,
                }

            # Ensure arena_elo rows exist so we can UPDATE below (defaults from Glicko-2 config)
            for m in (model_a, model_b):
                self.conn.execute(
                    "INSERT OR IGNORE INTO arena_elo (arena_name, model, criterion, rating, rating_deviation, volatility) "
                    "VALUES (?,?,?,?,?,?)",
                    (arena_name, m, criterion, initial, initial_rd, initial_vol),
                )

            state_a = _current_state(model_a)
            state_b = _current_state(model_b)
            new_a, new_b = _apply_match(state_a, state_b, winner, tau)

            if winner == "both_bad":
                outcome_a = outcome_b = "losses"
            else:
                outcome_a = "wins" if winner in ("a", "a_strong") else ("losses" if winner in ("b", "b_strong") else "ties")
                outcome_b = "wins" if winner in ("b", "b_strong") else ("losses" if winner in ("a", "a_strong") else "ties")

            log.debug(
                "Rating [%s]: %s %.1f/RD=%.0f -> %.1f/RD=%.0f, %s %.1f/RD=%.0f -> %.1f/RD=%.0f (winner=%s)",
                criterion, model_a, state_a[0], state_a[1], new_a[0], new_a[1],
                model_b, state_b[0], state_b[1], new_b[0], new_b[1], winner,
            )

            # outcome_a/outcome_b are always one of "wins", "losses", "ties" (hardcoded above)
            self.conn.execute(
                f"UPDATE arena_elo SET rating=?, rating_deviation=?, volatility=?, "
                f"matches=matches+1, {outcome_a}={outcome_a}+1 "
                "WHERE arena_name=? AND model=? AND criterion=?",
                (new_a[0], new_a[1], new_a[2], arena_name, model_a, criterion),
            )
            self.conn.execute(
                f"UPDATE arena_elo SET rating=?, rating_deviation=?, volatility=?, "
                f"matches=matches+1, {outcome_b}={outcome_b}+1 "
                "WHERE arena_name=? AND model=? AND criterion=?",
                (new_b[0], new_b[1], new_b[2], arena_name, model_b, criterion),
            )

        return {
            "model_a_name": model_a,
            "model_b_name": model_b,
            "criterion": criterion,
            "rating_a": round(new_a[0], 1),
            "rating_b": round(new_b[0], 1),
            "rd_a": round(new_a[1], 1),
            "rd_b": round(new_b[1], 1),
            "rating_change_a": round(new_a[0] - state_a[0], 1),
            "rating_change_b": round(new_b[0] - state_b[0], 1),
        }


    def get_leaderboard(self, arena_name, criterion=None):
        """Get Glicko-2 rating leaderboard. If *criterion* is a specific name, filter to it.
        If None (default), return aggregate: probability-space avg of ratings, mean of RDs,
        summed match/win/loss/tie counts."""
        manifest = self.get_manifest(arena_name)
        initial, initial_rd, _, _ = _glicko_cfg(manifest)

        with self.lock:
            if criterion:
                rows = self.conn.execute(
                    "SELECT model, rating, rating_deviation, matches, wins, losses, ties FROM arena_elo "
                    "WHERE arena_name=? AND criterion=? ORDER BY rating DESC",
                    (arena_name, criterion),
                ).fetchall()
                total = self.conn.execute(
                    "SELECT COUNT(*) as cnt FROM arena_votes WHERE arena_name=? AND criterion=?", (arena_name, criterion),
                ).fetchone()["cnt"]
                rows = [dict(r) for r in rows]
            else:
                per_crit = self.conn.execute(
                    "SELECT model, rating, rating_deviation, matches, wins, losses, ties "
                    "FROM arena_elo WHERE arena_name=?",
                    (arena_name,),
                ).fetchall()
                total = self.conn.execute(
                    "SELECT COUNT(*) as cnt FROM arena_votes WHERE arena_name=?", (arena_name,),
                ).fetchone()["cnt"]

                agg = {}  # model → {ratings, rds, matches, wins, losses, ties}
                for r in per_crit:
                    a = agg.setdefault(r["model"], {"ratings": [], "rds": [], "matches": 0, "wins": 0, "losses": 0, "ties": 0})
                    a["ratings"].append(r["rating"])
                    a["rds"].append(r["rating_deviation"])
                    a["matches"] += r["matches"]
                    a["wins"] += r["wins"]
                    a["losses"] += r["losses"]
                    a["ties"] += r["ties"]
                # Aggregate RD = arithmetic mean of per-criterion RDs. Deliberately conservative:
                # under fully-independent Gaussians the RD of a mean-of-ratings would be
                # sqrt(Σφ²)/n (tighter), but criteria are correlated (same annotators, same items)
                # so the independence assumption is optimistic. Mean-of-RDs gives a wider CI that
                # doesn't require modeling that correlation.
                rows = [
                    {"model": m,
                     "rating": _aggregate_ratings_prob_space(a["ratings"], initial),
                     "rating_deviation": sum(a["rds"]) / len(a["rds"]),
                     "matches": a["matches"], "wins": a["wins"], "losses": a["losses"], "ties": a["ties"]}
                    for m, a in agg.items()
                ]
                rows.sort(key=lambda r: -r["rating"])

        rankings = [
            {
                "rank": i + 1,
                "model": r["model"],
                "rating": round(r["rating"], 1),
                "rd": round(r["rating_deviation"], 1),
                "matches": r["matches"],
                "wins": r["wins"],
                "losses": r["losses"],
                "ties": r["ties"],
                "win_rate": round(r["wins"] / r["matches"] * 100, 1) if r["matches"] else 0,
            }
            for i, r in enumerate(rows)
        ]

        # Seed from manifest if no votes yet
        if not rankings and manifest:
            rankings = [
                {"rank": i + 1, "model": m, "rating": initial, "rd": initial_rd,
                 "matches": 0, "wins": 0, "losses": 0, "ties": 0, "win_rate": 0}
                for i, m in enumerate(manifest.get("models", []))
            ]

        log.debug("Leaderboard %s (criterion=%s): %d models, %d votes", arena_name, criterion, len(rankings), total)
        return {"rankings": rankings, "total_matches": total}

    def get_votes_json(self, arena_name):
        """Return all votes for an arena as a JSON-serializable list."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT match_id, input_id, model_a, model_b, winner, username, "
                "created_at, reasoning, criterion, duration_ms FROM arena_votes "
                "WHERE arena_name=? ORDER BY id", (arena_name,),
            ).fetchall()
        return [dict(r) for r in rows]


    def get_history(self, arena_name, limit=50, offset=0):
        """Return recent matches grouped by match_id.  Each entry includes
        ``criterion_votes`` — a list of per-criterion results."""
        with self.lock:
            # First get distinct match_ids in order, paginated
            match_rows = self.conn.execute(
                "SELECT match_id, MAX(created_at) as ts FROM arena_votes WHERE arena_name=? GROUP BY match_id ORDER BY ts DESC LIMIT ? OFFSET ?",
                (arena_name, limit, offset),
            ).fetchall()
            if not match_rows:
                return []
            match_ids = [r["match_id"] for r in match_rows]
            placeholders = ",".join("?" * len(match_ids))
            rows = self.conn.execute(
                f"SELECT match_id, input_id, model_a, model_b, winner, username, created_at, reasoning, criterion "
                f"FROM arena_votes WHERE arena_name=? AND match_id IN ({placeholders}) ORDER BY created_at ASC",
                [arena_name] + match_ids,
            ).fetchall()

        # Group rows by match_id
        grouped = {}
        for r in rows:
            mid = r["match_id"]
            if mid not in grouped:
                grouped[mid] = {
                    "match_id": mid,
                    "input_id": r["input_id"],
                    "model_a": r["model_a"],
                    "model_b": r["model_b"],
                    "username": r["username"],
                    "created_at": r["created_at"],
                    "criterion_votes": [],
                }
            grouped[mid]["criterion_votes"].append({
                "criterion": r["criterion"],
                "winner": r["winner"],
                "reasoning": r["reasoning"] or "",
            })

        # Return in the same order as the paginated match_ids
        return [grouped[mid] for mid in match_ids if mid in grouped]

    def _replay_history(self, votes, initial, initial_rd, initial_vol, tau, max_points=200):
        """Replay a list of vote rows into {model: [{match, rating, rd}, ...]}.
        One vote = one period. Skips are recorded but do not move state."""
        state = {}
        match_num = {}
        history = {}

        for v in votes:
            ma, mb, winner = v["model_a"], v["model_b"], v["winner"]
            for m in (ma, mb):
                if m not in state:
                    state[m] = (initial, initial_rd, initial_vol)
                    match_num[m] = 0
                    history[m] = [(0, initial, initial_rd)]

            if winner == "skip":
                continue

            state[ma], state[mb] = _apply_match(state[ma], state[mb], winner, tau)
            match_num[ma] += 1
            match_num[mb] += 1
            history[ma].append((match_num[ma], state[ma][0], state[ma][1]))
            history[mb].append((match_num[mb], state[mb][0], state[mb][1]))

        def _fmt(pts):
            return [{"match": p[0], "rating": round(p[1], 1), "rd": round(p[2], 1)} for p in pts]

        result = {}
        for model, pts in history.items():
            if len(pts) <= max_points:
                result[model] = _fmt(pts)
            else:
                step = len(pts) / max_points
                sampled = [pts[int(i * step)] for i in range(max_points - 1)]
                sampled.append(pts[-1])
                result[model] = _fmt(sampled)
        return result

    def get_elo_history(self, arena_name, criterion=None, max_points=200):
        """Replay votes to produce rating over time per model.

        If *criterion* is given, only replay votes for that criterion. If None, replay
        each criterion independently then aggregate per-model (prob-space rating, mean RD).
        Returns {model: [{match, rating, rd}, ...]} with at most max_points per model.
        """
        manifest = self.get_manifest(arena_name)
        initial, initial_rd, initial_vol, tau = _glicko_cfg(manifest)

        if criterion:
            with self.lock:
                votes = self.conn.execute(
                    "SELECT model_a, model_b, winner FROM arena_votes WHERE arena_name=? AND criterion=? ORDER BY id",
                    (arena_name, criterion),
                ).fetchall()
            if not votes:
                return {}
            result = self._replay_history(votes, initial, initial_rd, initial_vol, tau, max_points)
            log.debug("Rating history for %s (criterion=%s): %d votes, %d models", arena_name, criterion, len(votes), len(result))
            return result

        # Aggregate: replay each criterion independently, then combine
        with self.lock:
            rows = self.conn.execute(
                "SELECT model_a, model_b, winner, criterion FROM arena_votes WHERE arena_name=? ORDER BY id",
                (arena_name,),
            ).fetchall()

        if not rows:
            return {}

        by_criterion = {}
        for r in rows:
            by_criterion.setdefault(r["criterion"], []).append(r)

        per_criterion = {
            crit: self._replay_history(votes, initial, initial_rd, initial_vol, tau, max_points)
            for crit, votes in by_criterion.items()
        }

        # Combine across criteria per model: prob-space rating avg, mean RD, aligned by match index
        all_models = set()
        for h in per_criterion.values():
            all_models.update(h.keys())

        result = {}
        for model in all_models:
            crit_histories = [h[model] for h in per_criterion.values() if model in h]
            if not crit_histories:
                continue
            longest = max(crit_histories, key=len)
            averaged = []
            for i, pt in enumerate(longest):
                ratings_at_i, rds_at_i = [], []
                for ch in crit_histories:
                    idx = min(i, len(ch) - 1)
                    ratings_at_i.append(ch[idx]["rating"])
                    rds_at_i.append(ch[idx]["rd"])
                averaged.append({
                    "match": pt["match"],
                    "rating": round(_aggregate_ratings_prob_space(ratings_at_i, initial), 1),
                    "rd": round(sum(rds_at_i) / len(rds_at_i), 1),
                })
            result[model] = averaged

        log.debug("Rating history for %s (aggregate): %d criteria, %d models", arena_name, len(by_criterion), len(result))
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
            handler.send_header("Cache-Control", "public, max-age=86400, immutable")
            handler.end_headers()
            shutil.copyfileobj(resp["Body"], handler.wfile)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            log.warning("serve_asset %s: %s", key, e)
            handler.send_error(404, "Asset not found")
