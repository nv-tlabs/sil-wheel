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

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from dataclasses import dataclass
from threading import RLock

ITERATIONS = 200_000
SALT_BYTES = 16
SESSION_TTL_SECONDS = 72 * 3600  # 72 hours


@dataclass
class User:
    id: int
    username: str
    email: str | None
    role: str


class UsersDataStore:
    def __init__(self, db_path: str):
        self.lock = RLock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        # Default datasources granted to newly approved users
        self.default_datasources = [
            "AV V1 train",
            "AV V1 validation",
            "AV V2 train",
            "AV V2 validation",
            "AV V2.2. train",
            "AV V2.2. validation",
            "NVIQ",
            "Golden Dataset v2",
            "Golden Dataset v4",
            "HCM Dataset",
            "HCM Dataset v2",
            "MADS",
            "MADS-1M",
            "MultiCountry-800K",
            "Physical AI",
            "Physical AI private",
            "OpenDV-YouTube",
            "Waymo train",
            "Waymo test",
            "Waymo validation",
            "amo_cle_test_data_set",
            "celsius1_protoplus_1k",
            "celsius2_l3_55k",
            "celsius2_l3_wf10k_720p",
            "celsius2_sauron_15k_osm",
            "celsius2_sauron_nohighway_7k",
            "ncore-lidar-model-static-full",
            "vipe-dynpose100kpp",
        ]

    def _create_tables(self):
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT,
                    pass_hash BLOB NOT NULL,
                    pass_salt BLOB NOT NULL,
                    role TEXT DEFAULT 'user',
                    created_at INTEGER
                )
                """
            )
            # Per-user datasource permissions (optional). Empty set means no
            # access restrictions if role=admin, and "no access" for non-admin
            # users until configured by admin.
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_datasources (
                    user_id INTEGER,
                    data_source TEXT,
                    PRIMARY KEY (user_id, data_source),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS access_requests (
                    id INTEGER PRIMARY KEY,
                    username TEXT,
                    email TEXT,
                    pass_hash BLOB,
                    pass_salt BLOB,
                    reason TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at INTEGER
                )
                """
            )
            # Backfill/migrate: ensure pass_hash and pass_salt exist
            try:
                cols = [
                    r[1]
                    for r in self.conn.execute(
                        "PRAGMA table_info(access_requests)"
                    ).fetchall()
                ]
                if "pass_hash" not in cols:
                    self.conn.execute(
                        "ALTER TABLE access_requests ADD COLUMN pass_hash BLOB"
                    )
                if "pass_salt" not in cols:
                    self.conn.execute(
                        "ALTER TABLE access_requests ADD COLUMN pass_salt BLOB"
                    )
            except Exception:
                pass
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    created_at INTEGER,
                    expires_at INTEGER,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
                """
            )

    @staticmethod
    def _hash_password(password: str, salt: bytes) -> bytes:
        return hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, ITERATIONS
        )

    def create_user(
        self,
        username: str,
        password: str,
        email: str | None = None,
        role: str = "user",
    ) -> int:
        # Require a non-empty password
        if not isinstance(password, str) or not password.strip():
            raise ValueError("Password must be provided and non-empty")

        salt = os.urandom(SALT_BYTES)
        phash = self._hash_password(password, salt)
        ts = int(time.time())
        with self.lock, self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO users (username, email, pass_hash, pass_salt, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (username, email, phash, salt, role, ts),
            )
            return cur.lastrowid

    def verify_credentials(self, username: str, password: str) -> User | None:
        with self.lock:
            row = self.conn.execute(
                "SELECT * FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        if not row:
            return None
        salt = row["pass_salt"]
        expected = row["pass_hash"]
        actual = self._hash_password(password, salt)
        if not hmac.compare_digest(actual, expected):
            return None
        return User(
            id=row["id"],
            username=row["username"],
            email=row["email"],
            role=row["role"],
        )

    def create_access_request(
        self, username: str, email: str, reason: str, password: str
    ) -> int:
        ts = int(time.time())
        # Enforce NVIDIA email domain
        if not email or not re.match(
            r"^[^@\s]+@nvidia\.com$", email, re.IGNORECASE
        ):
            raise ValueError("Only @nvidia.com emails are accepted")
        if not isinstance(password, str) or not password.strip():
            raise ValueError("Password must be provided and non-empty")
        salt = os.urandom(SALT_BYTES)
        phash = self._hash_password(password, salt)
        with self.lock, self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO access_requests (username, email, reason, pass_hash, pass_salt, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (username, email, reason, phash, salt, ts),
            )
            return cur.lastrowid

    def approve_user(self, username: str, password: str | None = None) -> bool:
        # Optionally set password when approving (if not created yet)
        with self.lock, self.conn:
            row = self.conn.execute(
                "SELECT id FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            if row is None:
                if password is None:
                    return False
                # Create then approve
                self.create_user(username, password)
                return True
            return True

    def list_users(self) -> list[dict]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, username, email, role, created_at FROM users ORDER BY username ASC"
            ).fetchall()
            # Map user_id -> list of datasources
            ds_map: dict[int, list[str]] = {}
            for r in self.conn.execute(
                "SELECT user_id, data_source FROM user_datasources ORDER BY data_source ASC"
            ).fetchall():
                ds_map.setdefault(r["user_id"], []).append(r["data_source"])
        return [
            {
                "id": r["id"],
                "username": r["username"],
                "email": r["email"],
                "role": r["role"],
                "created_at": r["created_at"],
                "datasources": ds_map.get(r["id"], []),
            }
            for r in rows
        ]

    def set_user_role(self, username: str, role: str) -> bool:
        with self.lock, self.conn:
            cur = self.conn.execute(
                "UPDATE users SET role = ? WHERE username = ?",
                (role, username),
            )
            return cur.rowcount > 0

    def list_access_requests(self, status: str | None = None) -> list[dict]:
        with self.lock:
            if status is None:
                rows = self.conn.execute(
                    "SELECT id, username, email, reason, status, created_at FROM access_requests ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT id, username, email, reason, status, created_at FROM access_requests WHERE status = ? ORDER BY created_at DESC",
                    (status,),
                ).fetchall()
        return [
            {
                "id": r["id"],
                "username": r["username"],
                "email": r["email"],
                "reason": r["reason"],
                "status": r["status"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def set_access_request_status(self, request_id: int, status: str) -> bool:
        with self.lock, self.conn:
            cur = self.conn.execute(
                "UPDATE access_requests SET status = ? WHERE id = ?",
                (status, int(request_id)),
            )
            return cur.rowcount > 0

    def approve_request(self, request_id: int) -> tuple[bool, str | None]:
        """Approve access request and create/activate user.

        Returns (ok, generated_password or None)
        """
        with self.lock:
            r = self.conn.execute(
                "SELECT username, email, pass_hash, pass_salt FROM access_requests WHERE id = ?",
                (int(request_id),),
            ).fetchone()
        if not r:
            return False, None
        username = r["username"]
        email = r["email"]
        req_hash = r["pass_hash"]
        req_salt = r["pass_salt"]

        gen_pass = None
        with self.lock, self.conn:
            u = self.conn.execute(
                "SELECT id FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            if u is None:
                if req_hash is None or req_salt is None:
                    # Fallback: legacy request missing password, generate one
                    gen_pass = secrets.token_urlsafe(16)
                    self.create_user(
                        username, gen_pass, email=email, role="user"
                    )
                else:
                    # Insert user with requested password hash
                    ts = int(time.time())
                    self.conn.execute(
                        """
                        INSERT INTO users (username, email, pass_hash, pass_salt, role, created_at)
                        VALUES (?, ?, ?, ?, 'user', ?)
                        """,
                        (username, email, req_hash, req_salt, ts),
                    )
            else:
                # Ensure email is filled in if missing
                self.conn.execute(
                    "UPDATE users SET email = COALESCE(email, ?) WHERE username = ?",
                    (email, username),
                )
            # Grant default datasources to the approved user (idempotent)
            u2 = self.conn.execute(
                "SELECT id FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            if u2 is not None and self.default_datasources:
                uid = u2["id"]
                self.conn.executemany(
                    "INSERT OR IGNORE INTO user_datasources (user_id, data_source) VALUES (?, ?)",
                    [(uid, ds) for ds in self.default_datasources],
                )
            # Remove request after approval to keep queue clean
            self.conn.execute(
                "DELETE FROM access_requests WHERE id = ?", (int(request_id),)
            )
        return True, gen_pass

    def reject_request(self, request_id: int) -> bool:
        return self.set_access_request_status(request_id, "rejected")

    def list_admin_emails(self) -> list[str]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT email FROM users WHERE role='admin' AND email IS NOT NULL"
            ).fetchall()
        return [r["email"] for r in rows if r["email"]]

    def delete_user(self, username: str) -> bool:
        with self.lock, self.conn:
            cur = self.conn.execute(
                "DELETE FROM users WHERE username = ?", (username,)
            )
            return cur.rowcount > 0

    def update_user(
        self,
        username: str,
        email: str | None = None,
        role: str | None = None,
        password: str | None = None,
        datasources: list[str] | None = None,
    ) -> bool:
        """Update user fields. Any None parameter is left unchanged.
        Returns True if a user row was updated.
        """
        sets = []
        params: list = []
        if email is not None:
            sets.append("email = ?")
            params.append(email)
        if role is not None:
            sets.append("role = ?")
            params.append(role)
        if password is not None:
            salt = os.urandom(SALT_BYTES)
            phash = self._hash_password(password, salt)
            sets.append("pass_hash = ?")
            params.append(phash)
            sets.append("pass_salt = ?")
            params.append(salt)
        if not sets:
            # No change to core fields; still allow updating datasources below
            updated = False
        else:
            params.append(username)
        with self.lock, self.conn:
            updated = False
            if sets:
                cur = self.conn.execute(
                    f"UPDATE users SET {', '.join(sets)} WHERE username = ?",
                    params,
                )
                updated = cur.rowcount > 0
            if datasources is not None:
                # Replace allowed datasources set for this user
                urow = self.conn.execute(
                    "SELECT id FROM users WHERE username = ?",
                    (username,),
                ).fetchone()
                if urow is not None:
                    uid = urow["id"]
                    self.conn.execute(
                        "DELETE FROM user_datasources WHERE user_id = ?",
                        (uid,),
                    )
                    if datasources:
                        self.conn.executemany(
                            "INSERT OR IGNORE INTO user_datasources (user_id, data_source) VALUES (?, ?)",
                            [
                                (uid, ds.strip())
                                for ds in datasources
                                if ds and ds.strip()
                            ],
                        )
                    updated = True or updated
            return updated

    def get_allowed_datasources(self, user_id: int) -> list[str]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT data_source FROM user_datasources WHERE user_id = ? ORDER BY data_source ASC",
                (int(user_id),),
            ).fetchall()
        return [r["data_source"] for r in rows]

    def create_session(
        self, user_id: int, ttl_seconds: int = SESSION_TTL_SECONDS
    ) -> str:
        sid = secrets.token_urlsafe(32)
        now = int(time.time())
        exp = now + ttl_seconds
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO sessions (session_id, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (sid, user_id, now, exp),
            )
        return sid

    def delete_session(self, session_id: str) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )

    def get_user_by_session(self, session_id: str) -> User | None:
        if not session_id:
            return None
        now = int(time.time())
        with self.lock:
            row = self.conn.execute(
                """
                SELECT u.* FROM sessions s JOIN users u ON s.user_id = u.id
                WHERE s.session_id = ? AND s.expires_at > ?
                """,
                (session_id, now),
            ).fetchone()
        if not row:
            return None
        return User(
            id=row["id"],
            username=row["username"],
            email=row["email"],
            role=row["role"],
        )

    def grant_datasource_to_all_users(self, data_source: str):
        with self.lock, self.conn:
            # Fetch all existing user IDs
            user_ids = [
                row["id"]
                for row in self.conn.execute("SELECT id FROM users").fetchall()
            ]

            # 2. Bulk insert into user_datasources in case the user doesn't
            # already have access to this data source
            cur = self.conn.executemany(
                "INSERT OR IGNORE INTO user_datasources (user_id, data_source) VALUES (?, ?)",
                [(uid, data_source.strip()) for uid in user_ids],
            )

            return cur.rowcount
