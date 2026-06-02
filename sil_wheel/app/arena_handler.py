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

"""Mixin that adds all arena HTTP endpoints to RequestHandler."""

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger("arena")


class ArenaHandlerMixin:
    def _check_arena_access(self, arena_name):
        """Returns (user, allowed). Returns (None, False) if no user or access denied."""
        user = self._current_user()
        if not user:
            return None, False
        allowed = self.arena_store.can_access(arena_name, user.username, user.role)
        return user, allowed

    def _require_arena_user(self, parts, require_owner=False):
        """Auth helper for arena POST actions. Returns (user, arena_name) or sends error and returns (None, None)."""
        user = self._current_user()
        if not user:
            self._send_json({"error": "unauthorized"}, status=401)
            return None, None
        arena_name = parts[1] if len(parts) > 1 else None
        if not arena_name:
            self._send_json({"error": "missing arena_name"}, status=400)
            return None, None
        if require_owner:
            if not self.arena_store.is_owner_or_admin(arena_name, user.username, user.role):
                self._send_json({"error": "not found"}, status=404)
                return None, None
        else:
            if not self.arena_store.can_access(arena_name, user.username, user.role):
                self._send_json({"error": "not found"}, status=404)
                return None, None
        return user, arena_name

    def handle_arena_get(self, parsed_path, parsed_qs):
        """Handle all /arena/* GET requests. Returns True if the request was handled."""
        path = parsed_path.path

        if path == "/arena/list":
            user = self._current_user()
            if not user:
                return self._send_json({"error": "unauthorized"}, status=401)
            return self._send_json({"arenas": self.arena_store.list_arenas(user.id, user.username, user.role)})

        if path == "/arena/manifest":
            name = parsed_qs.get("name", [None])[0]
            if not name:
                return self._send_json({"error": "missing name"}, status=400)
            user, allowed = self._check_arena_access(name)
            if not allowed:
                return self._send_json({"error": "not found"}, status=404)
            manifest = self.arena_store.get_manifest(name)
            if not manifest:
                return self._send_json({"error": "not found"}, status=404)
            manifest["vlm_judge_available"] = getattr(self, "vlm_judge", None) is not None
            return self._send_json(manifest)

        if path == "/arena/leaderboard":
            name = parsed_qs.get("name", [None])[0]
            if not name:
                return self._send_json({"error": "missing name"}, status=400)
            user, allowed = self._check_arena_access(name)
            if not allowed:
                return self._send_json({"error": "not found"}, status=404)
            return self._send_json(self.arena_store.get_leaderboard(name))

        if path == "/arena/history":
            name = parsed_qs.get("name", [None])[0]
            if not name:
                return self._send_json({"error": "missing name"}, status=400)
            user, allowed = self._check_arena_access(name)
            if not allowed:
                return self._send_json({"error": "not found"}, status=404)
            limit = int(parsed_qs.get("limit", [50])[0])
            offset = int(parsed_qs.get("offset", [0])[0])
            return self._send_json(self.arena_store.get_history(name, limit, offset))

        if path == "/arena/export":
            name = parsed_qs.get("name", [None])[0]
            if not name:
                return self._send_json({"error": "missing name"}, status=400)
            user, allowed = self._check_arena_access(name)
            if not allowed:
                return self._send_json({"error": "not found"}, status=404)
            csv_data = self.arena_store.export_votes_csv(name)
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Disposition", f'attachment; filename="{name}_votes.csv"')
            self.end_headers()
            self.wfile.write(csv_data.encode("utf-8"))
            return True

        if path == "/arena/elo_confidence":
            name = parsed_qs.get("name", [None])[0]
            if not name:
                return self._send_json({"error": "missing name"}, status=400)
            user, allowed = self._check_arena_access(name)
            if not allowed:
                return self._send_json({"error": "not found"}, status=404)
            return self._send_json(self.arena_store.get_elo_confidence(name))

        if path == "/arena/elo_history":
            name = parsed_qs.get("name", [None])[0]
            if not name:
                return self._send_json({"error": "missing name"}, status=400)
            user, allowed = self._check_arena_access(name)
            if not allowed:
                return self._send_json({"error": "not found"}, status=404)
            return self._send_json(self.arena_store.get_elo_history(name))

        if path.startswith("/arena/asset/"):
            # /arena/asset/{arena_name}/{item_id}/{filename}
            path_parts = path.split("/")
            if len(path_parts) >= 6:
                arena_name = path_parts[3]
                user, allowed = self._check_arena_access(arena_name)
                if not allowed:
                    self.send_error(404)
                    return True
                item_id = path_parts[4]
                filename = "/".join(path_parts[5:])
                self.arena_store.serve_asset(self, arena_name, item_id, filename)
                return True
            self.send_error(404)
            return True

        self.send_error(404)

    def handle_arena_post(self, action, parts):
        """Handle all arena_* POST actions. Returns True if the action was handled."""
        if action == "arena_next_match":
            user, arena_name = self._require_arena_user(parts)
            if not user:
                return True
            match = self.arena_store.get_next_match(arena_name, user.id)
            if not match:
                return self._send_json({"error": "no match available"}, status=404)
            return self._send_json(match)

        if action == "arena_review_match":
            user, arena_name = self._require_arena_user(parts)
            if not user:
                return True
            match_id = parts[2] if len(parts) > 2 else None
            if not match_id:
                return self._send_json({"error": "missing match_id"}, status=400)
            match = self.arena_store.get_match(arena_name, match_id)
            if not match:
                return self._send_json({"error": "not found"}, status=404)
            return self._send_json(match)

        if action == "arena_submit_vote":
            user = self._current_user()
            if not user:
                return self._send_json({"error": "unauthorized"}, status=401)
            try:
                arena_name, match_id, item_id, model_a, model_b, winner = (
                    parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]
                )
            except (IndexError, ValueError):
                return self._send_json({"error": "invalid payload"}, status=400)
            reasoning = parts[7] if len(parts) > 7 else None
            if reasoning is not None:
                reasoning = reasoning.strip() or None
            if not self.arena_store.can_access(arena_name, user.username, user.role):
                return self._send_json({"error": "not found"}, status=404)
            if winner not in ("a", "b", "tie", "a_strong", "b_strong", "skip", "both_bad"):
                return self._send_json({"error": "invalid winner"}, status=400)
            result = self.arena_store.submit_vote(
                arena_name, match_id, item_id, model_a, model_b, winner, user.id, user.username, reasoning=reasoning
            )
            return self._send_json(result)

        if action == "arena_publish":
            user, arena_name = self._require_arena_user(parts, require_owner=True)
            if not user:
                return True
            self.arena_store.publish_arena(arena_name)
            return self._send_json({"ok": True})

        if action == "arena_unpublish":
            user, arena_name = self._require_arena_user(parts, require_owner=True)
            if not user:
                return True
            self.arena_store.unpublish_arena(arena_name)
            return self._send_json({"ok": True})

        if action == "arena_refresh_manifest":
            user, arena_name = self._require_arena_user(parts, require_owner=True)
            if not user:
                return True
            ok = self.arena_store.refresh_manifest(arena_name)
            if not ok:
                return self._send_json({"error": "failed to refresh"}, status=500)
            return self._send_json({"ok": True})

        if action == "arena_vlm_judge_batch":
            user, arena_name = self._require_arena_user(parts, require_owner=True)
            if not user:
                return True
            if not getattr(self, "vlm_judge", None):
                return self._send_json({"error": "VLM Judge not available"}, status=503)

            # Parse num_matches: default 10, max 100
            try:
                num_matches = int(parts[2]) if len(parts) > 2 else 10
            except (ValueError, IndexError):
                num_matches = 10
            num_matches = max(1, min(num_matches, 100))

            manifest = self.arena_store.get_manifest(arena_name)
            if not manifest or len(manifest.get("models", [])) < 2 or not manifest.get("items"):
                return self._send_json({"error": "arena has insufficient models or items"}, status=400)

            models = manifest["models"]
            items = manifest["items"]
            vlm_model_short = self.vlm_judge.vlm.model.split("/")[-1]
            vlm_username = f"vlm_judge:{vlm_model_short}"
            vlm_user_id = -1
            workers = getattr(self, "vlm_judge_workers", 16)

            # Sample matches using the same smart weighting as human matches
            matches = []
            for _ in range(num_matches):
                m1, m2, item_id = self.arena_store._sample_match(arena_name, models, items)
                matches.append((item_id, m1, m2))

            # Capture references for the background thread
            vlm_judge = self.vlm_judge
            arena_store = self.arena_store

            def run_batch():
                def do_one(triple):
                    item_id, m1, m2 = triple
                    match_id = str(uuid.uuid4())
                    try:
                        result = vlm_judge.judge_arena_match(
                            arena_store, arena_name, manifest, item_id, m1, m2
                        )
                        vote = result.get("vote", "tie")
                        reasoning = result.get("reasoning", "")
                        arena_store.submit_vote(
                            arena_name, match_id, item_id, m1, m2,
                            vote, vlm_user_id, vlm_username, reasoning=reasoning,
                        )
                    except Exception as exc:
                        log.warning("VLM judge failed for %s/%s vs %s: %s", item_id, m1, m2, exc)

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    list(executor.map(do_one, matches))
                log.info("VLM judge batch for %s complete: %d matches", arena_name, len(matches))

            thread = threading.Thread(target=run_batch, daemon=True)
            thread.start()

            return self._send_json({
                "started": True,
                "num_matches": num_matches,
            })

        self._send_json({"error": "unknown action"}, status=400)
