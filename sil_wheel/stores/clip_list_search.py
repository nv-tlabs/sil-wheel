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
import os
import re
import tempfile
from pathlib import Path

import orjson

from sil_wheel.stores.search_utils import project_dict
from sil_wheel.stores.time_utils import Timer
from sil_wheel.stores.utils import LRUDict


_HASH_LEN = 16
_HASH_RE = re.compile(r"^[0-9a-f]{16}$")


def _hash(clip_ids):
    canonical = orjson.dumps(sorted(set(clip_ids)))
    return hashlib.blake2b(canonical, digest_size=_HASH_LEN // 2).hexdigest()


class ClipListSearch:
    """Server-side store of content-addressed clip_id lists.

    Each list is uploaded once and identified by a 16-hex blake2b hash of
    the sorted+deduped contents. The hash flows through the URL as
    ``clip_id_list_hash``; ``search`` intersects ``current_results`` with
    the stored list. Storage is one ``<hash>.json`` file per list — no
    metadata, no owner. Content addressing makes uploads idempotent.
    """

    def __init__(self, clip_lists_dir):
        self.clip_lists_dir = Path(clip_lists_dir)
        self.clip_lists_dir.mkdir(parents=True, exist_ok=True)
        # hash -> frozenset[str]; LRU so repeated searches don't re-read.
        self._cache = LRUDict(size=32)
        self.timers = Timer()

    def _path(self, hash):
        if not _HASH_RE.match(hash):
            raise ValueError(f"Invalid clip-list hash: {hash!r}")
        return self.clip_lists_dir / f"{hash}.json"

    def save(self, clip_ids):
        """Store ``clip_ids`` under their content hash.

        Returns ``(hash, created)`` where ``created`` is False if the same
        content was already on disk.
        """
        clip_ids = sorted(set(clip_ids))
        if not clip_ids:
            raise ValueError("Cannot save empty clip-id list")
        h = _hash(clip_ids)
        path = self._path(h)
        if path.exists():
            return h, False
        payload = orjson.dumps(clip_ids)
        # Atomic install via temp file + rename. Two concurrent uploads of
        # the same content race harmlessly: whichever rename loses leaves
        # the same bytes on disk.
        fd, tmp = tempfile.mkstemp(
            prefix=f".tmp_{h}_", suffix=".json", dir=self.clip_lists_dir,
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
            os.replace(tmp, path)
        except Exception:
            if Path(tmp).exists():
                os.unlink(tmp)
            raise
        # Pre-warm the cache so the next search hits in-memory.
        self._cache[h] = frozenset(clip_ids)
        return h, True

    def _load_set(self, hash):
        if hash in self._cache:
            return self._cache[hash]
        path = self._path(hash)
        if not path.exists():
            return None
        with open(path, "rb") as f:
            ids = frozenset(orjson.loads(f.read()))
        self._cache[hash] = ids
        return ids

    def load(self, hash):
        """Return the stored list (sorted) or ``None`` if the hash is unknown."""
        ids = self._load_set(hash)
        return sorted(ids) if ids is not None else None

    def search(self, filters, current_results):
        if not filters.clip_id_list_hash:
            return current_results
        try:
            ids = self._load_set(filters.clip_id_list_hash)
        except ValueError:
            # Malformed hash — surface as no-match rather than 500.
            return {}
        if ids is None:
            # Unknown hash: explicit empty (filter is active but resolves
            # to no clips).
            return {}
        return project_dict(current_results, ids)
