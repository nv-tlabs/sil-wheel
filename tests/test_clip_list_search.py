# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import time
from types import SimpleNamespace

import orjson
import pytest

from sil_wheel.stores.clip_list_search import ClipListSearch


HASH_RE = re.compile(r"^[0-9a-f]{16}$")


@pytest.fixture()
def store(tmp_path):
    return ClipListSearch(tmp_path / "clip_lists")


def _filters(hash):
    return SimpleNamespace(clip_id_list_hash=hash)


def test_save_returns_16_hex_hash_and_writes_file(store):
    hash, created = store.save(["c1", "c2", "c3"])
    assert HASH_RE.match(hash)
    assert created is True
    assert (store.clip_lists_dir / f"{hash}.json").exists()


def test_save_is_order_insensitive(store):
    h1, _ = store.save(["b", "a", "c"])
    h2, _ = store.save(["a", "b", "c"])
    h3, _ = store.save(["c", "a", "b"])
    assert h1 == h2 == h3


def test_save_is_idempotent(store):
    h1, c1 = store.save(["a", "b"])
    mtime1 = (store.clip_lists_dir / f"{h1}.json").stat().st_mtime_ns
    time.sleep(0.01)
    h2, c2 = store.save(["a", "b"])
    mtime2 = (store.clip_lists_dir / f"{h2}.json").stat().st_mtime_ns
    assert (h1, c2) == (h2, False)
    assert c1 is True
    assert mtime1 == mtime2  # not rewritten


def test_save_dedupes_input(store):
    h1, _ = store.save(["a", "b"])
    h2, _ = store.save(["a", "b", "a", "b"])
    assert h1 == h2


def test_save_empty_raises(store):
    with pytest.raises(ValueError):
        store.save([])


def test_load_returns_sorted_list(store):
    hash, _ = store.save(["z", "a", "m"])
    assert store.load(hash) == ["a", "m", "z"]


def test_load_unknown_returns_none(store):
    assert store.load("0000000000000000") is None


def test_load_malformed_raises(store):
    with pytest.raises(ValueError):
        store.load("not-hex!!")


def test_search_noop_when_filter_absent(store):
    results = {"c1": object(), "c2": object()}
    assert store.search(_filters(None), results) is results


def test_search_intersects_when_filter_set(store):
    hash, _ = store.save(["c1", "c3", "c5"])
    results = {"c1": 1, "c2": 2, "c3": 3, "c4": 4}
    assert store.search(_filters(hash), results) == {"c1": 1, "c3": 3}


def test_search_returns_empty_when_hash_unknown(store):
    assert store.search(_filters("0000000000000000"), {"c1": 1}) == {}


def test_search_returns_empty_when_hash_malformed(store):
    assert store.search(_filters("not-hex!!"), {"c1": 1}) == {}


def test_cache_hits_dont_re_read_disk(store):
    hash, _ = store.save(["c1", "c2"])
    # Tamper with the on-disk file. A second search should see the
    # cached set (the pre-warmed save() cache), not the new bytes.
    (store.clip_lists_dir / f"{hash}.json").write_bytes(orjson.dumps(["other"]))
    results = {"c1": 1, "c2": 2, "other": 3}
    assert store.search(_filters(hash), results) == {"c1": 1, "c2": 2}
