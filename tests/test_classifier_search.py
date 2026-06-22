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

"""Tests for ClassifierSearch run-id-keyed filtering."""
import json
import time

import pytest

from sil_wheel.stores.classifier_search import (
    ClassifierSearch,
    _compile_expression,
)

RUN_ID = "run-aaa"
RUN_ID_MULTI = "run-bbb"

# 10 clip IDs with scores from 0.1 to 1.0
_SCORES = {f"clip-{i:02d}": round(0.1 * (i + 1), 1) for i in range(10)}


def _write_run(root, run_id, embed_type="cosmos", positive_labels=None,
               negative_labels=None, scores=None):
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "predicted_scores.json", "w") as fp:
        json.dump(scores if scores is not None else _SCORES, fp)
    metadata = {
        "run_id": run_id,
        "embed_type": embed_type,
        "positive_labels": positive_labels or ["label_A"],
        "negative_labels": negative_labels or [],
        "trained_by": "alice",
        "started_at": 0,
        "status": "done",
        "n_positive_clips": 0,
        "n_negative_clips": 0,
    }
    with open(run_dir / "metadata.json", "w") as fp:
        json.dump(metadata, fp)
    return run_dir


@pytest.fixture()
def classifier_store(tmp_path):
    _write_run(tmp_path, RUN_ID)
    return ClassifierSearch(str(tmp_path))


def _all_ids():
    return list(_SCORES.keys())


class TestFilterClips:
    def test_filter_clips_above_threshold(self, classifier_store):
        result = classifier_store.filter_clips(_all_ids(), RUN_ID, "p > 0.5")
        returned_clips = {c for c, _ in result}
        expected = {f"clip-{i:02d}" for i in range(5, 10)}
        assert returned_clips == expected

    def test_filter_clips_threshold_zero(self, classifier_store):
        result = classifier_store.filter_clips(_all_ids(), RUN_ID, "p > 0.0")
        assert len(result) == 10

    def test_filter_clips_threshold_one(self, classifier_store):
        result = classifier_store.filter_clips(_all_ids(), RUN_ID, "p > 1.0")
        assert len(result) == 0

    def test_scores_attached_to_results(self, classifier_store):
        result = classifier_store.filter_clips(_all_ids(), RUN_ID, "p > 0.5")
        for clip_id, score in result:
            assert score == pytest.approx(_SCORES[clip_id], abs=1e-5)
            assert score > 0.5

    def test_missing_run_returns_empty(self, classifier_store):
        result = classifier_store.filter_clips(_all_ids(), "nonexistent_run", "p > 0.0")
        assert result == []


class TestCacheInvalidation:
    def test_cache_invalidation(self, tmp_path, classifier_store):
        result_before = classifier_store.filter_clips(_all_ids(), RUN_ID, "p > 0.5")
        assert len(result_before) == 5

        new_scores = {"clip-08": 0.9, "clip-09": 1.0, "clip-00": 0.1}
        scores_file = tmp_path / RUN_ID / "predicted_scores.json"
        time.sleep(0.01)
        with open(scores_file, "w") as fp:
            json.dump(new_scores, fp)

        classifier_store.invalidate(RUN_ID)

        new_ids = list(new_scores.keys())
        result_after = classifier_store.filter_clips(new_ids, RUN_ID, "p > 0.5")
        returned = {c for c, _ in result_after}
        assert "clip-08" in returned
        assert len(result_after) >= 1

    def test_filter_cache_hit_returns_same_results(self, classifier_store):
        first = classifier_store.filter_clips(_all_ids(), RUN_ID, "p > 0.5")
        second = classifier_store.filter_clips(_all_ids(), RUN_ID, "p > 0.5")
        assert set(first) == set(second)

    def test_filter_cache_hit_is_faster(self, classifier_store):
        t0 = time.perf_counter()
        classifier_store.filter_clips(_all_ids(), RUN_ID, "p > 0.5")
        cold = time.perf_counter() - t0
        t0 = time.perf_counter()
        classifier_store.filter_clips(_all_ids(), RUN_ID, "p > 0.5")
        warm = time.perf_counter() - t0
        assert warm < cold

    def test_filter_cache_invalidated_with_scores(self, classifier_store):
        classifier_store.filter_clips(_all_ids(), RUN_ID, "p > 0.5")
        assert len(classifier_store._filter_cache) == 1
        classifier_store.invalidate(RUN_ID)
        assert len(classifier_store._filter_cache) == 0

    def test_filter_cache_respects_ids_subset(self, classifier_store):
        classifier_store.filter_clips(_all_ids(), RUN_ID, "p > 0.5")
        subset = [f"clip-{i:02d}" for i in range(7, 10)]
        result = classifier_store.filter_clips(subset, RUN_ID, "p > 0.5")
        returned = {c for c, _ in result}
        assert returned == {"clip-07", "clip-08", "clip-09"}


class TestExpression:
    def test_lt_expression(self, classifier_store):
        result = classifier_store.filter_clips(_all_ids(), RUN_ID, "p < 0.5")
        returned_clips = {c for c, _ in result}
        expected = {f"clip-{i:02d}" for i in range(4)}
        assert returned_clips == expected

    def test_lt_expression_threshold_zero(self, classifier_store):
        result = classifier_store.filter_clips(_all_ids(), RUN_ID, "p < 0.0")
        assert len(result) == 0

    def test_lt_expression_threshold_one(self, classifier_store):
        result = classifier_store.filter_clips(_all_ids(), RUN_ID, "p < 1.0")
        returned_clips = {c for c, _ in result}
        assert "clip-09" not in returned_clips
        assert len(result) == 9

    def test_gt_and_lt_are_disjoint(self, classifier_store):
        gt = {c for c, _ in classifier_store.filter_clips(_all_ids(), RUN_ID, "p > 0.5")}
        lt = {c for c, _ in classifier_store.filter_clips(_all_ids(), RUN_ID, "p < 0.5")}
        assert gt.isdisjoint(lt)

    def test_range_expression(self, classifier_store):
        result = classifier_store.filter_clips(_all_ids(), RUN_ID, "0.4 < p < 0.8")
        returned_clips = {c for c, _ in result}
        expected = {f"clip-{i:02d}" for i in range(4, 7)}
        assert returned_clips == expected

    def test_range_expression_inclusive_bounds(self, classifier_store):
        result = classifier_store.filter_clips(_all_ids(), RUN_ID, "0.5 <= p <= 0.7")
        returned_clips = {c for c, _ in result}
        expected = {f"clip-{i:02d}" for i in range(4, 7)}
        assert returned_clips == expected

    def test_missing_clip_excluded(self, classifier_store):
        ids = _all_ids() + ["unknown-clip-xyz"]
        for expr in ("p > 0.5", "p < 0.5", "0.3 < p < 0.8"):
            result = classifier_store.filter_clips(ids, RUN_ID, expr)
            returned = {c for c, _ in result}
            assert "unknown-clip-xyz" not in returned


class TestCompoundExpression:
    def test_or_expression(self, classifier_store):
        result = classifier_store.filter_clips(_all_ids(), RUN_ID, "p > 0.8 or p < 0.2")
        returned = {c for c, _ in result}
        assert returned == {"clip-00", "clip-08", "clip-09"}

    def test_and_expression(self, classifier_store):
        result = classifier_store.filter_clips(_all_ids(), RUN_ID, "p > 0.3 and p < 0.7")
        returned = {c for c, _ in result}
        assert returned == {"clip-03", "clip-04", "clip-05"}

    def test_or_with_nested_range(self, classifier_store):
        result = classifier_store.filter_clips(
            _all_ids(), RUN_ID, "p > 0.85 or 0.45 < p < 0.65"
        )
        returned = {c for c, _ in result}
        assert returned == {"clip-04", "clip-05", "clip-08", "clip-09"}

    def test_not_expression(self, classifier_store):
        result = classifier_store.filter_clips(_all_ids(), RUN_ID, "not (p > 0.5)")
        returned = {c for c, _ in result}
        assert returned == {f"clip-{i:02d}" for i in range(5)}


class TestExpressionSafety:
    @pytest.mark.parametrize("expr", [
        "__import__('os')",
        "p.tofile('/tmp/x')",
        "np.where(p > 0.5)",
        "len(p) > 0",
        "9 ** 9 ** 9",
        "p + 1 > 0.5",
        "().__class__",
    ])
    def test_unsafe_expressions_rejected(self, expr):
        with pytest.raises(ValueError):
            _compile_expression(expr)

    def test_unsafe_expression_rejected_through_filter(self, classifier_store):
        with pytest.raises(ValueError):
            classifier_store.filter_clips(_all_ids(), RUN_ID, "__import__('os')")

    def test_compiled_predicate_is_cached(self):
        assert _compile_expression("p > 0.5") is _compile_expression("p > 0.5")


class TestMultiLabel:
    def test_multi_label_classifier_loads(self, tmp_path):
        _write_run(
            tmp_path, RUN_ID_MULTI,
            positive_labels=["label_A", "label_B"],
        )
        store = ClassifierSearch(str(tmp_path))
        result = store.filter_clips(_all_ids(), RUN_ID_MULTI, "p > 0.5")
        returned = {c for c, _ in result}
        assert returned == {f"clip-{i:02d}" for i in range(5, 10)}

    def test_runs_with_same_labels_dont_collide(self, tmp_path):
        """Two runs over the same label combination keep separate score caches."""
        _write_run(tmp_path, "run-x", positive_labels=["label_A"])
        _write_run(
            tmp_path, "run-y", positive_labels=["label_A"],
            scores={k: v / 2 for k, v in _SCORES.items()},
        )
        store = ClassifierSearch(str(tmp_path))
        gt_x = {c for c, _ in store.filter_clips(_all_ids(), "run-x", "p > 0.4")}
        gt_y = {c for c, _ in store.filter_clips(_all_ids(), "run-y", "p > 0.4")}
        # run-y's scores are halved → strictly fewer clips above 0.4
        assert len(gt_y) < len(gt_x)
        assert store._score_cache.keys() == {"run-x", "run-y"}


class TestListRuns:
    def test_lists_runs_with_metadata(self, tmp_path):
        _write_run(tmp_path, "run-1", positive_labels=["a"])
        _write_run(tmp_path, "run-2", positive_labels=["b", "c"], embed_type="visual")
        # Old-format dir without metadata.json must stay invisible.
        legacy = tmp_path / "cosmos"
        (legacy / "snow").mkdir(parents=True)
        (legacy / "snow" / "predicted_scores.json").write_text("{}")

        store = ClassifierSearch(str(tmp_path))
        runs = store.list_runs()
        run_ids = {r["run_id"] for r in runs}
        assert run_ids == {"run-1", "run-2"}
        by_id = {r["run_id"]: r for r in runs}
        assert by_id["run-2"]["embed_type"] == "visual"
        assert by_id["run-2"]["positive_labels"] == ["b", "c"]


class TestLatency:
    def test_latency(self, classifier_store):
        t0 = time.perf_counter()
        classifier_store.filter_clips(_all_ids(), RUN_ID, "p > 0.5")
        elapsed = time.perf_counter() - t0
        assert elapsed <= 0.05, f"filter_clips took {elapsed:.4f}s"


class TestThreadSafety:
    def test_concurrent_load_scores(self, tmp_path):
        """Concurrent load_scores calls on a cold cache must not race or corrupt data.

        Regression test for the simdjson shared-parser segfault: when the
        ThreadingMixIn server handles multiple requests simultaneously, each
        thread calling load_scores on the same cold-cache run used to race
        through the module-level simdjson parser. Each thread now gets its own
        Parser instance, so concurrent loads are independent.
        """
        import threading

        # Large enough that the C-level parse overlaps across threads
        n_clips = 5_000
        scores = {f"clip-{i:06d}": round(i / n_clips, 6) for i in range(n_clips)}
        _write_run(tmp_path, RUN_ID, scores=scores)

        store = ClassifierSearch(str(tmp_path))

        n_threads = 16
        barrier = threading.Barrier(n_threads)
        results = [None] * n_threads
        errors = []
        lock = threading.Lock()

        def worker(idx):
            try:
                barrier.wait()
                clip_ids, scores_arr = store.load_scores(RUN_ID)
                results[idx] = (clip_ids, scores_arr)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Threads raised exceptions: {errors}"
        for clip_ids, scores_arr in results:
            assert clip_ids is not None
            assert len(clip_ids) == n_clips
            assert len(scores_arr) == n_clips
