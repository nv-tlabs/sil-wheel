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

"""Tests for TrajectoryStore expression-based search.

The TrajectoryStore constructor is bypassed via object.__new__ to avoid
loading files from disk. Only _inner_search_trajectory and search_trajectory
are tested here.
"""
import time
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest

from sil_wheel.stores.time_utils import Timer
from sil_wheel.stores.trajectory_store import (
    TRAJECTORY_EXPRESSIONS,
    TrajectoryStore,
    _compile_trajectory_predicate,
)
from sil_wheel.stores.utils import LRUDict

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

T = 300  # time steps per synthetic clip

COLS = 7  # [x, y, z, speed, acceleration, jerk, curvature]


def _make_clip(
    speed=None,
    acceleration=None,
    jerk=None,
    curvature=None,
) -> np.ndarray:
    """Create a (T, 7) trajectory array with the given column vectors.

    Any column not provided defaults to zeros.
    """
    arr = np.zeros((T, COLS), dtype=np.float32)
    if speed is not None:
        arr[:, 3] = speed
    if acceleration is not None:
        arr[:, 4] = acceleration
    if jerk is not None:
        arr[:, 5] = jerk
    if curvature is not None:
        arr[:, 6] = curvature
    return arr


def _make_store(trajectory_data: dict) -> TrajectoryStore:
    """Build a TrajectoryStore with injected trajectory_data (no file I/O)."""
    store = object.__new__(TrajectoryStore)
    store.lock = Lock()
    store.trajectory_data = trajectory_data
    store.searches = LRUDict(size=10)
    store.searches_shapes = LRUDict(size=10)
    store._search_predicates = {}
    store.timers = Timer()
    store.clip_to_idx = {}
    store.features_indexes = {}
    return store


# -------------------------------------------------------------------------
# Synthetic dataset
# -------------------------------------------------------------------------

# high_curvature: curvature > 0.15 for > 10 frames
_HIGH_CURV = np.full(T, 0.02, dtype=np.float32)
_HIGH_CURV[:50] = 0.2   # 50 frames > 0.15

# stop_go: speed < 0.5 then speed > 3.0, with stop before go
_STOP_GO_SPEED = np.zeros(T, dtype=np.float32)
_STOP_GO_SPEED[:50] = 0.2   # stopped (< 0.5)
_STOP_GO_SPEED[100:] = 5.0  # moving (> 3.0)

# hard_braking: acceleration < -3.0 for > 10 frames
_HARD_BRAKING_ACCEL = np.full(T, 0.0, dtype=np.float32)
_HARD_BRAKING_ACCEL[:50] = -4.0

# prolonged_stop: speed < 0.5 for > 150 frames
_PROLONGED_STOP_SPEED = np.full(T, 0.3, dtype=np.float32)

# neutral_speed: not stopped (>= 0.5 m/s) and not fast enough for moving_ego
# (< 5 kph = 1.39 m/s), so it doesn't trigger any trajectory expression.
_NEUTRAL_SPEED = np.full(T, 0.8, dtype=np.float32)

# moving_ego: speed_kph > 5 for > 10 frames (speed in m/s, 5 kph ≈ 1.39 m/s)
# Use neutral speed as the resting baseline so idle frames don't satisfy
# speed < 0.5 (which would falsely trigger prolonged_stop).
_MOVING_SPEED = np.full(T, 0.8, dtype=np.float32)
_MOVING_SPEED[:100] = 2.0   # 7.2 kph → > 5 kph

_CLIPS = {
    # high_curvature × 3 — speed is neutral (not stopped, not fast)
    "clip-hc-01": _make_clip(curvature=_HIGH_CURV, speed=_NEUTRAL_SPEED),
    "clip-hc-02": _make_clip(curvature=_HIGH_CURV, speed=_NEUTRAL_SPEED),
    "clip-hc-03": _make_clip(curvature=_HIGH_CURV, speed=_NEUTRAL_SPEED),
    # stop_go × 2
    "clip-sg-01": _make_clip(speed=_STOP_GO_SPEED),
    "clip-sg-02": _make_clip(speed=_STOP_GO_SPEED),
    # hard_braking × 1 — speed is neutral so it doesn't falsely match prolonged_stop
    "clip-hb-01": _make_clip(acceleration=_HARD_BRAKING_ACCEL, speed=_NEUTRAL_SPEED),
    # prolonged_stop × 2 (300 frames of speed < 0.5)
    "clip-ps-01": _make_clip(speed=_PROLONGED_STOP_SPEED),
    "clip-ps-02": _make_clip(speed=_PROLONGED_STOP_SPEED),
    # moving_ego × 3
    "clip-me-01": _make_clip(speed=_MOVING_SPEED),
    "clip-me-02": _make_clip(speed=_MOVING_SPEED),
    "clip-me-03": _make_clip(speed=_MOVING_SPEED),
    # neutral clips (none of the above) — speed=0.8 avoids all expressions
    "clip-nn-01": _make_clip(speed=_NEUTRAL_SPEED),
    "clip-nn-02": _make_clip(speed=_NEUTRAL_SPEED),
    "clip-nn-03": _make_clip(speed=_NEUTRAL_SPEED),
    "clip-nn-04": _make_clip(speed=_NEUTRAL_SPEED),
    "clip-nn-05": _make_clip(speed=_NEUTRAL_SPEED),
}


@pytest.fixture()
def traj_store():
    return _make_store(dict(_CLIPS))


# -------------------------------------------------------------------------
# Expression tests
# -------------------------------------------------------------------------


class TestExpressionHighCurvature:
    def test_count(self, traj_store):
        """3 clips with curvature > 0.15 for > 10 frames → count == 3."""
        expr = TRAJECTORY_EXPRESSIONS["high_curvature"]
        result = traj_store._inner_search_trajectory(expr, set(_CLIPS.keys()))
        hc_clips = {k for k in _CLIPS if k.startswith("clip-hc-")}
        assert result == hc_clips


class TestExpressionStopGo:
    def test_count(self, traj_store):
        """2 clips with stop then go → count == 2."""
        expr = TRAJECTORY_EXPRESSIONS["stop_go"]
        result = traj_store._inner_search_trajectory(expr, set(_CLIPS.keys()))
        sg_clips = {k for k in _CLIPS if k.startswith("clip-sg-")}
        assert result == sg_clips


class TestExpressionHardBraking:
    def test_count(self, traj_store):
        """1 clip with >10 frames of accel < -3.0 → count == 1."""
        expr = TRAJECTORY_EXPRESSIONS["hard_braking"]
        result = traj_store._inner_search_trajectory(expr, set(_CLIPS.keys()))
        hb_clips = {k for k in _CLIPS if k.startswith("clip-hb-")}
        assert result == hb_clips


class TestExpressionProlongedStop:
    def test_count(self, traj_store):
        """Clips with > 150 frames of speed < 0.5 → exactly 2."""
        expr = TRAJECTORY_EXPRESSIONS["prolonged_stop"]
        result = traj_store._inner_search_trajectory(expr, set(_CLIPS.keys()))
        ps_clips = {k for k in _CLIPS if k.startswith("clip-ps-")}
        assert result == ps_clips


class TestExpressionMovingEgo:
    def test_count(self, traj_store):
        """Clips with > 10 frames of speed_kph > 5 → moving clips."""
        expr = TRAJECTORY_EXPRESSIONS["moving_ego"]
        result = traj_store._inner_search_trajectory(expr, set(_CLIPS.keys()))
        # stop_go clips also spend time moving (speed > 3.0 m/s → > 10.8 kph)
        # moving_ego clips: clip-me-* and clip-sg-* (100 frames > 5 kph)
        expected = {
            k for k in _CLIPS
            if k.startswith("clip-me-") or k.startswith("clip-sg-")
        }
        assert result == expected


class TestCustomSpeedExpression:
    def test_custom_speed(self, traj_store):
        """sum(speed_kph > 80) > 5 → only high-speed clips."""
        # Add a high-speed clip (80 kph = 22.2 m/s)
        fast_speed = np.full(T, 23.0, dtype=np.float32)  # ~83 kph
        traj_store.trajectory_data["clip-fast-01"] = _make_clip(speed=fast_speed)

        all_ids = set(traj_store.trajectory_data.keys())
        traj_store.searches.clear()
        result = traj_store._inner_search_trajectory(
            "sum(speed_kph > 80) > 5", all_ids
        )
        assert "clip-fast-01" in result
        # No other fixture clip reaches 80 kph
        assert all(c == "clip-fast-01" for c in result)


class TestNoMatch:
    def test_no_match_returns_empty(self, traj_store):
        """Expression that matches nothing → empty set."""
        traj_store.searches.clear()
        result = traj_store._inner_search_trajectory(
            "sum(speed_kph > 9999) > 1", set(_CLIPS.keys())
        )
        assert result == set()


class TestResultSubset:
    def test_result_is_subset_of_ids(self, traj_store):
        """Result is always a subset of the input id set."""
        all_ids = set(_CLIPS.keys())
        expr = TRAJECTORY_EXPRESSIONS["high_curvature"]
        traj_store.searches.clear()
        result = traj_store._inner_search_trajectory(expr, all_ids)
        assert result.issubset(all_ids)

    def test_search_with_id_filter(self, traj_store):
        """Only 2 of 3 matching clips are in the id set → count ≤ 2."""
        all_hc = {k for k in _CLIPS if k.startswith("clip-hc-")}
        two_ids = set(list(all_hc)[:2])
        expr = TRAJECTORY_EXPRESSIONS["high_curvature"]
        traj_store.searches.clear()
        result = traj_store._inner_search_trajectory(expr, two_ids)
        assert result.issubset(two_ids)
        assert len(result) <= 2


class TestCacheHit:
    def test_cache_hit(self, traj_store):
        """Second call with same query hits cache and is fast (≤ 0.01 s)."""
        expr = TRAJECTORY_EXPRESSIONS["hard_braking"]
        all_ids = set(_CLIPS.keys())
        # Prime the cache
        traj_store._inner_search_trajectory(expr, all_ids)

        t0 = time.perf_counter()
        traj_store._inner_search_trajectory(expr, all_ids)
        elapsed = time.perf_counter() - t0
        assert elapsed <= 0.01, f"Cache hit took {elapsed:.4f}s"


class TestLatency:
    def test_latency_100_clips(self):
        """Expression search over 100 synthetic clips completes in ≤ 1.0 s."""
        # Build 100 clips
        data = {}
        for i in range(100):
            data[f"clip-lat-{i:03d}"] = _make_clip(
                curvature=_HIGH_CURV if i % 3 == 0 else None
            )
        store = _make_store(data)
        expr = TRAJECTORY_EXPRESSIONS["high_curvature"]

        t0 = time.perf_counter()
        store._inner_search_trajectory(expr, set(data.keys()))
        elapsed = time.perf_counter() - t0
        assert elapsed <= 1.0, f"Latency test took {elapsed:.4f}s"


# -------------------------------------------------------------------------
# idle_to_cruise: speed < 0.5 *before* speed > 10.0
# -------------------------------------------------------------------------

# Positive: idle first (< 0.5), then cruise (> 10.0)
_IDLE_CRUISE_SPEED = np.full(T, 0.8, dtype=np.float32)
_IDLE_CRUISE_SPEED[:50] = 0.2    # idle (< 0.5)
_IDLE_CRUISE_SPEED[100:] = 12.0  # cruise (> 10.0)

# Negative: cruise first, then idle — ordering reversed
_CRUISE_IDLE_SPEED = np.full(T, 0.8, dtype=np.float32)
_CRUISE_IDLE_SPEED[:100] = 12.0  # cruise first
_CRUISE_IDLE_SPEED[200:] = 0.2   # idle after (only 100 frames, avoids prolonged_stop)


class TestExpressionIdleToCruise:
    def test_count(self, traj_store):
        """Clips with idle then cruise → only those match."""
        traj_store.trajectory_data["clip-ic-01"] = _make_clip(
            speed=_IDLE_CRUISE_SPEED
        )
        traj_store.trajectory_data["clip-ic-02"] = _make_clip(
            speed=_IDLE_CRUISE_SPEED
        )
        traj_store.searches.clear()
        expr = TRAJECTORY_EXPRESSIONS["idle_to_cruise"]
        result = traj_store._inner_search_trajectory(
            expr, set(traj_store.trajectory_data.keys())
        )
        assert result == {"clip-ic-01", "clip-ic-02"}

    def test_reversed_does_not_match(self, traj_store):
        """Cruise first then idle does not satisfy the ordering constraint."""
        traj_store.trajectory_data["clip-ic-rev"] = _make_clip(
            speed=_CRUISE_IDLE_SPEED
        )
        traj_store.searches.clear()
        result = traj_store._inner_search_trajectory(
            TRAJECTORY_EXPRESSIONS["idle_to_cruise"], {"clip-ic-rev"}
        )
        assert "clip-ic-rev" not in result


# -------------------------------------------------------------------------
# high_speed_swerve: high curvature AND high speed
# -------------------------------------------------------------------------

# Curvature > 0.2 (stricter than _HIGH_CURV which uses > 0.15)
_HSS_CURVATURE = np.full(T, 0.02, dtype=np.float32)
_HSS_CURVATURE[:50] = 0.25  # 50 frames with curvature > 0.2

# Speed: 15 m/s ≈ 54 kph > 50 kph threshold
_HSS_SPEED = np.full(T, 15.0, dtype=np.float32)


class TestExpressionHighSpeedSwerve:
    def test_count(self, traj_store):
        """Clips with both high curvature and high speed → only those match."""
        traj_store.trajectory_data["clip-hss-01"] = _make_clip(
            curvature=_HSS_CURVATURE, speed=_HSS_SPEED
        )
        traj_store.trajectory_data["clip-hss-02"] = _make_clip(
            curvature=_HSS_CURVATURE, speed=_HSS_SPEED
        )
        traj_store.searches.clear()
        expr = TRAJECTORY_EXPRESSIONS["high_speed_swerve"]
        result = traj_store._inner_search_trajectory(
            expr, set(traj_store.trajectory_data.keys())
        )
        assert result == {"clip-hss-01", "clip-hss-02"}

    def test_speed_only_does_not_match(self, traj_store):
        """High speed alone (no curvature) does not satisfy high_speed_swerve."""
        traj_store.trajectory_data["clip-hss-fast-only"] = _make_clip(
            speed=_HSS_SPEED  # curvature column stays 0.0
        )
        traj_store.searches.clear()
        result = traj_store._inner_search_trajectory(
            TRAJECTORY_EXPRESSIONS["high_speed_swerve"], {"clip-hss-fast-only"}
        )
        assert "clip-hss-fast-only" not in result

    def test_curvature_only_does_not_match(self, traj_store):
        """High curvature alone (low speed) does not satisfy high_speed_swerve."""
        traj_store.trajectory_data["clip-hss-curv-only"] = _make_clip(
            curvature=_HSS_CURVATURE, speed=_NEUTRAL_SPEED
        )
        traj_store.searches.clear()
        result = traj_store._inner_search_trajectory(
            TRAJECTORY_EXPRESSIONS["high_speed_swerve"], {"clip-hss-curv-only"}
        )
        assert "clip-hss-curv-only" not in result


# -------------------------------------------------------------------------
# Ordering: reversed stop_go
# -------------------------------------------------------------------------

# Go (> 3.0 m/s) first, then stop (< 0.5 m/s) — should NOT match stop_go
_GO_STOP_SPEED = np.full(T, 0.8, dtype=np.float32)
_GO_STOP_SPEED[:100] = 5.0   # go first
_GO_STOP_SPEED[200:] = 0.2   # stop after (100 frames, avoids prolonged_stop)


class TestStopGoOrdering:
    def test_reversed_does_not_match(self, traj_store):
        """Go before stop does not satisfy the stop_go ordering constraint."""
        traj_store.trajectory_data["clip-sg-rev"] = _make_clip(
            speed=_GO_STOP_SPEED
        )
        traj_store.searches.clear()
        result = traj_store._inner_search_trajectory(
            TRAJECTORY_EXPRESSIONS["stop_go"], {"clip-sg-rev"}
        )
        assert "clip-sg-rev" not in result


# -------------------------------------------------------------------------
# Boundary / threshold edge cases
# -------------------------------------------------------------------------


class TestBoundaryThresholds:
    def test_high_curvature_at_threshold_does_not_match(self, traj_store):
        """Exactly 10 frames with curvature > 0.15 → sum == 10, not > 10."""
        curv_10 = np.full(T, 0.02, dtype=np.float32)
        curv_10[:10] = 0.2  # exactly 10 frames
        traj_store.trajectory_data["clip-hc-10"] = _make_clip(
            curvature=curv_10, speed=_NEUTRAL_SPEED
        )
        traj_store.searches.clear()
        result = traj_store._inner_search_trajectory(
            TRAJECTORY_EXPRESSIONS["high_curvature"], {"clip-hc-10"}
        )
        assert "clip-hc-10" not in result

    def test_high_curvature_just_above_threshold_matches(self, traj_store):
        """11 frames with curvature > 0.15 → sum == 11 > 10 → matches."""
        curv_11 = np.full(T, 0.02, dtype=np.float32)
        curv_11[:11] = 0.2
        traj_store.trajectory_data["clip-hc-11"] = _make_clip(
            curvature=curv_11, speed=_NEUTRAL_SPEED
        )
        traj_store.searches.clear()
        result = traj_store._inner_search_trajectory(
            TRAJECTORY_EXPRESSIONS["high_curvature"], {"clip-hc-11"}
        )
        assert "clip-hc-11" in result

    def test_prolonged_stop_at_threshold_does_not_match(self, traj_store):
        """Exactly 150 frames stopped → sum == 150, not > 150."""
        speed_150 = np.full(T, 0.8, dtype=np.float32)
        speed_150[:150] = 0.3
        traj_store.trajectory_data["clip-ps-150"] = _make_clip(speed=speed_150)
        traj_store.searches.clear()
        result = traj_store._inner_search_trajectory(
            TRAJECTORY_EXPRESSIONS["prolonged_stop"], {"clip-ps-150"}
        )
        assert "clip-ps-150" not in result

    def test_prolonged_stop_just_above_threshold_matches(self, traj_store):
        """151 frames stopped → sum == 151 > 150 → matches."""
        speed_151 = np.full(T, 0.8, dtype=np.float32)
        speed_151[:151] = 0.3
        traj_store.trajectory_data["clip-ps-151"] = _make_clip(speed=speed_151)
        traj_store.searches.clear()
        result = traj_store._inner_search_trajectory(
            TRAJECTORY_EXPRESSIONS["prolonged_stop"], {"clip-ps-151"}
        )
        assert "clip-ps-151" in result


# -------------------------------------------------------------------------
# search_trajectory (higher-level public method)
# -------------------------------------------------------------------------


class TestSearchTrajectory:
    def test_pattern_only(self, traj_store):
        """trajectory_pattern alone dispatches to the correct expression."""
        filters = SimpleNamespace(
            trajectory_pattern="high_curvature",
            search_speed=None,
        )
        traj_store.searches.clear()
        result = traj_store.search_trajectory(filters, set(_CLIPS.keys()))
        assert result == {k for k in _CLIPS if k.startswith("clip-hc-")}

    def test_speed_only(self, traj_store):
        """search_speed alone (no pattern) filters by the custom expression."""
        fast_speed = np.full(T, 23.0, dtype=np.float32)  # ~83 kph
        traj_store.trajectory_data["clip-fast-st"] = _make_clip(speed=fast_speed)
        filters = SimpleNamespace(
            trajectory_pattern=None,
            search_speed="sum(speed_kph > 80) > 5",
        )
        traj_store.searches.clear()
        result = traj_store.search_trajectory(
            filters, set(traj_store.trajectory_data.keys())
        )
        assert "clip-fast-st" in result
        assert all(c == "clip-fast-st" for c in result)

    def test_pattern_and_speed_combined(self, traj_store):
        """Pattern AND speed are AND-ed: only clips satisfying both match."""
        # clip-hc-fast: high curvature + high speed → should match combined
        traj_store.trajectory_data["clip-hc-fast"] = _make_clip(
            curvature=_HIGH_CURV,
            speed=np.full(T, 15.0, dtype=np.float32),  # 54 kph
        )
        # clip-hc-* in _CLIPS: high curvature but neutral speed (2.9 kph) → no match
        filters = SimpleNamespace(
            trajectory_pattern="high_curvature",
            search_speed="sum(speed_kph > 50) > 10",
        )
        traj_store.searches.clear()
        result = traj_store.search_trajectory(
            filters, set(traj_store.trajectory_data.keys())
        )
        assert result == {"clip-hc-fast"}


# -------------------------------------------------------------------------
# Empty edge cases
# -------------------------------------------------------------------------


class TestEmptyEdgeCases:
    def test_empty_id_set(self, traj_store):
        """An empty id set always produces an empty result."""
        result = traj_store._inner_search_trajectory(
            TRAJECTORY_EXPRESSIONS["high_curvature"], set()
        )
        assert result == set()

    def test_empty_store(self):
        """A store with no trajectory data returns an empty set."""
        store = _make_store({})
        result = store._inner_search_trajectory(
            TRAJECTORY_EXPRESSIONS["high_curvature"], set()
        )
        assert result == set()


# -------------------------------------------------------------------------
# has_trajectories
# -------------------------------------------------------------------------


class TestHasTrajectories:
    def test_known_clip_returns_true(self, traj_store):
        """A clip present in clip_to_idx is recognised."""
        traj_store.clip_to_idx["clip-hc-01"] = (0, 10)
        assert traj_store.has_trajectories("clip-hc-01") is True

    def test_unknown_clip_returns_false(self, traj_store):
        """A clip absent from clip_to_idx is not recognised."""
        assert traj_store.has_trajectories("clip-does-not-exist") is False


# -------------------------------------------------------------------------
# Expression safety
# -------------------------------------------------------------------------


class TestExpressionSafety:
    @pytest.mark.parametrize("expr", [
        "__import__('os')",
        "open('/etc/passwd')",
        "np.load('/etc/passwd')",
        "speed.tofile('/tmp/x')",
        "speed.__class__",
        "mean.__globals__",
        "().__class__",
        "lambda: 1",
        "[x for x in speed]",
        "'a string'",
    ])
    def test_unsafe_expressions_rejected(self, expr):
        with pytest.raises(ValueError):
            _compile_trajectory_predicate(expr)

    @pytest.mark.parametrize("name", list(TRAJECTORY_EXPRESSIONS))
    def test_predefined_patterns_compile(self, name):
        """Every shipped pattern must still parse under the allowlist."""
        _compile_trajectory_predicate(TRAJECTORY_EXPRESSIONS[name])

    def test_whitelisted_functions_evaluate(self):
        speed = np.concatenate([np.zeros(50), np.full(50, 10.0)])
        z = np.zeros(100, dtype=np.float32)
        # where + subscript, logical_and (UI example), gradient, mean
        for expr in [
            "mean(speed) > 1",
            "sum(logical_and(speed > 5, speed < 20)) > 10",
            "min(where(speed > 5)[0]) > 10",
            "max(abs(gradient(speed))) > 0",
        ]:
            assert _compile_trajectory_predicate(expr)(speed, z, z, z)

    def test_unsafe_expression_rejected_through_search(self):
        store = _make_store({})
        with pytest.raises(ValueError):
            store._inner_search_trajectory("speed.__class__", set())

    def test_compiled_predicate_is_cached(self):
        a = _compile_trajectory_predicate("mean(speed) > 5")
        b = _compile_trajectory_predicate("mean(speed) > 5")
        assert a is b
