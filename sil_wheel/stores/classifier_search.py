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
import re
from pathlib import Path

import numpy as np
import simdjson
from sil_wheel.stores.search_utils import project_starmap

# Maximum number of (run_id, expression) results held in the filter cache.
_FILTER_CACHE_SIZE = 10


def _compile_expression(expression):
    """Compile a probability expression string into a predicate.

    The expression uses ``p`` as the probability variable.  Any valid
    Python boolean expression is accepted, including chained comparisons.
    When ``p`` is a numpy array the predicate returns a boolean array,
    enabling vectorized evaluation over all scores at once.

    Chained comparisons like ``0.9 < p < 0.95`` are rewritten to
    ``(0.9 < p) & (p < 0.95)`` so they work correctly with numpy arrays.

    Examples::

        "p > 0.95"
        "p < 0.2"
        "0.9 < p < 0.95"
    """
    m = re.match(
        r"^([\d.]+)\s*(<=?)\s*p\s*(<=?)\s*([\d.]+)$", expression.strip()
    )
    if m:
        lo, lo_op, hi_op, hi = m.group(1), m.group(2), m.group(3), m.group(4)
        expression = f"({lo} {lo_op} p) & (p {hi_op} {hi})"
    _globals = {"__builtins__": {}, "np": np}
    fn = "def predicate(p):\n    return " + expression
    exec(fn, _globals)
    return _globals["predicate"]


class ClassifierSearch:
    def __init__(self, classifier_dir):
        self.classifier_dir = Path(classifier_dir)
        # run_id -> (mtime, clip_ids_array, scores_array)
        self._score_cache = {}
        # (run_id, expression) -> boolean mask over clip_ids_array
        self._filter_cache = {}
        # run_id -> dict (parsed metadata.json)
        self._metadata_cache = {}

    def _run_dir(self, run_id):
        return self.classifier_dir / run_id

    def load_metadata(self, run_id):
        """Return the parsed metadata.json for a run, or None if missing."""
        if run_id in self._metadata_cache:
            return self._metadata_cache[run_id]
        path = self._run_dir(run_id) / "metadata.json"
        if not path.exists():
            return None
        with open(path) as f:
            metadata = json.load(f)
        self._metadata_cache[run_id] = metadata
        return metadata

    def list_runs(self):
        """Return one dict per run on disk.

        Skips entries that don't carry a metadata.json so that pre-run-id
        directories left over from the old (embed_type, label) layout
        stay invisible to this listing.
        """
        if not self.classifier_dir.exists():
            return []
        runs = []
        for subdir in sorted(self.classifier_dir.iterdir()):
            if not subdir.is_dir():
                continue
            meta_path = subdir / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path) as f:
                    metadata = json.load(f)
            except Exception:
                continue
            run_id = subdir.name
            metadata = dict(metadata)
            metadata.setdefault("run_id", run_id)
            runs.append(metadata)
        return runs

    def load_scores(self, run_id):
        path = self._run_dir(run_id) / "predicted_scores.json"
        if not path.exists():
            return None, None

        modified = path.stat().st_mtime
        cached = self._score_cache.get(run_id)

        # Reload if the file has been modified since last load
        if cached is None or modified > cached[0]:
            parser = simdjson.Parser()
            with open(path, "rb") as f:
                scores = parser.parse(f.read())
            clip_ids = np.array(list(scores.keys()))
            scores_arr = np.array(list(scores.values()), dtype=np.float32)
            self._score_cache[run_id] = (modified, clip_ids, scores_arr)

        _, clip_ids, scores_arr = self._score_cache[run_id]
        return clip_ids, scores_arr

    def get_score(self, clip, run_id):
        clip_ids, scores_arr = self.load_scores(run_id)
        idx = np.where(clip_ids == clip)[0][0]
        return float(scores_arr[idx])

    def invalidate(self, run_id):
        """Drop cached state for ``run_id`` so the next call re-reads from disk.

        Called after a run's files are replaced on disk (e.g. by the
        ``/upload_classifier`` endpoint) so stale scores don't survive
        the swap.
        """
        self._score_cache.pop(run_id, None)
        self._metadata_cache.pop(run_id, None)
        stale = [k for k in self._filter_cache if k[0] == run_id]
        for k in stale:
            del self._filter_cache[k]

    def filter_clips(self, ids, run_id, expression):
        clip_ids, scores_arr = self.load_scores(run_id)
        if clip_ids is None:
            return []

        cache_key = (run_id, expression)
        if cache_key in self._filter_cache:
            expr_mask = self._filter_cache.pop(cache_key)
            self._filter_cache[cache_key] = expr_mask
        else:
            predicate = _compile_expression(expression)
            try:
                expr_mask = predicate(scores_arr)
            except ValueError as e:
                raise ValueError(
                    f"Invalid probability expression {expression!r}: {e}"
                ) from e
            if len(self._filter_cache) >= _FILTER_CACHE_SIZE:
                del self._filter_cache[next(iter(self._filter_cache))]
            self._filter_cache[cache_key] = expr_mask

        if len(ids) >= len(clip_ids):
            result_mask = expr_mask
        else:
            ids_mask = np.isin(clip_ids, np.asarray(list(ids)))
            result_mask = expr_mask & ids_mask

        return list(zip(
            clip_ids[result_mask].tolist(),
            scores_arr[result_mask].tolist(),
        ))

    def search(self, filters, current_results):
        if (
            filters.classifier_run_id is not None
            and filters.probability_expression is not None
        ):
            current_results = project_starmap(
                lambda r, s: r.with_classifier_score(s),
                current_results,
                self.filter_clips(
                    current_results.keys(),
                    filters.classifier_run_id,
                    filters.probability_expression,
                ),
            )
        return current_results
