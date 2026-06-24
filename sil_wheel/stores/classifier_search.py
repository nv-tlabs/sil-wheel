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

import ast
import functools
import json
from pathlib import Path

import numpy as np
import simdjson
from sil_wheel.stores.search_utils import project_starmap


# Expression elements a probability filter may use.
ALLOWED_NODES = (
    ast.Expression, ast.Compare, ast.BoolOp, ast.BinOp, ast.UnaryOp,
    ast.And, ast.Or, ast.Not, ast.Invert, ast.USub,
    ast.BitAnd, ast.BitOr, ast.BitXor,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
    ast.Name, ast.Load, ast.Constant,
)


class _NumpyBoolean(ast.NodeTransformer):
    """Rewrite Python boolean logic into numpy elementwise ops so a predicate
    evaluates correctly when ``p`` is an array of scores::

        a and b      -> a & b
        a or b       -> a | b
        not a        -> ~a
        lo < x < hi  -> (lo < x) & (x < hi)

    Done on the AST rather than the string, which also sidesteps the
    ``&``/``|`` operator-precedence trap of the textual form.
    """

    _BITWISE = {ast.And: ast.BitAnd, ast.Or: ast.BitOr}

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        op = self._BITWISE[type(node.op)]()
        combined = node.values[0]
        for value in node.values[1:]:
            combined = ast.BinOp(left=combined, op=op, right=value)
        return combined

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return ast.UnaryOp(op=ast.Invert(), operand=node.operand)
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) == 1:
            return node
        left = node.left
        comparisons = []
        for op, right in zip(node.ops, node.comparators):
            comparisons.append(
                ast.Compare(left=left, ops=[op], comparators=[right])
            )
            left = right
        combined = comparisons[0]
        for comparison in comparisons[1:]:
            combined = ast.BinOp(left=combined, op=ast.BitAnd(), right=comparison)
        return combined


@functools.lru_cache(maxsize=128)
def _compile_expression(expression):
    """Compile a probability filter string into a predicate ``f(p) -> mask``.

    ``p`` is the score (scalar or a numpy array of scores); the predicate
    returns a bool / bool array, so it vectorizes over all scores at once.
    Supports comparisons, chained comparisons, ``and`` / ``or`` / ``not``, and
    bitwise ``& | ~`` over ``p`` and numeric literals. Examples::

        "p > 0.95"
        "0.9 < p < 0.95"
        "p > 0.8 or p < 0.2"

    The expression is parsed and checked against a strict allowlist before it
    runs: it may only reference ``p`` and cannot call functions or reach
    attributes, so no arbitrary code can execute. Compiled predicates are
    memoised per expression (compilation is otherwise repeated on cache
    misses in ``filter_clips``).
    """
    tree = ast.parse(expression.strip(), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(
                f"disallowed element in probability expression: "
                f"{type(node).__name__}"
            )
        if isinstance(node, ast.Name) and node.id != "p":
            raise ValueError(f"only the variable 'p' is allowed, got {node.id!r}")
        if isinstance(node, ast.Constant) and not isinstance(
            node.value, (int, float)
        ):
            raise ValueError("only numeric literals are allowed")
    tree = ast.fix_missing_locations(_NumpyBoolean().visit(tree))
    code = compile(tree, "<probability_expression>", "eval")

    def predicate(p):
        return eval(code, {"__builtins__": {}}, {"p": p})

    return predicate


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
            try:
                predicate = _compile_expression(expression)
                expr_mask = predicate(scores_arr)
            except ValueError as e:
                raise ValueError(
                    f"Invalid probability expression {expression!r}: {e}"
                ) from e

            # Limit the _filter_cache to 10
            if len(self._filter_cache) >= 10:
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
