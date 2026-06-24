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

"""Caption quality metrics: registry + factory.

Example::

    from scorers import build_scorer
    judge = build_scorer("llm_judge", provider="auto", max_workers=20)
    judge.score_one(reference="Roadwork", prediction="The video shows ...")
"""
from typing import Any, Dict, List

# EVQA is registered lazily so this module imports without the [evqa] extra.
from judges import LingoJudge, LLMJudge
from metrics import (
    BERTScorer,
    NLGScorer,
    VLMJudgeScorer,
)

_REGISTRY: Dict[str, type] = {
    "nlg": NLGScorer,
    "bertscore": BERTScorer,
    "lingojudge": LingoJudge,
    "llm_judge": LLMJudge,
    "vlm_judge": VLMJudgeScorer,
}

#: Metrics that score predictions against the source video (need ``video_path`` on each pair).
REF_FREE_METRICS = frozenset({"vlm_judge", "evqa"})


def _evqa_available() -> bool:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return True


def available_metrics() -> List[str]:
    """Registered metric names; ``"evqa"`` only when ``[evqa]`` extra is installed."""
    out = list(_REGISTRY.keys())
    if _evqa_available():
        out.append("evqa")
    return out


def build_scorer(name: str, **kwargs: Any):
    """Instantiate a scorer by name. Reuse the returned object across many calls."""
    if name == "evqa":
        from evqa import EVQAScorer
        return EVQAScorer(**kwargs)
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown metric: {name!r}. Available: {available_metrics()}"
        )
    return _REGISTRY[name](**kwargs)


__all__ = ["REF_FREE_METRICS", "available_metrics", "build_scorer"]
