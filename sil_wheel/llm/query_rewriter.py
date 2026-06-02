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

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from sil_wheel.llm.llm_client import get_llm_client

logger = logging.getLogger(__name__)

_STOP_WORDS = frozenset(
    {"a", "an", "the", "of", "to", "in", "on", "at", "for", "with", "by", "is", "are"}
)

# Domain glossary: maps common AV query terms to caption-style synonyms.
# When a query matches a key (substring), the synonyms are injected into the
# prompt so the LLM has concrete anchors instead of guessing domain jargon.
_AV_GLOSSARY: Dict[str, List[str]] = {
    "nudge": ["lateral shift", "steering correction", "small swerve"],
    "cut-in": ["vehicle cutting in", "merge ahead", "sudden lane entry"],
    "cut in": ["vehicle cutting in", "merge ahead", "sudden lane entry"],
    "hard brake": ["sudden stop", "emergency braking", "abrupt deceleration"],
    "hard braking": ["sudden stop", "emergency braking", "abrupt deceleration"],
    "tailgating": ["close following", "following closely", "rear proximity"],
    "jaywalking": ["pedestrian crossing", "person crossing road"],
    "lane change": ["changing lanes", "lane switch", "lateral move"],
    "swerve": ["sudden steering", "evasive maneuver", "sharp turn"],
    "u-turn": ["u turn", "turning around", "reverse direction"],
    "roundabout": ["traffic circle", "rotary"],
    "construction zone": ["work zone", "road construction", "construction area"],
    "cyclist": ["bicycle rider", "bike rider", "person cycling"],
    "overtake": ["passing vehicle", "overtaking", "passing maneuver"],
    "red light": ["traffic signal red", "red signal", "stop signal"],
    "merge": ["lane merge", "merging traffic", "highway merge"],
}

_SYSTEM_PROMPT = """You generate short search rewrites for an autonomous-vehicle video caption retrieval system.

Retrieval context:
- Captions are descriptive text about road entities, traffic infrastructure, and ego behavior.
- Search uses strict keyword matching, so short phrases improve recall.
- The system unions results across all rewrites.

Task:
- Convert a user query into short, semantically equivalent alternatives that are likely to appear in VLM captions.
- Prefer concrete scene/action wording over abstract terms.

Rules:
1. Every rewrite must be 1 to 3 words.
2. Preserve the original intent; do not broaden into unrelated scenarios.
3. Use natural caption-like wording (e.g., "lane change", "pedestrian crossing", "steering correction").
4. Each rewrite must be a self-contained phrase. Never end with articles, prepositions, or stop words (a, an, the, of, to, in, on, at, for, with, by).
5. Keep rewrites concise. No explanations, no analysis, no markdown.
6. Return valid JSON only following the requested schema.
"""

_USER_PROMPT = """User query: {query}

Generate exactly {num_rewrites} rewrites.

Examples (style guide only):
- Input: "nudge"
  Output candidates: ["lateral shift", "steering correction", "veering slightly", "small swerve", "gentle drift"]
- Input: "pedestrian crossing the street"
  Output candidates: ["pedestrian crossing", "person crossing", "crossing pedestrian", "street crossing", "walking across"]
- Input: "lane change"
  Output candidates: ["lane change", "changing lanes", "lane switch", "lane merge", "lateral move"]

Output format (JSON only):
{{
  "rewrites": [
    "1-3 word rewrite",
    "1-3 word rewrite"
  ]
}}
"""


def _glossary_hints(query: str) -> str:
    """Return a prompt fragment with synonym hints for any glossary terms found in the query."""
    query_lower = query.lower()
    matches: List[str] = []
    for term, synonyms in _AV_GLOSSARY.items():
        if term in query_lower:
            matches.append(f'  "{term}": {", ".join(synonyms)}')
    if not matches:
        return ""
    return (
        "\nKnown domain synonyms (use as a starting point, also generate novel alternatives):\n"
        + "\n".join(matches)
        + "\n"
    )


def _normalize_rewrite(text: str) -> str:
    """Lowercase, collapse whitespace, truncate to 3 words, strip trailing stop words."""
    cleaned = re.sub(r"\s+", " ", str(text or "").strip().lower())
    words = [w for w in cleaned.split(" ") if w]
    if not words:
        return ""
    words = words[:3]
    while len(words) > 1 and words[-1] in _STOP_WORDS:
        words.pop()
    return " ".join(words)


@dataclass
class RewriteResult:
    original_input: str
    queries: List[str]


class QueryRewriter:
    def __init__(self, provider="auto", **kwargs):
        self.llm = get_llm_client(provider=provider, **kwargs)
        self._cache: Dict[Tuple[str, int], RewriteResult] = {}

    def rewrite_query(self, query: str, num_rewrites: int = 5) -> RewriteResult:
        cache_key = (query, num_rewrites)
        if cache_key in self._cache:
            logger.debug("Cache hit for query rewrite: %r", query)
            return self._cache[cache_key]

        hints = _glossary_hints(query)
        prompt = _USER_PROMPT.format(
            query=query,
            num_rewrites=num_rewrites,
        )
        if hints:
            prompt = prompt + hints

        try:
            response_text = self.llm.generate(
                prompt=prompt,
                system_prompt=_SYSTEM_PROMPT,
                response_format={"type": "json_object"},
            )
            response = self.llm.parse_json(response_text)
            rewrites = self._extract_rewrites(response)
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            rewrites = [query]

        result = RewriteResult(original_input=query, queries=rewrites)
        self._cache[cache_key] = result
        return result

    @staticmethod
    def _extract_rewrites(response) -> List[str]:
        """Pull rewrite strings from the LLM response, normalize, and deduplicate."""
        if isinstance(response, list):
            raw = response
        elif isinstance(response, dict):
            raw = response.get("rewrites") or response.get("queries") or []
        else:
            return []

        seen: set[str] = set()
        out: List[str] = []
        for item in raw:
            text = _normalize_rewrite(item if isinstance(item, str) else item.get("text", ""))
            if text and text not in seen:
                seen.add(text)
                out.append(text)
        return out
