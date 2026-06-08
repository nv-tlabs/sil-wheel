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

"""Reference-based judges: text-only LLM (binary match) and LingoJudge classifier."""
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _row(pair: Dict[str, Any], **scores: Any) -> Dict[str, Any]:
    """Build a result row: passthrough clip_id / data_source / scenario plus scores."""
    out: Dict[str, Any] = {
        "clip_id": pair.get("clip_id"),
        "data_source": pair.get("data_source"),
    }
    if pair.get("scenario"):
        out["scenario"] = pair["scenario"]
    out.update(scores)
    return out


# ---------------------------------------------------------------------------
# Text-only LLM-as-judge
# ---------------------------------------------------------------------------

LLM_JUDGE_SYSTEM = (
    "You are an expert evaluator of video captions for autonomous driving "
    "(dashcam / ego-vehicle) clips. You compare a CANDIDATE caption against a "
    "trusted REFERENCE and decide whether the candidate contains a description "
    "that matches the reference. Be objective and concise."
)

LLM_JUDGE_TEMPLATE = """\
## Task: Caption Match (Reference-Based, Binary)

You are given a REFERENCE caption (human ground truth) and a CANDIDATE \
caption (from a model). Decide whether the CANDIDATE contains a description \
that matches the REFERENCE — i.e. some part of the CANDIDATE accurately \
conveys the scene / event / content described by the REFERENCE. Only \
answer NO if the CANDIDATE contradicts the REFERENCE or fails to mention \
anything matching it.

### REFERENCE:
{reference}

### CANDIDATE:
{candidate}

### Output (single JSON object, nothing else):
{{
  "reasoning": "<1-2 sentences>",
  "match": "yes" or "no"
}}
"""


def _parse_match(value: Any) -> Optional[bool]:
    """Coerce an LLM ``match`` value (bool / int / str) to bool, else None."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"yes", "true", "1", "match"}
    return None


def _parse_json_block(text: str) -> Dict[str, Any]:
    """Extract a JSON object from text, tolerant of ``` fences and prose around it."""
    if not text:
        return {}
    text = text.strip()
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    return {}


class LLMJudge:
    """Text-only LLM-as-judge: binary match (yes/no) plus a short motivation."""

    SCORE_DIMS = ("match",)

    def __init__(
        self,
        provider: str = "auto",
        model: Optional[str] = None,
        max_workers: int = 20,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        from sil_wheel.llm.llm_client import get_llm_client

        kwargs: Dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
        if model:
            kwargs["model"] = model
        self.client = get_llm_client(provider=provider, **kwargs)
        self.max_workers = max_workers
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _score_one_pair(self, pair: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        user_prompt = LLM_JUDGE_TEMPLATE.format(
            reference=pair["reference"], candidate=pair["prediction"],
        )
        # No response_format: Gemini on NV inference returns content=null
        # under JSON mode. max_tokens is generous because Gemini-3-flash
        # burns most of the budget on reasoning_tokens before the JSON.
        try:
            raw = self.client.generate(
                prompt=user_prompt,
                system_prompt=LLM_JUDGE_SYSTEM,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            logger.warning("llm_judge failed on %s: %s", pair.get("clip_id"), exc)
            return None
        if not raw:
            logger.warning("llm_judge: empty response for %s", pair.get("clip_id"))
            return None

        parsed = _parse_json_block(raw) if isinstance(raw, str) else {}
        is_match = _parse_match(parsed.get("match"))
        if is_match is None:
            return None
        return _row(
            pair,
            llm_match=float(is_match),
            llm_motivation=parsed.get("reasoning") or "",
        )

    def score_one(
        self,
        reference: str,
        prediction: str,
        clip_id: str = "_",
        data_source: str = "_",
    ) -> Optional[Dict[str, Any]]:
        return self._score_one_pair({
            "clip_id": clip_id, "data_source": data_source,
            "reference": reference, "prediction": prediction,
        })

    def score_batch(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not pairs:
            return out
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(self._score_one_pair, p) for p in pairs]
            for fut in as_completed(futures):
                row = fut.result()
                if row is not None:
                    out.append(row)
        return out


def score_llm_judge(
    pairs: List[Dict[str, Any]],
    provider: str = "auto",
    model: Optional[str] = None,
    max_workers: int = 20,
) -> List[Dict[str, Any]]:
    return LLMJudge(provider=provider, model=model, max_workers=max_workers).score_batch(pairs)


# ---------------------------------------------------------------------------
# LingoJudge
# ---------------------------------------------------------------------------

LINGO_JUDGE_MODEL = "wayveai/Lingo-Judge"

# Default question used only when a pair carries no per-item question
# (caption-vs-caption / human modes). Lingo-Judge was trained on
# (question, answer, prediction) triples, so a pair's own question is preferred.
LINGO_JUDGE_QUESTION = "Describe what is happening in this driving video."


class LingoJudge:
    """Lingo-Judge truthfulness classifier from LingoQA (Marcu et al., "LingoQA:
    Visual Question Answering for Autonomous Driving", ECCV 2024,
    arXiv:2312.14115; code github.com/wayveai/LingoQA, model
    huggingface.co/wayveai/Lingo-Judge). Each pair is framed in LingoQA's
    question/answer/student format using the pair's own ``question`` when present
    (e.g. a QA dataset), else ``default_question``. Logit > 0 means correct."""

    def __init__(
        self,
        device: str = "cuda",
        max_length: int = 512,
        batch_size: int = 16,
        pretrained_model: str = LINGO_JUDGE_MODEL,
        default_question: str = LINGO_JUDGE_QUESTION,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not torch.cuda.is_available() and device == "cuda":
            logger.warning("CUDA unavailable; LingoJudge will run on CPU")
            device = "cpu"
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.default_question = default_question
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(pretrained_model)
            .eval()
            .to(device)
        )
        self._cls = self.tokenizer.cls_token
        self._torch = torch

    def _build(self, pair: Dict[str, Any]) -> str:
        question = (pair.get("question") or self.default_question).strip()
        ref = (pair.get("reference") or "").lower().strip()
        pred = (pair.get("prediction") or "").lower().strip()
        return f"{self._cls}\nQuestion: {question}\nAnswer: {ref}\nStudent: {pred}"

    def score_batch(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not pairs:
            return out
        torch = self._torch
        with torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start:start + self.batch_size]
                texts = [self._build(p) for p in batch]
                enc = self.tokenizer(
                    texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=self.max_length,
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits.squeeze(-1).cpu().tolist()
                if isinstance(logits, float):
                    logits = [logits]
                for p, score in zip(batch, logits):
                    out.append(_row(
                        p,
                        lingojudge_score=float(score),
                        lingojudge_correct=float(score > 0.0),
                    ))
        return out

    def score_one(
        self,
        reference: str,
        prediction: str,
        clip_id: str = "_",
        data_source: str = "_",
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.score_batch([{
            "clip_id": clip_id, "data_source": data_source,
            "reference": reference, "prediction": prediction, "question": question,
        }])[0]

    def close(self):
        import gc
        del self.model, self.tokenizer
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def score_lingojudge(
    pairs: List[Dict[str, Any]],
    device: str = "cuda",
    max_length: int = 512,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    return LingoJudge(
        device=device, max_length=max_length, batch_size=batch_size,
    ).score_batch(pairs)
