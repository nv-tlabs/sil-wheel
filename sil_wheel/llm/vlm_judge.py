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

"""
VLM judge for the Wheel server.
"""

import base64
import io
import json
import logging
import os
import re
import tempfile
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from sil_wheel.llm.base import BaseVLMClient
from sil_wheel.llm.vlm_client import get_vlm_client
from sil_wheel.llm.vlm_messages import (
    frames_message,
    merge_user_messages,
    text_message,
)

logger = logging.getLogger(__name__)


class LRUCache:
    def __init__(self, maxsize=2000):
        self._cache = OrderedDict()
        self._by_clip = defaultdict(set)
        self.maxsize = maxsize

    def put(self, key, value):
        clip_id = key[0]
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.maxsize:
                evicted, _ = self._cache.popitem(last=False)
                self._by_clip[evicted[0]].discard(evicted)
            self._by_clip[clip_id].add(key)
        self._cache[key] = value

    def get_for_clip(self, clip_id):
        return [self._cache[k] for k in self._by_clip.get(clip_id, ()) if k in self._cache]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

VIDEO_QUERY_MATCH_SYSTEM = """\
You are an expert video analyst for autonomous driving (dashcam / \
ego-vehicle) videos. Your task is to verify ground-truth consistency \
between a VIDEO scene and a textual QUERY description. You must be \
objective, precise, and specific."""

VIDEO_QUERY_MATCH_TEMPLATE = """\
## Task:
You have been provided with a VIDEO from an autonomous vehicle and a \
text QUERY. Your goal is to output a binary score (1 or 0) indicating \
if the VIDEO contains the visual content described in the QUERY.

## Instructions:
- Output 'prediction: 1' (YES) only if the VIDEO clearly shows what is \
described in the provided QUERY with enough visual evidence to \
confidently support it.
- Output 'prediction: 0' (NO) if the QUERY is not present, is \
contradicted, or if the VIDEO does not provide enough evidence \
(uncertain, too small/blurred/occluded/too brief, off-screen, \
implied only).
- If the QUERY has multiple requirements (A and B and C), output YES \
only if all are satisfied.
- Base the decision only on what is visible in the VIDEO frames provided \
(do not assume context not shown).
- Output only in JSON format. First describe your analysis, then your \
reasoning, then based on your analysis and reasoning, output the \
prediction.

## Output format (strict):
{{
  "analysis": "brief analysis of the VIDEO and QUERY",
  "reason": "brief reasoning citing concrete visual cues",
  "prediction": 0 or 1
}}

## Input QUERY:
"{query}"
"""

CAPTION_SCORE_SYSTEM = """\
You are an expert autonomous-driving video analyst. You score how well a \
caption describes a given dashcam video. You are objective and precise, \
grading each attribute independently."""

CAPTION_SCORE_TEMPLATE = """\
## Task: Caption Quality Scoring

You are given a driving video clip and a caption that attempts to \
describe it. Score the caption on the following attributes (each 1-10):

1. **scene** – How well does the caption describe the overall scene \
(weather, lighting, road type, environment)?
2. **action** – How well does the caption describe the ego-vehicle's \
actions and driving behavior?
3. **road_entities** – How well does the caption describe other road \
entities (vehicles, pedestrians, cyclists, traffic signs/lights)?
4. **temporal** – How well does the caption capture the temporal \
progression of events?
5. **overall** – Holistic quality of the caption as a faithful video \
description.

### Caption:
{caption}

### Instructions:
1. Watch the video carefully.
2. Read the caption and compare against what you see.
3. Score each attribute from 1 (very poor) to 10 (excellent).
4. Output your answer as a **single JSON block**:

```json
{{
  "reasoning": "<2-3 sentences justifying your scores>",
  "scores": {{
    "scene": <int>,
    "action": <int>,
    "road_entities": <int>,
    "temporal": <int>,
    "overall": <int>
  }}
}}
```

Output ONLY the JSON block after your reasoning. Nothing else after the JSON."""

ARENA_PAIRWISE_SYSTEM = """\
You are an expert evaluator performing blind A/B comparison. You will be \
shown an input context and two model outputs (A and B). Your task is to \
judge which output is better according to the provided evaluation \
criteria. Be objective, precise, and justify your choice with specific \
observations."""

ARENA_PAIRWISE_TEMPLATE = """\
## Evaluation Criteria:
{criteria_description}

## Input Context:
{inputs_description}

## Model A Output:
{output_a_description}

## Model B Output:
{output_b_description}

## Task:
Compare the two outputs above on EACH criterion listed. For every criterion, \
pick the better output and explain why. Output your answer as a single JSON block:

{{
  "criteria": [
    {{
      "name": "<criterion name, exactly as listed above>",
      "reasoning": "<2-4 sentences comparing the two outputs on this criterion>",
      "winner": "a" or "b" or "tie",
      "confidence": "strong" or "moderate" or "weak"
    }}
  ]
}}

- "a" means Output A is better
- "b" means Output B is better
- "tie" means they are roughly equal in quality
- confidence "strong" maps to a decisive preference, "moderate" to a slight \
preference, "weak" means nearly indistinguishable

You MUST include one entry per criterion. Output ONLY the JSON block. Nothing else."""


def build_video_query_match_prompt(query: str) -> tuple[str, str]:
    user_prompt = VIDEO_QUERY_MATCH_TEMPLATE.format(query=query)
    return VIDEO_QUERY_MATCH_SYSTEM, user_prompt


def build_caption_score_prompt(caption: str) -> tuple[str, str]:
    user_prompt = CAPTION_SCORE_TEMPLATE.format(caption=caption)
    return CAPTION_SCORE_SYSTEM, user_prompt


def encode_image(image: Union[str, bytes, np.ndarray]) -> str:
    if isinstance(image, bytes):
        return base64.b64encode(image).decode("utf-8")
    if isinstance(image, str):
        with open(image, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")
    if isinstance(image, np.ndarray):
        ok, buf = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("Failed to encode frame as JPEG")
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    raise ValueError(f"Unsupported image type: {type(image)}")


def extract_frames(
    video_bytes: bytes,
    max_frames: int = 64,
    fps: Optional[float] = 2.0,
    width: Optional[int] = 1280,
    height: Optional[int] = 720,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> List[np.ndarray]:
    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    try:
        os.write(fd, video_bytes)
        os.close(fd)

        cap = cv2.VideoCapture(tmp)
        if not cap.isOpened():
            raise ValueError("Cannot open video")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if total_frames <= 0:
            raise ValueError("Video has no frames")

        # Clamp to time range if specified
        start_frame = int(start_time * video_fps) if start_time is not None else 0
        end_frame = int(end_time * video_fps) if end_time is not None else total_frames
        start_frame = max(0, min(start_frame, total_frames))
        end_frame = max(start_frame, min(end_frame, total_frames))
        range_frames = end_frame - start_frame

        if fps is not None and fps > 0:
            interval = max(1, int(video_fps / fps))
            indices = list(range(start_frame, end_frame, interval))[:max_frames]
        elif range_frames <= max_frames:
            indices = list(range(start_frame, end_frame))
        else:
            indices = np.linspace(start_frame, end_frame - 1, max_frames, dtype=int).tolist()

        frames: List[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            if width or height:
                h, w = frame.shape[:2]
                nw = width or int(w * height / h)
                nh = height or int(h * width / w)
                frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        cap.release()
        return frames
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def parse_json_response(raw: str) -> Dict[str, Any]:
    match = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"raw": raw, "parse_error": True}


def parse_nested_json_response(raw: str) -> Dict[str, Any]:
    match = re.search(r"```json\s*(\{.*\})\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    depth, start = 0, None
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    start = None
    return {"raw": raw, "parse_error": True}


def judge_video_query_match(vlm: BaseVLMClient, frames: List[np.ndarray], query: str) -> Dict[str, Any]:
    system_prompt, user_prompt = build_video_query_match_prompt(query)
    sys_msg = text_message("system", system_prompt)
    vid_msg = frames_message(frames, encode_image)
    txt_msg = text_message("user", user_prompt)
    user_msg = merge_user_messages(vid_msg, txt_msg)

    resp = vlm.infer([sys_msg, user_msg])
    parsed = parse_json_response(resp["content"])
    parsed["prompt_tokens"] = resp["prompt_tokens"]
    parsed["response_tokens"] = resp["response_tokens"]
    return parsed


def judge_caption_score(vlm: BaseVLMClient, frames: List[np.ndarray], caption: str) -> Dict[str, Any]:
    system_prompt, user_prompt = build_caption_score_prompt(caption)
    sys_msg = text_message("system", system_prompt)
    vid_msg = frames_message(frames, encode_image)
    txt_msg = text_message("user", user_prompt)
    user_msg = merge_user_messages(vid_msg, txt_msg)

    resp = vlm.infer([sys_msg, user_msg])
    parsed = parse_nested_json_response(resp["content"])
    parsed["prompt_tokens"] = resp["prompt_tokens"]
    parsed["response_tokens"] = resp["response_tokens"]
    return parsed


def judge_arena_pairwise(
    vlm: BaseVLMClient,
    criteria: List[Dict],
    inputs: List[Dict],
    outputs_a: List[Dict],
    outputs_b: List[Dict],
) -> List[Dict[str, Any]]:
    """Judge a pairwise arena match. Entries have {name, type, label, content|frames}.

    Video/image entries should have 'frames' (list of np.ndarray).
    Text/json entries should have 'content' (str).

    *criteria* is a list of ``{"name": ..., "description": ...}`` dicts.
    Always returns a list of ``{criterion, vote, reasoning}`` dicts (one per criterion).
    """
    # Collect all frames with section labels for the visual message
    all_frames = []
    frame_labels = []

    for e in inputs:
        if "frames" in e and e["frames"]:
            frame_labels.append(f"--- Input: {e['label']} ({len(e['frames'])} frames) ---")
            all_frames.extend(e["frames"])

    for e in outputs_a:
        if "frames" in e and e["frames"]:
            frame_labels.append(f"--- Model A output: {e['label']} ({len(e['frames'])} frames) ---")
            all_frames.extend(e["frames"])

    for e in outputs_b:
        if "frames" in e and e["frames"]:
            frame_labels.append(f"--- Model B output: {e['label']} ({len(e['frames'])} frames) ---")
            all_frames.extend(e["frames"])

    # Build text descriptions
    def _describe_text(entries, fallback):
        parts = [f"[{e['label']}]: {e.get('content', '')}" for e in entries if e["type"] in ("text", "json")]
        return "\n".join(parts) if parts else f"(no text {fallback})"

    inputs_desc = _describe_text(inputs, "inputs")
    output_a_desc = _describe_text(outputs_a, "outputs")
    output_b_desc = _describe_text(outputs_b, "outputs")

    # Add frame layout note if there are visual entries
    if frame_labels:
        visual_note = "Visual content is provided as images in order:\n" + "\n".join(frame_labels)
        inputs_desc = visual_note + "\n\n" + inputs_desc

    # Build prompt
    criteria_desc = "\n".join(
        f"- **{c['name']}**: {c.get('description') or 'Judge which output is better overall.'}" for c in criteria
    )
    user_prompt = ARENA_PAIRWISE_TEMPLATE.format(
        criteria_description=criteria_desc,
        inputs_description=inputs_desc,
        output_a_description=output_a_desc,
        output_b_description=output_b_desc,
    )

    sys_msg = text_message("system", ARENA_PAIRWISE_SYSTEM)
    messages = [sys_msg]

    if all_frames:
        vid_msg = frames_message(all_frames, encode_image)
        txt_msg = text_message("user", user_prompt)
        messages.append(merge_user_messages(vid_msg, txt_msg))
    else:
        messages.append(text_message("user", user_prompt))

    resp = vlm.infer(messages)
    parsed = parse_nested_json_response(resp["content"])

    # Map (winner, confidence) → arena vote code
    vote_map = {
        ("a", "strong"): "a_strong", ("a", "moderate"): "a", ("a", "weak"): "tie",
        ("b", "strong"): "b_strong", ("b", "moderate"): "b", ("b", "weak"): "tie",
        ("tie", "strong"): "tie", ("tie", "moderate"): "tie", ("tie", "weak"): "tie",
    }

    def _to_vote(entry):
        w = str(entry.get("winner", "tie")).lower().strip()
        c = str(entry.get("confidence", "moderate")).lower().strip()
        return vote_map.get((w, c), "tie")

    # Parse into list of {criterion, vote, reasoning}
    results = []
    for c_entry in parsed.get("criteria", []):
        results.append({
            "criterion": c_entry.get("name", ""),
            "vote": _to_vote(c_entry),
            "reasoning": c_entry.get("reasoning", ""),
        })
    # If VLM missed some criteria, fill with tie
    returned_names = {r["criterion"] for r in results}
    for c in criteria:
        if c["name"] not in returned_names:
            results.append({"criterion": c["name"], "vote": "tie", "reasoning": "(no response from judge)"})
    return results


class VLMJudge:
    """VLM judge for the Wheel server."""

    def __init__(
        self,
        datastore,
        video_fetcher,
        provider: str = "auto",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        max_frames: int = 64,
        fps: float = 2.0,
        width: int = 1280,
        height: int = 720,
    ):
        self.datastore = datastore
        self.video_fetcher = video_fetcher
        # Build the VLM client via the factory so callers can pick between
        # nv_inference / openai / local_server / local without code changes.
        client_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if model is not None:
            client_kwargs["model"] = model
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self.vlm = get_vlm_client(provider=provider, **client_kwargs)
        self.max_frames = max_frames
        self.fps = fps
        self.width = width
        self.height = height
        self.caption_score_cache = LRUCache(maxsize=2000)

    def _load_frames(self, clip_id: str) -> List[np.ndarray]:
        video_path = self.datastore.get_video_path(clip_id)
        if not video_path:
            raise ValueError(f"No video path found for clip_id={clip_id}")
        video_bytes = self.video_fetcher.get_bytes(video_path)
        return extract_frames(
            video_bytes,
            max_frames=self.max_frames,
            fps=self.fps,
            width=self.width,
            height=self.height,
        )

    def score_caption(self, clip_id: str, caption: str, uid: int) -> Dict[str, Any]:
        frames = self._load_frames(clip_id)
        result = judge_caption_score(self.vlm, frames, caption)
        result["clip_id"] = clip_id
        self.caption_score_cache.put(
            (clip_id, uid),
            {
                "uid": uid,
                "scores": result.get("scores"),
                "reasoning": result.get("reasoning", "")
            },
        )
        return result

    def get_caption_scores_for_videos(
        self, captions_by_clip: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return cached caption scores structured to mirror the captions dict.

        Args:
            captions_by_clip: {clip_id: {model: [caption_text | {caption, ...}, ...]}}
        Returns:
            {clip_id: {model: [score_entry | None, ...]}}
        """
        result = {}
        for clip_id, captions in captions_by_clip.items():
            if not captions:
                continue
            entries = self.caption_score_cache.get_for_clip(clip_id)
            if not entries:
                continue
            score_by_uid = {e["uid"]: e for e in entries}
            result[clip_id] = {
                model: [
                    score_by_uid.get(item["uid"]) if isinstance(item, dict) else None
                    for item in (items if isinstance(items, list) else [items])
                ]
                for model, items in captions.items()
            }
        return result

    def match_query(self, clip_id: str, query: str) -> Dict[str, Any]:
        frames = self._load_frames(clip_id)
        out = judge_video_query_match(self.vlm, frames, query)
        out["clip_id"] = clip_id
        out["match"] = bool(out.get("prediction", 0))
        out.setdefault("reasoning", out.get("reason", "") or out.get("analysis", ""))
        return out

    def judge_arena_match(
        self,
        arena_store,
        arena_name: str,
        manifest: dict,
        item_id: str,
        model_a: str,
        model_b: str,
        max_frames_per_video: int = 64,
        max_total_frames: int = 256,
    ) -> List[Dict[str, Any]]:
        """Judge a pairwise arena match using the VLM.

        Fetches assets from S3 via arena_store, extracts frames from
        video/image assets, and runs pairwise comparison.
        Returns list of {criterion, vote, reasoning} dicts.
        """
        assets = arena_store.load_match_assets(
            arena_name, manifest, item_id, model_a, model_b
        )

        # Count total video entries to scale frames per video within budget
        all_entries = assets["inputs"] + assets["outputs_a"] + assets["outputs_b"]
        num_videos = sum(1 for e in all_entries if e["type"] == "video" and e.get("bytes"))
        if num_videos > 0:
            frames_per_video = min(max_frames_per_video, max_total_frames // num_videos)
            frames_per_video = max(frames_per_video, 4)  # at least 4 frames
        else:
            frames_per_video = max_frames_per_video

        def _prepare_entries(entries):
            prepared = []
            for e in entries:
                entry = {"name": e["name"], "type": e["type"], "label": e["label"]}
                if e["type"] in ("text", "json"):
                    entry["content"] = e.get("content", "")
                elif e["type"] == "video" and e.get("bytes"):
                    entry["frames"] = extract_frames(
                        e["bytes"],
                        max_frames=frames_per_video,
                        fps=2.0,
                        width=self.width,
                        height=self.height,
                        start_time=e.get("start_time"),
                        end_time=e.get("end_time"),
                    )
                elif e["type"] == "image" and e.get("bytes"):
                    img = cv2.imdecode(
                        np.frombuffer(e["bytes"], np.uint8), cv2.IMREAD_COLOR
                    )
                    if img is not None:
                        entry["frames"] = [img]
                prepared.append(entry)
            return prepared

        inputs = _prepare_entries(assets["inputs"])
        outputs_a = _prepare_entries(assets["outputs_a"])
        outputs_b = _prepare_entries(assets["outputs_b"])

        return judge_arena_pairwise(
            self.vlm, assets["criteria"], inputs, outputs_a, outputs_b
        )

