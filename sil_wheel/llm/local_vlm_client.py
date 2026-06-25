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

import base64
import io
import logging
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    # Optional local-backend dependency. Importing this module (and the
    # LocalVLMClient class) stays cheap so the cloud providers in
    # get_vlm_client don't require it; constructing LocalVLMClient raises
    # a clear error below if it's genuinely missing.
    process_vision_info = None

from sil_wheel.llm.base import BaseVLMClient

logger = logging.getLogger(__name__)


def _decode_image_url(url: str) -> Image.Image:
    """Turn an OpenAI-style ``data:image/...;base64,<...>`` URL into a PIL image."""
    if not url.startswith("data:image/"):
        raise ValueError(
            f"LocalVLMClient only accepts data: image URLs, got {url[:64]!r}"
        )
    _, b64 = url.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _to_qwen_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Translate OpenAI-format messages into the Qwen3-VL chat format.

    ``{"type": "image_url", "image_url": {"url": "data:..."}}`` becomes
    ``{"type": "image", "image": <PIL.Image>}``; text entries pass through.
    """
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        new_content = []
        for entry in content:
            if entry.get("type") == "image_url":
                new_content.append({
                    "type": "image",
                    "image": _decode_image_url(entry["image_url"]["url"]),
                })
            else:
                new_content.append(entry)
        out.append({"role": m["role"], "content": new_content})
    return out


class LocalVLMClient(BaseVLMClient):
    """VLM that runs in-process via HuggingFace transformers.

    Default model is Qwen3-VL-4B-Instruct; pick a larger family member by
    passing ``model=``. Used by the ``get_vlm_client`` factory's fallback
    path when no cloud API key is available.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-4B-Instruct",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._model = None
        self._processor = None
        # Load the weights now so the first request doesn't pay the cost.
        self._ensure_loaded()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        if process_vision_info is None:
            raise ImportError(
                "LocalVLMClient requires the 'qwen_vl_utils' package. Install it "
                "with `pip install qwen-vl-utils`, or use a cloud provider by "
                "setting NV_INFERENCE_API_KEY or OPENAI_API_KEY."
            )
        logger.info(
            "Loading local VLM %s (first call, may take a moment)...",
            self.model_name,
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self._model.eval()
        logger.info("Local VLM %s loaded.", self.model_name)

    def infer(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        self._ensure_loaded()

        qwen_messages = _to_qwen_messages(messages)
        text = self._processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(qwen_messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        tokens = max_tokens or self.max_tokens
        temp = temperature if temperature is not None else self.temperature
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=tokens,
                temperature=temp,
                do_sample=temp > 0,
            )

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0, prompt_len:]
        content = self._processor.batch_decode(
            [new_tokens], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        # max_new_tokens hit vs. natural EOS — mirror the OpenAI semantics.
        finish_reason = "length" if new_tokens.shape[0] >= tokens else "stop"
        return {
            "content": content,
            "finish_reason": finish_reason,
            "prompt_tokens": int(prompt_len),
            "response_tokens": int(new_tokens.shape[0]),
        }
