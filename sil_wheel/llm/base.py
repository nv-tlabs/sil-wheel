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
import logging
import re

logger = logging.getLogger(__name__)


class BaseLLMClient:
    """Abstract interface that every LLM backend must implement.

    Concrete subclasses:
      - LLMClient        – calls remote OpenAI-compatible APIs (NVIDIA, OpenAI, Azure).
      - LocalLLMClient   – runs a model on-device via HuggingFace transformers.

    Use the ``get_llm_client`` factory in ``llm_client.py`` to obtain the
    appropriate implementation based on the provider name or available API keys.
    """

    def generate(
        self,
        prompt,
        system_prompt,
        response_format=None,
        temperature=None,
        max_tokens=None,
    ):
        """Send a prompt to the model and return the generated text.

        Args:
            prompt: The user message to send to the model.
            system_prompt: A system-level instruction that guides the model's
                behaviour (e.g. persona, output constraints).
            response_format: Optional dict requesting a specific output format
                (e.g. ``{"type": "json_object"}``). Support varies by provider.
            temperature: Sampling temperature override. When *None*, the
                client's default temperature is used.
            max_tokens: Maximum number of tokens to generate. When *None*, the
                client's default limit is used.

        Returns:
            The raw text content of the model's response.
        """
        raise NotImplementedError

    def parse_json(self, text):
        """Extract a JSON object from raw model output.

        Handles three common wrappers:
          - ``<think>…</think>`` reasoning blocks emitted by Qwen3 / DeepSeek-R1
            -family models before the actual answer.
          - Markdown code fences (````json … ```` or plain ```` ``` ````).
          - Trailing whitespace.

        If those still don't yield valid JSON, falls back to extracting the
        first balanced ``{…}`` substring.

        Args:
            text: Raw string returned by ``generate``.

        Returns:
            A ``dict`` with the parsed JSON, or an empty ``dict`` ``{}`` if
            parsing fails.
        """
        text = (text or "").strip()
        # Strip any leading <think>...</think> reasoning blocks.
        text = re.sub(r"^<think>.*?</think>\s*", "", text, flags=re.DOTALL)

        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Last-ditch: pull the first balanced {...} object out of the
            # remaining text. Handles trailing model commentary after JSON.
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            logger.error("Failed to parse JSON. Text: %s", text)
            return {}


class BaseVLMClient:
    """Abstract interface for vision-language model backends.

    Concrete subclasses:
      - VLMClient        – OpenAI-compatible remote APIs (NVIDIA, OpenAI, or any
                           local server such as vLLM / Ollama / LM Studio).
      - LocalVLMClient   – runs a VLM on-device via HuggingFace transformers.

    Use ``get_vlm_client`` in ``vlm_client.py`` to obtain a client based on the
    provider name or available API keys. Construct message dicts with the free
    functions in ``vlm_messages`` (``text_message``, ``frames_message``,
    ``merge_user_messages``).
    """

    def infer(self, messages, max_tokens=None, temperature=None):
        """Run inference on a list of multimodal messages.

        Args:
            messages: A list of OpenAI-format chat messages. Each user
                content entry is either ``{"type": "text", "text": ...}`` or
                ``{"type": "image_url", "image_url": {"url": "data:image/...;
                base64,..."}}``.
            max_tokens: Override the client default cap on response tokens.
            temperature: Override the client default sampling temperature.

        Returns:
            A dict with keys ``content``, ``finish_reason``,
            ``prompt_tokens``, ``response_tokens``.
        """
        raise NotImplementedError
