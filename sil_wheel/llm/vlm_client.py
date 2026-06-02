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
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI

from sil_wheel.llm.base import BaseVLMClient
from sil_wheel.llm.vlm_messages import (
    frames_message,
    merge_user_messages,
    text_message,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 10
RETRY_WAIT = 30


@dataclass
class VLMConfig:
    """Configuration for VLM client."""

    provider: str = "openai"  # "openai" | "local_server"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.0

    _provider_configs: Dict[str, Dict] = field(
        default_factory=lambda: {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "env_key": "OPENAI_API_KEY",
                "default_model": "gpt-4o-mini",
            },
            "local_server": {
                # Any OpenAI-compatible local VLM server (vLLM, Ollama,
                # LM Studio, llama.cpp). Placeholder URL; user passes
                # base_url=... since each backend defaults to a different port.
                "base_url": "http://localhost:1234/v1",
                "default_model": None,
            },
        }
    )

    def __post_init__(self):
        config = self._provider_configs.get(self.provider, {})

        if not self.base_url:
            self.base_url = config.get("base_url")

        if not self.api_key:
            env_var = config.get("env_key")
            if env_var:
                self.api_key = os.environ.get(env_var)

        # Local servers ignore auth but the HTTP client needs a non-empty
        # Bearer token, so ship a placeholder.
        if not self.api_key and self.provider == "local_server":
            self.api_key = "not-needed"

        if not self.model:
            self.model = config.get("default_model")

    @property
    def default_headers(self) -> Dict[str, str]:
        return self._provider_configs.get(self.provider, {}).get(
            "default_headers", {}
        )


class VLMClient(BaseVLMClient):
    """OpenAI-compatible client for remote (or local-server) VLM endpoints."""

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        self.config = VLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if not self.config.api_key:
            raise ValueError(
                f"No API key for VLMClient (provider={provider}). "
                f"Set the appropriate env var or pass api_key=."
            )

        self.model = self.config.model
        self.max_tokens = self.config.max_tokens
        self.temperature = self.config.temperature
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            default_headers=self.config.default_headers or None,
        )

    def infer(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        tokens = max_tokens or self.max_tokens
        temp = temperature if temperature is not None else self.temperature

        raw = None
        for attempt in range(MAX_RETRIES):
            try:
                raw = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=tokens,
                    temperature=temp,
                )
                break
            except Exception as exc:
                logger.warning(
                    "VLMClient retry %d/%d: %s", attempt + 1, MAX_RETRIES, exc
                )
                time.sleep(RETRY_WAIT)
        if raw is None:
            raise RuntimeError(
                f"VLMClient: no response after {MAX_RETRIES} retries"
            )

        choice = raw.choices[0]
        return {
            "content": choice.message.content,
            "finish_reason": choice.finish_reason,
            "prompt_tokens": raw.usage.prompt_tokens,
            "response_tokens": raw.usage.total_tokens - raw.usage.prompt_tokens,
        }


def get_vlm_client(provider: str = "auto", **kwargs) -> BaseVLMClient:
    """Factory for a VLM client.

    Provider resolution when ``provider="auto"``:
      1. openai        (if OPENAI_API_KEY is set)
      2. local         (in-process transformers via LocalVLMClient)

    Explicit providers: "openai", "local_server", "local", or any string
    accepted by ``VLMClient`` with explicit ``base_url=`` and ``model=``.
    """
    # Imported lazily because LocalVLMClient pulls in heavy torch/transformers
    # imports at construction time.
    from sil_wheel.llm.local_vlm_client import LocalVLMClient

    if provider == "auto":
        for name, env_key in [
            ("openai", "OPENAI_API_KEY"),
        ]:
            if os.environ.get(env_key):
                logger.info(
                    "Auto-selected VLM provider: %s (via %s)", name, env_key
                )
                return VLMClient(provider=name, **kwargs)
        logger.info("No cloud API key found, falling back to local VLM.")
        return LocalVLMClient(**kwargs)

    if provider == "local":
        return LocalVLMClient(**kwargs)

    return VLMClient(provider=provider, **kwargs)
