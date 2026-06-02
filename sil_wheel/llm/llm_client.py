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
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests

from sil_wheel.llm.base import BaseLLMClient
from sil_wheel.llm.local_llm_client import LocalLLMClient

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    provider: str = "openai"  # "openai", "azure", "local"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.7

    # Provider-specific defaults
    _provider_configs: Dict[str, Dict] = field(
        default_factory=lambda: {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "env_key": "OPENAI_API_KEY",
                "default_model": "gpt-4o-mini",
            },
            "local_server": {
                # Any OpenAI-compatible local server (Ollama, MLX-LM,
                # LM Studio, vLLM, llama.cpp). The base_url here is just a
                # placeholder; users pass base_url=... since each backend
                # defaults to a different port.
                "base_url": "http://localhost:1234/v1",
                "default_model": None,
            },
        }
    )

    def __post_init__(self):
        """Initialize defaults based on provider."""
        config = self._provider_configs.get(self.provider, {})
        
        # Set defaults if not provided
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

class LLMClient(BaseLLMClient):
    """Simple LLM Client to handle interactions with OpenAI-compatible APIs."""
    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.config = LLMConfig(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Strip trailing slash from base_url if present
        if self.config.base_url and self.config.base_url.endswith("/"):
            self.config.base_url = self.config.base_url[:-1]

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        response_format: Optional[Dict[str, str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using the LLM."""
        if not self.config.api_key:
            logger.warning(
                f"No API key provided for LLMClient (provider: {self.config.provider}). "
                f"Returning empty response."
            )
            return "{}"

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        # Handle response_format
        if response_format:
            payload["response_format"] = response_format

        try:
            url = f"{self.config.base_url}/chat/completions"
            # (connect_timeout, read_timeout) — fail fast on unreachable hosts,
            # give the LLM a reasonable window to generate a response.
            response = requests.post(
                url, headers=headers, json=payload, timeout=(10, 30)
            )
            
            if response.status_code != 200:
                logger.error(f"LLM API Error: {response.status_code} - {response.text}")
                response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return content
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise e


def get_llm_client(provider="auto", **kwargs):
    """Factory function to get an LLM client.

    Provider resolution when provider="auto":
      1. openai        (if OPENAI_API_KEY is set)
      2. local         (mlx-lm fallback, always available)

    Explicit providers: "openai", "local", or "custom" with
    base_url= and model=.
    """

    api_providers = [
        ("openai", "OPENAI_API_KEY"),
    ]
    if provider == "auto":
        for name, env_key in api_providers:
            if os.environ.get(env_key):
                logger.info("Auto-selected LLM provider: %s (via %s)", name, env_key)
                return LLMClient(provider=name, **kwargs)
        logger.info("No cloud API key found, falling back to local LLM.")
        return LocalLLMClient(**kwargs)

    if provider == "local":
        return LocalLLMClient(**kwargs)

    return LLMClient(provider=provider, **kwargs)
