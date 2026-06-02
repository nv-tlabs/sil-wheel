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
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sil_wheel.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class LocalLLMClient(BaseLLMClient):
    """LLM client that runs a model locally via HuggingFace transformers."""

    def __init__(self, model=None, max_tokens=8192, temperature=0.7):
        # Default budget sized for Qwen3-family thinking models: a few thousand
        # reasoning tokens plus the final answer. Override per-call if needed.
        self.model_name = model or "Qwen/Qwen3-0.6B"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None
        # Load the weights now so the first request doesn't pay the cost.
        self._ensure_loaded()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        logger.info(
            "Loading local model %s (first call, may take a moment)...",
            self.model_name,
        )
        t0 = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self._model.eval()
        logger.info(
            "Local model %s loaded in %.1fs.",
            self.model_name, time.perf_counter() - t0,
        )

    def generate(
        self,
        prompt,
        system_prompt,
        response_format=None,
        temperature=None,
        max_tokens=None,
    ):
        self._ensure_loaded()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self._model.device)

        prompt_len = inputs["input_ids"].shape[1]
        max_new = max_tokens or self.max_tokens
        temp = temperature if temperature is not None else self.temperature
        logger.info(
            "LocalLLMClient.generate: prompt=%d tokens, max_new=%d, temp=%.2f",
            prompt_len, max_new, temp,
        )
        t0 = time.perf_counter()
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=temp,
                do_sample=temp > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0, prompt_len:]
        elapsed = time.perf_counter() - t0
        logger.info(
            "LocalLLMClient.generate: produced %d tokens in %.1fs (%.1f tok/s)",
            new_tokens.shape[0], elapsed,
            new_tokens.shape[0] / max(elapsed, 1e-6),
        )
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

