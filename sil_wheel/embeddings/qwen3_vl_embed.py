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

import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)


class Qwen3VLEmbed(torch.nn.Module):
    DEFAULT_VIDEO_INSTRUCTION = "Represent the user's input."
    DEFAULT_TEXT_INSTRUCTION = "Find a video that contains the following visual content."

    MODEL_MAP = {
        "qwen3_vl_embed_2b": "Qwen/Qwen3-VL-Embedding-2B",
        "qwen3_vl_embed_8b": "Qwen/Qwen3-VL-Embedding-8B",
    }

    def __init__(self, model_type="qwen3_vl_embed_8b", model_assets_dir=None, **kwargs):
        super().__init__()

        hf_name = self.MODEL_MAP.get(model_type, model_type)
        model_path = str(model_assets_dir) if model_assets_dir else hf_name

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
        ).cuda()
        self.processor = AutoProcessor.from_pretrained(
            model_path,
        )
        self.model.eval()

    def _build_messages(self, content, instruction):
        """Build the chat template the model was trained on.

        The paper (Section 3.1) defines the embedding template as:
            <|im_start|>system\n{instruction}<|im_end|>\n
            <|im_start|>user\n{content}<|im_end|>\n
            <|im_start|>assistant\n
        The hidden state at the last token position is the embedding.
        """
        return [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _pooling_last(hidden_state, attention_mask):
        """Extract hidden state at the last non-padding token.

        Matches the official Qwen3VLEmbedder._pooling_last() implementation.
        """
        flipped = attention_mask.flip(dims=[1])
        last_pos = flipped.argmax(dim=1)
        col = attention_mask.shape[1] - last_pos - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    def _embed(self, messages, videos=None, images=None):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        proc_kwargs = dict(text=[text], padding=True, return_tensors="pt")
        if videos is not None:
            proc_kwargs["videos"] = videos
        if images is not None:
            proc_kwargs["images"] = images

        inputs = self.processor(**proc_kwargs).to(self.model.device)

        outputs = self.model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        embeddings = self._pooling_last(hidden, inputs["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().to(torch.float32).numpy()

    @torch.no_grad()
    def get_video_embeddings(self, batch: np.ndarray) -> np.ndarray:
        """Embed video from BTCHW uint8 numpy array.

        Args:
            batch: shape (B, T, C, H, W) or (T, C, H, W) uint8.

        The underlying processor encodes one video at a time, so multi-clip
        batches are looped internally and the results stacked.
        """
        if batch.ndim == 4:
            batch = batch[None]

        # (B, T, C, H, W) -> (B, T, H, W, C)
        batch = batch.transpose(0, 1, 3, 4, 2)

        outs = []
        for clip in batch:
            frames = [Image.fromarray(f) for f in clip]
            content = [{"type": "video", "video": frames}]
            messages = self._build_messages(content, self.DEFAULT_VIDEO_INSTRUCTION)
            outs.append(self._embed(messages, videos=[frames]))

        return np.concatenate(outs, axis=0)

    @torch.no_grad()
    def get_text_embeddings(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        content = [{"type": "text", "text": text}]
        messages = self._build_messages(content, self.DEFAULT_TEXT_INSTRUCTION)
        return self._embed(messages)

    @torch.no_grad()
    def get_image_embeddings(self, image_input: np.ndarray) -> np.ndarray:
        """Embed an image from (H, W, C) uint8 numpy array."""
        image = Image.fromarray(image_input)
        content = [{"type": "image", "image": image}]
        messages = self._build_messages(content, self.DEFAULT_VIDEO_INSTRUCTION)
        return self._embed(messages, images=[image])
