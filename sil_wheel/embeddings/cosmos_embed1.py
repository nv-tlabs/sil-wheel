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

import numpy as np
import torch

# --- HF compatibility shim: restore moved helpers for old code ---
import transformers
from packaging import version

if version.parse(transformers.__version__) >= version.parse("4.42"):
    try:
        import transformers.pytorch_utils as _pt_utils
        import transformers.modeling_utils as _mu
        for _name in (
            "apply_chunking_to_forward",
            "find_pruneable_heads_and_indices",
            "prune_linear_layer",
        ):
            if not hasattr(_mu, _name) and hasattr(_pt_utils, _name):
                setattr(_mu, _name, getattr(_pt_utils, _name))
    except Exception:
        pass
# ----------------------------------------------------------------
from transformers import AutoProcessor, AutoModel


class CosmosEmbed1(torch.nn.Module):
    """A wrapper for the Cosmos-Embed1
    https://huggingface.co/nvidia/Cosmos-Embed1

    Cosmos Embed1 is a joint video-text embedder tailored for physical AI.
    It is optimized with 8 frames with 224x224 input resolution, 256-dim output
    text and video embeddings.
    """
    def __init__(self, model_type):
        super().__init__()
        embedder = {
            "cosmos_embed1_224p": "Cosmos-Embed1-224p",
            "cosmos_embed1_336p": "Cosmos-Embed1-336p",
            "cosmos_embed1_448p": "Cosmos-Embed1-448p"
        }[model_type]
        # load model and pre-processor
        self.model = AutoModel.from_pretrained(
            f"nvidia/{embedder}", trust_remote_code=True, token=True
        ).to("cuda", dtype=torch.bfloat16)
        self.model.eval()
        self.preprocess = AutoProcessor.from_pretrained(
            f"nvidia/{embedder}", trust_remote_code=True
        )

    @torch.no_grad()
    def get_video_embeddings(self, batch: np.ndarray) -> torch.Tensor:
        assert len(batch.shape) == 5
        assert batch.dtype == "uint8"
        video_inputs = self.preprocess(videos=batch).to("cuda", dtype=torch.bfloat16)
        video_out = self.model.get_video_embeddings(**video_inputs)
        return video_out.visual_proj.detach().to("cpu", dtype=torch.float32).numpy()

    @torch.no_grad()
    def get_text_embeddings(self, text: str) -> torch.Tensor:
        text_inputs = self.preprocess(text=text).to("cuda", dtype=torch.bfloat16)
        text_out = self.model.get_text_embeddings(**text_inputs)
        return text_out.text_proj
