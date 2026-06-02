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
from PIL import Image

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as pe_transforms


class PECore(torch.nn.Module):
    """Wrapper for Meta's Perception Encoder Core (PE-Core) CLIP models.

    PE-Core is a contrastive image-text encoder. Videos are embedded by
    encoding each frame independently with the image tower and mean-pooling
    across the time dimension before L2 normalization. Weights are pulled
    from the Hugging Face Hub via `pe.CLIP.from_config(..., pretrained=True)`.
    """

    MODEL_MAP = {
        "pe_core_b16_224p": "PE-Core-B16-224",
        "pe_core_l14_336p": "PE-Core-L14-336",
        "pe_core_g14_448p": "PE-Core-G14-448",
    }

    def __init__(self, model_type="pe_core_g14_448p", **kwargs):
        super().__init__()
        config_name = self.MODEL_MAP[model_type]

        self.model = pe.CLIP.from_config(config_name, pretrained=True).cuda()
        self.model.eval()

        self.preprocess = pe_transforms.get_image_transform(self.model.image_size)
        self.tokenizer = pe_transforms.get_text_tokenizer(self.model.context_length)

    @torch.no_grad()
    def get_video_embeddings(self, batch: np.ndarray) -> np.ndarray:
        """Embed video from BTCHW uint8 numpy array.

        Args:
            batch: shape (B, T, C, H, W) or (T, C, H, W) uint8.
        """
        if batch.ndim == 4:
            batch = batch[None]

        # (B, T, C, H, W) -> (B, T, H, W, C)
        batch = batch.transpose(0, 1, 3, 4, 2)
        B, T = batch.shape[:2]

        frames = [Image.fromarray(f) for clip in batch for f in clip]
        pixels = torch.stack(
            [self.preprocess(im) for im in frames]
        ).cuda()

        with torch.autocast("cuda"):
            feats = self.model.encode_image(pixels)

        feats = feats.view(B, T, -1).mean(dim=1)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.detach().cpu().to(torch.float32).numpy()

    @torch.no_grad()
    def get_text_embeddings(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).cuda()
        with torch.autocast("cuda"):
            feats = self.model.encode_text(tokens)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.detach().cpu().to(torch.float32).numpy()
