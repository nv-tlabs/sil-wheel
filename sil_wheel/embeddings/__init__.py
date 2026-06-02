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
from .cosmos_embed1 import CosmosEmbed1
from .pe_core import PECore
from .qwen3_vl_embed import Qwen3VLEmbed


def get_embedding_model(**kwargs) -> torch.nn.Module:
    model_classes = {
        "cosmos_embed1_224p": CosmosEmbed1,
        "cosmos_embed1_336p": CosmosEmbed1,
        "cosmos_embed1_448p": CosmosEmbed1,
        "qwen3_vl_embed_2b": Qwen3VLEmbed,
        "qwen3_vl_embed_8b": Qwen3VLEmbed,
        "pe_core_b16_224p": PECore,
        "pe_core_l14_336p": PECore,
        "pe_core_g14_448p": PECore,
    }
    if kwargs["model_type"] not in model_classes:
        raise ValueError(f"Unknown embedding type: {kwargs['model_type']}")
    return model_classes[kwargs["model_type"]](**kwargs)
