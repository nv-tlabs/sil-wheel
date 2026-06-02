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

"""Helpers to build the OpenAI-compatible multimodal message format used by
every VLM client. Module-level so both remote ``VLMClient`` and
``LocalVLMClient`` can build messages without circular imports."""
from typing import Any, Dict, List

import numpy as np


def text_message(role: str, text: str) -> Dict[str, Any]:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def frames_message(frames: List[np.ndarray], encode_image_fn) -> Dict[str, Any]:
    content = []
    for frame in frames:
        b64 = encode_image_fn(frame)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return {"role": "user", "content": content}


def merge_user_messages(*msgs: Dict[str, Any]) -> Dict[str, Any]:
    combined = []
    for m in msgs:
        combined.extend(m["content"])
    return {"role": "user", "content": combined}
