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

"""Load a SentenceTransformer with a zero-weight sanity check.

When the HuggingFace cache contains truncated or zero-byte safetensors
shards (common after an interrupted download, an OOM/disk-full event, or
two processes racing on ``from_pretrained`` for the same blob), the
model architecture loads fine but every weight tensor is zero. The
result is a model that encodes anything to the zero vector — no
exception, no warning, just silently broken downstream pipelines. This
helper catches that with a tiny encode probe and raises a clear error
telling the caller how to recover.
"""
import numpy as np
from sentence_transformers import SentenceTransformer


def load_sentence_transformer(model_name_or_path, **kwargs) -> SentenceTransformer:
    """Construct a ``SentenceTransformer`` and verify it has live weights.

    Forwards all kwargs to ``SentenceTransformer(...)``. Raises
    ``RuntimeError`` if the loaded model encodes a probe string to a
    zero-norm vector — the signature of a corrupted HF cache.
    """
    model = SentenceTransformer(model_name_or_path, **kwargs)
    probe = model.encode(["sanity check"], normalize_embeddings=False)
    if not np.linalg.norm(probe):
        raise RuntimeError(
            f"SentenceTransformer({model_name_or_path!r}) loaded with zero "
            "weights — the HuggingFace cache for this model is most likely "
            "corrupted (truncated or zero-byte safetensors shards from an "
            "interrupted download). Clear "
            "~/.cache/huggingface/hub/models--<org>--<name>/ for this model "
            "and retry, or pass force_download=True on the next load."
        )
    return model
