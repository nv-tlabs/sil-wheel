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

"""Factory for structured caption modes."""

from sil_wheel.captions import sil_av_benchmark, comprehensive_v3


def get_caption_mode(name: str) -> dict:
    """Return the caption mode dict for the given name.

    Available modes: 'sil_av_benchmark', 'comprehensive_v3'.
    """
    modes = {
        "sil_av_benchmark": sil_av_benchmark.mode,
        "comprehensive_v3": comprehensive_v3.mode,
    }
    if name not in modes:
        raise ValueError(
            f"Unknown caption mode: '{name}'. "
            f"Available: {sorted(modes)}"
        )
    return modes[name]

