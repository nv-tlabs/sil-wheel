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

import sys
from pathlib import Path

# caption_quality/ -> sibling modules (scorers, metrics, caption_quality, ...).
CAPTION_QUALITY_DIR = Path(__file__).resolve().parent.parent
# repo root -> the `sil_wheel` package, imported by run_caption_eval.py.
REPO_ROOT = CAPTION_QUALITY_DIR.parent.parent

for p in (CAPTION_QUALITY_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
