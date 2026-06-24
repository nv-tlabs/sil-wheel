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

import importlib.util
from pathlib import Path


def _load_setup_module():
    setup_path = Path(__file__).resolve().parents[1] / "setup.py"
    spec = importlib.util.spec_from_file_location("wheel_setup", setup_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_caption_quality_dependencies_are_extra_only():
    setup_module = _load_setup_module()
    caption_quality_deps = {"pycocoevalcap", "nltk", "rouge-score", "bert-score"}

    install_requires = set(setup_module.get_install_requirements())
    extras = setup_module.get_extras_require()

    assert caption_quality_deps.isdisjoint(install_requires)
    assert caption_quality_deps <= set(extras["caption-quality"])
