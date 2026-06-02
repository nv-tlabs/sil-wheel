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

"""Shared argparse blocks for the data-prep / extraction scripts."""


def add_hf_dataset_args(parser):
    """Add the three flags every processing script needs to point a
    ``dataset_factory`` call at a HuggingFace-hosted dataset.
    """
    parser.add_argument(
        "--hf-repo-id",
        default=None,
        type=str,
        help="HuggingFace dataset repo id (e.g. facebook/PE-Video). "
             "Mutually exclusive with --bucket; makes path_to_data "
             "optional.",
    )
    parser.add_argument(
        "--hf-allow-patterns",
        nargs="+",
        default=None,
        help="HuggingFace allow_patterns globs to restrict the "
             "download (e.g. test/*). Defaults to the whole repo.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        type=str,
        help="HuggingFace cache directory; defaults to $HF_HOME.",
    )
