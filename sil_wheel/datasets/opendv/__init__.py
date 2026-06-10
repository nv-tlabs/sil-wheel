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

"""OpenDV-YouTube acquisition and 20 s clip sampling for text→video retrieval.

Local-storage producer: ``download`` fetches videos from YouTube to disk and
``sampling`` cuts annotated 20 s clips. Outputs feed sil-wheel's existing
``scripts/prepare_data.py`` ingestion (which handles S3, compression, etc.).
"""
