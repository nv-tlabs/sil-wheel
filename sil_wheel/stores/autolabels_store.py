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

class AutolabelsDataStore:
    def __init__(self, clips_to_apis):
        self.clip_ids = []
        for clip, apis in clips_to_apis.items():
            if "Autolabels" in apis:
                self.clip_ids.append(clip)
        self.clip_ids = set(self.clip_ids)
        print(f"Loaded {len(self.clip_ids)} with autolabels")

    def get_depth_path(self, clip_id):
        if clip_id not in self.clip_ids:
            return None
        return f"depth_videos/{clip_id}_depth_with_mask_360p.mp4"

    def get_boxes_path(self, clip_id):
        if clip_id not in self.clip_ids:
            return None
        return f"bboxes_videos/{clip_id}_boxes.mp4"

    def get_pointmap_path(self, clip_id):
        if clip_id not in self.clip_ids:
            return None
        return f"pointmap_videos/{clip_id}_pointmap.mp4"

    def get_mfmr_path(self, clip_id):
        if clip_id not in self.clip_ids:
            return None
        return f"mfmhmr_videos/{clip_id}_mfmhmr_all.mp4"

    def get_vipe_path(self, clip_id):
        return f"vipe_videos/{clip_id}_vipe.mp4"
