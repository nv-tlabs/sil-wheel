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

"""Static configuration for the OpenDV-YouTube dataset."""

# Public OpenDV-YouTube Google Sheet (the dataset source of truth).
SHEET_ID = "1bHWWP_VXeEe5UzIG-QgKFBdH7mNlSC4GFSJkEhFnt2I"
SHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

# OpenDV-YouTube-Language annotations (HuggingFace).
LANG_REPO = "OpenDriveLab/OpenDV-YouTube-Language"
LANG_FILES = [f"10hz_YouTube_train_split{i}.json" for i in range(10)] + ["10hz_YouTube_val.json"]
FPS = 10
DEFAULT_BIN_SEC = 5

CLIP_SEC = 20
DEFAULT_INTERVAL = 60   # uniform: seconds between clip starts
DEFAULT_STRIDE = 10     # diverse: candidate window stride (s)

# Command (maneuver) category -> natural language (upstream utils/cmd2caption.py).
CMD_TEXT = {
    0: "go straight", 1: "pass the intersection", 2: "turn left", 3: "turn right",
    4: "change to the left lane", 5: "change to the right lane",
    6: "take the left lane branch", 7: "take the right lane branch",
    8: "pass the crosswalk", 9: "pass the railroad", 10: "merge",
    11: "make a U-turn", 12: "stop", 13: "deviate",
}
