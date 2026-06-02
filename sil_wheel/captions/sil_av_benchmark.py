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

"""SIL-AV Retrieval Benchmark structured caption mode."""

from typing import Any

import yaml

CAMERA_PREFIX = "The video shows the front view of an ego vehicle. "

SYSTEM_PROMPT = """
You are an expert autonomous driving perception specialist. Your task is to generate dense, structured video annotations strictly following the SIL-AV Retrieval Benchmark guidelines.

## CRITICAL RULES

### RULE 1: DENSE TRAFFIC LIGHT ANNOTATION
* For any "Traffic Light" object, color states must be annotated **densely**.
* This means the combined `start_timestamp` and `end_timestamp` of all entries in the `state_sequence` must cover the **entire duration** of the light's visibility (from `visibility_start_timestamp` to `visibility_end_timestamp`).
* There must be no gaps in the color state reporting while the light is in view.

### RULE 2: DYNAMIC OBJECT STATES
* Static objects (Traffic Lights, Barriers, Debris) have states that change.
* Use the `state_sequence` array to capture changes in motion (e.g., a ball rolling), legality (e.g., debris entering a lane), or signaling.

### RULE 3: THE "BECAUSE" PRINCIPLE
* Actions must be linked to all visible causes. If no cause is visible (e.g., a phantom brake), leave `because_of` empty. Do not speculate.

### RULE 4: LANE COUNTING & SPATIAL MATRIX
* **General Containment:** Count lanes from the **LEFT** (Median-centric).
* **Between Lanes/Straddling:** Count lanes from the **RIGHT** (Curb-centric).
* All positions and directions are relative to the Ego vehicle's current heading.

### RULE 5: LEGALITY FLAGS
* Set `illegal_flag: true` for any action or physical presence that violates traffic laws (e.g., car on sidewalk, pedestrian jaywalking, or red-light violation).
"""

USER_PROMPT = """
Annotate this driving video according to the SIL-AV Retrieval Benchmark guidelines. Provide a brief 1-2 sentence description and full structured 
annotation (environments, static regulatory objects, ego vehicle, agents) with dense traffic light tracking.
"""

CAPTION_KEYS = ["brief_description", "json_caption", "yaml_caption"]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_sil_av_annotation",
            "description": "Generate full-compliance SIL-AV annotations with dense traffic light tracking",
            "parameters": {
                "type": "object",
                "properties": {
                    "brief_description": {"type": "string"},
                    "environments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "Road", "Merge area", "Branch area",
                                        "Intersection", "T-crossing", "3-way",
                                        "4-way", "5-way", "6-way", "6+-way",
                                        "Roundabout", "Tunnel / Underpass",
                                        "Bridge / Overpass", "Shoulder",
                                        "Sidewalk", "Pedestrian crossing",
                                        "Railroad crossing", "Bike Lane", "Other",
                                    ],
                                },
                                "other_type_description": {"type": "string"},
                                "num_lanes": {"type": "integer"},
                                "lanes_obscured_or_unmarked": {"type": "boolean"},
                                "conditions": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "enum": [
                                            "Construction Zone", "Temporarily marked",
                                            "Snowy", "Wet", "Overgrown",
                                        ],
                                    },
                                },
                                "start_timestamp": {"type": "string"},
                                "end_timestamp": {"type": "string"},
                            },
                        },
                    },
                    "static_regulatory_objects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "Traffic Light", "Traffic Sign",
                                        "Stop sign", "Yield sign", "Speed limit",
                                        "Merge ahead", "Adjacent lanes ahead",
                                        "Barrier or Gate", "Railroad", "Garage",
                                        "Toll station", "Temporary road blockage",
                                        "Small Portable Traffic Indicator",
                                        "Traffic cones", "Warning sign",
                                        "Toy", "Ball", "Debris", "Dirt",
                                        "Trash", "Personnel", "Stroller",
                                    ],
                                },
                                "visibility_start_timestamp": {"type": "string"},
                                "visibility_end_timestamp": {"type": "string"},
                                "contained_in_env_id": {"type": "integer"},
                                "lane_number": {"type": "string"},
                                "state_sequence": {
                                    "type": "array",
                                    "description": "DENSE array: For Traffic Lights, intervals must cover 100% of visibility duration.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "start_timestamp": {"type": "string"},
                                            "end_timestamp": {"type": "string"},
                                            "motion_state": {
                                                "type": "string",
                                                "enum": ["Static", "Moving / Rolling"],
                                            },
                                            "illegal_flag": {"type": "boolean"},
                                            "influenced_agent_ids": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "traffic_light_details": {
                                                "type": "object",
                                                "properties": {
                                                    "current_color": {
                                                        "type": "string",
                                                        "enum": ["Red", "Green", "Yellow"],
                                                    },
                                                    "color_changed_in_this_interval": {"type": "boolean"},
                                                    "yellow_light_logic": {
                                                        "type": "object",
                                                        "properties": {
                                                            "ego_in_intersection_on_yellow": {"type": "boolean"},
                                                            "lead_vehicle_stopped": {"type": "boolean"},
                                                            "could_have_cleared_safely": {"type": "boolean"},
                                                        },
                                                    },
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "ego_vehicle": {
                        "type": "object",
                        "properties": {
                            "actions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "enum": [
                                                "Stops / Waits", "Yields", "Slows",
                                                "Follows Lane", "Follows Agent",
                                                "Nudges around", "Overtakes",
                                                "Changes lane", "Turns", "Reverses",
                                                "Enter roundabout", "Exit roundabout",
                                                "Pass through",
                                            ],
                                        },
                                        "because_of": {"type": "array", "items": {"type": "string"}},
                                        "illegal_flag": {"type": "boolean"},
                                        "is_aggressive_or_cut_in": {"type": "boolean"},
                                        "start_timestamp": {"type": "string"},
                                        "end_timestamp": {"type": "string"},
                                    },
                                },
                            },
                            "containment": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "env_id": {"type": "integer"},
                                        "lane_number": {"type": "string", "description": "Counted from LEFT"},
                                        "illegal_flag": {"type": "boolean"},
                                        "start_timestamp": {"type": "string"},
                                        "end_timestamp": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                    "agents": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "amount": {
                                    "type": "string",
                                    "enum": [
                                        "Single", "Row/group", "Light traffic",
                                        "Medium traffic", "Heavy traffic",
                                    ],
                                },
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "Car", "Heavy-duty vehicle", "Emergency Vehicle",
                                        "Truck", "Bus", "Cyclist", "Bike", "Scooter",
                                        "Pedestrian (Officer)", "Pedestrian (Personnel)",
                                        "Pedestrian (Adult)", "Pedestrian (Teen)",
                                        "Pedestrian (Child)", "Pedestrian (Stroller)",
                                        "Animal",
                                    ],
                                },
                                "actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "action_type": {
                                                "type": "string",
                                                "enum": [
                                                    "Parked", "Double-parked",
                                                    "Stops / Waits", "Yields", "Slows",
                                                    "Follows / Moves along",
                                                    "Nudges around", "Overtakes",
                                                    "Changes lane", "Turns", "Reverses",
                                                    "Walks / Runs", "Signals",
                                                    "Emergency Situation / Stalled",
                                                ],
                                            },
                                            "because_of": {"type": "array", "items": {"type": "string"}},
                                            "signaling_details": {
                                                "type": "object",
                                                "properties": {
                                                    "source": {
                                                        "type": "string",
                                                        "enum": ["Hand Gesture", "Light Indicator", "Physical Sign"],
                                                    },
                                                    "intent": {
                                                        "type": "string",
                                                        "enum": ["Stop", "Slow Down", "Proceed", "Caution", "Danger", "Unclear"],
                                                    },
                                                    "target_agent_id": {"type": "string"},
                                                },
                                            },
                                            "start_timestamp": {"type": "string"},
                                            "end_timestamp": {"type": "string"},
                                        },
                                    },
                                },
                                "spatial_details": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "position_rel_to_ego": {
                                                "type": "string",
                                                "enum": ["In front", "Left", "Right", "Behind"],
                                            },
                                            "direction_rel_to_ego": {
                                                "type": "string",
                                                "enum": ["Same", "Opposite", "Perpendicular", "Cross L-to-R", "Cross R-to-L"],
                                            },
                                            "near_env_id": {"type": "integer"},
                                            "between_lane_range": {
                                                "type": "object",
                                                "description": "Lane counting from RIGHT",
                                                "properties": {
                                                    "env_id": {"type": "integer"},
                                                    "right_most_lane": {"type": "integer"},
                                                    "left_most_lane": {"type": "integer"},
                                                },
                                            },
                                            "start_timestamp": {"type": "string"},
                                            "end_timestamp": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
                "required": [
                    "brief_description", "environments",
                    "static_regulatory_objects", "ego_vehicle", "agents",
                ],
            },
        },
    }
]


def process_caption(caption_data: dict[str, Any]) -> dict[str, Any]:
    brief_description = caption_data.get("brief_description", "")
    structured_data = {
        "environments": caption_data.get("environments", []),
        "static_regulatory_objects": caption_data.get("static_regulatory_objects", []),
        "ego_vehicle": caption_data.get("ego_vehicle", {}),
        "agents": caption_data.get("agents", []),
    }
    return {
        "brief_description": brief_description,
        "json_caption": structured_data,
        "yaml_caption": yaml.dump(structured_data, sort_keys=False),
    }


mode = {
    "system_prompt": SYSTEM_PROMPT,
    "user_prompt": USER_PROMPT,
    "tools": TOOLS,
    "caption_keys": CAPTION_KEYS,
    "process_caption": process_caption,
    "camera_prefix": CAMERA_PREFIX,
}
