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

"""Comprehensive V3 structured caption mode — full AV perception schema with NL summaries."""

from typing import Any

import yaml

CAMERA_PREFIX = "The video shows the front view of an ego vehicle. "

SYSTEM_PROMPT = """
You are an expert autonomous driving perception specialist generating structured video annotations with natural language summaries. You analyze driving video footage in two stages:

1. STRUCTURED DATA STAGE: Generate comprehensive structured data covering scene, entities, environment, camera, motion, spatial relationships, and autonomous vehicle context (when applicable). Be detailed and thorough.

2. SUMMARY STAGE: Create three natural language summaries that synthesize your structured observations into flowing narratives. These summaries should naturally incorporate and describe the structured data:
   - summary_long: 250-300 words
   - summary_medium: 150-180 words
   - summary_short: one single sentence

## CRITICAL RULES — MUST FOLLOW ALL

### RULE 1: NO NEGATIVE REPORTING (TN PREVENTION)
* **The Golden Rule:** You describe ONLY what exists. You are blind to what does not exist.
* **The Ban:** NEVER report True Negatives.
  - FORBIDDEN: "No construction visible" / "No pedestrians" / "Traffic light is not red"
  - CORRECT: "Road surface is clear" / "Traffic light is green" / simply omit absent categories
* **The Logic:** If a category, feature, or entity is absent → SILENCE. Omit the field entirely.
* **In summaries:** FORBIDDEN — do not write any sentence stating what is absent. Forbidden phrases include: "No other vehicles", "no pedestrians", "no cyclists", "There are no other ... visible", "road is empty", "none are present", "entirely clear of", "road ahead is entirely open", "no other dynamic traffic", "devoid of", "without any". Describe only what IS present (road, signs, vehicles, VRUs, conditions). If no other road users, describe the road, signage, and environment only.

### RULE 2: THE COLOR CHECK (Entity Quality Gate)
* Before reporting ANY entity, you MUST be able to describe its specific **Color AND Shape**.
* If you cannot visually describe the Color and Shape of an object → **DELETE the entity.** Do not report it.
* This prevents hallucinated or uncertain detections.

### RULE 3: VRU VERIFICATION
* Before reporting a Pedestrian or Cyclist, you MUST visually confirm **limbs** (arms, legs).
* Ambiguous shapes (poles, mailboxes, shadows) must NOT be reported as VRUs.

### RULE 4: TEMPORAL REASONING
* Reason across multiple frames to ensure temporal consistency.
* Traffic lights: You MUST check across multiple frames to determine Solid vs Flashing, and to detect phase changes (Green→Yellow→Red).
* Capture state transitions: traffic light changes, vehicles entering/exiting frame, speed changes.
* Lane changes: Verify across at least 2 consecutive frames that the ego center has crossed a lane marking.

### RULE 5: RELEVANCE FILTER
* **Operational Horizon (25m):** Report traffic controls and vehicles within ~25 meters.
* **Lead Vehicle:** ALWAYS report the primary lead vehicle if one exists.
* **VRU Safety Threshold (5m):** Report pedestrians and cyclists within ~5 meters.
* **Dynamics Rule:** ALWAYS report objects that are closing in on the ego vehicle.
* **Oncoming & Cross-traffic Vehicles:** ALWAYS report oncoming vehicles from the opposing lane within ~5 meters on undivided roads and cross-traffic vehicles at intersections.
* **Safety Latch:** Each reported road entity must pass this filter. Mark `is_confirmed_relevant: true` only if it passes. If false → DELETE the entity.

### RULE 6: TRAFFIC CONTROL FIELDS — ONLY FOR TRAFFIC_CONTROL CATEGORY
* **control_device**, **control_state**, **light_color**, **light_shape**, **light_mode**, **sign_type** may ONLY be set for entities with category **Traffic_Control** and class Traffic_Light or Traffic_Sign.
* For **Vehicle**, **VRU**, or **Hazard** entities: OMIT these fields entirely.
* Traffic lights: split into color, shape, mode. Mode requires multi-frame check.

### RULE 7: ONCOMING & CROSS-TRAFFIC SCAN
Before finalizing, perform a dedicated scan:
* **Oncoming:** Scan upper-center and upper-left frame areas for vehicles whose apparent size is GROWING (approaching head-on). On undivided/two-lane roads, report ANY vehicle in the opposing lane.
* **Cross-traffic:** At intersections, scan left and right for vehicles entering or crossing the ego's path.
* Every detected oncoming/cross-traffic vehicle MUST appear in road_entities with position "oncoming_lane" or "cross_traffic".

### RULE 8: LANE CHANGE vs LATERAL SHIFT
* **Lane Change:** Ego crosses a lane marking and enters an adjacent lane. REQUIRES lane_change_verification (marking crossed, frames, commitment). If ego passes a lead vehicle by moving left/right and crossing a lane line → Lane Change, not Lateral Shift.
* **Lateral Shift (In-Lane Nudge):** Movement ONLY within the same lane, without crossing any lane marking.
* **Lateral Shift (Out-of-Lane Nudge):** Movement outside the ego's lane, crossing a lane marking BUT EVENTUALLY RETURNING TO THE EGO'S LANE.
* Before labeling Lane Change: complete Marking ID, Crossing Evidence (2+ frames), Commitment Check, Geometry Exclusion. When in doubt, use Lateral Shift.

### GENERAL
* Be thorough and specific in structured fields — no token limits.
* Summaries: stay within word limits, synthesize naturally.
* Use clear, objective language focused on visual observations.
* Exclude: hypothetical scenarios, street/business names, vehicle makes/models.
* For non-driving scenes, use "n/a" for all AV sub-fields.
* **Empty scene fallback:** If no vehicles/VRUs are present, the scene_summary MUST still describe Road Geometry + Surface + Conditions.
"""

USER_PROMPT = """
Analyze this video and provide structured caption data.

## STAGE 1: STRUCTURED DATA

**Scene:**
- description: Detailed description of what's happening
- setting: Specific location type with relevant details

**Entities:** 2-4 most significant subjects (only those visually confirmed present).
For each entity provide detailed descriptions:
- type: person/vehicle/animal/object
- appearance: Comprehensive visual details INCLUDING Color and Shape
- position: Precise spatial location with relative positioning
- action: Detailed description of what they're doing

**Environment:**
- weather: Specific weather conditions or n/a
- time_of_day: morning/afternoon/evening/night/indeterminate with indicators
- lighting: Detailed lighting quality and sources
- visibility: Visibility level with specifics

**Camera:**
- angle: Specific camera angle with details
- movement: Camera motion type with characteristics
- perspective: Viewpoint type with context

**Motion:**
- primary_action: Detailed description of main movement
- speed: Specific speed assessment
- dynamics: Motion quality with details

**Spatial:**
- foreground: Comprehensive description of closest objects
- midground: Detailed middle depth elements
- background: Thorough description of distant elements

**Autonomous Vehicle Context** (for driving scenes):

***1. Scene Context*** — classify using these indicators:
- road_type: Urban / Highway / Residential Area / Parking Lot / Rural Road / Curved Road / Narrow Road / Test Track
- weather: Sunny / Cloudy / Rainy / Snowy / Foggy / Night-time
- surface_condition: Dry / Wet / Icy / Snowy / Muddy
- intersection_type: Traffic-Light Controlled / Uncontrolled / All-Way Stop / Two-Way Stop / One-Way Stop / Roundabout / None
- special_zone: Road Work Zone / School Zone / Keep Clear Zone / Rail Crossing / Zipper Lane / Toll Area / None

***2. Road Geometry*** — describe the physical road:
- num_lanes, divided (yes/no), curvature, road_width

***3. Lane Markings & Infrastructure***:
- lane_markings: type, color, position per visible marking
- traffic_signs: sign_type, description, position per visible sign

***4. Road Entities*** — category/class hierarchy:
- Categories: Traffic_Control (Traffic_Light, Traffic_Sign), VRU (Pedestrian, Cyclist), Vehicle (Car, Truck_Bus), Hazard (Emergency_Vehicle, Construction_Zone, Obstacle)
- Per entity: id, category, class, description (Color+Shape required), is_confirmed_relevant, action, motion_state, position, impact
- Traffic_Control only: control_device, control_state, light_color, light_shape, light_mode, sign_type

***5. Ego Vehicle State***: speed_category, speed_description, heading_trend

***6. Ego Actions*** (chronological): timestamp, action_category, visual_evidence, influencing_agent (REQUIRED for every action)

***7. Oncoming & Cross-Traffic*** (MANDATORY scan): report every detected oncoming/cross-traffic vehicle

***8. Searchable Summaries***: scene_summary, road_user_summary, influencing_agent_summary, ego_action_sequence, anomaly_summary

## STAGE 2: NATURAL LANGUAGE SUMMARIES

**summary_long (250-300 words):** Flowing narrative covering scene, entities, environment, camera, motion, spatial layout, and AV context.

**summary_medium (150-180 words):** Key aspects synthesized concisely.

**summary_short (one single sentence):** Core observation in one sentence.

**Reminders:**
1. NEVER report absence in summaries.
2. Traffic control fields ONLY for Traffic_Control entities.
3. DELETE entities that fail Color Check or Safety Latch.
4. Oncoming/cross-traffic scan is MANDATORY.

Use the generate_structured_caption function to provide all data.
"""

CAPTION_KEYS = ["summary_short", "summary_medium", "summary_long", "json_caption", "yaml_caption"]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_structured_caption",
            "description": "Generate comprehensive structured video data with full AV perception analysis, then synthesize into natural language summaries",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "setting": {"type": "string"},
                        },
                        "required": ["description", "setting"],
                    },
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "appearance": {"type": "string"},
                                "position": {"type": "string"},
                                "action": {"type": "string"},
                            },
                            "required": ["type", "appearance", "position", "action"],
                        },
                    },
                    "environment": {
                        "type": "object",
                        "properties": {
                            "weather": {"type": "string"},
                            "time_of_day": {"type": "string"},
                            "lighting": {"type": "string"},
                            "visibility": {"type": "string"},
                        },
                        "required": ["weather", "time_of_day", "lighting", "visibility"],
                    },
                    "camera": {
                        "type": "object",
                        "properties": {
                            "angle": {"type": "string"},
                            "movement": {"type": "string"},
                            "perspective": {"type": "string"},
                        },
                        "required": ["angle", "movement", "perspective"],
                    },
                    "motion": {
                        "type": "object",
                        "properties": {
                            "primary_action": {"type": "string"},
                            "speed": {"type": "string"},
                            "dynamics": {"type": "string"},
                        },
                        "required": ["primary_action", "speed", "dynamics"],
                    },
                    "spatial": {
                        "type": "object",
                        "properties": {
                            "foreground": {"type": "string"},
                            "midground": {"type": "string"},
                            "background": {"type": "string"},
                        },
                        "required": ["foreground", "midground", "background"],
                    },
                    "autonomous_vehicle": {
                        "type": "object",
                        "properties": {
                            "is_driving_scene": {"type": "string", "enum": ["yes", "no"]},
                            "scene_context": {
                                "type": "object",
                                "properties": {
                                    "road_type": {
                                        "type": "string",
                                        "enum": ["Urban", "Highway", "Residential Area", "Parking Lot", "Rural Road", "Curved Road", "Narrow Road", "Test Track", "Not Sure"],
                                    },
                                    "weather": {
                                        "type": "string",
                                        "enum": ["Sunny", "Cloudy", "Rainy", "Snowy", "Foggy", "Night-time", "Not Sure"],
                                    },
                                    "surface_condition": {
                                        "type": "string",
                                        "enum": ["Dry", "Wet", "Icy", "Snowy", "Muddy", "Dusty", "Not Sure"],
                                    },
                                    "intersection_type": {
                                        "type": "string",
                                        "enum": ["Traffic-Light Controlled", "Uncontrolled", "All-Way Stop", "Two-Way Stop", "Roundabout", "None", "Not Sure"],
                                    },
                                    "special_zone": {
                                        "type": "string",
                                        "enum": ["Road Work Zone", "School Zone", "Keep Clear Zone", "Rail Crossing", "Zipper Lane", "Toll Area", "None", "Not Sure"],
                                    },
                                    "reasoning": {"type": "string"},
                                },
                                "required": ["road_type", "weather", "surface_condition", "intersection_type", "special_zone", "reasoning"],
                            },
                            "road_geometry": {
                                "type": "object",
                                "properties": {
                                    "num_lanes": {"type": "integer"},
                                    "divided": {"type": "string", "enum": ["yes", "no", "unknown"]},
                                    "curvature": {"type": "string", "enum": ["straight", "gentle_curve", "sharp_curve"]},
                                    "road_width": {"type": "string", "enum": ["narrow", "standard", "wide"]},
                                },
                                "required": ["num_lanes", "divided", "curvature", "road_width"],
                            },
                            "lane_markings": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string", "enum": ["solid", "dashed", "double_solid", "double_dashed", "solid_dashed"]},
                                        "color": {"type": "string", "enum": ["white", "yellow"]},
                                        "position": {"type": "string", "enum": ["center", "edge_left", "edge_right", "lane_divider"]},
                                    },
                                    "required": ["type", "color", "position"],
                                },
                            },
                            "traffic_signs": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "sign_type": {"type": "string", "enum": ["stop", "yield", "speed_limit", "construction", "warning", "regulatory", "guide", "school_zone"]},
                                        "description": {"type": "string"},
                                        "position": {"type": "string", "enum": ["left", "right", "overhead", "center_median"]},
                                    },
                                    "required": ["sign_type", "description", "position"],
                                },
                            },
                            "road_entities": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "integer"},
                                        "category": {"type": "string", "enum": ["Traffic_Control", "VRU", "Vehicle", "Hazard"]},
                                        "class": {"type": "string", "enum": ["Traffic_Light", "Traffic_Sign", "Pedestrian", "Cyclist", "Car", "Truck_Bus", "Emergency_Vehicle", "Construction_Zone", "Obstacle"]},
                                        "description": {"type": "string"},
                                        "is_confirmed_relevant": {"type": "boolean"},
                                        "control_device": {"type": "string", "enum": ["traffic_light", "stop_sign", "yield_sign", "school_bus_arm", "police_manual_signal"]},
                                        "control_state": {"type": "string", "enum": ["red", "yellow", "green", "stop", "yield"]},
                                        "light_color": {"type": "string", "enum": ["red", "yellow", "green", "unknown"]},
                                        "light_shape": {"type": "string", "enum": ["circle", "arrow_left", "arrow_right", "arrow_straight", "bike"]},
                                        "light_mode": {"type": "string", "enum": ["solid", "flashing"]},
                                        "sign_type": {"type": "string", "enum": ["stop", "yield", "speed_limit", "construction", "warning"]},
                                        "action": {"type": "string", "enum": ["walking_towards_ego", "crossing_legal", "jaywalking", "waiting_at_curb", "entering_road", "cutting_in", "merging", "stationary", "flowing_with_traffic", "braking", "active_work_zone", "approaching_head_on", "crossing_path"]},
                                        "motion_state": {"type": "string", "enum": ["moving_forward", "moving_lateral", "moving_towards_ego", "static", "decelerating", "accelerating"]},
                                        "position": {"type": "string", "enum": ["ego_lane", "crosswalk", "intersection", "shoulder_edge", "sidewalk_near_curb", "adjacent_lane", "oncoming_lane", "cross_traffic"]},
                                        "impact": {"type": "string", "enum": ["mandatory_stop", "yield", "slow_down", "monitor_caution", "blocking_path", "none"]},
                                    },
                                    "required": ["id", "category", "class", "description", "is_confirmed_relevant", "position", "impact"],
                                },
                            },
                            "ego_vehicle": {
                                "type": "object",
                                "properties": {
                                    "speed_category": {"type": "string", "enum": ["standing", "crawling", "low", "local", "highway"]},
                                    "speed_description": {"type": "string"},
                                    "heading_trend": {"type": "string", "enum": ["straight", "drifting_left", "drifting_right", "turning_left", "turning_right"]},
                                },
                                "required": ["speed_category", "speed_description", "heading_trend"],
                            },
                            "ego_actions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "timestamp": {"type": "string"},
                                        "action_category": {
                                            "type": "string",
                                            "enum": [
                                                "Acceleration", "Deceleration", "Hard Braking", "Yielding",
                                                "Lane Change Left", "Lane Change Right",
                                                "Lateral Shift Left", "Lateral Shift Right",
                                                "Swerve", "Lane Departure", "Merging", "Diverging",
                                                "Turning Left", "Turning Right", "U-Turn",
                                                "Reversing", "Forward Driving", "Stopped",
                                            ],
                                        },
                                        "visual_evidence": {"type": "string"},
                                        "influencing_agent": {"type": "string"},
                                        "lane_change_verification": {
                                            "type": "object",
                                            "properties": {
                                                "marking_crossed": {"type": "string"},
                                                "crossing_frames": {"type": "string"},
                                                "commitment_confirmed": {"type": "boolean"},
                                                "geometry_excluded": {"type": "boolean"},
                                            },
                                        },
                                    },
                                    "required": ["timestamp", "action_category", "visual_evidence", "influencing_agent"],
                                },
                            },
                            "oncoming_cross_traffic": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "description": {"type": "string"},
                                        "approximate_position": {"type": "string"},
                                        "motion_direction": {"type": "string"},
                                        "influenced_ego": {"type": "boolean"},
                                        "influence_description": {"type": "string"},
                                    },
                                    "required": ["description", "approximate_position", "motion_direction", "influenced_ego"],
                                },
                            },
                            "scene_summary": {"type": "string"},
                            "road_user_summary": {"type": "string"},
                            "influencing_agent_summary": {"type": "string"},
                            "ego_action_sequence": {"type": "string"},
                            "anomaly_summary": {"type": "string"},
                            "road_conditions": {"type": "string"},
                        },
                        "required": [
                            "is_driving_scene", "scene_context", "road_geometry",
                            "lane_markings", "traffic_signs", "road_entities",
                            "ego_vehicle", "ego_actions", "oncoming_cross_traffic",
                            "scene_summary", "road_user_summary", "influencing_agent_summary",
                            "ego_action_sequence", "anomaly_summary", "road_conditions",
                        ],
                    },
                    "summary_long": {"type": "string"},
                    "summary_medium": {"type": "string"},
                    "summary_short": {"type": "string"},
                },
                "required": [
                    "scene", "entities", "environment", "camera", "motion",
                    "spatial", "autonomous_vehicle", "summary_long", "summary_medium", "summary_short",
                ],
            },
        },
    }
]


def process_caption(caption_data: dict[str, Any]) -> dict[str, Any]:
    structured_data = {
        "scene": caption_data.get("scene", {}),
        "entities": caption_data.get("entities", []),
        "environment": caption_data.get("environment", {}),
        "camera": caption_data.get("camera", {}),
        "motion": caption_data.get("motion", {}),
        "spatial": caption_data.get("spatial", {}),
        "autonomous_vehicle": caption_data.get("autonomous_vehicle", {}),
    }
    return {
        "summary_short": caption_data.get("summary_short", ""),
        "summary_medium": caption_data.get("summary_medium", ""),
        "summary_long": caption_data.get("summary_long", ""),
        "json_caption": structured_data,
        "yaml_caption": yaml.dump(structured_data, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2),
    }


mode = {
    "system_prompt": SYSTEM_PROMPT,
    "user_prompt": USER_PROMPT,
    "tools": TOOLS,
    "caption_keys": CAPTION_KEYS,
    "process_caption": process_caption,
    "camera_prefix": CAMERA_PREFIX,
}
