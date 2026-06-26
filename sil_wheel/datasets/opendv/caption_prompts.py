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

"""OpenDV Stage-2 reference-caption prompts (see docs/opendv.md, Stage 2).

A vision-language model runs ANALYZE then REFLECT in one conversation (system =
TWO_STEP_SYSTEM, the clip's frames provided once); a small text model then renders the
corrected JSON into short/medium/long captions via RENDER_SYSTEM and RENDER_TEMPLATE
(``RENDER_TEMPLATE.format(caption=corrected_json)``)."""

TWO_STEP_SYSTEM = "You are an expert visual analyst of egocentric driving and dashcam video. You produce careful, accurate, structured analyses in valid JSON, grounding every statement in what is actually visible."

ANALYZE = """You are an expert visual analyst of egocentric video data from a camera mounted on a vehicle, which we call the ego vehicle. Analyze the video and produce a SINGLE JSON object capturing BOTH (a) the key objects and the chronological events, AND (b) the structured scene attributes. Be thorough and specific — this analysis is used to write detailed captions, so do NOT omit any salient object or event.

Key objects: the ego vehicle and anything it must be aware of to navigate safely — other vehicles (especially in nearby lanes), people / children / pedestrians / construction or emergency workers / police, animals, cyclists, traffic lights, road signs.

Key events: all events concerning these objects IN CHRONOLOGICAL ORDER — at minimum every ego-vehicle action (lane change, stop, start, turn, accelerate, decelerate, collide) and any other notable action you see.

Motion cues: if static objects appear to move right, the ego vehicle is likely turning left and vice-versa; if the view bounces, the road is bumpy or a speed bump was crossed.

For each structured attribute write ONE OR TWO SENTENCES WITH REASONING for the value you chose (not a single word).

RECORD ONLY WHAT IS PRESENT OR WHAT ACTUALLY HAPPENS. State every field as a positive observation. Do NOT describe the absence of things and do NOT use negations such as "no", "none", "not", "without", "absent", "n't". Examples: write "clear, dry road surface with hazy distance" (NOT "no rain or snow"); for ego_meta_action list ONLY the maneuvers that occur (NOT "no left turn or lane change"); for rule_following_or_violation describe the compliant behaviors you actually observe such as "keeps its lane and maintains following distance" (NOT "no violation is visible"). If something is not present, simply omit it.

Output ONLY this JSON object (no preamble, no code fences):
{
  "key_objects": ["<object>: <description>", "..."],
  "key_events": ["<event, in chronological order>", "..."],
  "vehicle_density": "none/low/medium/high, with reasoning",
  "pedestrian_density": "none/low/medium/high, with reasoning",
  "weather": "clear/rain/snow/fog/etc., with reasoning",
  "illumination": "day/night plus any lighting sources, with reasoning",
  "ego_speed": "standing/low/city/highway, with reasoning",
  "road_curvature": "low/medium/high, with reasoning",
  "road_type": "rural/residential/urban/roundabout/highway entrance/etc., with reasoning",
  "road_information": "lane counts each direction, intersections, roundabouts, etc.",
  "ego_meta_action": "describe ALL of them temporally in chronological order — longitudinal [speed up, slow down, slow down rapidly, go straight slowly, go straight at a constant speed, stop, wait, reverse], lateral turning [left turn, right turn, turn around], lateral lane-control [keep lane, left lane change, right lane change, nudge slightly left, nudge slightly right]. There may be more than one.",
  "rule_following_or_violation": "e.g. ran red light, full stop or rolling stop at stop sign, lane change over a solid line, unsignaled turn, safe/unsafe unprotected turn, sudden braking for pedestrians, etc."
}

ONLY PROVIDE OUTPUT IN ENGLISH. Output only the JSON object."""

REFLECT = """Now critically review your JSON analysis against the same video. Internally identify the mistakes, inconsistencies, and — most importantly — the MISSING key objects or events you failed to capture the first time. Pay special attention to dynamic agents and ego-vehicle maneuvers that are easy to miss,
such as slight nudging to navigate around an object, pedestrian, blocked path, etc.

Then output a CORRECTED and COMPLETED version of the analysis: fix every error and add every previously-missing object or event, applying your corrections directly to the fields. Output EXACTLY the same JSON schema and keys as before — do NOT add, remove, or rename any key, and do NOT include a critique or commentary field. Return only the corrected analysis.

RECORD ONLY WHAT IS PRESENT OR WHAT ACTUALLY HAPPENS. State every field as a positive observation. Do NOT describe the absence of things and do NOT use negations such as "no", "none", "not", "without", "absent", "n't". For ego_meta_action list ONLY the maneuvers that occur; for rule_following_or_violation describe the compliant behaviors you actually observe rather than asserting the lack of a violation; for weather describe the conditions positively (e.g. "clear and dry") rather than what is absent. If something is not present, simply omit it.

Output ONLY this JSON object (no preamble, no code fences):
{
  "key_objects": ["<object>: <description>", "..."],
  "key_events": ["<event, in chronological order>", "..."],
  "vehicle_density": "...",
  "pedestrian_density": "...",
  "weather": "...",
  "illumination": "...",
  "ego_speed": "...",
  "road_curvature": "...",
  "road_type": "...",
  "road_information": "...",
  "ego_meta_action": "...",
  "rule_following_or_violation": "..."
}

ONLY PROVIDE OUTPUT IN ENGLISH. Output only the JSON object."""

RENDER_SYSTEM = "You write discriminative video captions from a structured JSON analysis of an egocentric driving video. Your captions are used as queries in a text-to-video retrieval benchmark, so each caption must capture what makes THIS specific clip different from a generic driving scene. ABOVE ALL you are STRICTLY FAITHFUL: every fact you write must be explicitly entailed by the structured JSON, and you NEVER add, infer, embellish, or upgrade any detail — especially the motion state of any object — beyond what the JSON states."

RENDER_TEMPLATE = """Below is a structured JSON analysis of an egocentric driving video: key
objects, chronological events, and scene attributes (weather, lighting, road,
ego speed, ego maneuvers, rule following/violations). Write THREE prose
captions of the scene at different lengths.

FAITHFULNESS — THIS IS THE OVERRIDING RULE. READ IT FIRST AND OBEY IT ABOVE
EVERYTHING ELSE BELOW:
1. Every fact you state — objects, their attributes (color, size, type),
   counts, spatial relations, actions, and ESPECIALLY motion states (moving,
   parked, stopped, stationary, turning, yielding, accelerating, braking) — MUST
   be explicitly entailed by the structured JSON. Do NOT add, infer, embellish,
   or upgrade any detail beyond what the JSON literally states.
2. MOTION STATES ARE THE #1 ERROR — NEVER INVENT ONE, FOR ANY OBJECT, AT ANY
   LENGTH. Give an object a motion state (moving, driving, passing, parked,
   stopped, stationary, idle, turning, braking, accelerating, yielding, merging,
   reversing) ONLY when the JSON explicitly states that motion for that exact
   object. A LOCATION is not a motion: if the JSON only places an object
   somewhere (at the curb, roadside, in the right lane, near buildings, on the
   shoulder) without stating motion, describe the location and say NOTHING about
   whether it moves — a roadside or curbside vehicle is NOT necessarily parked or
   stopped. This holds for every object type — e.g. do not turn a curbside van
   into "a parked van", a roadside SUV into "a stopped SUV", a truck near
   buildings into "a parked truck", a bus at the curb into "a parked bus", or a
   sedan in a lane into "a stopped sedan". If the JSON says an object is moving,
   never call it parked/stopped (and vice versa). When you shorten the long
   caption, carry each object's motion across UNCHANGED — you may drop an object
   entirely, but never harden a position into a parked or stopped state to save
   words.
3. NEVER transfer an attribute from one object to another. Every attribute —
   motion, color, size, type — belongs only to the object the JSON attaches it
   to; never carry an attribute from one object onto a different nearby object.
4. If the JSON is silent or ambiguous about an attribute or a motion state,
   OMIT it rather than guess. Faithfulness OUTRANKS salience and OUTRANKS hitting
   any word count: a shorter, fully-grounded caption is always better than a
   longer one containing a single unsupported detail.
5. If the JSON is internally inconsistent, prefer its explicit DYNAMIC
   statements (e.g. a "moving vehicles" list, or a verb like "passing" /
   "traveling") over any positional inference. If it is still ambiguous, describe
   the object's POSITION only, without asserting any motion.
6. "Salient" means SELECTING the most distinguishing facts that are ALREADY
   TRUE in the JSON and foregrounding them — it NEVER means adding new
   specificity, sharper attributes, or motion the JSON did not provide.
7. THESE FAITHFULNESS RULES APPLY EQUALLY TO ALL THREE LENGTHS. Shorter captions
   are abbreviations, NOT looser paraphrases: to compress, DROP whole facts,
   never invent, sharpen, or harden a detail you would not have written in the
   long caption. Compression may only REMOVE information, never add or
   strengthen it.

GOAL — subject to the faithfulness rule above, these captions are retrieval
queries that must DISTINGUISH this clip from thousands of other driving clips.
Lead with and emphasize what is SALIENT and SPECIFIC to this scene; spend as few
words as possible on generic scene-setting. Order your content by these
priorities:
1. EVENTS AND ACTIONS FIRST. Foreground what HAPPENS over time — the ego
   vehicle's maneuvers (turns, lane changes, braking, accelerating, stopping,
   merging, yielding) and the actions and motion of other agents (a pedestrian
   crossing, a cyclist merging, a truck cutting in, a car braking ahead), but
   ONLY as the JSON states them. Describe them concretely and in chronological
   order.
2. SALIENT, DISTINGUISHING DETAILS NEXT. Foreground the specific, unusual, or
   identifying elements the JSON already contains: a particular colored vehicle
   and what it does, a distinctive landmark or signage, an intersection type, a
   road-event, unusual weather or lighting, a specific maneuver. These are the
   details a person would use to pick this clip out of a crowd — but take them
   verbatim from the JSON, never sharpen them.
3. GENERIC SCENE-SETTING LAST AND BRIEFLY. Treat boilerplate that applies to
   almost any driving clip as low-value: keep it to a short phrase, not a
   paragraph. Avoid filler such as "soft ambient daylight", "visibility extends
   down the road", "traffic flows steadily", "maintains consistent lane
   control", "the vehicle centers itself in its lane", and similar padding that
   adds no distinguishing information. Do NOT pad to reach a word count — a
   tighter, more specific, fully-grounded caption is better than a longer generic
   one.

LENGTHS (these are CEILINGS, not targets — faithfulness and concision win):
- "long":   up to about 300 words, but only as long as there is SPECIFIC,
            DISTINGUISHING, JSON-GROUNDED content to fill it. Do not inflate.
- "medium": about 150 words. The key events in order plus the most
            distinguishing JSON-grounded subjects and attributes.
- "short":  fewer than 50 words. One to three sentences capturing the core
            scene and the most important ego action(s) and event(s).

CRITICAL RULE — describe only what IS present and what DOES happen. NEVER write
phrases describing the ABSENCE of things or what did NOT happen (e.g. "no
pedestrians", "no collision", "no lane changes", "no stops", "no construction",
"clear of traffic", "no rain or snow", "no traffic violation"). The JSON may
itself contain such absence statements or negations (e.g. "no clear violation
is visible", "there is no rain", "no left turn or lane change") — you MUST drop
them entirely and never carry them into the captions. Convert any
negatively-phrased fact into its positive form (e.g. JSON "no rain, dry road" →
write "dry road surface"; JSON "no violation, keeps lane" → write "keeps its
lane and follows the lead vehicle"). Do not use the words "no", "not", "none",
"without", "absent", or contractions ending in "n't" anywhere in the captions.

Return ONLY a JSON object with exactly three string keys: "long", "medium",
"short". No preamble, no code fences.

JSON analysis:
{caption}"""
