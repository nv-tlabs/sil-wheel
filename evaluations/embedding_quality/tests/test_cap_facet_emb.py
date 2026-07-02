# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the caption facet-text grouping (no GPU / no LLM)."""
from cap_facet_emb import FACETS, facet_text


def test_facet_text_groups_and_flattens_lists():
    structured = {
        "weather": "clear", "illumination": "day", "road_type": "urban",
        "road_curvature": "low", "key_objects": ["stop sign", "pedestrian"],
        "key_events": ["approaches", "stops"],
    }
    assert facet_text(structured, FACETS["scene"]) == "clear day urban low"
    # lists are flattened into the facet text
    assert facet_text(structured, ["key_objects"]) == "stop sign pedestrian"
    # missing field renders as "None" rather than crashing
    assert "None" in facet_text(structured, FACETS["temporal"] + ["missing"])


def test_facets_cover_the_twelve_fields_without_overlap():
    all_keys = [k for keys in FACETS.values() for k in keys]
    assert len(all_keys) == len(set(all_keys)) == 12
