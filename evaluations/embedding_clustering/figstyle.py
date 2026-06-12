# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Shared figure styling: register NVIDIA Sans and make it the default font.

Import for side effect at the top of any matplotlib figure script::

    import figstyle  # noqa: F401  (registers NVIDIA Sans, sets rcParams)

The TTF directory defaults to the whitepaper's ``NVIDIA-Sans-Font-TTF`` and is
overridable with ``NVIDIA_SANS_DIR``. ``figstyle.FAMILY`` holds the resolved
family name (e.g. for plotly ``font=dict(family=figstyle.FAMILY)``).
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
from matplotlib import font_manager

_DIR = Path(os.environ.get(
    "NVIDIA_SANS_DIR",
    "/home/april/sil-wheel-experiments/SIL_Wheel_Whitepaper/NVIDIA-Sans-Font-TTF"))

FAMILY = "DejaVu Sans"


def apply() -> str:
    """Register every NVIDIA Sans TTF and set it as the matplotlib default."""
    global FAMILY
    if not _DIR.is_dir():
        return FAMILY
    names = set()
    for ttf in sorted(_DIR.glob("*.ttf")):
        try:
            font_manager.fontManager.addfont(str(ttf))
            names.add(font_manager.FontProperties(fname=str(ttf)).get_name())
        except Exception:
            pass
    if names:
        FAMILY = "NVIDIA Sans" if "NVIDIA Sans" in names else sorted(names)[0]
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [FAMILY, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    return FAMILY


apply()
