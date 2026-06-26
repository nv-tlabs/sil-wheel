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

"""Fetch and normalize OpenDV-YouTube metadata from the public Google Sheet.

Mirrors the official meta_preprocess.py column mapping. The sheet is public, so
no Google API credentials are needed — we read the CSV export directly.
"""
import csv
import io
import logging
import urllib.request
from pathlib import Path

from sil_wheel.datasets.opendv.constants import SHEET_CSV_URL

log = logging.getLogger(__name__)

# Column-name (lowercased) -> normalized key. Matches the official meta_preprocess.py.
KEY_MAP = {
    "train / val": "split",
    "mini / full set": "subset",
    "nation or area (inferred by gpt)": "area",
    "state, province, or city (inferred by gpt and refined by human)": "state",
    "discarded length at the begininning (second)": "start_discard",
    "discarded length at the ending (second)": "end_discard",
}


def duration2length(duration: str) -> int:
    """'HH:MM:SS' or 'MM:SS' -> seconds."""
    parts = [int(p) for p in str(duration).split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"unrecognized duration: {duration!r}")


def parse_csv_text(csv_text: str) -> list[dict]:
    """Normalize the sheet CSV into records. Lowercases values for split/subset,
    coerces discard seconds to int, derives ``length`` from ``duration``."""
    records: list[dict] = []
    for row in csv.DictReader(io.StringIO(csv_text)):
        info: dict = {}
        for raw_key, value in row.items():
            if raw_key is None:
                continue
            key = KEY_MAP.get(raw_key.strip().lower(), raw_key.strip().lower())
            value = value.strip() if isinstance(value, str) else value
            if key in ("split", "subset"):
                value = value.lower()
            info[key] = value
        for dk in ("start_discard", "end_discard"):
            try:
                info[dk] = int(float(info.get(dk) or 0))
            except (TypeError, ValueError):
                info[dk] = 0
        try:
            info["length"] = duration2length(info["duration"]) if info.get("duration") else 0
        except ValueError:
            info["length"] = 0
        records.append(info)
    return records


def fetch_subset(subset: str = "mini", cache_csv: Path | None = None,
                 force: bool = False) -> list[dict]:
    """Download (or reuse cached) the sheet CSV and return ``subset`` records."""
    if cache_csv and cache_csv.exists() and not force:
        csv_text = cache_csv.read_text(encoding="utf-8")
    else:
        log.info("fetching OpenDV-YouTube sheet CSV")
        req = urllib.request.Request(
            SHEET_CSV_URL, headers={"User-Agent": "Mozilla/5.0 (sil-wheel)"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            csv_text = resp.read().decode("utf-8")
        if cache_csv:
            cache_csv.parent.mkdir(parents=True, exist_ok=True)
            cache_csv.write_text(csv_text, encoding="utf-8")
    records = parse_csv_text(csv_text)
    if subset == "mini":
        records = [r for r in records if r.get("subset") == "mini"]
    log.info("loaded %d %s records", len(records), subset)
    return records
