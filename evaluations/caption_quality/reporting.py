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

"""Markdown reporting helpers for evaluation scripts."""
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def path_size(path: str) -> str:
    """Human-readable size of a file or directory."""
    p = Path(path)
    if not p.exists():
        return "n/a"
    if p.is_file():
        total = p.stat().st_size
    else:
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} PB"


def write_markdown(
    path: str,
    title: str,
    headers: List[str],
    rows: List[Tuple[str, ...]],
    metadata: Optional[Dict[str, str]] = None,
    append: bool = False,
) -> None:
    """Write a markdown table with a metadata header line. ``append=True`` adds a section."""
    commit = _git_commit()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    meta_parts = [f"commit `{commit}`", f"date {timestamp}"]
    if metadata:
        meta_parts += [f"{k}: {v}" for k, v in metadata.items()]
    meta_line = "_" + " · ".join(meta_parts) + "_"

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells):
        return (
            "| "
            + " | ".join(c.ljust(w) for c, w in zip(cells, col_widths))
            + " |"
        )

    lines = [
        f"# {title}",
        "",
        meta_line,
        "",
        fmt_row(headers),
        "| " + " | ".join("-" * w for w in col_widths) + " |",
    ]
    for row in rows:
        lines.append(fmt_row(row))
    lines.append("")

    mode = "a" if append else "w"
    with open(path, mode) as f:
        if append:
            f.write("\n")
        f.write("\n".join(lines))
    print(f"Results {'appended to' if append else 'written to'} {path}")
