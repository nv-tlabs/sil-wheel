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

"""OpenDV-YouTube-Language: per-video dominant (caption, command) labels.

Bins the noisy per-frame sliding-window annotations into ``bin_sec`` buckets,
takes the dominant (caption, command) per bin, and collapses consecutive
identical bins into timed segments. It does NOT emit retrieval text — only the
labels the diverse sampler needs.
"""
import collections
import json
import logging
from pathlib import Path

from sil_wheel.datasets.opendv.constants import CMD_TEXT, DEFAULT_BIN_SEC, FPS, LANG_FILES, LANG_REPO

log = logging.getLogger(__name__)


def _parse_frame(fn: str) -> int:
    return int(Path(fn).stem)


def _videoid(folder: str) -> str:
    """'<split>_images/<youtuber>/<videoid>' -> '<videoid>'."""
    parts = [p for p in folder.split("/") if p]
    return parts[-1]


def segments_from_entries(entries, bin_sec: int = DEFAULT_BIN_SEC) -> dict:
    """Aggregate raw annotation entries into per-video timed segments:
    ``{video_id: {"caption_segments": [...], "command_timeline": [...]}}``.

    Times are in annotation (processed-frame) space, seconds.
    """
    binframes = bin_sec * FPS
    acc: dict = {}
    for e in entries:
        center = (_parse_frame(e["first_frame"]) + _parse_frame(e["last_frame"])) // 2
        w = center // binframes
        vid = acc.setdefault(_videoid(e["folder"]), {})
        b = vid.get(w)
        if b is None:
            b = vid[w] = {"cap": collections.Counter(), "cmd": collections.Counter(),
                          "lo": center, "hi": center}
        b["cap"][e["blip"]] += 1
        b["cmd"][e["cmd"]] += 1
        b["lo"] = min(b["lo"], center)
        b["hi"] = max(b["hi"], center)

    def t(frame_idx, end=False):
        return round((frame_idx + (1 if end else 0)) / FPS, 1)

    out: dict = {}
    for video_id, bins in acc.items():
        cap_segs, cmd_segs = [], []
        for w in sorted(bins):
            b = bins[w]
            cap = b["cap"].most_common(1)[0][0]
            cmd = b["cmd"].most_common(1)[0][0]
            command = CMD_TEXT.get(cmd, str(cmd))
            if cap_segs and cap_segs[-1]["caption"] == cap:
                cap_segs[-1]["end_sec"] = t(b["hi"], True)
            else:
                cap_segs.append({"start_sec": t(b["lo"]), "end_sec": t(b["hi"], True),
                                 "caption": cap})
            if cmd_segs and cmd_segs[-1]["command"] == command:
                cmd_segs[-1]["end_sec"] = t(b["hi"], True)
            else:
                cmd_segs.append({"start_sec": t(b["lo"]), "end_sec": t(b["hi"], True),
                                 "command": command})
        out[video_id] = {"caption_segments": cap_segs, "command_timeline": cmd_segs}
    return out


def _best_overlap(segs, a, b, key):
    best, best_ov = "", 0.0
    for s in segs:
        ov = min(b, s["end_sec"]) - max(a, s["start_sec"])
        if ov > best_ov:
            best_ov, best = ov, s[key]
    return best


def dominant(segments: dict, a: float, b: float) -> tuple[str, str]:
    """(caption, command) covering the most of window [a, b] (annotation space)."""
    return (_best_overlap(segments["caption_segments"], a, b, "caption"),
            _best_overlap(segments["command_timeline"], a, b, "command"))


def download_language_annotations(annotations_dir: Path, files=None) -> list[Path]:
    """Download OpenDV-YouTube-Language JSON splits from HuggingFace."""
    from huggingface_hub import hf_hub_download

    annotations_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fn in (files or LANG_FILES):
        log.info("downloading annotation split %s", fn)
        paths.append(Path(hf_hub_download(
            repo_id=LANG_REPO, repo_type="dataset", filename=fn,
            local_dir=str(annotations_dir))))
    return paths


def load_labels(annotations_dir: Path, bin_sec: int = DEFAULT_BIN_SEC, files=None) -> dict:
    """Load + aggregate all annotation splits present in ``annotations_dir`` into
    ``{video_id: {caption_segments, command_timeline}}``. Streams file-by-file."""
    paths = ([annotations_dir / f for f in files] if files
             else sorted(annotations_dir.glob("10hz_*.json")))
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise FileNotFoundError(
            f"No annotation JSONs in {annotations_dir}; download them first.")
    merged: dict = {}
    for p in paths:
        log.info("parsing %s", p.name)
        with open(p) as f:
            entries = json.load(f)
        for vid, segs in segments_from_entries(entries, bin_sec).items():
            merged[vid] = segs   # each video lives in exactly one split file
    return merged
