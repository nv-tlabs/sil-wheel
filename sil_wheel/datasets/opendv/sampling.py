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

"""Sample 20 s subclips from local OpenDV videos.

Two methods:
  - ``uniform``: one clip every ``interval`` seconds (content-blind baseline).
  - ``diverse``: rarity-weighted Maximal Marginal Relevance over candidate
    windows labeled with their dominant OpenDV (cmd, blip) — relevance is
    maneuver rarity, diversity penalizes caption overlap + same maneuver.
Clips are cut with ffmpeg and paired with a manifest keyed by ``clip_id`` =
``<video_id>__<video_id>_<start>-<end>`` (matches the caption benchmark).
"""
import json
import logging
import math
import shutil
import subprocess
from pathlib import Path

from sil_wheel.datasets.opendv import labels as opendv_labels
from sil_wheel.datasets.opendv.constants import CLIP_SEC, DEFAULT_BIN_SEC, DEFAULT_INTERVAL, DEFAULT_STRIDE

log = logging.getLogger(__name__)

_VIDEO_EXTS = ("mp4", "webm", "mkv")


def clip_id_for(video_id: str, a: float, b: float) -> str:
    """Benchmark-compatible id: ``<video_id>__<video_id>_<start>-<end>``."""
    return f"{video_id}__{video_id}_{int(round(a))}-{int(round(b))}"


def uniform_windows(start: float, end: float, interval: int, clip_sec: int = CLIP_SEC):
    """One [t, t+clip_sec] window at each ``interval`` mark within [start, end]."""
    out = []
    t = math.ceil(start / interval) * interval
    while t + clip_sec <= end:
        out.append((float(t), float(t + clip_sec)))
        t += interval
    return out


def _jaccard(x: str, y: str) -> float:
    sx, sy = set(x.lower().split()), set(y.lower().split())
    return len(sx & sy) / len(sx | sy) if (sx or sy) else 0.0


def _sim(c1: dict, c2: dict) -> float:
    same_cmd = 1.0 if c1["cmd"] == c2["cmd"] and c1["cmd"] else 0.0
    return 0.5 * _jaccard(c1["cap"], c2["cap"]) + 0.5 * same_cmd


def diverse_windows(candidates: list[dict], k: int | None = None,
                    total: int | None = None, lam: float = 0.5) -> list[dict]:
    """Greedy MMR selection over labeled candidates (each ``{a,b,cap,cmd}``).

    relevance = maneuver rarity ``log(N / (freq(cmd)+1))``; similarity =
    ``0.5*Jaccard(cap) + 0.5*[same cmd]``. Picks are non-overlapping in time.
    ``k`` caps the count; ``total`` is an alias for ``k`` here (per-video call).
    """
    cap_n = total if total is not None else (k if k is not None else len(candidates))
    if not candidates or cap_n <= 0:
        return []
    freq = {}
    for c in candidates:
        freq[c["cmd"]] = freq.get(c["cmd"], 0) + 1
    n = sum(freq.values()) or 1
    rel = {i: math.log(n / (freq[c["cmd"]] + 1)) for i, c in enumerate(candidates)}
    mx = max(rel.values()) or 1.0
    rel = {i: v / mx for i, v in rel.items()}

    selected: list[int] = []

    def overlaps(i):
        a, b = candidates[i]["a"], candidates[i]["b"]
        return any(a < candidates[j]["b"] and candidates[j]["a"] < b for j in selected)

    remaining = set(range(len(candidates)))
    while remaining and len(selected) < cap_n:
        best_i, best = None, -1e9
        for i in remaining:
            if overlaps(i):
                continue
            sim = max((_sim(candidates[i], candidates[j]) for j in selected), default=0.0)
            score = lam * rel[i] - (1 - lam) * sim
            if score > best:
                best, best_i = score, i
        if best_i is None:
            break
        selected.append(best_i)
        remaining.discard(best_i)
    return [candidates[i] for i in selected]


def find_video(videos_dir: Path, video_id: str) -> Path | None:
    """Locate ``<video_id>.<ext>`` under ``videos_dir``, in any layout.

    Tries a flat ``videos_dir/<video_id>.<ext>`` first, then searches
    recursively — so it works whether videos are stored flat or nested (e.g.
    the official OpenDV ``<youtuber>/<videoid>`` layout).
    """
    for ext in _VIDEO_EXTS:
        p = videos_dir / f"{video_id}.{ext}"
        if p.exists():
            return p
    for ext in _VIDEO_EXTS:
        matches = sorted(videos_dir.rglob(f"{video_id}.{ext}"))
        if matches:
            return matches[0]
    return None


def ffprobe_duration(video: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found on PATH")
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(video)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def cut_clip(video: Path, a: float, b: float, out: Path, cut: str = "nvenc") -> bool:
    """Cut [a, b] from ``video`` to ``out``. cut in {nvenc, libx264, copy}."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{a}", "-i", str(video), "-t", f"{b - a}"]
    if cut == "copy":
        cmd += ["-c", "copy"]
    elif cut == "libx264":
        cmd += ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
                "-pix_fmt", "yuv420p", "-an"]
    else:  # nvenc (default): exact GPU cut; -pix_fmt yuv420p downconverts 10-bit
        cmd += ["-c:v", "h264_nvenc", "-preset", "medium", "-rc", "vbr", "-cq", "23",
                "-b:v", "0", "-pix_fmt", "yuv420p", "-an"]
    cmd += ["-movflags", "+faststart", str(out)]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0


def _candidates(video_id, start, end, segments, clip_sec, stride):
    """Stride ``clip_sec`` windows over [start, end], label each with dominant
    (cap, cmd). Annotation times are offset by -start: annotation 0 == the
    original-video time ``start_discard``."""
    out = []
    t = math.ceil(start)
    while t + clip_sec <= end:
        cap, cmd = opendv_labels.dominant(segments, t - start, t + clip_sec - start)
        if cmd:                       # keep only windows with annotation coverage
            out.append({"a": float(t), "b": float(t + clip_sec), "cap": cap or "", "cmd": cmd})
        t += stride
    return out


def sample(records, videos_dir: Path, output_dir: Path, method: str = "uniform",
           interval: int = DEFAULT_INTERVAL, clip_sec: int = CLIP_SEC,
           stride: int = DEFAULT_STRIDE, select_k: int | None = None,
           total: int | None = None, lam: float = 0.5, cut: str = "nvenc",
           annotations_dir: Path | None = None, bin_sec: int = DEFAULT_BIN_SEC) -> Path:
    """Sample clips for all ``records`` into ``output_dir/<method>/`` and write a
    manifest. ``records`` are metadata dicts with videoid/start_discard/end_discard.
    """
    method_dir = output_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    label_map = (opendv_labels.load_labels(annotations_dir, bin_sec)
                 if method == "diverse" else {})

    # Pass 1: per usable video, build uniform windows directly, or collect the
    # labeled candidate windows that feed the diverse ranker.
    ranked_per_video: dict = {}
    cand_per_video: dict = {}
    for rec in records:
        vid = rec["videoid"]
        video = find_video(videos_dir, vid)
        if video is None:
            log.warning("%s: no local video, skipping", vid)
            continue
        dur = ffprobe_duration(video)
        if not dur:
            log.warning("%s: no duration, skipping", vid)
            continue
        start = float(rec.get("start_discard", 0) or 0)
        end = max(start, dur - float(rec.get("end_discard", 0) or 0))
        if method == "uniform":
            ranked_per_video[(vid, video)] = [
                {"a": a, "b": b, "cap": "", "cmd": ""}
                for a, b in uniform_windows(start, end, interval, clip_sec)]
        else:
            segs = label_map.get(vid)
            if not segs:
                log.warning("%s: no annotations, skipping", vid)
                continue
            cands = _candidates(vid, start, end, segs, clip_sec, stride)
            if cands:
                cand_per_video[(vid, video)] = cands

    # Pass 2 (diverse): rank each video's candidates. The per-video cap is based
    # on the number of videos that actually produced candidates (not len(records)),
    # so the global --total round-robin can reach its target even when only a
    # subset of videos is downloaded/annotated.
    if method == "diverse":
        n_vid = max(1, len(cand_per_video))
        per_cap = (math.ceil(total / n_vid) + 20) if total else select_k
        for key, cands in cand_per_video.items():
            ranked_per_video[key] = diverse_windows(cands, k=per_cap, lam=lam)

    chosen = _select_global(ranked_per_video, method, select_k, total)

    manifest = method_dir / "manifest.jsonl"
    rows = []
    for (vid, video), a, b, cap, cmd in chosen:
        cid = clip_id_for(vid, a, b)
        out = method_dir / vid / f"{vid}_{int(round(a))}-{int(round(b))}.mp4"
        if not cut_clip(video, a, b, out, cut):
            log.warning("cut failed: %s", cid)
            continue
        row = {"clip_id": cid, "video_id": vid, "clip_path": str(out),
               "start_sec": round(a, 2), "end_sec": round(b, 2), "method": method}
        if method == "diverse":
            row["dominant_command"] = cmd
            row["dominant_caption"] = cap
        rows.append(row)
    with open(manifest, "w") as f:
        for r in sorted(rows, key=lambda r: (r["video_id"], r["start_sec"])):
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
    if total and len(rows) < total:
        log.warning("requested --total %d but produced only %d clips "
                    "(not enough candidates across %d videos)",
                    total, len(rows), len(cand_per_video))
    log.info("wrote %d clips -> %s", len(rows), manifest)
    return manifest


def _select_global(ranked_per_video, method, select_k, total):
    """Flatten per-video windows into chosen ((vid, video), a, b, cap, cmd).
    For diverse --total: round-robin across videos to the cap."""
    flat = []
    if method == "diverse" and total:
        i = 0
        picked = 0
        items = list(ranked_per_video.items())
        while picked < total and any(i < len(w) for _, w in items):
            for key, wins in items:
                if i < len(wins) and picked < total:
                    flat.append((key, wins[i]))
                    picked += 1
            i += 1
    else:
        for key, wins in ranked_per_video.items():
            sel = wins[:select_k] if (method == "diverse" and select_k) else wins
            flat.extend((key, w) for w in sel)
    return [(key, w["a"], w["b"], w["cap"], w["cmd"]) for key, w in flat]
