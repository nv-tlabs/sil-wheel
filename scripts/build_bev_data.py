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

"""Build per-clip BEV msgpack files that wheel's BEV viewer can render.

Reads the Physical AI Autonomous Vehicles egomotion and obstacle labels and
writes one <clip_id>.msgpack per clip in the flat-array format documented at the
top of sil_wheel/app/static/js/bev/bev-binary-utils.js, so BEVFetcher can serve
them straight off local disk.

Inputs per clip, as laid down by the getting-started setup:

  <egomotion_dir>/<clip_id>.egomotion.offline.parquet   ego pose @ ~10 Hz
  <egomotion_dir>/<clip_id>.timestamps.parquet          camera frame times
  <obstacle_dir>/<clip_id>.obstacle.offline.parquet     3D boxes, rig frame

Coordinates
-----------
The dataset's egomotion is FLU (x forward, y left, z up), the same convention
scripts/extract_trajectory_stats.py documents for its physical_ai source, and
it is already clip-local: the first sample sits at the origin with an identity
rotation. The BEV renderer draws +Y up the screen and treats up as forward
(see renderFrame in bev-renderer.js), so every point is mapped FLU -> BEV as

    X_bev = -y_flu      (screen right)
    Y_bev =  x_flu      (screen forward)

Obstacles are stored in the rig frame at their own timestamp, so each box is
rotated into the clip-local frame by the ego pose at that moment.

Frames
------
Obstacle observations are per-track and staggered: every track is sampled at
~10 Hz but with its own phase, so almost every row carries a distinct
timestamp. Frames are therefore built on the ego clock, and each track
contributes the observation nearest that frame time within --match-tolerance.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import msgpack
import numpy as np
import pandas as pd


log = logging.getLogger("build_bev_data")

# Half of the ~10 Hz obstacle sampling period: a track contributes to a frame
# only when it was actually observed near that moment.
DEFAULT_MATCH_TOLERANCE_S = 0.05

# Used when calibration/vehicle_dimensions has no row for a clip.
FALLBACK_EGO_LENGTH_M = 4.872
FALLBACK_EGO_WIDTH_M = 2.121
FALLBACK_REAR_AXLE_TO_BBOX_CENTER_M = 1.327


def quat_to_yaw(qx, qy, qz, qw):
    """Yaw about +z from a quaternion, for a right-handed z-up frame."""
    return np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def corners_flu_to_bev(cx, cy):
    """Map FLU (x forward, y left) plane coordinates to the viewer's axes."""
    return -cy, cx


def box_corners(centre_x, centre_y, yaw, length, width):
    """Four BEV corners per pose, as (F, 4) arrays in the FLU plane.

    centre_x/centre_y/yaw are per-frame arrays; length/width are scalars or
    per-frame arrays. Corner order is front-left, front-right, rear-right,
    rear-left so the renderer traces a proper polygon.
    """
    half_l = np.asarray(length) / 2.0
    half_w = np.asarray(width) / 2.0
    # (4, 2) offsets in the box's own frame, broadcast over frames below.
    signs = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])
    local_x = signs[:, 0][None, :] * np.atleast_1d(half_l)[:, None]
    local_y = signs[:, 1][None, :] * np.atleast_1d(half_w)[:, None]

    cos_y = np.cos(yaw)[:, None]
    sin_y = np.sin(yaw)[:, None]
    x = centre_x[:, None] + cos_y * local_x - sin_y * local_y
    y = centre_y[:, None] + sin_y * local_x + cos_y * local_y
    return x, y


def flatten_corners(x, y):
    """Interleave (N, 4) corner arrays into the flat [x0,y0,x1,y1,...] form."""
    bev_x, bev_y = corners_flu_to_bev(x, y)
    out = np.empty((bev_x.shape[0], 8), dtype=np.float32)
    out[:, 0::2] = bev_x
    out[:, 1::2] = bev_y
    return out


def read_ego(egomotion_path: Path):
    """Ego clock, position and yaw, in the clip-local FLU frame."""
    df = pd.read_parquet(egomotion_path)
    t = df["timestamp"].astype(np.float64).to_numpy() / 1e6
    x = df["x"].astype(np.float64).to_numpy()
    y = df["y"].astype(np.float64).to_numpy()
    yaw = quat_to_yaw(
        df["qx"].astype(np.float64).to_numpy(),
        df["qy"].astype(np.float64).to_numpy(),
        df["qz"].astype(np.float64).to_numpy(),
        df["qw"].astype(np.float64).to_numpy(),
    )
    return t, x, y, yaw


def video_start_time(egomotion_path: Path, ego_t0: float) -> float:
    """First camera frame time, on the ego clock.

    Frame times are emitted relative to this rather than to the ego origin, so
    the viewer's ``bevDataTime = videoTime + video_offset`` works with an offset
    of zero. Baking the shift into the timestamps matters because the camera
    typically starts a few tens of milliseconds *before* the ego clock: leaving
    it in video_offset would make bevDataTime negative at videoTime 0, which the
    viewer rejects outright with "No BEV data at this timestamp" the instant a
    clip is opened.
    """
    ts_path = Path(str(egomotion_path).replace(
        ".egomotion.offline.parquet", ".timestamps.parquet"
    ))
    if not ts_path.exists():
        log.warning("No camera timestamps at %s; aligning BEV to the ego clock", ts_path)
        return ego_t0
    try:
        cam = pd.read_parquet(ts_path)
        cam_t = cam[cam.columns[0]].astype(np.float64).to_numpy() / 1e6
        return float(cam_t[0])
    except (KeyError, IndexError, OSError) as exc:
        log.warning("Unreadable camera timestamps %s (%s)", ts_path, exc)
        return ego_t0


def ego_dimensions(dims: pd.DataFrame | None, clip_id: str):
    """(length, width, rear_axle_to_bbox_center) for a clip, with fallbacks."""
    if dims is not None and clip_id in dims.index:
        row = dims.loc[clip_id]
        return (
            float(row["length"]),
            float(row["width"]),
            float(row.get("rear_axle_to_bbox_center", FALLBACK_REAR_AXLE_TO_BBOX_CENTER_M)),
        )
    return (
        FALLBACK_EGO_LENGTH_M,
        FALLBACK_EGO_WIDTH_M,
        FALLBACK_REAR_AXLE_TO_BBOX_CENTER_M,
    )


def obstacle_corners_per_frame(
    obstacles: pd.DataFrame,
    ego_t, ego_x, ego_y, ego_yaw,
    tolerance: float,
):
    """Per-frame lists of flat corner arrays for every observed obstacle.

    Each track is matched independently to the frame grid: for frame time T the
    track contributes its nearest observation when that observation is within
    `tolerance` seconds, and nothing otherwise. This keeps a track from being
    drawn at moments it was never seen, and avoids extrapolating stale boxes.
    """
    n_frames = len(ego_t)
    per_frame: list[list[np.ndarray]] = [[] for _ in range(n_frames)]
    if obstacles.empty:
        return per_frame

    for _, track in obstacles.groupby("track_id", sort=False):
        t_obs = track["timestamp_us"].astype(np.float64).to_numpy() / 1e6
        order = np.argsort(t_obs)
        t_obs = t_obs[order]

        # Nearest observation index for every frame time.
        right = np.searchsorted(t_obs, ego_t)
        lo = np.clip(right - 1, 0, len(t_obs) - 1)
        hi = np.clip(right, 0, len(t_obs) - 1)
        d_lo = np.abs(ego_t - t_obs[lo])
        d_hi = np.abs(ego_t - t_obs[hi])
        nearest = np.where(d_lo <= d_hi, lo, hi)
        visible = np.minimum(d_lo, d_hi) <= tolerance
        if not visible.any():
            continue

        frames = np.flatnonzero(visible)
        sel = nearest[frames]
        cx_rig = track["center_x"].astype(np.float64).to_numpy()[order][sel]
        cy_rig = track["center_y"].astype(np.float64).to_numpy()[order][sel]
        length = track["size_x"].astype(np.float64).to_numpy()[order][sel]
        width = track["size_y"].astype(np.float64).to_numpy()[order][sel]
        yaw_rig = quat_to_yaw(
            track["orientation_x"].astype(np.float64).to_numpy()[order][sel],
            track["orientation_y"].astype(np.float64).to_numpy()[order][sel],
            track["orientation_z"].astype(np.float64).to_numpy()[order][sel],
            track["orientation_w"].astype(np.float64).to_numpy()[order][sel],
        )

        # rig -> clip-local, using the ego pose at each matched frame.
        e_yaw = ego_yaw[frames]
        cos_e, sin_e = np.cos(e_yaw), np.sin(e_yaw)
        world_x = ego_x[frames] + cos_e * cx_rig - sin_e * cy_rig
        world_y = ego_y[frames] + sin_e * cx_rig + cos_e * cy_rig

        cx, cy = box_corners(world_x, world_y, e_yaw + yaw_rig, length, width)
        flat = flatten_corners(cx, cy)
        for row, frame in enumerate(frames):
            per_frame[frame].append(flat[row])

    return per_frame


def build_clip(
    clip_id: str,
    egomotion_path: Path,
    obstacle_path: Path | None,
    dims: pd.DataFrame | None,
    tolerance: float,
) -> dict | None:
    ego_t, ego_x, ego_y, ego_yaw = read_ego(egomotion_path)
    if len(ego_t) < 2:
        log.warning("%s: fewer than 2 ego samples; skipping", clip_id)
        return None

    n_frames = len(ego_t)
    t0 = float(ego_t[0])
    # Frame times are relative to the first video frame, so video_offset is 0.
    video_start = video_start_time(egomotion_path, t0)
    rel_t = (ego_t - video_start).astype(np.float32)

    length, width, axle_to_centre = ego_dimensions(dims, clip_id)
    # The ego pose tracks the rear axle, so the body box sits forward of it.
    ego_cx = ego_x + np.cos(ego_yaw) * axle_to_centre
    ego_cy = ego_y + np.sin(ego_yaw) * axle_to_centre
    ex, ey = box_corners(ego_cx, ego_cy, ego_yaw, length, width)
    ego_flat = flatten_corners(ex, ey)

    obstacles = pd.DataFrame()
    if obstacle_path is not None and obstacle_path.exists():
        try:
            obstacles = pd.read_parquet(obstacle_path)
        except OSError as exc:
            log.warning("%s: unreadable obstacles (%s)", clip_id, exc)
    other = obstacle_corners_per_frame(
        obstacles, ego_t, ego_x, ego_y, ego_yaw, tolerance
    )

    bev_ego_x, bev_ego_y = corners_flu_to_bev(ego_x, ego_y)
    ego_positions = np.empty(n_frames * 2, dtype=np.float32)
    ego_positions[0::2] = bev_ego_x
    ego_positions[1::2] = bev_ego_y

    # The dataset ships no map layer, so these stay empty for every frame. The
    # renderer handles count == 0 and simply draws nothing.
    empty_segments = [{"coords": [], "count": 0} for _ in range(n_frames)]

    clip = {
        "clip_id": clip_id,
        "num_frames": int(n_frames),
        "duration": float(ego_t[-1] - video_start),
        "video_offset": 0.0,
        "t_origin": int(round(t0 * 1e6)),
        "base_timestamp": 0.0,
        "timestamps": rel_t.tolist(),
        "road_boundaries": empty_segments,
        "lane_lines": list(empty_segments),
        "ego_vehicles": [
            {"corners": ego_flat[i].tolist(), "count": 1} for i in range(n_frames)
        ],
        "other_vehicles": [
            {
                "corners": np.concatenate(boxes).tolist() if boxes else [],
                "count": len(boxes),
            }
            for boxes in other
        ],
        "ego_positions": ego_positions.tolist(),
    }
    return clip


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("egomotion_dir", type=Path,
                        help="Directory of <clip>.egomotion.offline.parquet files.")
    parser.add_argument("obstacle_dir", type=Path,
                        help="Directory of <clip>.obstacle.offline.parquet files.")
    parser.add_argument("output_dir", type=Path,
                        help="Where to write <clip>.msgpack files.")
    parser.add_argument("--vehicle-dimensions", type=Path, default=None,
                        help="calibration/vehicle_dimensions parquet, indexed by "
                             "clip_id. Falls back to typical dimensions when absent.")
    parser.add_argument("--match-tolerance", type=float,
                        default=DEFAULT_MATCH_TOLERANCE_S,
                        help="Max seconds between a frame and an obstacle "
                             "observation for that track to be drawn "
                             "(default %(default)s).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild clips whose .msgpack already exists.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    ego_paths = sorted(args.egomotion_dir.glob("*.egomotion.offline.parquet"))
    if not ego_paths:
        log.error("No egomotion parquet under %s", args.egomotion_dir)
        sys.exit(1)

    dims = None
    if args.vehicle_dimensions is not None and args.vehicle_dimensions.exists():
        dims = pd.read_parquet(args.vehicle_dimensions)
        log.info("Loaded ego dimensions for %d clips", len(dims))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written, skipped, no_obstacles = 0, 0, 0
    for ego_path in ego_paths:
        clip_id = ego_path.name.split(".")[0]
        out_path = args.output_dir / f"{clip_id}.msgpack"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        obstacle_path = args.obstacle_dir / f"{clip_id}.obstacle.offline.parquet"
        if not obstacle_path.exists():
            no_obstacles += 1
            obstacle_path = None

        clip = build_clip(clip_id, ego_path, obstacle_path, dims, args.match_tolerance)
        if clip is None:
            continue
        out_path.write_bytes(
            msgpack.packb({"clips": [clip], "batch_size": 1}, use_single_float=True)
        )
        written += 1

    log.info(
        "BEV written: %d clips -> %s (%d already present, %d without obstacle labels)",
        written, args.output_dir, skipped, no_obstacles,
    )
    if written == 0 and skipped == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
