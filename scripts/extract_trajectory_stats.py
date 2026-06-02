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

import argparse
import io
from io import BytesIO
import numpy as np
import pandas as pd
from pathlib import Path
import tarfile
import tempfile

from safetensors.numpy import save_file, load_file
from safetensors import SafetensorError
from scipy.signal import savgol_filter
from tqdm import tqdm

from sil_wheel.datasets.base_dataset import S3ObjectFetcher


def compute_speed(x, y, z, ts):
    """Computes the instantaneous speed of the ego-vehicle.

    Args:
        x (np.array): Array of x-coordinates.
        y (np.array): Array of y-coordinates.
        z (np.array): Array of z-coordinates.
        ts (np.array): Array of timestamps in seconds.

    Returns:
        Return the norm of the speed
    """
    vx = np.gradient(x, ts)
    vy = np.gradient(y, ts)
    vz = np.gradient(z, ts)
    return np.sqrt(vx**2 + vy**2 + vz**2)


def compute_curvature(x, y, ts):
    """
    Computes the 2D curvature of the ego-vehicle's path.

    Curvature measures how sharply a curve bends. A higher curvature value
    indicates a sharper turn. This statistic is crucial for identifying
    turning maneuvers (e.g., left/right turns, U-turns) and for
    characterizing the curviness of a road.

    The formula used here is based on the derivative of the heading angle
    with respect to arc length, approximated using finite differences.
    For 2D curvature, we project the 3D path onto the XY plane.

    Args:
        x (np.array): Array of x-coordinates.
        y (np.array): Array of y-coordinates.
        ts (np.array): Array of timestamps in seconds.

    Returns:
        An array containing the absolute curvature with 0s when poorly defined
    """
    # Calculate first derivatives (velocities)
    vx = np.gradient(x, ts)
    vy = np.gradient(y, ts)

    # Calculate second derivatives (accelerations)
    ax = np.gradient(vx, ts)
    ay = np.gradient(vy, ts)

    # Avoid division by zero in case of zero speed
    denominator = (vx**2 + vy**2)**1.5
    denominator = np.maximum(denominator, 1e-6)

    # Curvature for well defined points only
    kappa = np.where(
        np.sqrt(vx**2 + vy**2) < 2/3.6,
        0,
        np.abs(vx * ay - vy * ax) / denominator
    )

    return kappa


def get_trajectory_data(path_to_recordings):
    path_to_frame_ts = path_to_recordings + ".timestamps"
    with open(path_to_frame_ts, "r") as f:
        frame_ts = f.readlines()
    frame_ts = np.array([int(ts.strip().split("\t")[-1]) for ts in frame_ts]) / 1e6

    base_dir_to_ego_data = Path(path_to_recordings).parent.parent.parent
    path_to_ego = list(base_dir_to_ego_data.rglob("egomotion_estimate.parquet"))
    if len(path_to_ego) == 0:
        return None

    if path_to_ego[0].exists() and path_to_ego[0].stat().st_size == 0:
        return None

    try:
        df = pd.read_parquet(path_to_ego)
    except (pyarrow.lib.ArrowInvalid, FileNotFoundError) as e:
        print(f"Failed to read {path_to_ego}: {e}")
        return None

    # Convert timestamps from microseconds to seconds
    try:
        ts = df["key.timestamp_micros"].astype(np.float64).to_numpy() / 1e6
        x = df["egomotion_estimate.location.x"].astype(np.float64).to_numpy()
        y = df["egomotion_estimate.location.y"].astype(np.float64).to_numpy()
        z = df["egomotion_estimate.location.z"].astype(np.float64).to_numpy()
    except KeyError as e:
        return None

    return x, y, z, ts, frame_ts


def get_trajectory_data_from_vipe_poses(path_to_recordings):
    """Load VIPE camera-to-world poses and extract the ego trajectory.

    The .npz contains:
      - 'data': (N, 4, 4) camera-to-world transforms, X_w = R @ X_c + t,
        with R = c2w[:, :3, :3] and t = c2w[:, :3, 3]
      - 'inds': (N,) frame indices for each pose

    VIPE produces one pose per video frame at a fixed 30 fps, so the
    pose timestamps and the frame timestamps are in sync; both are
    returned as the same evenly spaced array.

    Returned x/y/z are in VIPE camera convention (x->right, y->down,
    z->forward). The caller rotates them into the Alpamayo convention.
    """
    npz = np.load(path_to_recordings)
    cam2world = npz["data"].astype(np.float64)
    inds = npz["inds"]

    order = np.argsort(inds)
    if not np.array_equal(order, np.arange(len(inds))):
        cam2world = cam2world[order]

    num_frames = len(cam2world)
    ts = np.arange(num_frames, dtype=np.float64) / 30.0
    frame_ts = ts.copy()

    initial_transform = np.linalg.inv(cam2world[0])
    relative_transforms = initial_transform @ cam2world
    xyz = relative_transforms[:, :3, 3]

    return xyz[:, 0], xyz[:, 1], xyz[:, 2], ts, frame_ts


def get_trajectory_data_clipgt(path_to_recordings):
    path_to_frame_ts = str(path_to_recordings).replace(".mp4", ".timestamps")
    with open(path_to_frame_ts, "r") as f:
        frame_ts = f.readlines()
    frame_ts = np.array([int(ts.strip().split("\t")[-1]) for ts in frame_ts]) / 1e6

    base_dir_to_ego_data = path_to_recordings.parent
    path_to_ego = list(base_dir_to_ego_data.rglob("egomotion_estimate.parquet"))
    if len(path_to_ego) == 0:
        return None

    if path_to_ego[0].exists() and path_to_ego[0].stat().st_size == 0:
        return None

    try:
        df = pd.read_parquet(path_to_ego)
    except (pyarrow.lib.ArrowInvalid, FileNotFoundError) as e:
        print(f"Failed to read {path_to_ego}: {e}")
        return None

    # Convert timestamps from microseconds to seconds
    try:
        ts = df["key.timestamp_micros"].astype(np.float64).to_numpy() / 1e6
        x = df["egomotion_estimate.location.x"].astype(np.float64).to_numpy()
        y = df["egomotion_estimate.location.y"].astype(np.float64).to_numpy()
        z = df["egomotion_estimate.location.z"].astype(np.float64).to_numpy()
    except KeyError as e:
        return None

    return x, y, z, ts, frame_ts


def get_sample_from_bytes(tar_bytes: bytes) -> dict:
    """
    Get a sample from in-memory tar bytes with decoding.

    This parses the tar file directly from memory without writing to disk,
    matching the behavior of WebDataset's decode() for .npy files.

    Returns:
        Dictionary mapping filenames to decoded numpy arrays
    """
    sample = {}
    with tarfile.open(tar_bytes, mode='r:*') as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            data = f.read()

            # Get the key (filename without extension for matching WebDataset behavior)
            name = member.name

            # Decode .npy files (the vehicle_pose format)
            if name.endswith('.npy'):
                sample[name[:-4]] = np.load(io.BytesIO(data))
            else:
                # Store raw bytes for other formats
                sample[name] = data

    return sample


def extract_poses_from_tar(tar) -> np.ndarray:
    """
    Extract poses from the tar file

    Returns:
        numpy array of shape (T, 4, 4) containing all pose matrices
    """
    sample = get_sample_from_bytes(tar)
    # Filter for vehicle_pose entries and sort by frame index
    pose_data = {k: v for k, v in sample.items() if "vehicle_pose" in k}
    if not pose_data:
        return None
    pose_all_frames = np.stack([
        pose_data[k] for k in sorted(pose_data.keys())
    ])
    return pose_all_frames.astype(np.float64)


def extract_timestamps(ts_path) -> np.ndarray:
    if Path(ts_path).exists():
        with open(ts_path, "r") as f:
            lines = f.readlines()
        parsed = [int(line.strip().split("\t")[-1]) for line in lines]
        ts = np.array(parsed).astype(np.float64) / 1e6
    else:
        return None


def compute_stats_from_clip(
    path_to_recordings,
    data_type=None,
    vipe_trajectories=False
):
    path_to_recordings = str(path_to_recordings)
    if data_type == "clipgt":
        x, y, z, ts, frame_ts = get_trajectory_data_clipgt(
            path_to_recordings
        )
    elif data_type == "vipe":
        x, y, z, ts, frame_ts = get_trajectory_data_from_vipe_poses(
            path_to_recordings
        )
        # Set the flag in case it's not already set
        vipe_trajectories = True

    elif data_type == "waymo":
        ego_path = path_to_recordings.replace("camera_front_50fov_fps10.mp4", "xyzt.npz")
        data = np.load(ego_path)
        ts = data["egomotion_estimate.timestamp_micros"].astype(np.float64) / 1e6
        x = data["egomotion_estimate.location.x"].astype(np.float64)
        y = data["egomotion_estimate.location.y"].astype(np.float64)
        z = data["egomotion_estimate.location.z"].astype(np.float64)
        frame_ts = data["camera_front_left_50fov.timestamp_micros"].astype(np.float64) / 1e6
    elif data_type == "mads":
        # For MADS data we follow the FLU convention
        # x -> forward, y -> left and z -> upward
        df = pd.read_parquet(path_to_recordings)
        ts = df["key.timestamp"].astype(np.float64).to_numpy()
        x = df["egomotion_estimate.location.x"].astype(np.float64).to_numpy()
        y = -df["egomotion_estimate.location.y"].astype(np.float64).to_numpy()
        z = df["egomotion_estimate.location.z"].astype(np.float64).to_numpy()
        # The video and pose timestamp are synchronized
        frame_ts = None
    elif data_type == "mads-1M":
        pose_tar_path = Path(path_to_recordings).parents[2] / "vehicle_pose" / f"{local_mp4.stem}.tar"
        if not pose_tar_path.exists():
            return None

        poses = extract_poses_from_tar(pose_tar_path)
        if poses is None:
            return None

        # For MADS data we follow the FLU convention
        # x -> forward, y -> left and z -> upward
        translation = poses[:, :3, 3]
        x = translation[:, 0].astype(np.float64)
        y = -translation[:, 1].astype(np.float64)
        z = translation[:, 2].astype(np.float64)

        ts_path = str(pose_tar_path).replace(".tar", ".timestamps")
        ts = extract_timestamps(ts_path)
        if ts is None:
            # We extract the timestamps using the video legth and the fact that
            # fps=30
            ts = np.linspace(0, x.shape[0] / 30.0, x.shape[0])
            t = ts.astype(np.float64)
        if len(poses) != len(ts):
            x = x[:len(ts)]
            y = y[:len(ts)]
            z = z[:len(ts)]
        # The video and pose timestamp are synchronized
        frame_ts = None

        # Delete the temp files
        if pose_tar_path.exists():
            pose_tar_path.unlink()

        ts_file_path = Path(ts_path)
        if ts_file_path.exists():
            ts_file_path.unlink()

        Path(path_to_recordings).unlink()
    else:
        x, y, z, ts, frame_ts = get_trajectory_data(path_to_recordings)

    # Compute the full stats wrt the ego-motion timestamps
    if vipe_trajectories:
        # Rearranging axes from Vipe coordinates:
        #       x -> right, y -> downward, z -> forward
        # to Alpamayo coordinates:
        #       x -> forward, z -> upward, y -> right
        x, y, z = z, x, -y

    speed = compute_speed(x, y, z, ts)
    speed = savgol_filter(speed, window_length=11, polyorder=3)
    acceleration = np.gradient(speed, ts)
    acceleration = savgol_filter(acceleration, window_length=11, polyorder=3)
    jerk = np.gradient(acceleration, ts)
    curvature = compute_curvature(x, y, ts)

    # We assume that the camera and the ego are already in sync
    if frame_ts is None:
        stats = np.stack(
            [x, y, z, speed, acceleration, jerk, curvature], axis=1
        )
        return stats.astype(np.float32)

    # Convert them to the frame timestamps and return them
    try:
        ego_ts = ts - frame_ts[0]
    except IndexError:
        return None

    frame_ts = frame_ts - frame_ts[0]
    stats = np.stack(
        [
            np.interp(frame_ts, ego_ts, x),
            np.interp(frame_ts, ego_ts, y),
            np.interp(frame_ts, ego_ts, z),
            np.interp(frame_ts, ego_ts, speed),
            np.interp(frame_ts, ego_ts, acceleration),
            np.interp(frame_ts, ego_ts, jerk),
            np.interp(frame_ts, ego_ts, curvature),
        ],
        axis=1
    )

    return stats.astype(np.float32)


def get_clip_id_index_from_paths(path_to_data, bucket=None):
    if bucket is not None:
        return -1, "mads-1M"
    if str(path_to_data[0]).endswith(".parquet"):
        return -1, "mads"
    if str(path_to_data[0]).endswith(".npz"):
        return -1, "vipe"
    if "recordings" in str(path_to_data[0]):
        paths_parts = [Path(p).parts for p in path_to_data[:3]]
        cols = list(zip(*paths_parts))

        # Find the index where all paths have 'recordings'
        rec_idx = next(i for i, col in enumerate(cols) if all(x == "recordings"  for x in col))

        clip_id_idx = rec_idx - 1
        return clip_id_idx, "av_prod_v2"
    else:
        # We assume that all paths have the same format, if not this won't work
        paths_parts = []
        for clip in path_to_data[:3]:
            parts = Path(clip).parts
            paths_parts.append(parts)

        # All indices where at least one tuple differs. If a part from the path
        # contains the word "camera" we also ignore it.
        diff_ind = [
            i for i, col in enumerate(zip(*paths_parts))
            if len(set(col)) > 1 and not any("camera" in item for item in col)
        ]
        # There should be a difference only on one part of the path
        assert len(diff_ind) == 1
        if diff_ind[0] == 11:
            return diff_ind[0], "clip_gt"
        elif diff_ind[0] == 12:
            return diff_ind[0], "waymo"
        else:
            return diff_ind[0], None


def mads_pose_data_for_front_wide(mp4_key: str, local_mp4_path: Path):
    """
    For an mp4 key like:
      Austria_20s/chunk_1/ftheta_camera_front_wide_120fov/<stem>.mp4

    Also downloads:
      Austria_20s/chunk_1/vehicle_pose/<stem>.tar
      Austria_20s/chunk_1/ftheta_camera_front_wide_120fov_timestmap/<stem>.timestamps

    and stores timestamps locally as: <local_mp4_path>.timestamps
    """
    k = str(mp4_key).lstrip("/")
    p = Path(k)

    stem = p.stem  # uuid_t0_t1
    # e.g. Austria_20s/chunk_1
    prefix = Path(*p.parts[:2])

    # vehicle_pose tar
    pose_key = str(prefix / "vehicle_pose" / f"{stem}.tar")
    pose_local = (local_mp4_path.parents[2] / "vehicle_pose" / f"{stem}.tar").resolve()

    # per-frame timestamps (exact .timestamps filename)
    ts_key = str(prefix / "ftheta_camera_front_wide_120fov_timestmap" / f"{stem}.timestamps")
    ts_local = Path(str(pose_local).replace(".tar", ".timestamps"))

    return [
        (pose_key, pose_local),
        (ts_key, ts_local),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get trajectory related captions i.e. speed, location, acceleration etc."
    )
    parser.add_argument(
        "path_to_data",
        help="List of absolute paths of clips to be processed"
    )
    parser.add_argument(
        "path_to_output",
        help="Path to save the output"
    )
    parser.add_argument(
        "cnt",
        type=int,
        help="Number used to diferentiate between different outputs"
    )
    parser.add_argument(
        "--vipe_trajectories",
        action="store_true",
        help="Trajectory data were computed with vipe"
    )
    parser.add_argument(
        "--start",
        default=0,
        type=int,
        help="Start index if processing the data in chunks"
    )
    parser.add_argument(
        "--end",
        default=None,
        type=int,
        help="End index if processing the data in chunks"
    )
    parser.add_argument(
        "--bucket",
        default=None,
        type=str,
        help="S3 bucket name containing the videos (enables S3 mode)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        type=str,
        help="AWS profile name for credentials (e.g., sil-wheel)",
    )
    parser.add_argument(
        "--endpoint",
        default="https://s3.example.com",
        type=str,
        help="S3 endpoint URL (default https://s3.example.com)",
    )
    args = parser.parse_args()

    with open(args.path_to_data, "r") as f:
        video_paths = f.readlines()
    video_paths = sorted(set([Path(ci.strip()) for ci in video_paths]))
    print(f"Loading {len(video_paths)} files")

    # Get only the slicee of data we will be processing.
    start = args.start or 0
    video_paths = video_paths[start:args.end]

    # Get the index of the clip_id within the path
    clip_id_index, data_type = get_clip_id_index_from_paths(
        video_paths[:5], args.bucket
    )

    path_to_safetensors = sorted(Path(args.path_to_output).glob("*.safetensors"))
    processed = []
    for pts in tqdm(path_to_safetensors):
        try:
            sft = load_file(pts)
        except SafetensorError:
            print(pts)
            continue
        processed.extend(list(sft.keys()))
    processed = set(processed)
    print(f"Skipping {len(processed)} clips out of the {len(video_paths)}")

    if clip_id_index == -1:
        video_paths = [
            p for p in video_paths
            if Path(p).stem not in processed
        ]
    else:
        video_paths = [
            p for p in video_paths
            if p.parts[clip_id_index] not in processed
        ]
    print(f"Computing the trajectories for {len(video_paths)} video clips")

    # Load the data corresponding to this output
    output_file = f"{args.path_to_output}/trajectory_stats_smoothed_{args.cnt}.safetensors"
    all_stats = {}
    if Path(output_file).exists():
        all_stats = load_file(output_file)
        print(f"Loading {len(all_stats)} trajectories from {output_file}")

    save_every = 1000
    skipped = 0
    update_processed_every = 1000000
    if args.bucket:
        fetcher = S3ObjectFetcher(
            bucket=args.bucket,
            profile=args.profile,
            endpoint=args.endpoint
        )
        with tempfile.TemporaryDirectory(prefix="s3_mads_traj_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            for i, local_mp4 in tqdm(
                enumerate(fetcher.stream_downloads(
                    video_paths,
                    tmpdir_path,
                    num_workers=4,
                    max_queue=1000,
                    also_download=mads_pose_data_for_front_wide
                )),
                total=len(video_paths),
                desc=f"Downloading + processing {len(video_paths)} from s3://{args.bucket}",
            ):
                local_mp4 = Path(local_mp4)
                clip = local_mp4.stem
                if clip in processed:
                    skipped += 1
                    continue

                stats = compute_stats_from_clip(local_mp4, data_type, args.vipe_trajectories)
                if stats is None:
                    continue

                processed.add(clip)
                all_stats[clip] = stats

                if i % save_every == 0:
                    save_file(all_stats, output_file)
                    print(f"Saving {len(all_stats)}")
    else:
        for i, clip_to_path in tqdm(enumerate(video_paths)):
            clip = clip_to_path.parts[clip_id_index]
            if clip in processed:
                skipped += 1
                continue

            stats = compute_stats_from_clip(
                clip_to_path, data_type, args.vipe_trajectories
            )
            if stats is None:
                continue

            processed.add(clip)
            all_stats[clip] = stats

            if i % save_every == 0:
                save_file(all_stats, output_file)
                print(f"Saving {len(all_stats)}")

            if i % update_processed_every == 0 and i > 0:
                path_to_safetensors = sorted(Path(args.path_to_output).glob("*.safetensors"))
                processed = []
                for pts in path_to_safetensors:
                    try:
                        sft = load_file(pts)
                    except SafetensorError:
                        print(pts)
                    processed.extend(list(sft.keys()))
                processed = set(processed)
                print(f"Skipping {len(processed)} clips out of the {len(video_paths)}")

    save_file(all_stats, output_file)
    print(f"Saved trajectory stats to {output_file}")
