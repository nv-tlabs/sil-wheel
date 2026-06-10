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
    """Load an ego trajectory and the camera frame timestamps for a recording.

    Args:
        path_to_recordings (str): Path to the recording.

    Returns:
        Tuple (x, y, z, ts, frame_ts) of np.arrays (locations in meters, ego
        timestamps and frame timestamps in seconds), or None if no
        egomotion_estimate.parquet is found, or it is empty or unreadable.
    """
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
    VIPE produces one pose per video frame at a fixed 30 fps, so the
    pose timestamps and the frame timestamps are in sync; both are
    returned as the same evenly spaced array.

    Args:
        path_to_recordings (str): Path to the recording, that is an npz
        file that contains
            - data: (N, 4, 4) camera-to-world transforms, X_w = R @ X_c + t,
              with R = c2w[:, :3, :3] and t = c2w[:, :3, 3]
            - inds: (N,) frame indices for each pose

    Returns:
        Tuple (x, y, z) in VIPE camera convention (x->right, y->down,
        z->forward).
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
    """Load an ego trajectory and the camera frame timestamps for a recording.

    Args:
        path_to_recordings (Path): Path to the clip's .mp4 recording.

    Returns:
        Tuple (x, y, z, ts, frame_ts) of np.arrays (locations in meters, ego
        timestamps and frame timestamps in seconds), or None if no
        egomotion_estimate.parquet is found, or it is empty or unreadable.
    """
    path_to_frame_ts = str(path_to_recordings).replace(".mp4", ".timestamps")
    with open(path_to_frame_ts, "r") as f:
        frame_ts = f.readlines()
    frame_ts = np.array(
        [int(ts.strip().split("\t")[-1]) for ts in frame_ts]
    ) / 1e6

    base_dir_to_ego_data = path_to_recordings.parent
    path_to_ego = list(
        base_dir_to_ego_data.rglob("egomotion_estimate.parquet")
    )
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

            name = member.name
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
    """Read tab-delimited frame timestamps from a .timestamps file.

    Args:
        ts_path (str | Path): Path to the .timestamps file; each line ends
            with an integer timestamp in microseconds.

    Returns:
        np.ndarray of timestamps in seconds, or None if the file does not
        exist.
    """
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
    """Compute the per-frame trajectory statistics for a single clip.

    Loads the ego trajectory for the given data_type (each source has its own
    reader and axis convention) and computes speed, acceleration, jerk and
    curvature. If the reader returns frame timestamps, the statistics are
    interpolated onto them (one row per frame); otherwise they are returned at
    the ego-pose timestamps.

    Args:
        path_to_recordings: Path to the clip recording or trajectory file; its
            meaning depends on data_type.
        data_type (str | None): Trajectory source type (e.g. "physical_ai",
            "mads", "vipe", "waymo"); selects the reader and axis handling.
            None uses get_trajectory_data.
        vipe_trajectories (bool): If True, rotate VIPE camera-convention axes
            into the viewer frame (also implied by data_type == "vipe").

    Returns:
        np.ndarray of shape (T, 7) float32 with columns
        [x, y, z, speed, acceleration, jerk, curvature], or None if the
        trajectory could not be loaded.
    """
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
    elif data_type == "physical_ai":
        # Physical AI / Alpamayo egomotion.offline parquet. Columns:
        #   timestamp -> microseconds, x/y/z -> ego location in meters, in a
        #   right-handed FLU frame (x forward, y left, z up; verified via the
        #   quaternion: positive yaw / a left turn moves toward +y).
        # The offline egomotion is ~10 Hz while the camera runs at ~30 Hz, both
        # on the same clip clock. We resample the ego trajectory onto the camera
        # frame timestamps (read below from the .timestamps.parquet file written
        # next to this one) so there is one trajectory row per video frame --
        # unlike MADS we do not treat the ego samples as frames.
        frame_ts_path = str(path_to_recordings).replace(
            ".egomotion.offline.parquet", ".timestamps.parquet"
        )
        try:
            df = pd.read_parquet(path_to_recordings)
            ts = df["timestamp"].astype(np.float64).to_numpy() / 1e6
            # FLU (x forward, y left, z up) -> the viewer's frame (y right), so
            # negate y.
            x = df["x"].astype(np.float64).to_numpy()
            y = -df["y"].astype(np.float64).to_numpy()
            z = df["z"].astype(np.float64).to_numpy()
            frame_ts = (
                pd.read_parquet(frame_ts_path)["timestamp"]
                .astype(np.float64).to_numpy() / 1e6
            )
        except (KeyError, FileNotFoundError, OSError) as e:
            print(f"Failed to read egomotion/timestamps {path_to_recordings}: {e}")
            return None
        if len(ts) < 2:
            return None
    elif data_type == "mads":
        df = pd.read_parquet(path_to_recordings)
        ts = df["key.timestamp"].astype(np.float64).to_numpy()

        # MADS follows the FLU convention i.e.
        # x -> forward, y -> left and z -> upward
        # so we need to negate y, to adher to what the viewer expects
        # (y -> right)
        x = df["egomotion_estimate.location.x"].astype(np.float64).to_numpy()
        y = -df["egomotion_estimate.location.y"].astype(np.float64).to_numpy()
        z = df["egomotion_estimate.location.z"].astype(np.float64).to_numpy()
        # The video and ego pose timestamp are synchronized
        frame_ts = None
    elif data_type == "mads-1M":
        pose_tar_path = Path(path_to_recordings).parents[2] / "vehicle_pose" / f"{local_mp4.stem}.tar"
        if not pose_tar_path.exists():
            return None

        poses = extract_poses_from_tar(pose_tar_path)
        if poses is None:
            return None

        # MADS follows the FLU convention i.e.
        # x -> forward, y -> left and z -> upward
        # so we need to negate y, to adher to what the viewer expects
        # (y -> right)
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
        # to the viewer coordinates:
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


def clip_id_from_path(path, clip_id_index):
    """Derive the clip_id used as the trajectory key from a source path."""
    if clip_id_index == -1:
        # Drop the suffix i.e. <clip_id>.parquet, <clip_id>.egomotion.offline.parquet
        return Path(path).name.split(".")[0]
    # If it's not filename, we assume it doesn't have a suffix
    return Path(path).parts[clip_id_index]


def get_clip_id_index_from_paths(path_to_data, bucket=None):
    """Infer where the clip ID appears in a path and identify the dataset type.

    The function handles two types of layouts:

    1. File-based datasets:
       The clip ID is encoded in the filename itself, so the returned index is -1.

    2. Directory-based datasets:
       The clip ID is one of the path components. We infer its index by comparing
       a few input paths and finding the component that varies across clips, while
       ignoring camera-specific path components.

    Returns:
        tuple[int, str | None]: A pair (clip_id_index, data_type) where
        clip_id_index is either -1 for file-based datasets or the index of
        the clip-ID path component for directory-based datasets. data_type
        is the inferred dataset type when known.
    """
    # file-based sources: clip_id is the filename token (index -1)
    if bucket is not None:
        return -1, "mads-1M"
    first = str(path_to_data[0])
    if first.endswith(".egomotion.offline.parquet"):
        return -1, "physical_ai"
    if first.endswith(".parquet"):
        return -1, "mads"
    if first.endswith(".npz"):
        return -1, "vipe"

    # directory-based sources: clip_id is a path component
    paths_parts = [Path(p).parts for p in path_to_data[:3]]
    diff_ind = [
        i for i, col in enumerate(zip(*paths_parts))
        if len(set(col)) > 1 and not any("camera" in item for item in col)
    ]
    assert len(diff_ind) == 1
    if diff_ind[0] == 11:
        return diff_ind[0], "clip_gt"
    elif diff_ind[0] == 12:
        return diff_ind[0], "waymo"
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

    # Get only the slice of data we will be processing.
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

    video_paths = [
        p for p in video_paths
        if clip_id_from_path(p, clip_id_index) not in processed
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
    # If trajectory data are fetched from S3
    if args.bucket:
        fetcher = S3ObjectFetcher(
            bucket=args.bucket,
            profile=args.profile,
            endpoint=args.endpoint
        )
        with tempfile.TemporaryDirectory(prefix="s3_traj_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Stream each clip from S3 into the temp dir
            for i, local_mp4 in tqdm(
                enumerate(fetcher.stream_downloads(
                    video_paths,
                    tmpdir_path,
                    num_workers=4,
                    max_queue=1000,
                    also_download=mads_pose_data_for_front_wide
                )),
                total=len(video_paths),
                desc=f"Downloading {len(video_paths)} from s3://{args.bucket}",
            ):
                local_mp4 = Path(local_mp4)
                clip = local_mp4.stem
                if clip in processed:
                    skipped += 1
                    continue

                stats = compute_stats_from_clip(
                    local_mp4, data_type, args.vipe_trajectories
                )
                if stats is None:
                    continue

                processed.add(clip)
                all_stats[clip] = stats

                if i % save_every == 0:
                    save_file(all_stats, output_file)
                    print(f"Saving {len(all_stats)}")
    else:
        # If trajectory data are stored locally
        for i, clip_to_path in tqdm(enumerate(video_paths)):
            clip = clip_id_from_path(clip_to_path, clip_id_index)
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
                path_to_safetensors = sorted(
                    Path(args.path_to_output).glob("*.safetensors")
                )
                processed = []
                for pts in path_to_safetensors:
                    try:
                        sft = load_file(pts)
                    except SafetensorError:
                        print(pts)
                    processed.extend(list(sft.keys()))
                processed = set(processed)
                print(
                    f"Skipping {len(processed)} clips "
                    f"out of the {len(video_paths)}"
                )

    save_file(all_stats, output_file)
    print(f"Saved trajectory stats to {output_file}")
