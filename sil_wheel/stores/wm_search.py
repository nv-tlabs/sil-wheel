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

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# To translate the angle range string to the index of the angle range in the loc_data array
ANGLE_RANGE_IDX = {
    "FRONT": 5,
    "FRONT_RIGHT": 0,
    "BACK_RIGHT": 1,
    "BACK": 2,
    "BACK_LEFT": 3,
    "FRONT_LEFT": 4,
}

# TODO: Figure out a better way of exposing this. Putting it here to import into the frontend UI
CLASS_NAMES = [
    "VEHICLE_CAR",
    "VEHICLE_TRUCK",
    "VEHICLE_BUS",
    "BIKE_WITH_RIDER",
    "BIKE_TRICYCLE",
    "PEDESTRIAN_UNKNOWN",
]


class WMSearch:
    def __init__(
        self, wm_data: pd.DataFrame, clip_ids: list[str] = None
    ) -> None:
        self.wm_data = wm_data
        self.clip_ids = (
            clip_ids if clip_ids is not None else set(self.wm_data["clip_id"])
        )

        self.preprocess_wm_stats_data()

        print(f"Found {len(self.wm_data)} clips with WM data")

    def preprocess_wm_stats_data(self) -> None:
        # drop columns with clip_id that are not in self.clip_ids
        self.wm_data = self.wm_data[self.wm_data["clip_id"].isin(
            self.clip_ids)]

        self.wm_data["clip_id"] = [sys.intern(c) for c in self.wm_data["clip_id"]]

        # convert all loc columns to numpy arrays. They were stored as lists in the parquet file
        for col in self.wm_data.columns:
            if col.startswith("loc_"):
                self.wm_data[col] = self.wm_data[col].apply(
                    lambda x: np.vstack(x) if x is not None else None
                )

    def search_count(
        self,
        class_name: str,
        min_count: int = 1,
        max_count: Optional[int] = None,
    ) -> list[str]:
        """
        Search for clips with counts >= min_count and optionally <= max_count for a given class.
        """
        # Return clip_ids with counts >= min_count and optionally <= max_count
        counts = self.wm_data[f"num_{class_name}"].values
        found = counts >= min_count

        # Apply max_count filter if specified
        if max_count is not None:
            found = found & (counts <= max_count)

        clip_ids = self.wm_data[found]["clip_id"].values

        return clip_ids

    def search_range(
        self,
        class_name: str,
        angle_range: list[str],
        max_dist: float = 10.0,
        min_time: Optional[float] = 0,
    ) -> list[str]:
        """
        Search for clips where there exists an object of the given class in the given angle range (list of strings) within max_dist (in meters) and
        for time greater than min_time (in seconds).
        """
        for angle in angle_range:
            if angle not in ANGLE_RANGE_IDX.keys():
                print(f"Invalid angle range: {angle}")
                return []

        # Get the index of the angle range
        angle_range_idx = [ANGLE_RANGE_IDX[angle] for angle in angle_range]

        # Filter data which has a non-None range representation for the given class
        mask = self.wm_data[f"loc_{class_name}"].notna()
        filtered_data = self.wm_data[mask]

        # Vectorized filtering: get all location data as a numpy array
        loc_data = np.stack(filtered_data[f"loc_{class_name}"].values)

        # N x T x A (A = len(ANGLE_RANGE_IDX.keys()))

        # Extract distances for the specified angle ranges
        distances_in_range = loc_data[:, :, angle_range_idx]

        # First compute an or operation along the angles axis, then sum across the time axis and divide by fps to convert to seconds
        # TODO: fix hardcoded fps
        times = (
            np.sum(np.any(distances_in_range < max_dist, axis=2), axis=1) / 10
        )
        time_condition = times > min_time

        # Get clip_ids for rows that meet the condition
        valid_clip_ids = filtered_data[time_condition]["clip_id"].values

        # sort clip_ids based on times array in descending order
        #   valid_clip_ids = valid_clip_ids[np.argsort(times[time_condition])]

        # Make sure to free up loc_data memory
        del loc_data

        return valid_clip_ids


def consolidate_wm_stats_data(
    data_dir: str = "./",
    out_file: str = "/path/to/lustre/datasets/alpamayo/wm_stats_data.parquet",
):
    """
    Consolidate the WM stats data of individual clips into a single parquet file to use with the WMStatsSearch class.
    """
    fnames = [fname for fname in os.listdir(
        data_dir) if fname.endswith(".npz")]

    # initialize a dataframe with columns clip_id
    # We will add the count and range representation per class as we encounter them
    df = pd.DataFrame(columns=["clip_id"])

    for fname in tqdm(fnames, desc="Consolidating WM stats data"):
        clip_id = fname.split(".")[0].split("_")[1]
        wm_stats = np.load(os.path.join(data_dir, fname), allow_pickle=True)

        for class_name in wm_stats["num_obj"].item().keys():
            if f"num_{class_name}" not in df.columns:
                df[f"num_{class_name}"] = None
            if f"loc_{class_name}" not in df.columns:
                # just in case, even though num and loc should have the same class keys per clip
                df[f"loc_{class_name}"] = None

        # convert the loc to list since parquet does not support 2d arrays
        clip_data = {
            "clip_id": clip_id,
            **{
                f"num_{class_name}": int(wm_stats["num_obj"].item()[class_name])
                for class_name in wm_stats["num_obj"].item().keys()
            },
            **{
                f"loc_{class_name}": wm_stats["obj_loc"]
                .item()[class_name]
                .tolist()
                for class_name in wm_stats["obj_loc"].item().keys()
            },
        }
        df = pd.concat([df, pd.DataFrame([clip_data])], ignore_index=True)

    df.to_parquet(out_file)


if __name__ == "__main__":
    # consolidate_wm_stats_data(data_dir="/path/to/lustre/gitrepo/alpamayo/out/v1_wm_stats", out_file="/path/to/lustre/datasets/alpa/wm_stats_data.parquet")
    wm_data = pd.read_parquet(
        "/path/to/eval_benchmark/eval_benchmark/wm_stats_data.parquet"
    )
    wm_search = WMSearch(wm_data)

    res = wm_search.search_count("VEHICLE_CAR", min_count=50)
    print(res)
    res = wm_search.search_range(
        "BIKE_WITH_RIDER", ["FRONT"], max_dist=10, min_time=10
    )
    print(res)
    import pdb

    pdb.set_trace()
