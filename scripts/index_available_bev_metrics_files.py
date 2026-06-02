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

"""
This script indexes the available BEV and metrics files.
Then writes them to file for quick metrics and bev filtering.
Export as python sets of clip_ids.
"""
import argparse
import os
import pickle
from pathlib import Path
import subprocess


def index_available_bev_files(bucket, s3_prefix, index_out_dir):
    prefix = f"s3://{bucket}/{s3_prefix}/"
    s3_glob = prefix + "*"

    print("Running s5cmd ls to get BEV files. This might take a while...")
    res = subprocess.run(
        ["s5cmd", "--profile", "sil-wheel", "ls", s3_glob],
        capture_output=True, text=True, check=True
    )

    bev_files = []
    for line in res.stdout.splitlines():
        url = line.split()[-1]  # last column
        if url.startswith(prefix):
            bev_files.append(url[len(prefix):])
        else:
            bev_files.append(url)
    # remove .msgpack file ending
    bev_files = [file.replace(".msgpack", "") for file in bev_files]
    print(f"Found {len(bev_files)} clips with BEV files")
    print(bev_files[:10])
    set_bev_files = set(bev_files)
    with open(Path(index_out_dir) / "clips_with_bev_set.pkl", "wb") as f:
        pickle.dump(set_bev_files, f)
    print(f"Successfully saved BEV index to {Path(index_out_dir) / 'clips_with_bev_set.pkl'}")


def index_available_metrics_files(predictions_dir, index_out_dir):
    metrics_dir = Path(predictions_dir) / "ground_truth" / "eval"
    paths = metrics_dir.glob("*.parquet")
    clips_with_metrics = [path.stem for path in paths]
    set_clips_with_metrics = set(clips_with_metrics)
    print(f"Found {len(set_clips_with_metrics)} clips with metrics")
    print(clips_with_metrics[:10])

    with open(Path(index_out_dir) / "clips_with_metrics_set.pkl", "wb") as f:
        pickle.dump(set_clips_with_metrics, f)
    print(f"Successfully saved metrics index to {Path(index_out_dir) / 'clips_with_metrics_set.pkl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index available BEV and metrics files for filtering"
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions directory",
    )
    parser.add_argument(
        "--bev_s3_prefix",
        default="bev_data/v0",
        help="S3 prefix for BEV data (default: bev_data/v0)",
    )
    parser.add_argument(
        "--bev_s3_bucket",
        default="processed_data",
        help="S3 bucket name for BEV data (default: processed_data)",
    )
    parser.add_argument(
        "--bev_metrics_index_dir",
        required=True,
        help="Output directory for index files",
    )
    args = parser.parse_args()

    os.makedirs(args.bev_metrics_index_dir, exist_ok=True)

    index_available_bev_files(
        args.bev_s3_bucket, args.bev_s3_prefix, args.bev_metrics_index_dir
    )
    index_available_metrics_files(args.predictions, args.bev_metrics_index_dir)