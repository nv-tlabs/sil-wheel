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

"""Sample 20 s clips from local OpenDV videos (uniform or diverse)."""
import argparse
import logging
from pathlib import Path

from sil_wheel.datasets.opendv.constants import DEFAULT_INTERVAL, DEFAULT_STRIDE
from sil_wheel.datasets.opendv.labels import download_language_annotations
from sil_wheel.datasets.opendv.metadata import fetch_subset
from sil_wheel.datasets.opendv.sampling import sample

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Sample 20 s OpenDV clips.")
    parser.add_argument("--subset", choices=["mini", "full"], default="mini")
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=["uniform", "diverse"], default="uniform")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    parser.add_argument("--clip-sec", type=int, default=20)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--select-k", type=int, default=None)
    parser.add_argument("--total", type=int, default=None)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.5)
    parser.add_argument("--cut", choices=["nvenc", "copy", "libx264"], default="nvenc",
                        help="ffmpeg cut backend. 'copy' is fastest but NOT frame-accurate "
                             "(snaps to the previous keyframe), so clips may start before the "
                             "labeled time; nvenc/libx264 are exact.")
    parser.add_argument("--annotations-dir", type=Path, default=None,
                        help="OpenDV-YouTube-Language dir (diverse). Downloaded if absent.")
    args = parser.parse_args()

    records = fetch_subset(args.subset, cache_csv=args.output_dir / "meta" / "OpenDV-YouTube.csv")
    ann_dir = args.annotations_dir
    if args.method == "diverse":
        ann_dir = ann_dir or (args.output_dir / "annotations")
        has_ann = ann_dir.exists() and any(ann_dir.glob("10hz_*.json"))
        if not has_ann:
            download_language_annotations(ann_dir)
    sample(records, args.videos_dir, args.output_dir, method=args.method,
           interval=args.interval, clip_sec=args.clip_sec, stride=args.stride,
           select_k=args.select_k, total=args.total, lam=args.lam, cut=args.cut,
           annotations_dir=ann_dir)


if __name__ == "__main__":
    main()
