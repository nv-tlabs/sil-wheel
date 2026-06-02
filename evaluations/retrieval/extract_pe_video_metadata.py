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

"""Extract per-video metadata (caption, category, ...) from the locally-cached
``facebook/PE-Video`` tar shards into a parquet.

Each shard is a WebDataset tar containing pairs ``<key>.mp4`` /
``<key>.json``; this script walks the JSON sidecars and ignores the
mp4 bytes. The resulting parquet feeds downstream benchmarks (e.g.
PVD-Bench retrieval) and can also be uploaded into Wheel's caption
store.
"""
import argparse
import json
import tarfile
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download
from tqdm import tqdm


REPO_ID = "facebook/PE-Video"

# Columns we keep from each JSON sidecar. The first two are the only
# ones the retrieval benchmark needs; the rest are kept for analysis
# (per-category breakdowns, motion stats, source attributions).
KEEP_FIELDS = [
    "human_caption",
    "model_caption",
    "category",
    "category_name",
    "video_duration_in_s",
    "fps",
    "num_frames",
    "height",
    "width",
    "keywords",
    "description",
    "has_face",
]


def extract_split(local_dir, split):
    """Yield one row dict per ``<key>.json`` found in ``local_dir/<split>/*.tar``."""
    tar_paths = sorted((Path(local_dir) / split).glob("*.tar"))
    if not tar_paths:
        raise FileNotFoundError(
            f"No tar shards under {local_dir}/{split}. "
            "Did you run prepare_data.py (or otherwise prime the HF "
            f"cache) for split '{split}'?"
        )
    for tar_path in tqdm(tar_paths, desc=f"Scanning {split}"):
        with tarfile.open(tar_path, "r") as tf:
            for member in tf:
                if not member.name.endswith(".json"):
                    continue
                meta = json.loads(tf.extractfile(member).read())
                row = {"clip_id": member.name[: -len(".json")], "split": split}
                for field in KEEP_FIELDS:
                    row[field] = meta.get(field)
                yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "output",
        type=Path,
        help="Path to the output .parquet (will be overwritten).",
    )
    parser.add_argument(
        "--split",
        choices=["test", "train", "extended", "all"],
        default="test",
        help="Which PE-Video split(s) to scan. 'all' concatenates test+train+extended.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help="HuggingFace cache directory. Defaults to $HF_HOME.",
    )
    args = parser.parse_args()

    splits = (
        ["test", "train", "extended"] if args.split == "all" else [args.split]
    )

    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=[f"{s}/*" for s in splits],
        cache_dir=args.hf_cache_dir,
        max_workers=8,
    )

    rows = []
    for split in splits:
        rows.extend(extract_split(local_dir, split))

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")
    print(df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
