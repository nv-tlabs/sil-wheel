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

import pickle
import time
from pathlib import Path
from typing import Dict, Tuple

import faiss
import numpy as np
from tqdm import tqdm


def parse_clip_embeddings_from_dir(
    path_to_clip_embeddings: str,
    index_factory: str = "l2",
    with_gpu: bool = False,
) -> Tuple[faiss.Index, Dict[int, str]]:
    start = time.time()
    path_to_embeddings = Path(path_to_clip_embeddings)
    pkl_files = sorted(Path(path_to_embeddings).glob("**/clip_group_*.pkl"))

    path_to_faiss_index = path_to_embeddings / "clip_embeddings.index"
    path_to_clip_index = path_to_embeddings / "clip_to_index.pkl"

    # If precomputed, load and return
    if path_to_faiss_index.exists() and path_to_clip_index.exists():
        features_index = faiss.read_index(str(path_to_faiss_index))
        with open(path_to_clip_index, "rb") as f:
            clip_to_index = pickle.load(f)
        print(
            f"Loading feature index from {path_to_faiss_index} "
            f"CLIP embeddings of size {features_index.ntotal}..."
        )
        return features_index, clip_to_index

    # Build index
    quantizer = faiss.IndexFlatIP(512)
    features_index = faiss.IndexIVFPQ(quantizer, 512, 4096, 16, 8)
    features_index.metric_type = faiss.METRIC_INNER_PRODUCT
    features_index.nprobe = 64

    clip_to_index = {}
    all_features = []
    offset = 0
    for fi in tqdm(pkl_files):
        with open(fi, "rb") as f:
            data = pickle.load(f)

        clip_index = data["clip_index"]  # maps local idx to clip_id
        clip_features = data["embeddings"].astype(np.float32)
        assert len(clip_features) == len(clip_index)
        all_features.append(clip_features)

        clip_to_index.update(
            {key + offset: val for key, val in clip_index.items()}
        )
        offset += len(clip_index)
        print(len(set(clip_to_index.values())))

        total_so_far = sum(len(f) for f in all_features)
        if total_so_far > 500000 and not features_index.is_trained:
            print("Train the index")
            features = np.vstack(all_features)
            features_index.train(features)
            features_index.add(features)
            all_features.clear()

        total_so_far = sum(len(f) for f in all_features)
        if features_index.is_trained and total_so_far > 500000:
            features = np.vstack(all_features)
            features_index.add(features)
            all_features.clear()

    if len(all_features) > 0:
        features = np.vstack(all_features)
        features_index.add(features)
    all_features.clear()
    elapsed = time.time() - start
    print(f"Finished indexing in {elapsed:.2f} seconds")

    # Save FAISS index
    faiss.write_index(
        features_index, str(path_to_embeddings / "clip_embeddings.index")
    )

    # Save clip_to_index mapping
    with open(path_to_embeddings / "clip_to_index.pkl", "wb") as f:
        pickle.dump(clip_to_index, f)
    return features_index, clip_to_index
