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

from typing import Tuple
import time

import faiss
import numpy as np
import torch


def normalize_features(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / norms


def text_to_video_retrieval(
    query_text: str,
    video_features_index: faiss.Index,
    internvideo2_model: torch.nn.Module,
    n_neighbors: int = 2048,
    params: faiss.SearchParametersIVF = None,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve videos conditioned on features computed from a text query."""
    start = time.time()

    # Get the features from the query text and normalize them
    text_features = internvideo2_model.get_text_embeddings(query_text)
    text_features = torch.nn.functional.normalize(text_features, dim=-1)
    if text_features.dtype == torch.bfloat16:
        text_features = text_features.float()
    text_features = text_features.cpu().numpy()

    # Namely take all neighbors
    if n_neighbors == -1:
        n_neighbors = video_features_index.ntotal

    # Now compute the distances
    distances, indices = video_features_index.search(
        text_features, n_neighbors, params=params
    )
    # Only keep the indices and distances, for indices != -1
    # distances = distances[indices != -1]
    # indices = indices[indices != -1]
    if len(indices) <= 0:
        import pdb
        pdb.set_trace()
    # assert len(indices) > 0
    elapsed = time.time() - start
    if verbose:
        print(
            f"t2v : Computed {len(indices)} nearest neighbors between the text query "
            f"and the {video_features_index.ntotal} video features in {elapsed:.4f}s"
        )
    return distances, indices


def text_to_video_retrieval_with_negative_prompt(
    query_text: str,
    negative_query_text: str,
    npfeatures: np.ndarray,
    video_features_index: faiss.Index,
    internvideo2_model: torch.nn.Module,
    n_neighbors: int = 2048,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve videos conditioned on text features computed for a positive and
    negative text. We want to exclude results similar to the negative text prompt.
    """

    # We start by doing the retrieval for the negative prompt
    _, negative_indices = text_to_video_retrieval(
        negative_query_text, video_features_index,
        internvideo2_model, n_neighbors=-1
    )
    # Now remove the negative indices from the index
    keep_indices = np.setdiff1d(
        np.arange(video_features_index.ntotal), negative_indices
    )
    # Create an IDSelectorBatch to select the subset of the index
    params = faiss.SearchParametersIVF(
        sel=faiss.IDSelectorBatch(keep_indices), nprobe=1000
    )
    distances, indices = text_to_video_retrieval(
        query_text, video_features_index, internvideo2_model,
        n_neighbors, params=params, verbose=verbose
    )
    return distances, indices


def video_to_video_retrieval(
    query_video_clip: str,
    clips: np.ndarray,
    npfeatures: np.ndarray,
    video_features_index: faiss.Index,
    n_neighbors: int = 2048,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve videos conditioned on video features computed for a given
    query_video_clip."""
    start = time.time()

    try:
        query_idx = np.where(clips == query_video_clip)[0][0]
    except IndexError:
        print(f"Query clip {query_video_clip} not found.")
        return None, None
    query_features = npfeatures[query_idx][None]
    # Make sure that features are normalized
    norms = np.linalg.norm(query_features, axis=-1, keepdims=True)
    query_video_features = query_features / norms

    # Namely take all neighbors
    if n_neighbors == -1:
        n_neighbors = video_features_index.ntotal
    # Now compute the distances
    distances, indices = video_features_index.search(query_video_features, n_neighbors)

    elapsed = time.time() - start
    if verbose:
        print(
            f"v2v : Computed {n_neighbors} nearest neigbors between the query "
            f"video and the video features in {elapsed:.4f}s"
        )
    return distances, indices


def text_and_video_to_video_retrieval(
    query_video_clip: str,
    video_description: str,
    clips: np.ndarray,
    npfeatures: np.ndarray,
    video_features_index: faiss.Index,
    internvideo2_model: torch.nn.Module,
    n_neighbors: int = 2048,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve videos conditioned on both video and text features."""
    start = time.time()

    # Step 1: Video-to-Video Retrieval
    _, indices_v2v = video_to_video_retrieval(
        query_video_clip, clips, npfeatures,
        video_features_index, n_neighbors=-1, verbose=verbose
    )

    # Step 2: Filter clips and features using the indices from video-to-video retrieval
    positive_indices = indices_v2v.flatten()
    candidate_clips = clips[positive_indices]
    candidate_features = npfeatures[positive_indices]

    if len(candidate_clips) == 0:
        print("No candidates found after filtering with video-to-video retrieval.")
        return None, None

    # Step 3: Create a new FAISS index for the candidate features from video-to-video retrieval
    # This ensures we perform text-to-video retrieval only on these candidates
    # TODO: This is slow and we need to be replaced with faiss.SearchParametersIVF
    candidate_video_features_index = faiss.IndexFlatL2(candidate_features.shape[1])
    candidate_video_features_index.add(candidate_features)

    distances_t2v, indices_t2v = text_to_video_retrieval(
        video_description, candidate_video_features_index, internvideo2_model,
        n_neighbors, verbose=verbose
    )
    # Map the resulting indices back to the original clip indices
    final_indices = positive_indices[indices_t2v.flatten()]

    elapsed = time.time() - start
    if verbose:
        print(
            f"tv2v : Computed {n_neighbors} nearest neighbors using both video and text "
            f"on positive video indices in {elapsed:.4f}s"
        )
    return distances_t2v, final_indices[None]


def query_to_video_retrieval(
    query_features: np.ndarray,
    video_features_index: faiss.Index,
    n_neighbors: int = 4096,
    verbose: bool = False,
    params=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve videos conditioned on video features computed for a given
    query_video_clip."""
    start = time.time()

    # Make sure that features are normalized
    norms = np.linalg.norm(query_features, axis=-1, keepdims=True)
    query_video_features = query_features / norms

    # Namely take all neighbors
    if n_neighbors == -1:
        n_neighbors = video_features_index.ntotal
    # Now compute the distances
    if len(query_video_features.shape) == 1:
        query_video_features = query_video_features[None, :]
    distances, indices = video_features_index.search(
        query_video_features, n_neighbors, params=params
    )

    elapsed = time.time() - start
    if verbose:
        print(
            f"v2v : Computed {n_neighbors} nearest neigbors between the query "
            f"video and the video features in {elapsed:.4f}s"
        )
    return distances, indices
