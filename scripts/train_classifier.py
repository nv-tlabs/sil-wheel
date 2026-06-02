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
import logging
import time
from pathlib import Path

import faiss
import numpy as np
import simdjson as json
from tqdm import tqdm

from embed_io import load_clip_to_index
from sil_wheel.classifier_build import build_classifier_run
from sil_wheel.cluster_build import generate_run_id


def gather_clips_with_condition(annotations, labels, use_autolabels):
    clips = []
    for anns in tqdm(annotations, desc="Gathering clips"):
        clip = anns["clip_id"]
        for ann in anns["annotations"]:
            if ann["key"] in labels and ann["label_type"] in (
                ["autolabel", "manual"] if use_autolabels else ["manual"]
            ):
                clips.append(clip)
    return clips


def clips_to_embeddings(clip_ids, clip_to_index, features_index):
    indices = [clip_to_index[c] for c in clip_ids if c in clip_to_index]
    vectors = features_index.reconstruct_batch(indices)
    return np.array(vectors, dtype=np.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a binary classifier run")
    parser.add_argument(
        "classifier_output_dir",
        help="Path to root classifier dir; a <run_id>/ subdir is created under it"
    )
    parser.add_argument(
        "annotation_file",
        help="Path to annotations file"
    )
    parser.add_argument(
        "label",
        help="Positive label name. For multi-label, use sorted labels joined with '&&', e.g. 'cat&&dog'.",
    )
    parser.add_argument(
        "embeddings_file",
        help="Path to the embeddings dir"
    )
    parser.add_argument(
        "--trained_by",
        required=True,
        help="Username of the person training this classifier (recorded in metadata)",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Run id for this classifier; auto-generated if omitted",
    )
    parser.add_argument(
        "--n_positive_samples",
        type=int,
        default=-1,
        help="Cap on positive samples used for training",
    )
    parser.add_argument(
        "--n_negative_samples",
        type=int,
        default=100,
        help="Number of negative samples used for training",
    )
    parser.add_argument(
        "--use_autolabels",
        action="store_true",
        help="Include autolabels among positive training samples",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=50000,
        help="Number of clips to score per batch during inference",
    )
    parser.add_argument(
        "--negative_labels",
        type=lambda x: tuple(map(str, x.split(","))),
        default=None,
        help="Comma-separated annotation keys to use as negatives. If omitted, negatives are sampled from all clips not in the positive set.",
    )
    parser.add_argument(
        "--save_threshold",
        type=float,
        default=0.3,
        help="Only save predictions with probability strictly greater than this threshold",
    )
    parser.add_argument(
        "--max_clips",
        type=int,
        default=7_000_000,
        help="Maximum number of clips to save (top by score). Use -1 for no limit",
    )
    parser.add_argument(
        "--embed_type",
        choices=["cosmos", "caption", "visual"],
        default="cosmos",
    )
    parser.add_argument("--index_tag", default="ivf4096_pq96x8")
    parser.add_argument(
        "--search_params",
        default=None,
        help="Optional URL query that produced the corpus, persisted in metadata.json",
    )
    args = parser.parse_args(argv)

    run_id = args.run_id or generate_run_id()
    run_dir = Path(args.classifier_output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "train.log"
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    start = time.time()
    with open(args.annotation_file, "r") as f:
        annotations = json.load(f)
    path_to_embeddings = Path(args.embeddings_file)
    path_to_faiss_index = path_to_embeddings / f"{args.embed_type}_embeddings_{args.index_tag}.index"

    features_index = faiss.read_index(
        str(path_to_faiss_index),
        faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,
    )
    logging.info(
        f"Loaded {args.embed_type} index from {path_to_faiss_index} ({features_index.ntotal} vectors)"
    )
    clip_to_index = load_clip_to_index(
        path_to_embeddings, args.embed_type, args.index_tag,
    )
    features_index.make_direct_map()
    all_clips = set(clip_to_index.keys())

    positive_labels = args.label.split("&&")
    negative_labels = list(args.negative_labels) if args.negative_labels else []

    positive_clips = gather_clips_with_condition(
        annotations, positive_labels, args.use_autolabels,
    )
    max_positive = 50_000
    if args.n_positive_samples != -1:
        max_positive = min(args.n_positive_samples, max_positive)
    if len(positive_clips) > max_positive:
        np.random.shuffle(positive_clips)
        positive_clips = positive_clips[:max_positive]

    logging.info(f"Found {len(positive_clips)} positive clips")

    if not negative_labels:
        negative_clips = all_clips - set(positive_clips)
    else:
        negative_clips = gather_clips_with_condition(
            annotations, negative_labels, args.use_autolabels,
        )

    negative_clips = sorted(set(negative_clips))
    logging.info(f"Found {len(negative_clips)} negative clips")

    if len(negative_clips) >= args.n_negative_samples:
        np.random.shuffle(negative_clips)
        negative_clips = negative_clips[: args.n_negative_samples]
    else:
        remaining_needed = args.n_negative_samples - len(negative_clips)
        already_sampled = set(negative_clips)
        available_pool = list(all_clips - set(positive_clips) - already_sampled)
        np.random.shuffle(available_pool)
        sampled_additional = available_pool[:remaining_needed]
        negative_clips += sampled_additional

    # Drop training clips that aren't in the embedding index — feature
    # reconstruction would skip them anyway, and the `*_clips.json`
    # outputs should reflect what was actually trained on.
    positive_clips = [c for c in positive_clips if c in clip_to_index]
    negative_clips = [c for c in negative_clips if c in clip_to_index]

    X_pos = clips_to_embeddings(positive_clips, clip_to_index, features_index)
    X_neg = clips_to_embeddings(negative_clips, clip_to_index, features_index)

    build_classifier_run(
        output_dir=run_dir,
        positive_clips=positive_clips,
        negative_clips=negative_clips,
        positive_features=X_pos,
        negative_features=X_neg,
        corpus_items=list(clip_to_index.items()),
        features_index=features_index,
        embed_type=args.embed_type,
        positive_labels=positive_labels,
        negative_labels=negative_labels,
        trained_by=args.trained_by,
        use_autolabels=args.use_autolabels,
        save_threshold=args.save_threshold,
        max_clips=args.max_clips,
        chunk_size=args.chunk_size,
        run_id=run_id,
        search_params=args.search_params,
        logger=logging.getLogger(),
    )

    elapsed = time.time() - start
    logging.info(f"Training and evaluating the classifier took {elapsed:.4f}s")


if __name__ == "__main__":
    main()
