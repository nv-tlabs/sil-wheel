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
import sqlite3
import time
from pathlib import Path

import pandas as pd
from sil_wheel.embeddings.sentence_transformer_loader import (
    load_sentence_transformer,
)
from tqdm import tqdm


def fetch_captions_bulk(conn, clip_ids):
    """Return dict: clip_id -> list of individual caption strings."""
    if not clip_ids:
        return {}

    result = {}
    with conn:
        placeholders = ",".join("?" * len(clip_ids))
        rows = conn.execute(
            f"""
            SELECT clip_id, caption
            FROM captions
            WHERE clip_id IN ({placeholders})
            ORDER BY clip_id
            """,
            clip_ids,
        ).fetchall()
    for clip_id, caption in rows:
        result.setdefault(clip_id, []).append(caption)
    return result


def load_captions_from_parquet(parquet_path):
    """Return {clip_id: [caption, ...]} from a qwen-captions parquet.

    One caption per sub-clip; the 'summary' column is the final text.
    Multiple sub-clips per clip_id naturally become multiple captions.
    """
    df = pd.read_parquet(parquet_path, columns=["clip_id", "summary"])
    result = {}
    for cid, summary in df.itertuples(index=False, name=None):
        result.setdefault(cid, []).append(summary)
    return result, list(set(df["clip_id"]))


def flush_parts(path_to_output, existing_df, pending_parts):
    """Merge pending parts with existing data and write to parquet."""
    if not pending_parts:
        return existing_df
    new_df = pd.concat(pending_parts, ignore_index=True)

    if existing_df is not None:
        data = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        data = new_df

    data.to_parquet(path_to_output)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Extract caption embeddings from a caption DB"
    )
    parser.add_argument(
        "captions_source",
        help="Either a SQLite caption DB (*.db/*.sqlite) or a parquet file "
             "produced by extract_captions.py (*.parquet).",
    )
    parser.add_argument(
        "--embedding_model",
        choices=[
            "Qwen/Qwen3-Embedding-0.6B",
            "Qwen/Qwen3-Embedding-4B",
            "Qwen/Qwen3-Embedding-8B",
        ],
        default="Qwen/Qwen3-Embedding-8B",
        help="SentenceTransformer model name or path",
    )
    parser.add_argument(
        "--path_to_all_clips",
        default="/path/to/tmp/all_clip_ids.txt",
        help="Path to clip ids to be processed from the DB"
    )
    parser.add_argument(
        "--process_id",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of captions to encode in one model forward pass",
    )
    parser.add_argument(
        "--output",
        default="/path/to/datasets/caption_embeddings/group_{process_id}_{n_processes}.parquet",
        help="Output parquet path. Supports {process_id} and {n_processes} placeholders.",
    )
    args = parser.parse_args()

    path_to_output = args.output.format(
        process_id=args.process_id, n_processes=args.n_processes
    )
    Path(path_to_output).parent.mkdir(parents=True, exist_ok=True)

    if Path(path_to_output).is_file():
        existing_df = pd.read_parquet(path_to_output, columns=["clip_id", "embedding"])
        processed_clips = set(existing_df["clip_id"].tolist())
        print(
            f"Resuming: {len(processed_clips)} clips already processed at {path_to_output}"
        )
    else:
        existing_df = None
        processed_clips = set()

    source = args.captions_source
    if source.endswith(".parquet"):
        captions_by_clip, all_clip_ids = load_captions_from_parquet(source)
        fetch_batch = lambda ids: {c: captions_by_clip.get(c, []) for c in ids}
        conn = None
    else:
        conn = sqlite3.connect(source)
        fetch_batch = lambda ids: fetch_captions_bulk(conn, ids)
        with open(args.path_to_all_clips, "r") as f:
            all_clip_ids = [line.strip() for line in f]

    clip_ids = all_clip_ids[args.process_id :: args.n_processes]
    clip_ids = [c for c in clip_ids if c not in processed_clips]
    print(
        f"Process {args.process_id}/{args.n_processes}: {len(clip_ids)} clips to process"
    )

    model = load_sentence_transformer(args.embedding_model)
    model.max_seq_length = 512

    pending_parts = []
    total_new = 0
    start = time.time()
    chunk_size = 100
    save_every = 5

    for chunk_idx, i in enumerate(tqdm(range(0, len(clip_ids), chunk_size)), start=1):
        batch_clip_ids = clip_ids[i : i + chunk_size]
        captions_map = fetch_batch(batch_clip_ids)

        ordered_ids = []
        texts = []
        for c in batch_clip_ids:
            for caption in captions_map.get(c, []):
                ordered_ids.append(c)
                texts.append(caption)

        if not texts:
            continue

        embeddings = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
        )

        pending_parts.append(
            pd.DataFrame(
                {
                    "clip_id": ordered_ids,
                    "embedding": [emb.tolist() for emb in embeddings],
                }
            )
        )
        total_new += len(ordered_ids)

        if chunk_idx % save_every == 0:
            existing_df = flush_parts(path_to_output, existing_df, pending_parts)
            pending_parts = []
            print(
                f"Checkpoint saved after {chunk_idx} chunks "
                f"({len(existing_df)} total rows on disk)"
            )

        elapsed = time.time() - start
        print(
            f"Prepared {total_new} new clips so far — last chunk took {elapsed:.2f}s"
        )
        start = time.time()

    if conn is not None:
        conn.close()

    existing_df = flush_parts(path_to_output, existing_df, pending_parts)

    if existing_df is None:
        existing_df = pd.DataFrame(columns=["clip_id", "embedding"])

    print(f"Done. {len(existing_df)} clips saved at {path_to_output}")
