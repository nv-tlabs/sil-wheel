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

"""Train a classifier from wheel search results and upload it to the server.

Workflow
--------
1. Open the wheel UI in a browser, build up two searches: one for clips you
   want as positives (e.g. ``hard braking at intersection``) and optionally
   one for clips you want as negatives (e.g. ``smooth highway driving``).
   Copy each URL.
2. Run this script with both URLs. It pulls clip_ids from each search over
   HTTP, reconstructs the matching embeddings from the local FAISS index,
   trains a logistic regression classifier, scores the entire corpus,
   and uploads the run back to the same server.
3. Refresh the wheel UI's classifier panel — the new run is there under
   the matching ``(positive_labels, negative_labels)`` combination.

If ``--negative-search-url`` is omitted, ``--n-negative-samples`` random
clips from the embeddings index (excluding the positives) are used as
negatives instead.

Requires read access to the wheel's FAISS embeddings directory (the
notebook/HPC node and the server can be different machines, but they
must share the embedding files).

Usage
-----
::

    export WHEEL_PASSWORD=...
    python examples/classifier_from_search.py \\
        --server-url http://wheel-host:8012 \\
        --username alice \\
        --positive-search-url 'http://wheel-host:8012/?search=hard+braking' \\
        --negative-search-url 'http://wheel-host:8012/?search=highway+cruising' \\
        --positive-labels hard_braking \\
        --embeddings-dir /path/to/cosmos_embeddings
"""
import argparse
import os
import pickle
from pathlib import Path

import faiss
import numpy as np

from sil_wheel.classifier_build import build_classifier_run
from sil_wheel.http_client import WheelHTTPClient


def load_clip_to_index(embeddings_dir: Path, embed_type: str, tag: str) -> dict:
    """Return ``{clip_id: faiss_row}`` for the given embed_type+tag.

    Mirrors ``scripts/embed_io.load_clip_to_index`` but inlined so this
    example doesn't depend on the scripts/ pythonpath. Tries the modern
    .npy sidecars first, then the .pkl form.
    """
    clip_ids_npy = embeddings_dir / f"{embed_type}_clip_ids_{tag}.npy"
    position_npy = embeddings_dir / f"{embed_type}_position_of_row_{tag}.npy"
    if clip_ids_npy.exists() and position_npy.exists():
        clip_ids = np.load(clip_ids_npy, allow_pickle=True)
        position_of_row = np.asarray(np.load(position_npy))
        _, first_rows = np.unique(position_of_row, return_index=True)
        return dict(zip(
            [str(c) for c in clip_ids.tolist()], first_rows.tolist(),
        ))

    cti_pkl = embeddings_dir / f"{embed_type}_clip_to_index_{tag}.pkl"
    if cti_pkl.exists():
        with open(cti_pkl, "rb") as f:
            return pickle.load(f)

    raise FileNotFoundError(
        f"No clip mapping found under {embeddings_dir} for {embed_type}/{tag}"
    )


def _embeddings_for_clips(clip_ids, clip_to_index, index):
    """Return (kept_clip_ids, embeddings) for clips present in the index."""
    kept = [c for c in clip_ids if c in clip_to_index]
    if not kept:
        return [], np.zeros((0, index.d), dtype=np.float32)
    rows = np.array([clip_to_index[c] for c in kept], dtype="int64")
    embs = index.reconstruct_batch(rows)
    return kept, np.asarray(embs, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server-url", required=True,
                        help="e.g. http://wheel-host:8012")
    parser.add_argument("--username")
    parser.add_argument("--positive-search-url", required=True,
                        help="URL whose results define the positive training set")
    parser.add_argument("--negative-search-url", default=None,
                        help="Optional URL whose results define negatives. If omitted, "
                             "negatives are random clips disjoint from positives.")
    parser.add_argument("--positive-labels", required=True, nargs="+",
                        help="Annotation keys these positives represent (e.g. 'hard_braking'). "
                             "Persisted in metadata so the UI lists this run under the matching label combination.")
    parser.add_argument("--negative-labels", nargs="*", default=[],
                        help="Annotation keys for negatives. Empty when --negative-search-url is omitted.")
    parser.add_argument("--embeddings-dir", required=True, type=Path,
                        help="Directory holding the FAISS index + clip map")
    parser.add_argument("--embed-type", default="cosmos",
                        choices=["cosmos", "caption", "visual", "other"])
    parser.add_argument("--index-tag", default=None,
                        help="Index tag suffix; auto-detected if omitted")

    parser.add_argument("--n-positive-samples", type=int, default=-1,
                        help="Cap on positives sampled from the search (-1 = no cap)")
    parser.add_argument("--n-negative-samples", type=int, default=100,
                        help="Number of negatives. When --negative-search-url is omitted, "
                             "this many random clips are used.")

    parser.add_argument("--save-threshold", type=float, default=0.3)
    parser.add_argument("--max-clips", type=int, default=7_000_000)
    parser.add_argument("--C", type=float, default=100.0)
    parser.add_argument("--chunk-size", type=int, default=50_000)

    parser.add_argument("--output-dir", type=Path, default=Path("./offline_classifier_runs"))
    parser.add_argument("--run-id", default=None,
                        help="Defaults to a 10-char alphanumeric string")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    username = args.username
    password = None
    if secrets := os.environ.get("WHEEL_SECRETS"):
        secrets = Path(secrets)
        if secrets.exists() and secrets.is_file():
            with secrets.open("r") as f:
                username, password = f.read().split(":")
    password = os.environ.get("WHEEL_PASSWORD", password)
    if not password:
        raise SystemExit("Set WHEEL_PASSWORD or WHEEL_SECRETS in the environment")
    if not username:
        raise SystemExit("Pass --username or set WHEEL_SECRETS")

    tag = args.index_tag
    if tag is None:
        prefix = f"{args.embed_type}_embeddings_"
        candidates = sorted(args.embeddings_dir.glob(f"{prefix}*.index"))
        if not candidates:
            raise SystemExit(
                f"No {prefix}*.index files in {args.embeddings_dir}"
            )
        if len(candidates) > 1:
            raise SystemExit(
                f"Multiple {args.embed_type} indexes in {args.embeddings_dir}: "
                f"{[p.name for p in candidates]} — pass --index-tag to pick one"
            )
        tag = candidates[0].name[len(prefix):-len(".index")]
        print(f"Auto-detected --index-tag {tag}")

    client = WheelHTTPClient(
        server_url=args.server_url, username=username, password=password,
    )

    print(f"Positive search: {args.positive_search_url}")
    pos_result = client.search_from_url(args.positive_search_url)
    positive_clip_ids = pos_result.clip_ids
    if args.n_positive_samples > 0:
        positive_clip_ids = positive_clip_ids[: args.n_positive_samples]
    print(f"  → {len(positive_clip_ids)} positive clips")

    if args.negative_search_url:
        print(f"Negative search: {args.negative_search_url}")
        neg_result = client.search_from_url(args.negative_search_url)
        negative_clip_ids = neg_result.clip_ids[: args.n_negative_samples]
        print(f"  → {len(negative_clip_ids)} negative clips")

    index_path = args.embeddings_dir / f"{args.embed_type}_embeddings_{tag}.index"
    print(f"Loading FAISS index: {index_path}")
    index = faiss.read_index(
        str(index_path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,
    )
    print("Building direct map (silent; ~30s-2min for million-scale indexes)...")
    index.make_direct_map()
    clip_to_index = load_clip_to_index(args.embeddings_dir, args.embed_type, tag)

    if not args.negative_search_url:
        rng = np.random.default_rng(args.seed)
        pool = [c for c in clip_to_index if c not in set(positive_clip_ids)]
        n = min(args.n_negative_samples, len(pool))
        negative_clip_ids = list(rng.choice(pool, size=n, replace=False))
        print(f"Sampled {len(negative_clip_ids)} random negatives from the index")

    positive_clip_ids, X_pos = _embeddings_for_clips(
        positive_clip_ids, clip_to_index, index,
    )
    negative_clip_ids, X_neg = _embeddings_for_clips(
        negative_clip_ids, clip_to_index, index,
    )
    if not positive_clip_ids:
        raise SystemExit("No positive results were present in the embedding index")
    if not negative_clip_ids:
        raise SystemExit("No negative results were present in the embedding index")

    search_params = WheelHTTPClient.query_from_url(args.positive_search_url)

    print(f"Training classifier ({len(positive_clip_ids)} pos, {len(negative_clip_ids)} neg)...")
    run_dir = build_classifier_run(
        output_dir=args.output_dir,
        positive_clips=positive_clip_ids,
        negative_clips=negative_clip_ids,
        positive_features=X_pos,
        negative_features=X_neg,
        corpus_items=list(clip_to_index.items()),
        features_index=index,
        embed_type=args.embed_type,
        positive_labels=args.positive_labels,
        negative_labels=args.negative_labels,
        trained_by=username,
        save_threshold=args.save_threshold,
        max_clips=args.max_clips,
        C=int(args.C),
        chunk_size=args.chunk_size,
        run_id=args.run_id,
        search_params=search_params,
    )
    print(f"  → wrote {run_dir}")

    print("Uploading to server...")
    response = client.upload_classifier_run(run_dir, overwrite=args.overwrite)
    print(f"  → run_id={response['run_id']}")
    print(f"  → server path: {response.get('path', '<unknown>')}")
    print(f"  → files={response['files_written']}")


if __name__ == "__main__":
    main()
