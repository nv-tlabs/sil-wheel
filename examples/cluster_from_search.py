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

"""Cluster the result of a wheel search and upload the run back to the server.

Workflow
--------
1. Open the wheel UI in a browser, build up a search you like (caption FTS,
   semantic, classifier, country filter, …), and copy the URL.
2. Run this script with that URL — it pulls the clip_id list over HTTP,
   reconstructs the matching embeddings from the local FAISS index,
   builds a clustering run, and uploads it back to the same server.
3. Refresh the wheel UI's clustering panel — the new run is there.

Requires read access to the wheel's FAISS embeddings directory (the
notebook/HPC node and the server can be different machines, but they
must share the embedding files).

Expected ``--embeddings-dir`` layout
------------------------------------
The script expects the directory to follow the wheel's data-prep naming
convention. For a given ``--embed-type`` (``cosmos``/``caption``/``visual``)
and ``--index-tag`` (e.g. ``ivf4096_pq96x8``), it looks for:

- ``<embed_type>_embeddings_<tag>.index`` — FAISS index with
  ``make_direct_map()`` / ``reconstruct_batch()`` support (the default
  ``IVF + PQ`` and ``Flat`` indexes both work).
- One of the following clip-id ↔ row maps (tried in order):
  ``<embed_type>_clip_ids_<tag>.npy`` + ``<embed_type>_position_of_row_<tag>.npy``,
  or ``<embed_type>_clip_to_index_<tag>.pkl``.

The vectors inside the index must be the same dimensionality the wheel's
search uses for that ``embed_type`` (cosmos/caption/visual stores expose
their own dim — the script reconstructs whatever the index returns and
clusters that). ``--index-tag`` is auto-detected when only one
``<embed_type>_embeddings_*.index`` file is present in the directory.

Use ``--embed-type other`` if your embeddings don't match the three known
types. Clustering still works; the server's "Top clusters for…" feature
will be disabled for that run since it has no encoder for ``other``.

Usage
-----
::

    export WHEEL_PASSWORD=...
    python examples/cluster_from_search.py \
        --server-url http://wheel-host:8012 \
        --username alice \
        --search-url 'http://wheel-host:8012/?search=hard+braking&search_country=DE' \
        --embeddings-dir /path/to/cosmos_embeddings \
        --n-clusters 200
"""
import argparse
import os
import pickle
from pathlib import Path

import faiss
import numpy as np

from sil_wheel.cluster_build import build_clustering_run
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server-url", required=True,
                        help="e.g. http://wheel-host:8012")
    parser.add_argument("--username")
    parser.add_argument("--search-url", required=True,
                        help="URL copied from the wheel UI")
    parser.add_argument("--embeddings-dir", required=True, type=Path,
                        help="Directory holding the FAISS index + clip map")
    parser.add_argument("--embed-type", default="cosmos",
                        choices=["cosmos", "caption", "visual", "other"],
                        help="Use 'other' for custom embeddings — clustering "
                             "works, but 'Top clusters for…' on the resulting "
                             "run will return empty since the server has no "
                             "encoder registered for that type.")
    parser.add_argument("--index-tag", default=None,
                        help="Index tag suffix; auto-detected from --embeddings-dir if omitted")
    parser.add_argument("--n-clusters", type=int, default=200)
    parser.add_argument("--max-clips", type=int, default=None,
                        help="Optional cap on the search result before clustering; default no cap")
    parser.add_argument("--output-dir", type=Path, default=Path("./offline_runs"))
    parser.add_argument("--run-id", default=None,
                        help="Defaults to a 10-char alphanumeric string")
    parser.add_argument("--overwrite", action="store_true")

    # K-means knobs (forwarded to build_clustering_run)
    parser.add_argument("--spherical", action="store_true",
                        help="Use spherical (cosine-style) k-means")
    parser.add_argument("--max-points-per-centroid", type=int, default=256,
                        help="FAISS k-means training cap = n_clusters × this")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-iter", type=int, default=25,
                        help="K-means iterations")
    parser.add_argument("--n-redo", type=int, default=1,
                        help="K-means restarts")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose K-means logging")

    # UMAP knobs
    parser.add_argument("--umap-n-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-max-clips", type=int, default=50_000,
                        help="Subsample at most this many clips for UMAP")

    # Topic extraction (optional)
    parser.add_argument("--captions-db", type=Path, default=None,
                        help="SQLite captions DB; if given, runs cluster_topics")
    parser.add_argument("--caption-model", default=None,
                        help="Caption model name for topic extraction")
    args = parser.parse_args()

    # Grab the user info from args or the environment
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

    if args.embed_type == "other":
        print(
            "Warning: --embed-type=other → 'Top clusters for…' will be "
            "disabled on this run (server has no encoder for that type)."
        )

    client = WheelHTTPClient(
        server_url=args.server_url,
        username=username,
        password=password,
    )

    print(f"Search: {args.search_url}")
    result = client.search_from_url(args.search_url)
    clip_ids = (
        result.clip_ids if args.max_clips is None
        else result.clip_ids[: args.max_clips]
    )
    print(f"  → {len(result.clip_ids)} clips matched, clustering top {len(clip_ids)}")

    index_path = args.embeddings_dir / f"{args.embed_type}_embeddings_{tag}.index"
    print(f"Loading FAISS index: {index_path}")
    index = faiss.read_index(
        str(index_path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,
    )
    print("Building direct map (silent; ~30s-2min for million-scale indexes)...")
    index.make_direct_map()
    clip_to_index = load_clip_to_index(
        args.embeddings_dir, args.embed_type, tag,
    )

    kept_clip_ids = [c for c in clip_ids if c in clip_to_index]
    if not kept_clip_ids:
        raise SystemExit("No search results were present in the embedding index")
    row_ids = np.array(
        [clip_to_index[c] for c in kept_clip_ids], dtype="int64",
    )
    print(f"Reconstructing {len(row_ids)} embeddings...")
    embeddings = index.reconstruct_batch(row_ids)

    print(f"Clustering into {args.n_clusters} clusters...")
    run_dir = build_clustering_run(
        output_dir=args.output_dir,
        embeddings=embeddings,
        clip_ids=kept_clip_ids,
        n_clusters=args.n_clusters,
        embed_type=args.embed_type,
        spherical=args.spherical,
        max_points_per_centroid=args.max_points_per_centroid,
        seed=args.seed,
        n_iter=args.n_iter,
        n_redo=args.n_redo,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_max_clips=args.umap_max_clips,
        captions_db_path=args.captions_db,
        caption_model=args.caption_model,
        run_id=args.run_id,
        search_params=WheelHTTPClient.query_from_url(args.search_url),
        verbose=args.verbose,
    )
    print(f"  → wrote {run_dir}")

    print("Uploading to server...")
    response = client.upload_clustering_run(run_dir, overwrite=args.overwrite)
    print(f"  → run_id={response['run_id']}")
    print(f"  → server path: {response.get('path', '<unknown>')}")
    print(f"  → files={response['files_written']}")


if __name__ == "__main__":
    main()
