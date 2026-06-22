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

"""Recursive (hierarchical) k-means with always-on per-level topic labels.

Re-clusters each cluster into finer sub-clusters and extracts topics at every
node, so a node's children can be read to decide which to descend into: a
progressively finer cluster-topic taxonomy. Splits reuse
:class:`~sil_wheel.cluster_build.FaissKMeans`; topics reuse
:func:`~sil_wheel.stores.cluster_topics.extract_topics_for_run`. A captions DB
(or explicit ``topic_fn``) is required -- topic-less hierarchies are refused.
Both backends are injectable for testing or custom split/topic backends.
"""

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sil_wheel.cluster_build import FaissKMeans, write_cluster_assignments
from sil_wheel.stores.cluster_topics import extract_topics_for_run, read_topics


@dataclass
class HierNode:
    """One node in the cluster hierarchy.

    ``path`` is the dotted id ("" at root, then "3", "3.7"); ``keywords`` /
    ``description`` describe this node within its parent (empty at root); leaves
    carry their members in ``clip_ids`` (so total clip storage is exactly n).
    """

    path: str
    depth: int
    size: int
    keywords: list[str] = field(default_factory=list)
    description: str = ""
    children: dict[str, "HierNode"] = field(default_factory=dict)
    clip_ids: list[str] = field(default_factory=list)  # leaf members only


def _resolve_branching(branching, depth):
    """k to use when splitting a node at ``depth`` into level ``depth+1``."""
    if isinstance(branching, int):
        return branching
    return branching[depth] if depth < len(branching) else branching[-1]


def build_hierarchical_clustering(
    embeddings,
    clip_ids,
    *,
    branching=10,
    max_depth=3,
    min_cluster_size=50,
    cluster_fn=None,
    topic_fn=None,
    captions_db_path=None,
    caption_model=None,
    samples_per_cluster=50,
    spherical=True,
    seed=1234,
    output_dir=None,
):
    """Recursively k-means ``embeddings`` (aligned to ``clip_ids``) and label
    every level with topics; returns the hierarchy root (root has no topics).

    ``branching``: children per split (int = same k everywhere; sequence =
    per-level k). ``max_depth``: split levels below the depth-0 root.
    ``min_cluster_size``: nodes smaller than this stay leaves. ``cluster_fn``
    ``(X, k, seed) -> (labels, centroids)`` and ``topic_fn`` ``(clip_ids, labels,
    k) -> {cid: {keywords, description}}`` default to the faiss / caption-topic
    backends; ``captions_db_path`` is required unless ``topic_fn`` is given.
    ``output_dir`` (optional) writes ``hier_assignments.parquet`` + ``hier_topics.json``.
    """
    if topic_fn is None:
        if captions_db_path is None:
            raise ValueError(
                "topics are mandatory at every level: pass captions_db_path or topic_fn"
            )
        topic_fn = _wheel_topic_fn(captions_db_path, caption_model, samples_per_cluster)
    if cluster_fn is None:
        cluster_fn = _wheel_cluster_fn(spherical=spherical)

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    clip_ids = list(clip_ids)
    if len(embeddings) != len(clip_ids):
        raise ValueError(
            f"embeddings rows ({len(embeddings)}) != clip_ids ({len(clip_ids)})"
        )

    root = HierNode(path="", depth=0, size=len(clip_ids))
    _split_node(
        root,
        embeddings=embeddings,
        clip_ids=clip_ids,
        branching=branching,
        max_depth=max_depth,
        min_cluster_size=min_cluster_size,
        cluster_fn=cluster_fn,
        topic_fn=topic_fn,
        seed=seed,
    )

    if output_dir is not None:
        _write_outputs(root, Path(output_dir))
    return root


def _split_node(
    node,
    *,
    embeddings,
    clip_ids,
    branching,
    max_depth,
    min_cluster_size,
    cluster_fn,
    topic_fn,
    seed,
):
    """Split ``node`` in place, recursing into each child."""
    k = _resolve_branching(branching, node.depth)
    n = len(clip_ids)
    # Need enough members to form k meaningful clusters, and to be allowed to.
    if node.depth >= max_depth or n < min_cluster_size or n < 2 * k:
        node.clip_ids = list(clip_ids)  # this node is a leaf
        return

    labels, _centroids = cluster_fn(embeddings, k, seed)
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    topics = topic_fn(clip_ids, labels, k) or {}  # always-on labels for the k children

    for cid in range(k):
        mask = labels == cid
        n_members = int(mask.sum())
        if n_members == 0:
            continue
        child_path = f"{node.path}.{cid}" if node.path else str(cid)
        topic = topics.get(str(cid), topics.get(cid, {}))
        child = HierNode(
            path=child_path,
            depth=node.depth + 1,
            size=n_members,
            keywords=list(topic.get("keywords", [])),
            description=topic.get("description", ""),
        )
        node.children[child_path] = child
        idx = np.flatnonzero(mask)
        _split_node(
            child,
            embeddings=embeddings[idx],
            clip_ids=[clip_ids[i] for i in idx],
            branching=branching,
            max_depth=max_depth,
            min_cluster_size=min_cluster_size,
            cluster_fn=cluster_fn,
            topic_fn=topic_fn,
            seed=seed,
        )


def flatten_leaf_assignments(root):
    """``{clip_id: leaf_path}`` for every clip (leaves carry their members)."""
    out = {}

    def walk(n):
        if not n.children:
            for cid in n.clip_ids:
                out[cid] = n.path
            return
        for c in n.children.values():
            walk(c)

    walk(root)
    return out


def _write_outputs(root, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    topics_out = {}

    def walk(n):
        if n.path:  # root has no topics
            topics_out[n.path] = {
                "keywords": n.keywords,
                "description": n.description,
                "size": n.size,
                "depth": n.depth,
            }
        for c in n.children.values():
            walk(c)

    walk(root)
    (output_dir / "hier_topics.json").write_text(json.dumps(topics_out, indent=2))

    leaf_map = flatten_leaf_assignments(root)
    depth_by_path = {p: t["depth"] for p, t in topics_out.items()}
    clip_ids = list(leaf_map)
    pd.DataFrame(
        {
            "clip_id": clip_ids,
            "path": [leaf_map[c] for c in clip_ids],
            "depth": [depth_by_path.get(leaf_map[c], 0) for c in clip_ids],
        }
    ).to_parquet(output_dir / "hier_assignments.parquet", index=False)


def _wheel_cluster_fn(spherical=True):
    """Default split backend: :class:`~sil_wheel.cluster_build.FaissKMeans`."""

    def cluster_fn(X, k, seed):
        km = FaissKMeans(
            feature_dim=X.shape[1],
            n_clusters=k,
            spherical_kmeans=spherical,
            seed=seed,
            verbose=False,
        )
        km.fit(X)
        labels, _distances = km.predict(np.ascontiguousarray(X, dtype=np.float32))
        return labels, km.centroids

    return cluster_fn


def _wheel_topic_fn(
    captions_db_path,
    caption_model,
    samples_per_cluster,
):
    """Default topic backend: extract_topics_for_run over a temp run dir built
    from this node's assignments."""

    def topic_fn(clip_ids, labels, k):
        labels = np.asarray(labels).astype(np.int64).reshape(-1)
        distances = np.zeros(len(labels), dtype=np.float32)  # unused for topics
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_cluster_assignments(run_dir, labels, distances, list(clip_ids), k)
            extract_topics_for_run(
                run_dir,
                str(captions_db_path),
                model_name=caption_model,
                samples_per_cluster=samples_per_cluster,
            )
            return read_topics(run_dir).get("topics", {})

    return topic_fn
