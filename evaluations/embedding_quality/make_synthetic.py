#!/usr/bin/env python
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

"""Create a tiny synthetic embedding-quality dataset for smoke tests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


LABELS = [
    "Animal Crossing",
    "dog",
    "Stop sign",
    "Barrier gate",
    "Fog",
    "Person Holding Traffic Sign",
    "Person with Hand Gestures",
    "U-turn",
]

EMBEDDINGS = {
    "cosmos": 32,
    "qwen3_vl_8b": 48,
    "pe_core_g14": 40,
    "caption": 36,
    "florence2_sigclip": 32,
    "trajectory": 24,
    "random": 32,
}

SIGNAL = {
    "cosmos": {
        "Animal Crossing": 1.0,
        "Stop sign": 1.2,
        "Barrier gate": 2.0,
        "Fog": 2.0,
        "Person with Hand Gestures": 0.9,
        "U-turn": 0.5,
    },
    "qwen3_vl_8b": {
        "Animal Crossing": 1.3,
        "Stop sign": 1.2,
        "Barrier gate": 2.2,
        "Fog": 2.1,
        "Person with Hand Gestures": 1.0,
        "U-turn": 0.6,
    },
    "pe_core_g14": {
        "Animal Crossing": 1.0,
        "Stop sign": 1.1,
        "Barrier gate": 1.8,
        "Fog": 1.6,
        "Person with Hand Gestures": 0.8,
        "U-turn": 0.5,
    },
    "caption": {
        "Animal Crossing": 2.0,
        "dog": 1.8,
        "Stop sign": 1.4,
        "Person Holding Traffic Sign": 1.8,
        "Person with Hand Gestures": 2.0,
        "Barrier gate": 1.0,
    },
    "florence2_sigclip": {
        "Animal Crossing": 1.5,
        "dog": 1.2,
        "Stop sign": 1.6,
        "Person Holding Traffic Sign": 1.4,
        "Person with Hand Gestures": 1.2,
        "Barrier gate": 1.0,
    },
    "trajectory": {
        "U-turn": 3.0,
        "Barrier gate": 0.4,
    },
    "random": {},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-neg", type=int, default=160)
    parser.add_argument("--n-per-label", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _slug(label: str) -> str:
    return (
        label.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def _unit_directions(
    rng: np.random.Generator,
    labels: list[str],
    dim: int,
) -> dict[str, np.ndarray]:
    dirs = {}
    for label in labels:
        vec = rng.standard_normal(dim).astype(np.float32)
        dirs[label] = vec / np.linalg.norm(vec)
    neg = rng.standard_normal(dim).astype(np.float32)
    dirs["__negative__"] = neg / np.linalg.norm(neg)
    return dirs


def _embedding_matrix(
    rng: np.random.Generator,
    clip_labels: dict[str, str | None],
    encoder: str,
    dim: int,
) -> tuple[list[str], np.ndarray]:
    directions = _unit_directions(rng, LABELS, dim)
    ids = sorted(clip_labels)
    rows: list[np.ndarray] = []
    for clip_id in ids:
        label = clip_labels[clip_id]
        vec = rng.normal(scale=0.45, size=dim).astype(np.float32)
        vec += 0.4 * directions["__negative__"]
        if label is not None:
            vec -= 0.25 * directions["__negative__"]
            vec += SIGNAL[encoder].get(label, 0.2) * directions[label]
        if encoder == "random":
            vec = rng.standard_normal(dim).astype(np.float32)
        rows.append(vec)
    return ids, np.stack(rows).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    embeddings_dir = args.out / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    clip_labels: dict[str, str | None] = {}
    label_rows: list[dict[str, str]] = []
    for label in LABELS:
        for i in range(args.n_per_label):
            clip_id = f"{_slug(label)}_{i:04d}"
            clip_labels[clip_id] = label
            label_rows.append({"clip_id": clip_id, "label": label})
    neg_rows: list[dict[str, str]] = []
    for i in range(args.n_neg):
        clip_id = f"negative_{i:04d}"
        clip_labels[clip_id] = None
        neg_rows.append({"clip_id": clip_id})

    labels_csv = args.out / "labels.csv"
    with open(labels_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "label"])
        writer.writeheader()
        writer.writerows(label_rows)

    negatives_csv = args.out / "negatives.csv"
    with open(negatives_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id"])
        writer.writeheader()
        writer.writerows(neg_rows)

    for encoder, dim in EMBEDDINGS.items():
        ids, X = _embedding_matrix(rng, clip_labels, encoder, dim)
        np.savez(
            embeddings_dir / f"{encoder}.npz",
            clip_ids=np.array(ids, dtype=object),
            embeddings=X,
        )

    meta = {
        "labels_csv": str(labels_csv),
        "negative_csv": str(negatives_csv),
        "embeddings_dir": str(embeddings_dir),
        "embeddings": list(EMBEDDINGS),
        "n_neg": args.n_neg,
        "n_per_label": args.n_per_label,
        "seed": args.seed,
    }
    (args.out / "README.json").write_text(json.dumps(meta, indent=2))
    print(f"[synthetic] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
