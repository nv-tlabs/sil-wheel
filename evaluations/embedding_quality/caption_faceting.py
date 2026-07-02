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

"""Caption facet encoders: split the dense caption into four aligned views.

The dense caption is split into four facets aligned with the caption-quality
axes (scene / road entities / action / temporal; the twelve structured fields
produced by ``facet_segment.py`` grouped into four texts), each embedded
separately with Qwen3-Embedding-8B. Emitted as ordinary
``caption_facet_{scene,road_entities,action,temporal}.npz`` encoders read by
``embeddings_io.load_embeddings`` and scored on the standard vector path.

The related common-component-removal ablation lives in
``evaluations/embedding_clustering/caption_pc_ablation.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Paper's caption-quality axes -> the 12 structured_caption fields.
FACETS: dict[str, list[str]] = {
    "scene": ["weather", "illumination", "road_type", "road_curvature"],
    "road_entities": ["key_objects", "road_information",
                      "vehicle_density", "pedestrian_density"],
    "action": ["ego_meta_action", "ego_speed", "rule_following_or_violation"],
    "temporal": ["key_events"],
}


def facet_text(structured: dict, keys: list[str]) -> str:
    """Join a facet's structured fields into one text (lists flattened)."""
    parts: list[str] = []
    for k in keys:
        v = structured.get(k)
        parts.append(" ".join(map(str, v)) if isinstance(v, list) else str(v))
    return " ".join(parts)


def _save_npz(path: Path, clip_ids: list[str], embeddings: np.ndarray) -> None:
    np.savez(path, clip_ids=np.array(clip_ids, dtype=object),
             embeddings=embeddings.astype(np.float32))
    print(f"[faceting] wrote {path.name}: {embeddings.shape}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--facets", type=Path, required=True,
                    help="JSONL from facet_segment.py: clip_id + 12 structured fields.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(l) for l in args.facets.open()]
    seen: set[str] = set()
    recs = [r for r in recs if not (r["clip_id"] in seen or seen.add(r["clip_id"]))]
    clip_ids = [r["clip_id"] for r in recs]
    print(f"[faceting] {len(clip_ids)} clips with facets", flush=True)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model, model_kwargs={"torch_dtype": "float16"})
    for facet, keys in FACETS.items():
        texts = [facet_text(r, keys) for r in recs]
        emb = model.encode(texts, normalize_embeddings=True,
                           batch_size=args.batch_size, show_progress_bar=True)
        _save_npz(args.out_dir / f"caption_facet_{facet}.npz", clip_ids, emb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
