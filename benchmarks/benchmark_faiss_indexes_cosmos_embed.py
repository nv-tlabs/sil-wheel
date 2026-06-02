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
import gc
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from sil_wheel.stores.cosmos_embeddings_store import CosmosEmbeddingsStore


@dataclass
class BuildStats:
    train_s: float
    add_s: float
    size_bytes: int


def normalize_feats(feats: np.ndarray) -> np.ndarray:
    feats = np.ascontiguousarray(feats.astype(np.float32, copy=False))
    faiss.normalize_L2(feats)
    return feats


def make_index(
    d: int,
    index_spec: str,
    metric=faiss.METRIC_INNER_PRODUCT,
    nprobe: int = 256,
) -> faiss.Index:
    if index_spec.upper() == "FLAT":
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.index_factory(d, index_spec, metric)
        if hasattr(index, "nprobe"):
            index.nprobe = nprobe
        if hasattr(index, "hnsw"):
            try:
                index.hnsw.efSearch = max(64, int(nprobe))
            except Exception:
                pass
    return index


def iter_embedding_batches(
    source: Path,
    shuffle_files: bool = True,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yields (clip_ids, feats) one parquet file at a time.

    *source* can be a single parquet file or a directory.
    """
    if source.is_file():
        parquet_files = [source]
    else:
        parquet_files = list(source.glob("**/*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {source}")
        if shuffle_files:
            random.shuffle(parquet_files)

    for pf in tqdm(parquet_files, desc="[data] Reading parquet files", leave=True):
        df = pd.read_parquet(pf, columns=["clip_id", "embeddings"])
        df = df.drop_duplicates(subset=["clip_id"], ignore_index=True)

        if len(df) == 0:
            del df
            continue

        clip_ids = df["clip_id"].to_numpy()
        feats = np.vstack(df["embeddings"].values)
        feats = normalize_feats(feats)

        yield clip_ids, feats

        del df
        del clip_ids
        del feats
        gc.collect()


def collect_training_sample(
    source: Path,
    train_samples: int,
    d: int = 768,
) -> np.ndarray:
    """
    Collect up to train_samples vectors using reservoir sampling, so peak memory
    is bounded by train_samples * d * 4 bytes plus one file batch.
    """
    if train_samples <= 0:
        raise ValueError("train_samples must be positive")

    reservoir = np.empty((train_samples, d), dtype=np.float32)
    seen = 0

    for _, feats in iter_embedding_batches(source, shuffle_files=True):
        n = feats.shape[0]

        if seen < train_samples:
            take = min(train_samples - seen, n)
            reservoir[seen : seen + take] = feats[:take]
            seen += take

            if take == n:
                continue

            feats = feats[take:]
            n = feats.shape[0]

        # Reservoir sampling for the remaining vectors
        for i in range(n):
            j = random.randint(0, seen)
            if j < train_samples:
                reservoir[j] = feats[i]
            seen += 1

    if seen == 0:
        raise RuntimeError("No training vectors collected")

    if seen < train_samples:
        reservoir = reservoir[:seen]

    return np.ascontiguousarray(reservoir, dtype=np.float32)


def add_embeddings_to_index(
    index: faiss.Index,
    source: Path,
) -> Dict[str, int]:
    """
    Second pass: add vectors file by file. Keeps only one batch in memory.
    """
    clip_to_index: Dict[str, int] = {}
    offset = 0

    for clip_ids, feats in tqdm(
        iter_embedding_batches(source, shuffle_files=False),
        desc="[add] Adding to index",
        leave=True,
    ):
        # If cross-file duplicates are possible, skip already seen clip_ids.
        if clip_to_index:
            keep_mask = np.array([cid not in clip_to_index for cid in clip_ids], dtype=bool)
            if not keep_mask.all():
                clip_ids = clip_ids[keep_mask]
                feats = feats[keep_mask]

        if len(clip_ids) == 0:
            continue

        index.add(feats)

        for cid in clip_ids.tolist():
            clip_to_index[str(cid)] = offset
            offset += 1

    return clip_to_index


def build_faiss_index(
    source: Path,
    index_spec: str,
    train_samples: int,
    pretrained_index_path: Optional[Path] = None,
    metric=faiss.METRIC_INNER_PRODUCT,
    nprobe: int = 256,
) -> Tuple[faiss.Index, Dict[str, int], BuildStats]:
    d = 768
    index = make_index(d=d, index_spec=index_spec, metric=metric, nprobe=nprobe)

    train_s = 0.0

    if not index.is_trained:
        if pretrained_index_path is not None and pretrained_index_path.exists():
            print(f"[build] Loading pretrained structure from {pretrained_index_path} ...")
            index = faiss.read_index(str(pretrained_index_path))
            index.reset()
            if hasattr(index, "nprobe"):
                index.nprobe = nprobe
        else:
            train_data = collect_training_sample(source, train_samples)
            print(
                f"[build] Training {index_spec} on {train_data.shape[0]:,} sampled vectors ..."
            )
            t0 = time.perf_counter()
            index.train(train_data)
            train_s = time.perf_counter() - t0
            print(f"[build] Training done in {train_s:.2f}s")
            del train_data
            gc.collect()

    t0 = time.perf_counter()
    clip_to_index = add_embeddings_to_index(index, source)
    add_s = time.perf_counter() - t0
    print(f"[build] Add done in {add_s:.2f}s ({len(clip_to_index):,} total clips)")

    return (
        index,
        clip_to_index,
        BuildStats(train_s=train_s, add_s=add_s, size_bytes=0),
    )


@dataclass
class QueryStats:
    recall_at_k: float
    total_search_s: float
    mean_ms: float
    p50_ms: float
    p95_ms: float


def percentile(arr: List[float], p: float) -> float:
    if not arr:
        return 0.0
    return float(np.percentile(np.array(arr, dtype=np.float64), p))


def run_queries(
    store: CosmosEmbeddingsStore,
    queries: List[str],
    k: int,
) -> Tuple[Dict[str, List[str]], float, List[float]]:
    print("[query] Warmup ...")
    try:
        store.search_with_text("warmup query for faiss benchmark")
    except Exception:
        pass
    print(f"[query] Warmup complete. Running {len(queries)} queries ...")

    per_query_ids: Dict[str, List[str]] = {}
    latencies_ms: List[float] = []

    t_all0 = time.perf_counter()
    for q in queries:
        t0 = time.perf_counter()
        res = store.search_with_text(q)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        per_query_ids[q] = [cid for cid, _ in res[:k]]
    total_s = time.perf_counter() - t_all0

    mean_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    print(f"[query] Completed in {total_s:.2f}s. Mean {mean_ms:.1f} ms/query")
    return per_query_ids, total_s, latencies_ms


def compute_recall(
    baseline: Dict[str, List[str]], tested: Dict[str, List[str]], k: int
) -> float:
    recalls = []
    for q, base_ids in baseline.items():
        test_ids = tested.get(q, [])
        base_k = base_ids[:k]
        test_k = test_ids[:k]
        recalls.append(
            len(set(base_k) & set(test_k)) / len(base_k) if base_k else 0.0
        )
    return float(np.mean(recalls)) if recalls else 0.0


def format_seconds(s: float) -> str:
    return f"{s:.3f}"


def format_ms(ms: float) -> str:
    return f"{ms:.1f}"


def human_size(n: int) -> str:
    return f"{n / (1024.0 * 1024.0):.1f} MB"


def print_table(rows: List[Dict[str, str]], headers: List[str]) -> None:
    widths = {c: len(c) for c in headers}
    for r in rows:
        for c in headers:
            widths[c] = max(widths[c], len(str(r.get(c, ""))))

    def line(char: str = "-") -> str:
        return "+" + "+".join(char * (widths[c] + 2) for c in headers) + "+"

    def fmt_row(vals: List[str]) -> str:
        return (
            "|"
            + "|".join(f" {v:<{widths[c]}} " for v, c in zip(vals, headers))
            + "|"
        )

    print(line("="))
    print(fmt_row(headers))
    print(line("="))
    for r in rows:
        print(fmt_row([str(r.get(c, "")) for c in headers]))
    print(line("-"))


def collect_embeddings_for_clip_ids(
    emb_dir: Path, clip_ids: set, out_path: Path
) -> Path:
    """Collect embeddings for a known set of clip IDs, save to out_path."""
    rows = []
    remaining = set(clip_ids)
    parquet_files = list(emb_dir.glob("**/*.parquet"))
    for pf in tqdm(parquet_files, desc="[data] Collecting embeddings for clip IDs", leave=True):
        if not remaining:
            break
        df = pd.read_parquet(pf, columns=["clip_id", "embeddings"])
        df = df[df["clip_id"].isin(remaining)]
        df = df.drop_duplicates(subset=["clip_id"], ignore_index=True)
        remaining -= set(df["clip_id"].tolist())
        rows.append(df)
        del df
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["clip_id", "embeddings"])
    result.to_parquet(out_path, index=False)
    print(f"[data] Saved {len(result):,} shared clips to {out_path}")
    return out_path


def collect_and_cache_shared_clips(
    emb_dir: Path, max_clips: int, out_path: Path
) -> Path:
    """Collect max_clips embeddings from emb_dir, save to out_path, return out_path."""
    rows = []
    parquet_files = list(emb_dir.glob("**/*.parquet"))
    random.shuffle(parquet_files)
    seen: set = set()
    for pf in tqdm(parquet_files, desc="[data] Collecting shared clips", leave=True):
        if len(seen) >= max_clips:
            break
        df = pd.read_parquet(pf, columns=["clip_id", "embeddings"])
        df = df.drop_duplicates(subset=["clip_id"], ignore_index=True)
        df = df[~df["clip_id"].isin(seen)]
        remaining = max_clips - len(seen)
        df = df.iloc[:remaining]
        seen.update(df["clip_id"].tolist())
        rows.append(df)
        del df
    result = pd.concat(rows, ignore_index=True)
    result.to_parquet(out_path, index=False)
    print(f"[data] Saved {len(result):,} shared clips to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FAISS indexes vs Flat on Cosmos embeddings with Recall@k, "
            "build times, index size, and query latency."
        )
    )
    parser.add_argument(
        "embeddings_dir",
        type=str,
        help="Path to directory containing parquet embeddings (clip_id, embeddings)",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-clips", type=int, default=3_000_000)
    parser.add_argument("--nprobe", type=int, default=256)
    parser.add_argument(
        "--train-samples",
        type=int,
        default=3_000_000,
        help="Training sample size for IVF/OPQ/PQ indexes",
    )
    parser.add_argument("--queries", type=str, nargs="*", default=None)
    args = parser.parse_args()

    emb_dir = Path(args.embeddings_dir)
    k = args.k

    default_queries = [
        "dog",
        "ambulance",
        "pedestrian crossing the street",
        "dog on the sidewalk",
        "wheelchair user crossing road",
        "traffic cones on the street",
        "bicyclist passing by parked cars",
        "child near a crosswalk",
        "car turning left at an intersection",
        "bus stop with waiting passengers",
        "nighttime street with headlights",
        "rainy day pedestrians with umbrellas",
        "construction zone with cones and barriers",
        "person pushing a stroller on the sidewalk",
        "car stopping for a school bus",
        "night scene",
        "officer holding a sign",
        "officer signaling to stop",
        "officer waving to go",
    ]
    queries = args.queries if args.queries else default_queries

    configs = [
        # (spec,                          tag,                           nprobe)
        ("FLAT",                          "flat",                        args.nprobe),
        ("IVF4096,PQ96x8",               "ivf4096_pq96x8",              256),
        ("OPQ96_768,IVF4096,PQ96x8",     "opq96_768_ivf4096_pq96x8",   256),
        ("IVF8192,PQ96x8",               "ivf8192_pq96x8",              512),
        ("OPQ96_768,IVF8192,PQ96x8",     "opq96_768_ivf8192_pq96x8",   512),
        ("IVF16384,PQ96x8",              "ivf16384_pq96x8",             1024),
        ("OPQ96_768,IVF16384,PQ96x8",    "opq96_768_ivf16384_pq96x8",  1024),
        ("IVF8192,PQ96x8",               "ivf8192_pq96x8_np1024",       1024),
        ("IVF8192,PQ96x8",               "ivf8192_pq96x8_np2048",       2048),
        ("HNSW32",                        "hnsw32",                      128),
        ("HNSW64",                        "hnsw64",                      256),
    ]

    # Shared embeddings — collected once, reused for every index build
    shared_parquet_path = emb_dir / f"shared_embeddings_{args.max_clips // 1_000_000}m.parquet"
    shared_ids_path = emb_dir / f"shared_clip_ids_{args.max_clips // 1_000_000}m.pkl"
    if shared_parquet_path.exists():
        print(f"[data] Using cached shared embeddings from {shared_parquet_path}")
    elif shared_ids_path.exists():
        with open(shared_ids_path, "rb") as f:
            shared_clip_ids = pickle.load(f)
        print(f"[data] Loaded {len(shared_clip_ids):,} clip IDs from {shared_ids_path}, collecting embeddings ...")
        collect_embeddings_for_clip_ids(emb_dir, shared_clip_ids, shared_parquet_path)
    else:
        collect_and_cache_shared_clips(emb_dir, args.max_clips, shared_parquet_path)

    # Build phase — skip if both files already exist
    # For configs sharing the same spec (e.g. different nprobe), the second build
    # reuses the trained structure from the first via pretrained_index_path.
    built: Dict[str, Path] = {}  # spec → idx_path of already-built index
    for spec, tag, index_nprobe in configs:
        idx_path = emb_dir / f"cosmos_embeddings_{tag}.index"
        map_path = emb_dir / f"cosmos_clip_to_index_{tag}.pkl"
        if idx_path.exists() and map_path.exists():
            print(f"[{tag}] Index exists, skipping build")
            built[spec] = idx_path
            continue
        print(f"\n[{tag}] Building {spec} (nprobe={index_nprobe}) ...")
        index, clip_to_index, stats = build_faiss_index(
            source=shared_parquet_path,
            index_spec=spec,
            train_samples=args.train_samples,
            pretrained_index_path=built.get(spec),
            nprobe=index_nprobe,
        )
        faiss.write_index(index, str(idx_path))
        with open(map_path, "wb") as f:
            pickle.dump({str(cid): int(i) for cid, i in clip_to_index.items()}, f)
        print(f"[{tag}] Saved (train={stats.train_s:.1f}s, add={stats.add_s:.1f}s)")
        built[spec] = idx_path
        del index, clip_to_index
        gc.collect()

    # Query phase
    results = []
    baseline_results: Optional[Dict[str, List[str]]] = None

    for spec, tag, index_nprobe in configs:
        idx_path = emb_dir / f"cosmos_embeddings_{tag}.index"
        map_path = emb_dir / f"cosmos_clip_to_index_{tag}.pkl"
        if not idx_path.exists() or not map_path.exists():
            print(f"[{tag}] Index files missing, skipping")
            continue

        store = CosmosEmbeddingsStore(
            path_to_embeddings=args.embeddings_dir,
            index_spec=spec,
            use_flat=spec == "FLAT",
        )
        if hasattr(store.features_index, "nprobe"):
            store.features_index.nprobe = index_nprobe
        if hasattr(store.features_index, "hnsw"):
            try:
                store.features_index.hnsw.efSearch = max(64, index_nprobe)
            except Exception:
                pass

        n_clips = len(store.clips_to_index)
        size_bytes = idx_path.stat().st_size

        print(f"\n[{tag}] Running {len(queries)} queries (k={k}) ...")
        per_query_ids, total_s, lat_ms = run_queries(store, queries, k)

        recall = 1.0
        if spec == "FLAT":
            baseline_results = per_query_ids
            recall_str = "baseline"
        elif baseline_results is not None:
            recall = compute_recall(baseline_results, per_query_ids, k)
            recall_str = f"{recall:.3f}"
        else:
            recall_str = "n/a"

        mean_ms = float(np.mean(lat_ms)) if lat_ms else 0.0
        row = {
            "index": tag,
            "nprobe": str(index_nprobe),
            "clips": f"{n_clips:,}",
            "size": human_size(size_bytes),
            "recall@k": recall_str,
            "total_q_s": format_seconds(total_s),
            "mean_ms": format_ms(mean_ms),
            "p50_ms": format_ms(percentile(lat_ms, 50)),
            "p95_ms": format_ms(percentile(lat_ms, 95)),
        }
        results.append((recall, float(percentile(lat_ms, 95)), row))

        del store
        gc.collect()

    results.sort(key=lambda x: (-x[0], x[1]))
    rows = [r for _, _, r in results]

    print(f"\nBenchmark results (queries={len(queries)}, k={k}, clips={args.max_clips:,}):")
    print_table(
        rows,
        ["index", "nprobe", "clips", "size", "recall@k", "total_q_s", "mean_ms", "p50_ms", "p95_ms"],
    )


if __name__ == "__main__":
    main()
