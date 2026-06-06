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

"""EVQAScore (arXiv 2411.06908): reference-free video caption quality.

Coarse score (global frame ↔ caption) + fine score (YOLO crops ↔ LLM
keywords), combined via bidirectional max-matching + harmonic mean. Heavy
deps live behind the ``[evqa]`` extra (ultralytics, SigLIP/CLIP weights).
"""
import gc
import hashlib
import logging
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _row(pair: Dict[str, Any], **scores: float) -> Dict[str, Any]:
    """Build a result row: passthrough clip_id / data_source / scenario plus scores."""
    out = {"clip_id": pair.get("clip_id"), "data_source": pair.get("data_source")}
    if pair.get("scenario"):
        out["scenario"] = pair["scenario"]
    out.update(scores)
    return out


KEYWORD_PROMPT = (
    "Extract a comma-separated list of concrete visual keywords (objects, "
    "agents, road features, weather/lighting cues) from the following driving "
    "scene caption. Return only the comma-separated list — no preamble, no "
    "explanation.\n\nCaption: {text}\n\nKeywords:"
)

_BACKEND_MODELS = {
    "siglip": "google/siglip-so400m-patch14-384",
    "clip": "openai/clip-vit-large-patch14",
}


# ---------------------------------------------------------------------------
# Pure scoring helpers
# ---------------------------------------------------------------------------

def _hmean(values: List[float]) -> float:
    pos = [v for v in values if v > 0]
    if len(pos) < len(values) or not pos:
        return 0.0
    return len(pos) / sum(1.0 / v for v in pos)


def _bidir_score(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 0.0
    sim = a @ b.T
    precision = torch.mean(torch.max(sim, dim=0)[0]).item()
    recall = torch.mean(torch.max(sim, dim=1)[0]).item()
    return _hmean([precision, recall])


def _clean_keywords(raw: str) -> str:
    if raw is None:
        return ""
    for prefix in ("Keywords:", "keywords:"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    items = [
        k.strip().strip('"').strip("'")
        for k in raw.replace("\n", ",").split(",")
        if k.strip()
    ]
    return ", ".join(items)


def _sample_frames(video_path: str, interval: int) -> List[np.ndarray]:
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return []
    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Class-based scorer
# ---------------------------------------------------------------------------

class EVQAScorer:
    """EVQAScore reference-free scorer holding encoder + YOLO + keyword LLM client.

    Loads heavy resources once; ``score_batch`` deduplicates by ``video_path``.
    Visual features are cached in-memory (LRU) and optionally on-disk via
    ``cache_dir`` so repeat runs skip SigLIP+YOLO.
    """

    SCORE_DIMS = ("evqa_coarse", "evqa_fine", "evqa_score")

    def __init__(
        self,
        backend: str = "siglip",
        model_path: Optional[str] = None,
        yolo_path: str = "yolo11x-seg.pt",
        frame_interval: int = 30,
        keyword_provider: str = "auto",
        keyword_model: Optional[str] = None,
        keyword_workers: int = 20,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        mem_cache_size: int = 2000,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "EVQAScore requires the 'evqa' extra. Install with: "
                "pip install -e .[evqa]"
            ) from exc

        from sil_wheel.llm.llm_client import get_llm_client
        from transformers import AutoModel, AutoProcessor

        if not torch.cuda.is_available() and device == "cuda":
            logger.warning("CUDA unavailable; EVQAScore will run on CPU (slow)")
            device = "cpu"

        self.frame_interval = frame_interval
        self.keyword_workers = keyword_workers
        self.device = device

        repo = model_path or _BACKEND_MODELS[backend]
        logger.info("Loading %s visual encoder: %s", backend, repo)
        self.processor = AutoProcessor.from_pretrained(repo)
        self.model = AutoModel.from_pretrained(repo).eval().to(device)

        logger.info("Loading YOLO: %s", yolo_path)
        self.yolo = YOLO(yolo_path)

        kwargs: Dict[str, Any] = {"temperature": 0.7, "max_tokens": 1024}
        if keyword_model:
            kwargs["model"] = keyword_model
        self.keyword_client = get_llm_client(provider=keyword_provider, **kwargs)

        # In-memory LRU → on-disk sidecar → recompute. Tag in the filename
        # invalidates on encoder/interval change (cf. visual_embeddings_store).
        self._cache_tag = f"{backend}_i{frame_interval}"
        self._mem_cache: "OrderedDict[str, Dict[str, torch.Tensor]]" = OrderedDict()
        self._mem_cache_size = mem_cache_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ----- encoder helpers -----

    @torch.inference_mode()
    def _encode_images(self, images, batch_size: int = 16) -> torch.Tensor:
        if not images:
            return torch.empty(0, 0)
        feats: List[torch.Tensor] = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            out = self.model.get_image_features(**inputs)
            feats.append(torch.nn.functional.normalize(out, dim=-1).cpu())
        return torch.cat(feats, dim=0)

    @torch.inference_mode()
    def _encode_texts(self, texts: List[str]) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for text in texts:
            items = [t.strip() for t in (text or "").split(",") if t.strip()] or [text or ""]
            inputs = self.processor(
                text=items, padding="max_length",
                return_tensors="pt", truncation=True,
            ).to(self.device)
            feats = self.model.get_text_features(**inputs)
            out.append(torch.nn.functional.normalize(feats, dim=-1).cpu())
        return out

    def _yolo_crops(self, frames: List[np.ndarray], batch_size: int = 64) -> List[np.ndarray]:
        crops: List[np.ndarray] = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            for j, r in enumerate(self.yolo(batch, stream=True, verbose=False)):
                for xyxy in r.boxes.xyxy:
                    x1, y1, x2, y2 = xyxy.cpu().numpy().astype(int)
                    crop = batch[j][y1:y2, x1:x2]
                    if crop.size > 0:
                        crops.append(crop)
        return crops or frames

    # ----- video features (cached) -----

    def _sidecar_path(self, video_path: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        # Stem keeps filenames readable; sha1 disambiguates same-stem paths.
        h = hashlib.sha1(video_path.encode()).hexdigest()[:16]
        return self.cache_dir / f"{Path(video_path).stem}_{h}_{self._cache_tag}.pt"

    def _video_features(self, video_path: str) -> Optional[Dict[str, torch.Tensor]]:
        feats = self._mem_cache.get(video_path)
        if feats is not None:
            self._mem_cache.move_to_end(video_path)
            return feats

        sidecar = self._sidecar_path(video_path)
        if sidecar and sidecar.exists():
            try:
                feats = torch.load(sidecar, weights_only=False, map_location="cpu")
            except Exception as exc:
                logger.warning("Bad sidecar %s, recomputing: %s", sidecar, exc)
                feats = None
            if feats is not None:
                self._cache_put(video_path, feats)
                return feats

        feats = self._compute_video_features(video_path)
        if feats is None:
            return None
        self._cache_put(video_path, feats)
        if sidecar:
            # Atomic write so concurrent readers see either old or new.
            tmp = sidecar.with_suffix(".pt.tmp")
            torch.save(feats, tmp)
            tmp.rename(sidecar)
        return feats

    def _cache_put(self, key: str, value: Dict[str, torch.Tensor]) -> None:
        self._mem_cache[key] = value
        self._mem_cache.move_to_end(key)
        while len(self._mem_cache) > self._mem_cache_size:
            self._mem_cache.popitem(last=False)

    def _compute_video_features(
        self, video_path: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        from PIL import Image
        frames = _sample_frames(video_path, self.frame_interval)
        if not frames:
            return None
        pil_frames = [Image.fromarray(f) for f in frames]
        frame_feats = self._encode_images(pil_frames)
        if frame_feats.numel() == 0:
            return None
        global_feat = torch.nn.functional.normalize(
            frame_feats.mean(dim=0, keepdim=True), dim=-1,
        )
        crops = self._yolo_crops(frames)
        local_feats = self._encode_images([Image.fromarray(c) for c in crops])
        return {"g": global_feat, "l": local_feats}

    # ----- keyword extraction (parallel LLM API) -----

    def _extract_keyword_one(self, text: str) -> str:
        for attempt in range(3):
            try:
                resp = self.keyword_client.generate(
                    prompt=KEYWORD_PROMPT.format(text=text),
                    system_prompt="You extract visual keywords from captions.",
                    temperature=0.7,
                    max_tokens=1024,
                )
                if resp:
                    return _clean_keywords(resp.strip())
            except Exception as exc:
                if attempt == 2:
                    logger.warning("Keyword extraction failed for %r: %s", text[:80], exc)
                    return text
                time.sleep(1)
                continue
            time.sleep(1)
        logger.warning("Keyword extraction empty for %r — using full caption", text[:80])
        return text

    def _extract_keywords(self, captions: List[str]) -> List[str]:
        out: List[Optional[str]] = [None] * len(captions)
        with ThreadPoolExecutor(max_workers=self.keyword_workers) as pool:
            futures = {pool.submit(self._extract_keyword_one, captions[i]): i for i in range(len(captions))}
            for fut in as_completed(futures):
                out[futures[fut]] = fut.result()
        return [s or "" for s in out]

    # ----- public API -----

    def score_batch(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not pairs:
            return []
        captions = [p["prediction"] for p in pairs]
        keywords = self._extract_keywords(captions)

        unique_paths = list({p["video_path"] for p in pairs})
        video_feats: Dict[str, Optional[Dict[str, torch.Tensor]]] = {}
        for vpath in unique_paths:
            video_feats[vpath] = self._video_features(vpath)

        caption_feats = self._encode_texts(captions)
        keyword_feats = self._encode_texts(keywords)

        out: List[Dict[str, Any]] = []
        for i, p in enumerate(pairs):
            vf = video_feats.get(p["video_path"])
            if vf is None:
                continue
            c = _bidir_score(vf["g"], caption_feats[i])
            f = _bidir_score(vf["l"], keyword_feats[i])
            out.append(_row(
                p,
                evqa_coarse=c,
                evqa_fine=f,
                evqa_score=_hmean([c, f]),
            ))
        return out

    def score_one(
        self,
        video_path: str,
        prediction: str,
        clip_id: str = "_",
        data_source: str = "_",
    ) -> Optional[Dict[str, Any]]:
        rows = self.score_batch([{
            "clip_id": clip_id, "data_source": data_source,
            "video_path": video_path, "prediction": prediction,
            "reference": None,
        }])
        return rows[0] if rows else None

    def close(self):
        self._mem_cache.clear()
        del self.model, self.processor, self.yolo
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def score_evqa(
    pairs: List[Dict[str, Any]],
    backend: str = "siglip",
    model_path: Optional[str] = None,
    yolo_path: str = "yolo11x-seg.pt",
    frame_interval: int = 30,
    keyword_provider: str = "auto",
    keyword_model: Optional[str] = None,
    keyword_workers: int = 20,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    scorer = EVQAScorer(
        backend=backend, model_path=model_path, yolo_path=yolo_path,
        frame_interval=frame_interval, keyword_provider=keyword_provider,
        keyword_model=keyword_model, keyword_workers=keyword_workers,
        device=device, cache_dir=cache_dir,
    )
    try:
        return scorer.score_batch(pairs)
    finally:
        scorer.close()
