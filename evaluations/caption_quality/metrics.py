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

"""NLG (BLEU/ROUGE/METEOR/CIDEr), BERTScore, and reference-free VLM judge.

A pair is {"clip_id", "reference", "prediction", "data_source", "video_path" (vlm_judge only)}. Each scorer exposes score_batch(pairs), score_one(...), and (where applicable) close().
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _row(pair: Dict[str, Any], **scores: float) -> Dict[str, Any]:
    """Build a result row: passthrough clip_id / data_source / scenario plus scores."""
    out = {"clip_id": pair.get("clip_id"), "data_source": pair.get("data_source")}
    if pair.get("scenario"):
        out["scenario"] = pair["scenario"]
    out.update(scores)
    return out


# ---------------------------------------------------------------------------
# NLG: BLEU-4, ROUGE-1/2/L, CIDEr, METEOR
# ---------------------------------------------------------------------------

class NLGScorer:
    """BLEU-4 / ROUGE-1/2/L / METEOR / CIDEr per pair, computed via pycocoevalcap (BLEU, CIDEr), google-research rouge_score, and NLTK (METEOR). Metrics: BLEU (Papineni et al. 2002), ROUGE (Lin 2004), METEOR (Banerjee & Lavie 2005), CIDEr (Vedantam et al. 2015). CIDEr uses TF-IDF over the batch, so small batches degrade to ~0."""

    SCORE_DIMS = ("bleu4", "rouge1_f", "rouge2_f", "rougeL_f", "meteor", "cider")

    def __init__(self):
        # Lazy imports keep pycocoevalcap / nltk out of the import path for callers who only need other scorers.
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from rouge_score import rouge_scorer
        import nltk
        from nltk.translate.meteor_score import meteor_score as _nltk_meteor

        for path, pkg in (("corpora/wordnet", "wordnet"),
                          ("tokenizers/punkt_tab", "punkt_tab")):
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)

        self._Bleu = Bleu
        self._Cider = Cider
        self._rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True,
        )
        self._meteor = _nltk_meteor

    def score_batch(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not pairs:
            return []

        predictions = [p["prediction"] for p in pairs]
        references = [[p["reference"]] for p in pairs]
        gts = {i: refs for i, refs in enumerate(references)}
        res = {i: [predictions[i]] for i in range(len(predictions))}

        _, bleu_per = self._Bleu(4).compute_score(gts, res)
        _, cider_per = self._Cider().compute_score(gts, res)

        out: List[Dict[str, Any]] = []
        for i, p in enumerate(pairs):
            rouge = self._rouge.score(p["reference"], p["prediction"])
            try:
                meteor = float(self._meteor(
                    [p["reference"].split()], p["prediction"].split(),
                ))
            except Exception:
                meteor = 0.0
            out.append(_row(
                p,
                bleu4=float(bleu_per[3][i]),
                rouge1_f=float(rouge["rouge1"].fmeasure),
                rouge2_f=float(rouge["rouge2"].fmeasure),
                rougeL_f=float(rouge["rougeL"].fmeasure),
                meteor=meteor,
                cider=float(cider_per[i]),
            ))
        return out

    def score_one(
        self,
        reference: str,
        prediction: str,
        clip_id: str = "_",
        data_source: str = "_",
    ) -> Dict[str, Any]:
        return self.score_batch([{
            "clip_id": clip_id, "data_source": data_source,
            "reference": reference, "prediction": prediction,
        }])[0]


def score_nlg(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return NLGScorer().score_batch(pairs)


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

class BERTScorer:
    """BERTScore P / R / F1 (Zhang et al., "BERTScore: Evaluating Text Generation with BERT", ICLR 2020; via the bert_score package), holding the HF model (default microsoft/deberta-xlarge-mnli) across calls."""

    SCORE_DIMS = ("bert_precision", "bert_recall", "bert_f1")

    def __init__(
        self,
        model_type: str = "microsoft/deberta-xlarge-mnli",
        lang: str = "en",
        batch_size: int = 64,
        rescale_with_baseline: bool = False,
        device: Optional[str] = None,
        max_length: Optional[int] = 512,
    ):
        from bert_score import BERTScorer as _BS

        self._scorer = _BS(
            model_type=model_type, lang=lang, batch_size=batch_size,
            rescale_with_baseline=rescale_with_baseline,
            device=device,
        )
        tokenizer = getattr(self._scorer, "_tokenizer", None)
        if tokenizer is not None and max_length is not None:
            tokenizer.model_max_length = int(max_length)

    def score_batch(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not pairs:
            return []
        predictions = [p["prediction"] for p in pairs]
        references = [p["reference"] for p in pairs]
        P, R, F1 = self._scorer.score(predictions, references)
        out: List[Dict[str, Any]] = []
        for i, p in enumerate(pairs):
            out.append(_row(
                p,
                bert_precision=float(P[i]),
                bert_recall=float(R[i]),
                bert_f1=float(F1[i]),
            ))
        return out

    def score_one(
        self,
        reference: str,
        prediction: str,
        clip_id: str = "_",
        data_source: str = "_",
    ) -> Dict[str, Any]:
        return self.score_batch([{
            "clip_id": clip_id, "data_source": data_source,
            "reference": reference, "prediction": prediction,
        }])[0]

    def close(self):
        import gc
        del self._scorer
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def score_bertscore(
    pairs: List[Dict[str, Any]],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    batch_size: int = 64,
    lang: str = "en",
    max_length: Optional[int] = 512,
) -> List[Dict[str, Any]]:
    return BERTScorer(
        model_type=model_type, lang=lang, batch_size=batch_size,
        max_length=max_length,
    ).score_batch(pairs)


# ---------------------------------------------------------------------------
# VLM judge (reference-free)
# ---------------------------------------------------------------------------

class VLMJudgeScorer:
    """Reference-free VLM scorer over (video_path, caption) pairs.

    Decodes video, samples frames at fps / max_frames, sends frames + caption to the VLM, returns 5 attributes (1-10 each): scene, action, road_entities, temporal, overall. Distinct from sil_wheel.llm.vlm_judge.VLMJudge (the live server judge); this is the eval-side variant that takes pre-resolved local video paths.
    """

    SCORE_DIMS = ("scene", "action", "road_entities", "temporal", "overall")

    def __init__(
        self,
        model: str = "gcp/google/gemini-3-flash-preview",
        api_key: Optional[str] = None,
        max_workers: int = 20,
        max_frames: int = 8,
        fps: float = 1.0,
        temperature: float = 0.0,
    ):
        from sil_wheel.llm.vlm_client import VLMClient
        from sil_wheel.llm.vlm_judge import extract_frames, judge_caption_score

        self.max_frames = max_frames
        self.fps = fps
        self.max_workers = max_workers
        api_key = api_key or os.environ.get("NV_INFERENCE_API_KEY")
        if not api_key:
            raise ValueError(
                "NV_INFERENCE_API_KEY not set. Pass api_key= or export the env var."
            )
        self.vlm = VLMClient(
            model=model, api_key=api_key, temperature=temperature,
        )
        self._extract_frames = extract_frames
        self._judge_caption_score = judge_caption_score

    def _score_one_pair(self, pair: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            with open(pair["video_path"], "rb") as f:
                video_bytes = f.read()
            frames = self._extract_frames(
                video_bytes, max_frames=self.max_frames, fps=self.fps,
            )
            if not frames:
                return None
            result = self._judge_caption_score(self.vlm, frames, pair["prediction"])
        except Exception as exc:
            logger.warning("vlm_judge failed on %s: %s", pair.get("clip_id"), exc)
            return None

        scores = result.get("scores") or {}
        if not scores:
            return None
        score_fields = {
            f"vlm_{dim}": float(scores[dim])
            for dim in self.SCORE_DIMS
            if isinstance(scores.get(dim), (int, float))
        }
        return _row(pair, **score_fields) if score_fields else None

    def score_one(
        self,
        video_path: str,
        prediction: str,
        clip_id: str = "_",
        data_source: str = "_",
    ) -> Optional[Dict[str, Any]]:
        return self._score_one_pair({
            "clip_id": clip_id, "data_source": data_source,
            "video_path": video_path, "prediction": prediction,
            "reference": None,
        })

    def score_batch(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not pairs:
            return out
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(self._score_one_pair, p) for p in pairs]
            for fut in as_completed(futures):
                row = fut.result()
                if row is not None:
                    out.append(row)
        return out


def score_vlm_judge(
    pairs: List[Dict[str, Any]],
    model: str = "gcp/google/gemini-3-flash-preview",
    max_workers: int = 20,
    max_frames: int = 8,
    fps: float = 1.0,
) -> List[Dict[str, Any]]:
    try:
        scorer = VLMJudgeScorer(
            model=model, max_workers=max_workers,
            max_frames=max_frames, fps=fps,
        )
    except ValueError as exc:
        logger.error("vlm_judge skipped: %s", exc)
        return []
    return scorer.score_batch(pairs)
