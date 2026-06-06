# Caption Quality Evaluation

Evaluate model-generated captions against human annotations and (optionally)
the source video. CLI: [`caption_quality.py`](caption_quality.py).
Implementations: `metrics.py`, `judges.py`, `evqa.py` (registry in `scorers.py`,
reporting in `reporting.py`). Run from the repo root so the sibling modules and
`sil_wheel` both import.

## Setup

```bash
pip install -e .                                   # the sil_wheel package
pip install pycocoevalcap nltk rouge-score bert-score   # reference-based metrics
pip install ultralytics                            # only for EVQAScore (evqa)
export NV_INFERENCE_API_KEY=...                    # required for llm_judge / vlm_judge / evqa
```

## Reference modes

**Captions vs captions** (`--reference-model <model_name>`) — joins the
captions SQLite DB on `clip_id`, with one `model_name` as reference and
another as prediction. Useful for relative model comparison.

**Human annotations** (`--reference-model human`) — pulls reference text
from the annotations DB one scenario at a time. Emits one row per
`(clip, scenario)` pair; aggregation groups by scenario.

Scenarios are either explicit (`--scenarios "Roadwork" --scenarios "U-turn"`)
or auto-selected (`--num-scenarios`, default 20): `select_scenarios()` ranks
by `(n_versions, n_pred_clips)` and stratified-samples by clip count so the
chosen set spans common and rare. `--annotation-project` (default `Alpamayo`)
scopes the labels; empty string merges all projects. A denylist drops
automation-emitted keys (`vlm_distill_*`, `reason_*`, `distill_*`,
`scenario_*`).

Annotation keys are short tags, so n-gram metrics (BLEU / ROUGE / CIDEr /
METEOR) will be near zero against long captions. **BERTScore, LingoJudge,
and LLM judge are the right metrics in human mode.**

Reference-free metrics (`vlm_judge`, `evqa`) additionally resolve a local
video path via the annotations DB's `video_paths` table.

## Metrics

### Reference-based

| Metric       | What it measures                                            | Range / output                         | Deps                          |
| ------------ | ----------------------------------------------------------- | -------------------------------------- | ----------------------------- |
| `nlg`        | BLEU-4, ROUGE-1/2/L F, METEOR, CIDEr                        | per-clip floats                        | `pycocoevalcap`, `nltk`, `rouge-score` |
| `bertscore`  | Contextual embedding P / R / F1 (DeBERTa)                   | per-clip floats in `[-1, 1]`           | `bert-score`                  |
| `lingojudge` | Wayve Lingo-Judge classifier (LingoQA QA format adapter)    | logit + binary `correct` flag (>0)     | `transformers`, GPU recommended |
| `llm_judge`  | LLM decides whether candidate contains a description matching reference | binary `llm_match` (0/1) + `llm_motivation` | `sil_wheel.llm.LLMClient`     |

### Reference-free

| Metric      | What it measures                                                      | Range / output                                        | Deps                  |
| ----------- | --------------------------------------------------------------------- | ----------------------------------------------------- | --------------------- |
| `vlm_judge` | VLM scores 5 attributes against the source video                      | per-clip 1-10 (`scene`, `action`, `road_entities`, `temporal`, `overall`) | wheel `VLMClient`     |
| `evqa`      | EVQAScore (arXiv 2411.06908): SigLIP/CLIP × YOLO crops × LLM keywords | per-clip `evqa_coarse`, `evqa_fine`, `evqa_score`     | `[evqa]` extra        |

`vlm_judge` reuses `judge_caption_score` from `sil_wheel.llm.vlm_judge`, so
scores are directly comparable to the live server's VLM judge.

## Usage

Human tags as reference, auto-selecting 20 Alpamayo scenarios (default):

```bash
python evaluations/caption_quality/caption_quality.py \
    config/wheel_launch_dev_server_config.yaml \
    caption_quality_results.md \
    --reference-model human \
    --prediction-model "Qwen2.5-VL-7B-Instruct" \
    --metrics bertscore,lingojudge,llm_judge \
    --num-samples 200
```

Pin specific scenarios:

```bash
python evaluations/caption_quality/caption_quality.py config/... results.md \
    --reference-model human \
    --prediction-model "Qwen2.5-VL-7B-Instruct" \
    --scenarios "Roadwork" --scenarios "U-turn" --scenarios "VRU crossing - pedestrian" \
    --metrics llm_judge --num-samples 60
```

Model vs model (silver standard, useful for relative ranking):

```bash
python evaluations/caption_quality/caption_quality.py \
    config/wheel_launch_dev_server_config.yaml \
    caption_quality_results.md \
    --reference-model "Gemini-2.5-pro (comprehensive v3 - summary long)" \
    --prediction-model "Qwen2.5-VL-7B-Instruct" \
    --metrics nlg,bertscore,llm_judge \
    --num-samples 200
```

Notable flags (see `--help` for the full list):

- `--metrics` selects any subset of the names above; failing metrics are
  logged and skipped while the rest continue.
- `--llm-provider auto` auto-detects from `NV_INFERENCE_API_KEY` /
  `NVIDIA_API_KEY` / `OPENAI_API_KEY`, else falls back to local.
- `--evqa-cache-dir <dir>` enables per-video visual-feature sidecars
  (`.pt`, ~50-200 KB each); warm runs skip SigLIP+YOLO. Server use also
  gets an in-memory LRU automatically (`EVQAScorer(mem_cache_size=...)`).

Output: one markdown section per metric. Rows are grouped by `scenario` in
human mode (per auto-selected scenario plus an `all` row), or by
`data_source` in caption-vs-caption mode. Numbers are means across clips.

## References & attribution

The reference-based n-gram metrics, BERTScore, Lingo-Judge, and EVQAScore
re-use published metrics / external models. `llm_judge` and `vlm_judge` are our
own prompts (the latter mirrors `sil_wheel.llm.vlm_judge`, the live server's
VLM judge).

| Metric | Source | Model / package |
| --- | --- | --- |
| `nlg` (BLEU) | Papineni et al., *BLEU*, ACL 2002 | `pycocoevalcap` |
| `nlg` (ROUGE) | Lin, *ROUGE*, 2004 | `rouge-score` (google-research) |
| `nlg` (METEOR) | Banerjee & Lavie, *METEOR*, 2005 | `nltk` |
| `nlg` (CIDEr) | Vedantam et al., *CIDEr*, CVPR 2015 | `pycocoevalcap` |
| `bertscore` | Zhang et al., *BERTScore*, ICLR 2020 | `bert-score`; `microsoft/deberta-xlarge-mnli` (MIT) |
| `lingojudge` | Marcu et al., *LingoQA*, ECCV 2024 ([arXiv:2312.14115](https://arxiv.org/abs/2312.14115), [code](https://github.com/wayveai/LingoQA)) | `wayveai/Lingo-Judge` |
| `evqa` | Liu et al., *EVQAScore*, [arXiv:2411.06908](https://arxiv.org/abs/2411.06908) (builds on EMScore, Shi et al., CVPR 2022) | independent reimplementation; SigLIP/CLIP + Ultralytics YOLO11 |

**Licensing note (for the open-source release).** Code here is Apache-2.0 and
does not bundle any third-party model weights — encoders/classifiers are pulled
at runtime from Hugging Face under their own model-card licenses (CLIP: MIT,
DeBERTa: MIT, SigLIP: Apache-2.0). The one exception to watch:
**Ultralytics YOLO11 (`ultralytics`) is AGPL-3.0**, used only by `evqa`. It is
an optional dependency (the `[evqa]` extra), lazy-imported, and neither its code
nor its `yolo11x-seg.pt` weight is committed to this repo. If EVQAScore is run,
that happens in the user's own environment; review AGPL obligations before
redistributing anything that bundles it. `evqa` is the only metric with this
constraint — the rest are permissive.
