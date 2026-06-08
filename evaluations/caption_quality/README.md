# Caption Quality Evaluation

Evaluate model-generated captions against human annotations and (optionally) the source video. CLI: [`caption_quality.py`](caption_quality.py). Implementations: `metrics.py`, `judges.py`, `evqa.py` (registry in `scorers.py`, reporting in `reporting.py`). Run from the repo root so the sibling modules and `sil_wheel` both import.

## Setup

```sh
pip install -e .                                   # the sil_wheel package
pip install pycocoevalcap nltk rouge-score bert-score   # reference-based metrics
pip install ultralytics                            # only for EVQAScore (evqa)
export NV_INFERENCE_API_KEY=...                    # required for llm_judge / vlm_judge / evqa
```

## Quickstart

No data of your own yet? `make_lingoqa_starter.py` pulls the public **LingoQA** validation split (Marcu et al., ECCV 2024, [arXiv:2312.14115](https://arxiv.org/abs/2312.14115)) via the HuggingFace `datasets` loader and writes a ready-to-score captions DB + `config.yaml`. Each LingoQA question carries two human answers, loaded as the reference and prediction model — a real human-vs-human agreement baseline, with no model inference, video, or API key needed (for `nlg`).

```sh
pip install datasets
python evaluations/caption_quality/make_lingoqa_starter.py --out ./lingoqa_starter --limit 100
python evaluations/caption_quality/caption_quality.py \
    ./lingoqa_starter/config.yaml ./lingoqa_starter/out.md \
    --reference-model lingoqa_human_a --prediction-model lingoqa_human_b \
    --metrics nlg
```

Add `bertscore` (downloads DeBERTa) or `llm_judge` (needs `NV_INFERENCE_API_KEY`) to `--metrics`. `lingojudge` runs too, but note this caption-centric CLI frames every pair with a generic question rather than LingoQA's per-item question.

## Reference modes

**Captions vs captions** (`--reference-model <model_name>`) — joins the captions SQLite DB on `clip_id`, with one `model_name` as reference and another as prediction. Useful for relative model comparison.

**Human annotations** (`--reference-model human`) — pulls reference text from the annotations DB one scenario at a time. Emits one row per `(clip, scenario)` pair; aggregation groups by scenario.

Scenarios are either explicit (`--scenarios "Roadwork" --scenarios "U-turn"`) or auto-selected (`--num-scenarios`, default 20): `select_scenarios()` ranks by `(n_versions, n_pred_clips)` and stratified-samples by clip count so the chosen set spans common and rare. `--annotation-project` (default `Alpamayo`) scopes the labels; empty string merges all projects. A denylist drops automation-emitted keys (`vlm_distill_*`, `reason_*`, `distill_*`, `scenario_*`).

Annotation keys are short tags, so n-gram metrics (BLEU / ROUGE / CIDEr / METEOR) will be near zero against long captions. **BERTScore, LingoJudge, and LLM judge are the right metrics in human mode.**

Reference-free metrics (`vlm_judge`, `evqa`) additionally resolve a local video path via the annotations DB's `video_paths` table.

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

`vlm_judge` reuses `judge_caption_score` from `sil_wheel.llm.vlm_judge`, so scores are directly comparable to the live server's VLM judge.

## Run on your own data

Point a `config.yaml` at your stores (`datastores.captions_db`, plus `annotations_db` for human / reference-free modes), then:

```sh
# Model vs model (caption-vs-caption): rank one model against a trusted one
python evaluations/caption_quality/caption_quality.py config.yaml results.md \
    --reference-model "<reference model_name>" \
    --prediction-model "<prediction model_name>" \
    --metrics nlg,bertscore,llm_judge --num-samples 200

# Human annotations as reference (auto-selects 20 scenarios; rows grouped by scenario)
python evaluations/caption_quality/caption_quality.py config.yaml results.md \
    --reference-model human --prediction-model "<prediction model_name>" \
    --metrics bertscore,lingojudge,llm_judge --num-samples 200
```

Pin scenarios with repeated `--scenarios "Roadwork" --scenarios "U-turn"`. A failing metric is logged and skipped (the rest continue); `--llm-provider auto` picks the provider from your API-key env vars; `--evqa-cache-dir <dir>` caches per-video features so warm runs skip SigLIP+YOLO. Output is one markdown section per metric — rows grouped by `scenario` (human mode) or `data_source` (caption-vs-caption), means across clips.

## References & attribution

The reference-based n-gram metrics, BERTScore, Lingo-Judge, and EVQAScore re-use published metrics / external models. `llm_judge` and `vlm_judge` are our own prompts (the latter mirrors `sil_wheel.llm.vlm_judge`, the live server's VLM judge).

| Metric | Source | Model / package |
| --- | --- | --- |
| `nlg` (BLEU) | Papineni et al., *BLEU*, ACL 2002 | `pycocoevalcap` |
| `nlg` (ROUGE) | Lin, *ROUGE*, 2004 | `rouge-score` (google-research) |
| `nlg` (METEOR) | Banerjee & Lavie, *METEOR*, 2005 | `nltk` |
| `nlg` (CIDEr) | Vedantam et al., *CIDEr*, CVPR 2015 | `pycocoevalcap` |
| `bertscore` | Zhang et al., *BERTScore*, ICLR 2020 | `bert-score`; `microsoft/deberta-xlarge-mnli` (MIT) |
| `lingojudge` | Marcu et al., *LingoQA*, ECCV 2024 ([arXiv:2312.14115](https://arxiv.org/abs/2312.14115), [code](https://github.com/wayveai/LingoQA)) | `wayveai/Lingo-Judge` |
| `evqa` | Liu et al., *EVQAScore*, [arXiv:2411.06908](https://arxiv.org/abs/2411.06908) (builds on EMScore, Shi et al., CVPR 2022) | independent reimplementation; SigLIP/CLIP + Ultralytics YOLO11 |

**Licensing note (for the open-source release).** Code here is Apache-2.0 and does not bundle any third-party model weights — encoders/classifiers are pulled at runtime from Hugging Face under their own model-card licenses (CLIP: MIT, DeBERTa: MIT, SigLIP: Apache-2.0). The one exception to watch: **Ultralytics YOLO11 (`ultralytics`) is AGPL-3.0**, used only by `evqa`. It is an optional dependency (the `[evqa]` extra), lazy-imported, and neither its code nor its `yolo11x-seg.pt` weight is committed to this repo. If EVQAScore is run, that happens in the user's own environment; review AGPL obligations before redistributing anything that bundles it. `evqa` is the only metric with this constraint — the rest are permissive.
