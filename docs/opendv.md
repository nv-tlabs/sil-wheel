# OpenDV retrieval pipeline

[OpenDV-YouTube](https://github.com/OpenDriveLab/DriveAGI/tree/main/opendv) is a large open
driving-video dataset (OpenDriveLab, *GenAD*, CVPR 2024). This guide turns it into a
text↔video **retrieval benchmark** in three stages:

| Stage | What you run | What you get |
| --- | --- | --- |
| 1. Clips | `scripts/opendv_sample_clips.py` | 20-second clips + a `manifest.jsonl` |
| 2. Captions | a VLM (see Stage 2) | one caption per clip, in three lengths |
| 3. Retrieval | `evaluations/retrieval/run_benchmark.py` | a Recall@K / median-rank leaderboard |

**Before you start:** clone and install this repo (see the top-level `README`); the scripts
need `ffmpeg` on your `PATH`. Each clip has a `clip_id` of the form
`<video_id>__<video_id>_<start>-<end>` — the video id appears twice (once as a prefix, once in
the clip name) — e.g. `abc123__abc123_600-620` for seconds 600–620 of video `abc123`. That id
links a clip to its caption and its embeddings through all three stages.

## Stage 1 — Get 20-second clips

OpenDV is published as a *list of YouTube videos*, not as hosted files, so you download the
source videos yourself and then cut them into clips. (The sampler reads the OpenDV metadata
sheet — split membership and per-video trim points — automatically; you don't fetch it.)

**1. Download the source videos.** Follow the official
[OpenDriveLab/DriveAGI](https://github.com/OpenDriveLab/DriveAGI/tree/main/opendv) guide,
which provides the video list and download scripts. Save the files anywhere on local disk: the
sampler finds each video by its id regardless of folder layout, and skips ids you don't have
(videos get removed from YouTube over time, so the exact clip set isn't fixed).

**2. Cut clips** with `scripts/opendv_sample_clips.py`. It defaults to the small `mini`
split, so pass `--subset full` for the whole dataset; clips are 20 s (`--clip-sec`) and
encoded with NVIDIA's GPU encoder by default (`--cut libx264` if you have no NVIDIA GPU):

```bash
# uniform — one clip per minute of video; a simple, content-blind baseline
python scripts/opendv_sample_clips.py --subset full \
    --videos-dir /data/opendv/videos --output-dir /data/opendv/clips \
    --method uniform --interval 60

# diverse — prefer rare, non-redundant clips (the first run downloads the
# OpenDriveLab/OpenDV-YouTube-Language maneuver/caption labels from HuggingFace)
python scripts/opendv_sample_clips.py --subset full \
    --videos-dir /data/opendv/videos --output-dir /data/opendv/clips \
    --method diverse --total 1000
```

Both samplers cut `--clip-sec`-second windows (default 20 s) from each video's usable range
`[start, end]` (the trims come from the metadata sheet); they differ only in which windows they
keep.

**`uniform`** emits a window `[t, t + clip_sec]` at every interval mark — starting at
`t = ⌈start / interval⌉ · interval` and stepping by `--interval` while `t + clip_sec ≤ end`.
Simple and reproducible, but it inherits the dataset's bias toward common scenes (driving is
mostly going straight).

**`diverse`** targets rare, non-redundant clips. It slides candidate windows (stride
`--stride`), labels each window `c` with its dominant maneuver `cmd(c)` and caption `cap(c)`
(from the OpenDV-YouTube-Language labels), then greedily grows a selection `S` by **Maximal
Marginal Relevance** (Carbonell & Goldstein, 1998) — each step adds the window that maximizes

    score(c) = λ·rel(c) − (1−λ)·max_{s∈S} sim(c, s)
    rel(c)   = log( N / (freq(cmd(c)) + 1) ),  normalized so the rarest maneuver = 1
    sim(c,s) = 0.5·Jaccard(cap(c), cap(s)) + 0.5·1[cmd(c) = cmd(s)]

with `N` the number of candidate windows and `freq(cmd)` how many share that maneuver — so a
rare maneuver (high `rel`) that differs from clips already chosen (low `sim`) wins. `--lambda`
(0–1, default 0.5) trades relevance against diversity; overlapping windows are skipped so clips
never overlap in time, and `--total N` (round-robin across videos) or `--select-k` per video
sets the budget.

A run writes `<output-dir>/<method>/` (e.g. `clips/uniform/`): the clip files at
`<video_id>/<video_id>_<start>-<end>.mp4`, plus `manifest.jsonl` with one row per clip —
`clip_id`, `video_id`, `clip_path`, `start_sec`, `end_sec`, `method` (and, for `diverse`,
`dominant_command` and `dominant_caption`).

## Stage 2 — Generate reference captions

OpenDV has no captions, so the benchmark uses *synthetic* ones (auto-generated, not
human-written). Stage 3 needs three captions per clip, so the goal of this stage is a
**`captions.jsonl`** with one object per clip:

```json
{"clip_id": "abc123__abc123_600-620", "short": "...", "medium": "...", "long": "..."}
```

(`clip_id` matches Stage 1; Stage 3 strips the `<video_id>__` prefix to align with the
embeddings.) The reference captions are produced like this — reproducible with any
vision-language model (VLM) API:

1. **Sample frames.** Decode each clip at a low frame rate (~4 fps) and send the frames to a
   VLM as images.
2. **Analyze, then reflect.** Ask for one JSON object describing the clip — key objects, the
   order of events, and scene attributes (weather, road type, ego maneuver, traffic density)
   — describing only what is visible. Then, in the same conversation, have the VLM re-check and
   correct that JSON.
3. **Render to three lengths.** Give the corrected JSON to a cheap text model and have it write
   the `short` / `medium` / `long` captions from that JSON alone.

All three prompts are deliberately strict. The **analysis** prompt asks for a fixed JSON schema
— key objects, chronological events, and ten scene attributes (vehicle and pedestrian density,
weather, illumination, ego speed, road curvature, road type, road layout, ego maneuvers, and
rule-following or violations) — under a *positive-observation* rule: state only what is present,
never what is absent. The **reflection** prompt re-checks that JSON against the video and fills
in whatever the first pass missed. The **render** prompt is strictly faithful — it may drop
detail to fit a shorter length but never adds, infers, or sharpens a fact (especially an
object's motion state) beyond the JSON. The exact prompts are in
[`../sil_wheel/datasets/opendv/caption_prompts.py`](../sil_wheel/datasets/opendv/caption_prompts.py):
`TWO_STEP_SYSTEM`, `ANALYZE`, and `REFLECT` for the analyze→reflect pass, and `RENDER_SYSTEM`
with `RENDER_TEMPLATE` for the render step (`RENDER_TEMPLATE.format(caption=...)` substitutes
the corrected JSON).

The repo's `scripts/extract_structured_captions.py` is a useful building block — it calls a
tool-calling VLM API and writes a structured caption per clip — but its output columns differ
from the JSONL above, so map its fields to `{clip_id, short, medium, long}` before Stage 3. The
official reference set used a GPT-5-class VLM to analyze and a small text model (Gemini
Flash-Lite) to render; any equivalent models work.

## Stage 3 — Run the retrieval benchmark

`evaluations/retrieval/run_benchmark.py` scores 1:1 text↔video retrieval — **Recall@{1, 5,
10}** (is the matching clip in the top K results?) and **median rank** — for each embedding
model on its own and for fused combinations (RRF and z-score). It needs two inputs:

1. **Video embeddings**, one parquet per encoder in a single directory. Build them from your
   clips with `scripts/extract_video_text_embeddings.py`,
   `scripts/extract_florence2_sigclip_embeddings.py`, and
   `scripts/extract_captions_embeddings.py`; the exact filenames the benchmark expects are in
   [`../evaluations/retrieval/README.md`](../evaluations/retrieval/README.md#expected-embeddings-format).
2. **The `captions.jsonl`** from Stage 2.

The script takes four positional arguments — `dataset`, `embeddings_dir`, `cache_dir`,
`gt_path` — then the caption length:

```bash
python evaluations/retrieval/run_benchmark.py \
    opendv \
    /data/opendv/embeddings \
    /data/opendv/text_cache \
    /data/opendv/captions.jsonl \
    --caption_length long \
    --results_md results/opendv_long.md
```

`cache_dir` is scratch space for encoded query text (reused across runs). `--caption_length`
chooses which caption field becomes the query — produce all three in Stage 2 to compare them.
See [`../evaluations/retrieval/README.md`](../evaluations/retrieval/README.md) for the full
encoder list, the fusion options, and how to dump per-query failures.
