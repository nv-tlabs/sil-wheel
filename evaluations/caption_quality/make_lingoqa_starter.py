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

"""Build a runnable starter dataset from the public LingoQA validation set.

Downloads the LingoQA val split (Marcu et al., "LingoQA: Visual Question
Answering for Autonomous Driving", ECCV 2024, arXiv:2312.14115;
https://github.com/wayveai/LingoQA) and writes a small captions SQLite +
``config.yaml`` that the ``caption_quality.py`` CLI can score out of the box.

Each LingoQA eval question carries two independent human answers; we load them
as the reference model and the prediction model, i.e. a real human-vs-human
agreement baseline. So the demo uses 100%% real public text and needs no model
inference, no video, and no API key (for the ``nlg`` metric).

Data is pulled from a community Hugging Face mirror of the official split
(which Wayve distributes via Google Drive); pass ``--hf-repo`` to use another.

    python evaluations/caption_quality/make_lingoqa_starter.py --out ./lingoqa_starter --limit 100
    python evaluations/caption_quality/caption_quality.py \
        ./lingoqa_starter/config.yaml ./lingoqa_starter/out.md \
        --reference-model lingoqa_human_a --prediction-model lingoqa_human_b --metrics nlg
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

REF_MODEL = "lingoqa_human_a"
PRED_MODEL = "lingoqa_human_b"
DATA_SOURCE = "lingoqa_val"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("lingoqa_starter"))
    ap.add_argument("--limit", type=int, default=100,
                    help="number of questions (clips) to include")
    ap.add_argument("--hf-repo", default="wyddmw/lingoqa-val",
                    help="HF dataset repo holding val.parquet (LingoQA val mirror)")
    ap.add_argument("--hf-file", default="val.parquet")
    args = ap.parse_args(argv)

    from huggingface_hub import hf_hub_download
    import pandas as pd

    path = hf_hub_download(args.hf_repo, args.hf_file, repo_type="dataset")
    df = pd.read_parquet(path, columns=["question_id", "segment_id", "question", "answer"])

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    db = out / "captions.db"
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE captions(uid INTEGER PRIMARY KEY, clip_id TEXT NOT NULL, "
        "model_name TEXT NOT NULL, caption TEXT NOT NULL, data_source TEXT, "
        "start_time REAL, end_time REAL)"
    )
    rows, n = [], 0
    for qid, g in df.groupby("question_id"):
        answers = g["answer"].tolist()
        if len(answers) < 2:  # need two human answers to form (reference, prediction)
            continue
        rows.append((qid, REF_MODEL, answers[0], DATA_SOURCE))
        rows.append((qid, PRED_MODEL, answers[1], DATA_SOURCE))
        n += 1
        if n >= args.limit:
            break
    con.executemany(
        "INSERT INTO captions(clip_id, model_name, caption, data_source) VALUES (?,?,?,?)",
        rows,
    )
    con.commit()
    con.close()

    (out / "config.yaml").write_text(f"datastores:\n  captions_db: {db.resolve()}\n")
    print(f"wrote {db} ({n} questions x 2 human answers) and {out/'config.yaml'}")
    print("\nrun the eval:")
    print("  python evaluations/caption_quality/caption_quality.py \\")
    print(f"      {out/'config.yaml'} {out/'out.md'} \\")
    print(f"      --reference-model {REF_MODEL} --prediction-model {PRED_MODEL} --metrics nlg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
