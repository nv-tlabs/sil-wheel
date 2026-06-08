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

"""Build a runnable caption-quality starter from the public LingoQA val split.

Loads LingoQA (Marcu et al., "LingoQA", ECCV 2024, arXiv:2312.14115) via the
HuggingFace `datasets` loader and writes a captions SQLite + config.yaml. Each
question has two human answers, loaded as the reference and prediction model --
a real human-vs-human baseline that needs no model, video, or API key (`nlg`).

    python evaluations/caption_quality/make_lingoqa_starter.py --out ./lingoqa_starter
"""
from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

REF, PRED = "lingoqa_human_a", "lingoqa_human_b"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("lingoqa_starter"))
    ap.add_argument("--limit", type=int, default=100, help="questions (clips) to include")
    ap.add_argument("--hf-repo", default="wyddmw/lingoqa-val", help="HF dataset (LingoQA val mirror)")
    args = ap.parse_args(argv)

    from datasets import load_dataset

    ds = load_dataset(args.hf_repo, split="validation")
    by_q: dict[str, list[str]] = defaultdict(list)
    for qid, ans in zip(ds["question_id"], ds["answer"]):
        by_q[qid].append(ans)

    args.out.mkdir(parents=True, exist_ok=True)
    db = args.out / "captions.db"
    db.unlink(missing_ok=True)
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE captions(uid INTEGER PRIMARY KEY, clip_id TEXT, model_name TEXT, "
                "caption TEXT, data_source TEXT, start_time REAL, end_time REAL)")
    pairs = [(q, a) for q, a in by_q.items() if len(a) >= 2][: args.limit]
    con.executemany(
        "INSERT INTO captions(clip_id, model_name, caption, data_source) VALUES (?,?,?,?)",
        [row for q, a in pairs for row in
         ((q, REF, a[0], "lingoqa_val"), (q, PRED, a[1], "lingoqa_val"))],
    )
    con.commit()
    con.close()
    (args.out / "config.yaml").write_text(f"datastores:\n  captions_db: {db.resolve()}\n")
    print(f"wrote {db} ({len(pairs)} questions x 2 human answers) + config.yaml\n"
          f"run: python evaluations/caption_quality/caption_quality.py "
          f"{args.out}/config.yaml {args.out}/out.md "
          f"--reference-model {REF} --prediction-model {PRED} --metrics nlg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
