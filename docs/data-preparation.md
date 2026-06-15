<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Data Preparation

We provide various scripts to prepare the videos and metadata for the server:
- `scripts/prepare_data.py` for processing the raw videos
- `scripts/extract_trajectory_stats.py` for extracting ego-trajectory data and aligning them with the camera timestamps
- `scripts/extract_video_text_embeddings.py` for extracting text-to-video embeddings using Cosmos-Embed, Qwen3-VL-Embed, or PE-Core features
- `scripts/extract_captions.py` for extracting captions either for the entire clip or sub-clips
- `scripts/extract_florence2_sigclip_embeddings.py` for extracting image-based embeddings using Florence2 and SigCLIP
- `scripts/extract_captions_embeddings.py` for extracting caption embeddings using text embeddings such as Qwen3-Embedding-8B

## Table of Contents
- [Supported Sources and Formats](#supported-sources-and-formats)
- [Process Videos](#process-videos)
    - [Videos on Local Filesystem](#videos-on-local-filesystem)
    - [Videos on S3](#videos-on-s3)
    - [Videos on the Hugging Face Hub](#videos-on-the-hugging-face-hub)
    - [Add Data to S3 and DB Updates](#add-data-to-s3-and-db-updates)
- [Grant User Access to a New Dataset](#grant-user-access-to-a-new-dataset)
- [Process Ego-trajectories](#process-ego-trajectories)
    - [Updating the full-trajectory index](#updating-the-full-trajectory-index)
    - [Updating the sub-trajectory indexes (5 s and 10 s)](#updating-the-sub-trajectory-indexes-5-s-and-10-s)
    - [Building an index from scratch (rare)](#building-an-index-from-scratch-rare)
- [Extract Text-to-Video Embeddings](#extract-text-to-video-embeddings)
    - [Update Text-to-Video FAISS Index](#update-text-to-video-faiss-index)
- [Extract Captions](#extract-captions)
    - [Extract Structured Captions via API Calls](#extract-structured-captions-via-api-calls)
    - [Add Captions to DB](#add-captions-to-db)
- [Extract Caption Embeddings](#extract-caption-embeddings)
    - [Update Caption Embedding Index](#update-caption-embedding-index)
- [Extract Visual (Florence2 SigCLIP) Embeddings](#extract-visual-florence2-sigclip-embeddings)
    - [Update Visual FAISS Index](#update-visual-faiss-index)
- [Statistics Artifacts](#statistics-artifacts)
    - [Trajectory Statistics](#trajectory-statistics)
    - [Dataset Statistics](#dataset-statistics)

## Supported Sources and Formats

Every data-preparation and extraction script reads its input the same way, so
you can point any of them at the same dataset. The videos can live on the local
filesystem, in an S3 object store, or on the Hugging Face Hub, and they can be
raw `.mp4` files or `.tar` or `.zip` archives. Each script detects the source
and file type and picks the right reader automatically. The supported
combinations are:

| Source | `.mp4` (raw files) | `.tar` archive | `.zip` archive |
| :--- | :---: | :---: | :---: |
| Local filesystem | yes | yes | no |
| S3 object store | yes | yes | no |
| Hugging Face Hub | no | yes | yes |

In short, `.zip` archives are read only from the Hugging Face Hub, and raw
`.mp4` files only from the local filesystem or S3; on the Hub the videos must be
packaged into `.tar` or `.zip` shards. You also work with one source at a time:
`--bucket` (S3) and `--hf-repo-id` (Hugging Face) cannot be combined, and a local
or S3 file list must use a single type, because the reader is chosen from its
first entry. Inside an archive, each video is named `<clip_id>.<camera>.mp4`, and
`--camera` limits processing to a single camera.

The source flags are the same across every script (`--bucket` / `--profile` /
`--endpoint` for S3, and `--hf-repo-id` / `--hf-allow-patterns` /
`--hf-cache-dir` for the Hugging Face Hub), so a dataset on any source flows
through the whole pipeline. The sections below cover each step in turn, starting
with video processing.

## Process Videos

`scripts/prepare_data.py` copies and compresses source videos into an
`output_dir`. For each video it downscales (default width 640), encodes to
H.264/AAC, adds faststart, and targets a size or CRF with ffmpeg. If compression
fails, the original video is saved uncompressed to the output directory. *The
script skips already-processed files and runs in parallel (default 16 workers).*

For distributed processing use `--process_id` and `--n_processes` to shard
work across multiple jobs (same pattern as `extract_video_text_embeddings.py`).

### Videos on Local Filesystem

```python
from pathlib import Path

root = "/path/to/videos_root"
# For raw .mp4 inputs this is a plain list of absolute .mp4 paths, one per line.
videos = sorted(Path(root).glob("**/*.mp4"))

with open("/tmp/video_clips.txt", "w") as f:
    for p in videos:
        f.write(str(p) + "\n")
```
The example above writes a plain list of `.mp4` paths, which is the input
format for raw local `.mp4` files. Local `.tar` archives use a different
contract: instead of a list of archive paths, pass a JSON file that maps each
`clip_id` to the `.tar` archive holding it, for example
`{"clip_000001": "/path/to/shard_00.tar"}`. The reader opens each archive and
extracts the matching `<clip_id>.<camera>.mp4` member.

Optionally, you can split the list of clips into chunks for parallel processing as follows:
```bash
in=/tmp/video_clips.txt
outdir=/tmp/chunks
mkdir -p "$outdir"
split -l 3000 -d -a 3 --additional-suffix=.txt "$in" "$outdir/chunk_"
```
This is recommended for maximum throughput. Once your list of paths is ready,
you can run the script in an interactive job or submit it with Slurm. Examples
of both follow:

- Run locally (interactive job)

```bash
python scripts/prepare_data.py /tmp/video_paths.txt /path/to/output_dir
```

To distribute across multiple processes:

```bash
python scripts/prepare_data.py /tmp/video_paths.txt /path/to/output_dir \
  --process_id 0 --n_processes 8
```

- Run as a Slurm job

```bash
sbatch --job-name data_processing_0 \
  launch_cpu_job.sbatch \
  scripts/prepare_data.py \
  /tmp/video_paths.txt \
  /path/to/output_dir/ \
  --process_id 0 --n_processes 8

# Submit multiple jobs
# Make sure to modify OUTPUT_ROOT, VIDEO_LIST and N_PROCESSES in the script
./submit_prepare_data_processing_jobs.sh
```

### Videos on S3

`scripts/prepare_data.py` can also process data stored on an S3-compatible
object store. The input is the paths of the files (`.mp4` or `.tar`) relative
to the bucket root. You can build that list with a command like:

```bash
s5cmd --endpoint-url=https://<your-s3-endpoint> --profile <your-aws-profile> ls 's3://<your-source-bucket>/*/chunk_*/<your-camera-folder>/*.mp4' | awk '{print $NF}' | tee video_paths.txt | pv -l >/dev/null
# s5cmd --endpoint-url=https://<your-s3-endpoint> --profile <your-aws-profile> ls 's3://<your-source-bucket>/*/chunk_*/*.tar' | awk '{print $NF}' | tee video_tar_paths.txt | pv -l >/dev/null
```


The command above collects every key matching
`s3://<your-source-bucket>/*/chunk_*/<your-camera-folder>/*.mp4`
and writes them to `video_paths.txt`. Once the file is ready, launch the
processing script as before, setting `--bucket` and `--profile` to point at the
S3 bucket you are downloading from.

### Videos on the Hugging Face Hub

You can also process a dataset straight from the Hugging Face Hub without
downloading it by hand. Point the script at a dataset repo with `--hf-repo-id`:
it lists the repo files, picks the archive format (`.zip` or `.tar`) from the
first matching shard, downloads only this process's shards with
`snapshot_download`, and reads the videos out of them. `--hf-repo-id` replaces
the positional `path_to_data` and cannot be combined with `--bucket`.

```bash
# .zip-packaged dataset (e.g. nvidia/PhysicalAI-Autonomous-Vehicles)
python scripts/prepare_data.py /path/to/output_dir \
  --hf-repo-id nvidia/PhysicalAI-Autonomous-Vehicles \
  --hf-allow-patterns "val/*" \
  --process_id 0 --n_processes 8

# .tar-packaged dataset (e.g. facebook/PE-Video)
python scripts/prepare_data.py /path/to/output_dir \
  --hf-repo-id facebook/PE-Video \
  --process_id 0 --n_processes 8
```

Key arguments:
- `--hf-repo-id`: Hugging Face dataset repo id (e.g. `facebook/PE-Video`). Makes
  `path_to_data` optional and is mutually exclusive with `--bucket`.
- `--hf-allow-patterns`: one or more glob patterns that limit which files are
  downloaded (e.g. `val/*`). Defaults to the whole repo.
- `--hf-cache-dir`: download cache location; defaults to `$HF_HOME`.

The archives must contain `<clip_id>.<camera>.mp4` members, and `--camera`
filters to a single camera as with the other sources. The same `--hf-*` flags
are available on the extraction scripts (captions, embeddings, trajectory
stats), so a Hugging Face dataset can be carried through the entire pipeline.

### Add Data to S3 and DB Updates

**Upload processed videos to your S3 bucket/folder**

```bash
aws s3 --endpoint=https://<your-s3-endpoint> sync /path/to/output_dir s3://<your-target-bucket>/path/to/folder
```

If you have a large amount of data to copy, we recommend
[s5cmd](https://github.com/peak/s5cmd). For reference, this is what we typically
run:

```bash
find path/to/videos_root -type f -name "*.mp4" > path/to/output_dir/mp4s.txt
awk '{print "cp \"" $0 "\" s3://<your-target-bucket>/your_dataset/"}' path/to/output_dir/mp4s.txt > path/to/output_dir/mp4s.s5cmd
s5cmd --endpoint-url=https://<your-s3-endpoint> --profile <your-aws-profile> --numworkers 128 --retry-count 10 --log error run path/to/output_dir/mp4s.s5cmd
```

**Create a mapping `clip_id -> relative_s3_path` (relative to your bucket root/folder) and save as JSON**

```python
import json
from pathlib import Path
from tqdm import tqdm

local_path = "/path/to/output_dir"
dataset_name = "path/to/folder/"  # S3 folder

def get_clipid_to_relpath(local_path, dataset_name, output_path):
    video_paths = {}
    for p in tqdm(sorted(Path(local_path).glob("*.mp4"))):
        clip_id = p.stem
        video_paths[clip_id] = dataset_name + "/" + p.name
    print(local_path, len(video_paths))

    with open(output_path, "w") as f:
        json.dump(video_paths, f)
```
Expected JSON format for `clipid_to_relpath`:

```json
{
  "clip_000001": "path/to/folder/clip_000001.mp4",
  "clip_000002": "path/to/folder/clip_000002.mp4"
}
```
Here `path/to/folder` must match `dataset_name`. It does not have to be the
dataset name shown in the tool, just the name of the folder that holds this data
source's videos.


If you have your data on S3 you can do the following to generate this mapping:
```python
import json
import subprocess

bucket = "your_dataset_v1"
output_path = "your_dataset_v1_video_paths.json"

result = subprocess.run(
    ["aws", "s3", "--profile", "<your-aws-profile>", "ls", f"s3://<your-target-bucket>/{bucket}/"],
    capture_output=True, text=True, check=True,
)
video_paths = {}
for line in result.stdout.strip().splitlines():
    fname = line.split()[-1]
    if not fname.endswith(".mp4"):
        continue
    video_paths[fname.removesuffix(".mp4")] = f"{bucket}/{fname}"

print(f"Found {len(video_paths)} clips")
with open(output_path, "w") as f:
    json.dump(video_paths, f)
```

**Upsert into SQLite tables `video_paths` and `clips`**

```python
import json, sqlite3

with open("/tmp/clipid_to_relpath.json", "r") as f:
    video_paths = json.load(f)

conn = sqlite3.connect("/path/to/annotations.db")

upsert_paths = """
INSERT INTO video_paths (clip_id, path)
VALUES (?, ?)
ON CONFLICT(clip_id) DO UPDATE SET path = excluded.path
"""

with conn:
    for clip_id, rel_path in video_paths.items():
        if rel_path:
            conn.execute(upsert_paths, (clip_id, rel_path))

upsert_clips = """
INSERT INTO clips (clip_id, data_source)
VALUES (?, ?)
ON CONFLICT(clip_id) DO UPDATE SET
  data_source = CASE
    WHEN clips.data_source IS NULL OR clips.data_source = ''
      THEN excluded.data_source
    WHEN instr(',' || clips.data_source || ',', ',' || excluded.data_source || ',') > 0
      THEN clips.data_source
    ELSE clips.data_source || ',' || excluded.data_source
  END
"""

# Replace YourDataSourceName with the dataset name that will appear in the tool
with conn:
    for clip_id in video_paths.keys():
        conn.execute(upsert_clips, (clip_id, "YourDataSourceName"))


# To validate that the correct number of clips were updated, run the following
# to count the clips containing a specific dataset name
sql = """
SELECT COUNT(*)
FROM clips
WHERE instr(',' || data_source || ',', ',' || ? || ',') > 0;
"""

cur = conn.execute(sql, ("YourDataSourceName",))
(count,) = cur.fetchone()

print(count)
```

**Registering the dataset's category and license**

Dataset grouping (optgroup in the UI) and the licensing icon (🔒 for
licensed, NVIDIA mark for internal, 🌐 for public) are driven by
the `datasets` table. Every `data_source` name that can appear in the
UI should have a row; names without one fall back to `Autonomous
Driving (AV)` + `internal`.

```python
query = """
INSERT INTO datasets (name, category, license)
VALUES (?, ?, ?)
ON CONFLICT(name) DO UPDATE SET
    category = excluded.category,
    license  = excluded.license;
"""

cur = conn.execute(sql, ("YourDataSourceName", "General Video", "public"))
conn.close()
```

- `category`: one of `Autonomous Driving (AV)`, `General Video`,
  `Graphics / Synthetic`, `Robotics / Embodied AI`.
- `license`: `public` (🌐), `licensed` (🔒), or `internal`
  (NVIDIA mark).

Rows can be edited later without a code change, since the server
re-reads this table on every `/videos` request.

**Updating country information**

Country data is stored directly in the `clips` table as an ISO 3166-1 alpha-2
code (e.g. `"US"`, `"DE"`, `"GB"`). It powers the country and driving-side
filters in the server. To set or update it:

```python
import sqlite3

conn = sqlite3.connect("/path/to/annotations.db")

upsert_country = "UPDATE clips SET country = ? WHERE clip_id = ?"

# Map of clip_id -> ISO 3166-1 alpha-2 country code
clip_id_to_country = {
    "clip_000001": "US",
    "clip_000002": "DE",
}

with conn:
    for clip_id, country_code in clip_id_to_country.items():
        conn.execute(upsert_country, (country_code, clip_id))
```

## Grant User Access to a New Dataset

After ingesting a new dataset (steps 1-3 above), users need explicit permission
to see it. There are two things to do:

**1. Add the dataset to `default_datasources`** in `sil_wheel/stores/users_data_store.py`
so that any user approved going forward automatically gets access:

```python
# users_data_store.py → UsersDataStore.__init__
self.default_datasources = [
    ...
    "YourDataSourceName",   # <-- add the new dataset name here
]
```

**2. Grant access to all existing users** by running the following one-off
script against the live users DB:

```python
from sil_wheel.stores.users_data_store import UsersDataStore

store = UsersDataStore("path/to/users.db")
store.grant_datasource_to_all_users("YourDataSourceName")
```

This uses `INSERT OR IGNORE` so it is safe to run multiple times.
Alternatively, individual users' datasource lists can be updated from the
admin panel via **Admin → Edit user → Datasources**.

## Process Ego-trajectories

`scripts/extract_trajectory_stats.py` parses the ego-trajectory signals from the
raw logs and aligns them to the video frames, writing a safetensors file the
server can consume. Specifically, it computes speed, acceleration, jerk, and
curvature from the ego-trajectory observations and interpolates them to the
frame timestamps for per-frame display.

Run locally:

```bash
python scripts/extract_trajectory_stats.py path/to/video_paths.txt path/to/output_dir 0
```
As with the other processing scripts, you pass the file that lists the absolute
video paths to process (`path_to_data`), the output directory
(`path_to_output`), and a `cnt` integer that disambiguates output filenames
across shards (`trajectory_stats_smoothed_{cnt}.safetensors`). The script expects
the related sensor files (for example the egomotion parquet) in a matching
directory layout. If your format is not supported, parse the trajectory data
into these arrays: `[x, y, z]` for location (Nx3), `ts` for the timestamps (N,)
at which the locations were recorded, and `frame_ts` for the timestamps (M,) at
which the camera frames were recorded.

Once the trajectory data is processed, you update (i) the memory map that holds
all trajectory data and (ii) the FAISS indexes used for search. For the first
step, run the snippet below, which builds the memory map and the mapping from
`clip_id` to its row range in it.

```python
import json
import numpy as np
from pathlib import Path
from safetensors.numpy import load_file, safe_open
from tqdm import tqdm

clip_to_idx = {}
cnt = 0
path_to_output = "path/to/trajectory_data/trajectory_data.dat"

trajectory_data = "path/to/trajectory_data/"
path_to_trajectory_data = sorted(Path(trajectory_data).rglob("*/*.safetensors"))
print(f"Loading {len(path_to_trajectory_data)} safetensors")

json_path = "path/to/trajectory_data/clip_to_idx.json"
SAVE_EVERY = 10000

# Precompute total number of rows across ALL safetensors.
total_rows = 0
for path_to_data in tqdm(path_to_trajectory_data):
    with safe_open(path_to_data, framework="np") as f:
        for k in f.keys():
            total_rows += f.get_tensor(k).shape[0]

# Create the memmap with the correct final shape.
fp = np.memmap(path_to_output, dtype="float32", mode="w+", shape=(total_rows, 7))

# Iterate all safetensors and write in chunks.
for path_to_data in path_to_trajectory_data:
    print(path_to_data)
    with safe_open(path_to_data, framework="np") as f:
        for k in tqdm(f.keys()):
            if k in clip_to_idx:
                continue

            data = f.get_tensor(k)
            assert data.shape[1] == 7
            if np.isnan(data).any():
                continue

            start = cnt
            end = cnt + data.shape[0]
            fp[start:end, :] = data
            cnt = end

            clip_to_idx[k] = (start, end)

            # periodic JSON save
            if len(clip_to_idx) % SAVE_EVERY == 0:
                with open(json_path, "w") as jf:
                    json.dump(clip_to_idx, jf)
fp.flush()
# final JSON save
with open(json_path, "w") as f:
    json.dump(clip_to_idx, f)
```
Tip: back up `trajectory_data.dat` and `clip_to_idx.json` before updating.

Now, the next step is to update the FAISS indexes used for search. There are
three indexes in total, each covering a different time granularity:

| Index | Spec | `sec` | `M` | Windows per clip |
| :---- | :--- | :---: | :-: | :-------------- |
| Full trajectory | `OPQ121,IVF4096,PQ121x8` | n/a | n/a | 1 (entire clip) |
| 10-second sub-trajectories | `OPQ40,IVF4096,PQ40x8` | 10 | 40 | 3 (0-10 s, 5-15 s, 10-20 s) |
| 5-second sub-trajectories | `OPQ20,IVF4096,PQ20x8` | 5 | 20 | 4 (0-5 s, 5-10 s, 10-15 s, 15-20 s) |

**p0 / p1 versioning.** Index files use a two-version scheme. The `_p0` file
is the trained but not-yet-populated seed index (it is never overwritten).
Incremental update functions read from `_p0` and write the result to `_p1`.
The server always loads `_p1`. Training is only needed once (it is slow for
high-dimensional data); afterwards only `update_*` is needed.

### Updating the full-trajectory index

Reads the trained seed index (`_p0`) and appends any clips not already present.
Clips already in the index are silently skipped.

```python
from sil_wheel.stores.trajectory_store import update_index

DATA = "path/to/trajectory_data/"
update_index(DATA)
# Writes: trajectory_data_OPQ121_IVF4096_PQ121x8_p1.index
#         trajectory_clip_to_index_OPQ121_IVF4096_PQ121x8_p1.pkl
```

### Updating the sub-trajectory indexes (5 s and 10 s)

Each sub-trajectory index has its own `update_subtrajectory_index` call.
Run both after adding new safetensors files.

```python
from sil_wheel.stores.trajectory_store import update_subtrajectory_index

DATA = "path/to/trajectory_data/"

# 10-second windows (M=40)
update_subtrajectory_index(DATA, sec=10, M=40)
# Reads:  trajectory_data_10s_OPQ40,IVF4096,PQ40x8_p0.index
#         trajectory_clip_to_index_10s_OPQ40,IVF4096,PQ40x8_p0.pkl
# Writes: trajectory_data_10s_OPQ40,IVF4096,PQ40x8_p1.index
#         trajectory_clip_to_index_10s_OPQ40,IVF4096,PQ40x8_p1.pkl

# 5-second windows (M=20)
update_subtrajectory_index(DATA, sec=5, M=20)
# Reads:  trajectory_data_5s_OPQ20,IVF4096,PQ20x8_p0.index
#         trajectory_clip_to_index_5s_OPQ20,IVF4096,PQ20x8_p0.pkl
# Writes: trajectory_data_5s_OPQ20,IVF4096,PQ20x8_p1.index
#         trajectory_clip_to_index_5s_OPQ20,IVF4096,PQ20x8_p1.pkl
```

### Building an index from scratch (rare)

Only needed if the `_p0` seed file does not exist or has been lost. This
performs full training, which can take a significant amount of time.

```python
from sil_wheel.stores.trajectory_store import parse_trajectory_data_from_dir, parse_subtrajectory_data_from_dir

DATA = "path/to/trajectory_data/"

# Full trajectory
parse_trajectory_data_from_dir(DATA)

# Sub-trajectories
parse_subtrajectory_data_from_dir(DATA, sec=10, M=40)
parse_subtrajectory_data_from_dir(DATA, sec=5, M=20)
```

These functions write `_p1` files directly and can be used as a drop-in
replacement if the seed index is unavailable. Training requires at least
1 000 000 vectors to be buffered before it starts.

Tip: keep backups of all `_p0` seed index files. They are never regenerated
automatically and are required as input by the incremental update functions.

## Extract Text-to-Video Embeddings

This section summarizes how to extract text-to-video semantic features with
`scripts/extract_video_text_embeddings.py` and keep the Cosmos embedding index
up to date. It is intentionally brief, since the main README covers more. As
with the other processing scripts, you pass the file that lists the absolute
video paths to process; it can list raw `.mp4` files or `.tar` archives, and it
can be the same list you prepared for [Process Videos](#process-videos).

Key arguments
- `--model_type`: One of `cosmos_embed1_224p`, `cosmos_embed1_336p`, `cosmos_embed1_448p`, `qwen3_vl_embed_2b`, `qwen3_vl_embed_8b`, `pe_core_b16_224p`, `pe_core_l14_336p`, `pe_core_g14_448p` (default `cosmos_embed1_448p`). The `pe_core_*` variants encode each frame with Meta's Perception Encoder Core CLIP and mean-pool across the time dimension; weights are pulled from the Hugging Face Hub on first use and require the `perception_models` package.
- `--batch_size`: Number of clips per model forward pass (default 8). Higher values amortize GPU launch overhead for `cosmos_embed1_*` and `pe_core_*`. The Qwen3-VL-Embed wrappers accept `B>1` but loop one clip at a time internally, so raising `--batch_size` does not improve throughput for those models.
- `--n_frames`: Frames sampled per clip (default 8).
- `--fps`: Target sampling rate in frames per second. Ignored when `--n_frames` is set.
- `--camera`: Restrict processing to a single camera (e.g. `camera_front_wide_120fov`); defaults to all cameras.
- `--bucket` / `--profile` / `--endpoint`: S3 credentials, set together to stream videos from S3 instead of reading local files.
- `--n_processes` / `--process_id`: Shard the work for distributed runs (same pattern as the other extraction scripts).

- Run locally (interactive job)

```bash
python scripts/extract_video_text_embeddings.py \
    path/to/video_paths.txt --model_type cosmos_embed1_448p \
    --output path/to/output_dir/{model}_embeddings/{split}/group_{process_id}_{n_processes}.parquet \
    --process_id 0 --n_processes 5
```

- Run with Slurm Job

```bash
# Submit multiple jobs
# Make sure to modify the OUTPUT_ROOT, VIDEO_LIST and N_PROCS to
# reflect your data configuration and output
../submit_video_text_embed_processing_jobs.sh
```

### Update Text-to-Video FAISS Index

After generating new parquet shards of text-video embeddings, append them to the existing FAISS index. From the repo root:

```python
from pathlib import Path
from sil_wheel.stores.cosmos_embeddings_store import CosmosEmbeddingsStore

# Initialize the store with your embeddings root (loads/creates the FAISS index)
cosmos = CosmosEmbeddingsStore(path_to_cosmos_embed_indexes)

# Point to the root directory that contains your Cosmos parquet shards
path_to_embed = "/abs/path/to/cosmos_embeddings/"
# Option A: Append all parquet files under a directory
for pi in sorted(Path(path_to_embed).glob("**/*.parquet")):
    print(pi)
    cosmos.append_embeddings_parquet(pi)

# Option B:
# cosmos.append_multiple_embeddings_with_path(path_to_embed)
```

## Extract Captions

`scripts/extract_captions.py` generates either one caption per 20-second clip or
per-subclip captions, splitting each clip into fixed windows. As with the other
scripts, you pass the file that lists the video paths to process.

Key arguments
- Positional `path_to_data`: text/JSON file listing absolute video paths (one per line).
- `--model_family`: VLM family, one of `qwen2.5-vl` or `qwen3-vl` (default `qwen2.5-vl`).
- `--model_size`: Model size in billions of parameters. Valid sizes depend on `--model_family`: `qwen2.5-vl` accepts `3`, `7`, `32`, `72`; `qwen3-vl` accepts `2`, `4`, `8`, `30`, `235`.
- `--prompt_factory_type`: Caption prompt template, one of `yotta_prompt_long` (default), `video_caption_dense`, `reason_prompt`, `msrvtt_prompt`.
- `--batch_size`: Batch size for model inference.
- `--clip_duration`: Window size in seconds (e.g., 5 for 4 subclips in a 20s video).
- `--clip_overlap`: Overlap in seconds between consecutive windows.
- `--process_id` / `--n_processes`: shard index and total shard count.
- `--output`: per-shard output parquet; `{process_id}` and `{n_processes}` placeholders are substituted at runtime.

Run locally:

```bash
python scripts/extract_captions.py path/to/video_paths.txt \
  --output path/to/output_dir/group_{process_id}_{n_processes}.parquet \
  --batch_size 20 --clip_duration 5 --clip_overlap 0 \
  --process_id 0 --n_processes 1
```

### Extract Structured Captions via API Calls

Calls an OpenAI-compatible tool-calling API (e.g. NVIDIA inference) to produce
structured captions. No GPU required, since parallelism comes from concurrent
API calls (`--workers`). Two modes:

| Mode | Description |
| :--- | :--- |
| `sil_av_benchmark` | Dense SIL-AV annotation (environments, agents, ego vehicle, traffic lights) |
| `comprehensive_v3` | Full AV perception schema with short/medium/long NL summaries |

Key arguments:
- Positional `path_to_data`: explicit file list (`.json` or `.txt`) of absolute video paths.
- `--caption_mode`: `sil_av_benchmark` or `comprehensive_v3`.
- `--workers`: concurrent API calls (default: 4).
- `--model`: API model identifier (default: `gpt-4o`).
- `--api_key`: falls back to `OPENAI_API_KEY` env var.

```bash
export OPENAI_API_KEY="your-key"
python scripts/extract_structured_captions.py path/to/video_paths.txt \
  --caption_mode sil_av_benchmark \
  --output path/to/output_dir/{caption_mode}/group_{process_id}_{n_processes}.parquet \
  --n_processes 4 --process_id 0 --workers 8
```

### Add Captions to DB
After export, you can load captions into the SQLite FTS5 store for search as follows.

**Qwen captions**:
```python
from pathlib import Path
import pandas as pd
path_to_captions = "path/to/qwen_captions/"
parquet_files = sorted(Path(path_to_captions).glob("**/*.parquet"))
# Gather all captions into a single data frame
df = pd.concat([pd.read_parquet(fp) for fp in parquet_files], ignore_index=True)

# Convert the captions into the following format. If your captions are already
# in this format, skip the next steps and load them as is.
#                                       clip_id                                            summary  start_time  end_time
# 0000098d-473e-4def-8e75-d3acd31df0a7  The video captures a scene at a suburban inter...          15        20
# 00000b83-a648-4afc-9872-e1a639555e6f  The video depicts a sunny day on a tree-lined ...           0         5
# 00000b83-a648-4afc-9872-e1a639555e6f  The video depicts a sunny day on a tree-lined ...           5        10
# 00000b83-a648-4afc-9872-e1a639555e6f  The video depicts a sunny day on Roxbury Drive...          10        15
# 00000b83-a648-4afc-9872-e1a639555e6f  The video depicts a sunny day on a tree-lined ...          15        20
out = df.sort_values("clip_id", kind="mergesort").copy()
suffix = out["clip_key"].astype(str).str.strip().str[-1]
import numpy as np

# Suppose suffix is your Series
suffix = out["clip_key"].astype(str).str.strip().str[-1]

# Define a mapping from suffix → (start_time, end_time)
intervals = {
    "0": (0, 5),
    "1": (5, 10),
    "2": (10, 15),
    "3": (15, 20),
    "4": (20, 25),
    "5": (25, 30),
    "6": (30, 35),
    "7": (35, 40),
    "8": (40, 45),
    "9": (45, 50),
}

# Map start_time and end_time using .map
out["start_time"] = suffix.map({k: v[0] for k, v in intervals.items()})
out["end_time"]   = suffix.map({k: v[1] for k, v in intervals.items()})
out.drop(["captions", "clip_key"], axis=1)

# Now we can update the captions
from sil_wheel.stores.sqlite_caption_store import FTSCaptionStore
caption_store = FTSCaptionStore("path/to/captions.db")
caption_store.insert_from_dataframe(out, "Qwen2.5-7B", "YourDataSourceName")
```

**Structured captions**:
```python
from pathlib import Path
import pandas as pd

# Example for sil_av_benchmark; replace column name for comprehensive_v3
path_to_captions = "/path/to/structured_captions/sil_av_benchmark/"
parquet_files = sorted(Path(path_to_captions).glob("**/*.parquet"))
df = pd.concat([pd.read_parquet(fp) for fp in parquet_files], ignore_index=True)

#         clip_id      brief_description                              camera  model                       caption_mode
# abc123  The ego vehicle drives through a ...    front   gpt-4o   sil_av_benchmark
# def456  Ego approaches a signalized intersection...  front   gpt-4o   sil_av_benchmark
out = df[["clip_id", "brief_description"]].copy()
out = out.rename(columns={"brief_description": "summary"})
out["start_time"] = 0
out["end_time"] = 20  # adjust to match your clip duration

model_name = df["model"].iloc[0]  # e.g. "gpt-4o"
from sil_wheel.stores.sqlite_caption_store import FTSCaptionStore
caption_store = FTSCaptionStore("/path/to/captions.db")
caption_store.insert_from_dataframe(out, model_name, "YourDataSourceName")
```

## Extract Caption Embeddings

`scripts/extract_captions_embeddings.py` encodes each caption with a
SentenceTransformer text model (default Qwen3-Embedding-8B) and writes
the per-caption embeddings to a parquet file. Like the other extraction
scripts, it shards work via `--process_id` / `--n_processes` so the job
can be split across a Slurm array.

The captions source can be either the SQLite captions DB used by the
server, or a parquet file produced by
`scripts/extract_captions.py` (one row per sub-clip; the
`summary` column is embedded). The script branches on the input
file extension:

```bash
# From the captions DB
python scripts/extract_captions_embeddings.py \
    path/to/captions.db \
    --path_to_all_clips path/to/all_clip_ids.txt \
    --process_id 0 --n_processes 128 \
    --output path/to/caption_embeddings/group_{process_id}_{n_processes}.parquet

# From a Qwen-captions parquet
python scripts/extract_captions_embeddings.py \
    path/to/qwen_captions/group_0_128.parquet \
    --path_to_all_clips path/to/all_clip_ids.txt \
    --process_id 0 --n_processes 128 \
    --output path/to/caption_embeddings/group_{process_id}_{n_processes}.parquet
```

Key arguments:
- Positional `captions_source`: SQLite caption DB (`*.db` / `*.sqlite`) or a parquet produced by `extract_captions.py` (`*.parquet`).
- `--embedding_model`: SentenceTransformer model name; one of `Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Embedding-4B`, or `Qwen/Qwen3-Embedding-8B` (default).
- `--path_to_all_clips`: text file with one clip id per line; the script processes the slice `[process_id::n_processes]`.
- `--process_id` / `--n_processes`: shard index and total shard count.
- `--batch_size`: captions per model forward (default 64).
- `--output`: per-shard output parquet; the literal `{process_id}` and `{n_processes}` placeholders are substituted at runtime.

### Update Caption Embedding Index

After generating the new parquet shards for caption embeddings, append them to the existing FAISS index. From the repo root:
```python
from sil_wheel.stores.caption_embeddings_store import CaptionEmbeddingsStore
embedding_store = CaptionEmbeddingsStore(
    path_to_caption_embeddings,
    index_spec="IVF4096,PQ128x8",
)

from pathlib import Path
parquet_paths = sorted(
    Path(path_to_caption_embeddings).glob("**/group_*.parquet")
)
for p in parquet_paths:
    embedding_store.append_embeddings_parquet(p)
```

## Extract Visual (Florence2 SigCLIP) Embeddings

This section describes how to extract frame-level visual embeddings using the
`extract_florence2_sigclip_embeddings.py` script. The script samples `--n_frames`
frames per clip, encodes them with a Florence2 SigCLIP model, and writes the
results to a pickle file. Like the other processing scripts, it uses a shard
index (`--process_id` / `--n_processes`) so the work can be split across many
parallel jobs against the same video-paths file.

```bash
python scripts/extract_florence2_sigclip_embeddings.py \
    path/to/video_paths.txt \
    --process_id 100 \
    --n_processes 3000 \
    --output path/to/visual_embeddings/florence2_sigclip_group_{process_id}_{n_processes}.pkl \
    --n_frames 16
```

Key arguments:
- Positional: path to a text file listing absolute video paths to process (one per line).
- `--process_id`: zero-based shard index for this job.
- `--n_processes`: total number of shards; each job processes `1/n_processes` of the file.
- `--output`: output path for the pickle file. The literal strings `{process_id}` and `{n_processes}` in the path are replaced at runtime.
- `--n_frames`: number of frames sampled uniformly from each clip for embedding.

### Update Visual FAISS Index

After generating the new pkl shards for visual embeddings, append them to the existing FAISS index. From the repo root:
```python
from sil_wheel.stores.visual_embeddings_store import Florence2SigCLIPEmbeddingStore
embedding_store = Florence2SigCLIPEmbeddingStore(path_to_visual_embeddings)

from pathlib import Path
pkl_paths = sorted(Path(path_to_visual_embeddings).glob("**/florence2_sigclip_group_*.pkl"))
embedding_store.append_pkl(pkl_paths)
```


## Statistics Artifacts

Two offline scripts compute per-dataset statistics for the `/data_stats` view
and related analyses.

### Trajectory Statistics

Script: `scripts/analyze_trajectory_stats.py`

Purpose
- Consume the trajectory memory map and reverse index to produce per-dataset artifacts (PNGs + JSON) used by the `/data_stats` page.

How to run

```
python scripts/analyze_trajectory_stats.py \
  --data_database path/to/annotations.db \
  --trajectory_data path/to/trajectory_data_dir \
  --output_dir path/to/TRAJECTORY_STATS_DIR \
  --samples 100000 --seed 42
```

Inputs
- `--trajectory_data` directory must contain the memory map and index files:
  - `trajectory_data.dat`: float32 memmap with columns `[x,y,z,speed,acceleration,jerk,curvature]`.
  - `clip_to_idx.json`: mapping `clip_id -> [start, end]` rows in the memmap.
- `--data_database`: SQLite annotations DB to group clips by `data_source`.

Prerequisite: build the memmap and reverse index
- See the detailed walk-through in [Process Ego-trajectories](#process-ego-trajectories) for creating `trajectory_data.dat` and `clip_to_idx.json`.

Outputs
- Writes one set of artifacts per dataset into `--output_dir` using the dataset slug in the filenames:

The server endpoint `/data_stats_list` discovers datasets by looking for these four files under `TRAJECTORY_STATS_DIR` and shows them in `/data_stats`.

### Dataset Statistics

Script: `scripts/analyze_data_stats.py`

Purpose
- Compute per-dataset statistics directly from the SQLite databases (annotations and captions). Complements trajectory statistics with label and caption analytics.

What it computes
- Clips per dataset and annotation coverage via `clips.has_manual_annotations` / `clips.has_autolabels`.
- Annotations per clip: total, manual-only, autolabel-only distributions (mean/std/min/max/median).
- Timed annotation durations: distribution and percentiles (p10/p25/p50/p75/p90/p95).
- Captions (optional, with `--captions-db`): captions per clip, words per caption, caption durations, and percentiles.
- Overlap metric: count and ratio of timed annotations that overlap at least one caption interval.
- Artifacts: barplot of annotation counts per label key (PNG preferred or SVG fallback).

Defaults
- `--db` and `--captions-db` have hardcoded developer-machine defaults; in
  practice always pass them explicitly.

How to run

```
python scripts/analyze_data_stats.py \
  --output_dir path/to/out \
  --datasource YourDataSourceName   # repeat flag to analyze multiple datasets
```

If `--datasource` is omitted, the script enumerates all datasets found in `clips.data_source` and processes each one.

Outputs
- JSON per dataset: `data_stats_summary_<slug>.json` written to `--output_dir` with:
  - `dataset`, `n_clips_sampled`
  - `features` with `{mean,std,min,max,median}` for each distribution
  - `percentiles` where applicable
  - optional `artifacts.labels_barplot` path to the label frequency plot

Performance
- Prints per-dataset elapsed time and total runtime.