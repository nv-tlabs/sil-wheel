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

"""GPU environment smoke test.

Verifies the active conda environment can run the full retrieval/captioning stack
that requires the latest vLLM (0.22.x) for the newest Qwen3-VL captioner, while still
holding numpy<2 / opencv<4.13 so nuscenes-devkit and cosmos-embed work in the same env.
It exercises the three GPU capabilities on synthetic inputs (no nuScenes data needed):

  imports          versions/pins are right, torch+CUDA work, and the cosmos-embed
                   remote code constructs on this transformers (no weight download)
  qwen3vl_caption  Qwen/Qwen3-VL-2B-Instruct via vLLM -> a caption (the
                   scripts/extract_captions.py --model_family qwen3-vl path)
  cosmos_embed     sil_wheel CosmosEmbed1("cosmos_embed1_448p") -> video+text embeddings
  qwen3vl_embed    sil_wheel Qwen3VLEmbed("qwen3_vl_embed_2b") -> video/text/image embeddings

Each GPU check runs in its OWN subprocess so the model + vLLM KV cache are released
before the next loads (all three would not co-reside in 24 GiB).

Run it:
    # as a script (recommended) -- prints a PASS/FAIL summary:
    python tests/test_environment.py
    python tests/test_environment.py --checks imports cosmos_embed
    python tests/test_environment.py --quick            # imports only, no downloads
    # via pytest (gated; needs a GPU + ~15 GB of model downloads):
    RUN_ENV_SMOKE_TEST=1 pytest tests/test_environment.py -s
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path

import pytest

# Resolve sil_wheel to THIS checkout (tests/ -> repo root), independent of any
# editable install the active env may point at.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ALL_CHECKS = ["imports", "qwen3vl_caption", "cosmos_embed", "qwen3vl_embed"]


# --------------------------------------------------------------------------- #
# Synthetic inputs
# --------------------------------------------------------------------------- #
def _synthetic_frames_thwc(t=8, h=224, w=224):
    """t frames, (T, H, W, 3) uint8 -- a moving colored band so it isn't pure noise."""
    import numpy as np

    frames = np.zeros((t, h, w, 3), dtype=np.uint8)
    for i in range(t):
        x0 = int((w - 40) * i / max(t - 1, 1))
        frames[i, :, x0 : x0 + 40, i % 3] = 255
        frames[i, h // 3 : 2 * h // 3, :, (i + 1) % 3] = 96
    return frames


# --------------------------------------------------------------------------- #
# Checks (each returns None on success, raises on failure)
# --------------------------------------------------------------------------- #
def check_imports():
    import importlib.metadata as md

    import numpy as np
    import torch

    def ver(p):
        try:
            return md.version(p)
        except Exception:
            return "MISSING"

    print(f"  python                 {sys.version.split()[0]}")
    for p in ["vllm", "torch", "transformers", "numpy", "opencv-python-headless",
              "numba", "nuscenes-devkit", "sentence-transformers", "qwen-vl-utils"]:
        print(f"  {p:22s} {ver(p)}")

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    print(f"  cuda                   {torch.version.cuda} | {torch.cuda.get_device_name(0)}")

    # numpy<2 for nuscenes-devkit; opencv<4.13 to match; vllm>=0.22 / transformers>=4.57 for Qwen3-VL
    assert int(np.__version__.split(".")[0]) < 2, f"numpy must be <2, got {np.__version__}"
    cv = ver("opencv-python-headless")
    assert cv != "MISSING" and tuple(map(int, cv.split(".")[:2])) < (4, 13), f"opencv must be <4.13, got {cv}"
    assert tuple(map(int, ver("vllm").split(".")[:2])) >= (0, 22), f"vllm must be >=0.22, got {ver('vllm')}"
    assert tuple(map(int, ver("transformers").split(".")[:2])) >= (4, 57), f"transformers must be >=4.57, got {ver('transformers')}"

    # everything else must import
    import cv2  # noqa: F401
    import faiss  # noqa: F401
    import nuscenes  # noqa: F401
    from sil_wheel.embeddings.cosmos_embed1 import CosmosEmbed1  # noqa: F401
    from sil_wheel.embeddings.qwen3_vl_embed import Qwen3VLEmbed  # noqa: F401

    # cosmos remote code must at least construct on this transformers (no weight download)
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.from_pretrained("nvidia/Cosmos-Embed1-448p", trust_remote_code=True)
    AutoModel.from_config(cfg, trust_remote_code=True)
    print("  cosmos-embed1 remote code constructs on this transformers: OK")


def check_qwen3vl_caption():
    """Qwen3-VL-2B-Instruct via vLLM, mirroring the scripts/extract_captions.py request shape."""
    # flashinfer's sampler JIT-compiles against the CUDA-13 CCCL and fails on hosts whose
    # system nvcc is older (e.g. CUDA 12.0); use vLLM's native top-k/top-p sampler instead.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    import numpy as np
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    print(f"  loading {model_id} via vLLM ...")
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 10, "video": 10},
    )
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    frames = _synthetic_frames_thwc(t=8)
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": [Image.fromarray(f) for f in frames], "fps": 2.0},
            {"type": "text", "text": "Briefly describe what happens in this video."},
        ],
    }]
    text = processor.apply_chat_template([messages[0]], add_generation_prompt=True, tokenize=False)
    _, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    vid = np.asarray(videos[0])
    n = int(vid.shape[0])
    per_kwargs = {k: (v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v)
                  for k, v in (video_kwargs or {}).items()}
    per_kwargs.setdefault("fps", 2.0)
    per_kwargs.setdefault("do_sample_frames", False)
    metadata = {"fps": float(per_kwargs.get("fps", 2.0)), "total_num_frames": n,
                "duration": n / 2.0, "video_backend": "decord",
                "frames_indices": list(range(n))}
    request = {"prompt": text, "multi_modal_data": {"video": (vid, metadata)},
               "mm_processor_kwargs": per_kwargs}

    out = llm.generate([request], SamplingParams(temperature=0.1, top_p=0.001, max_tokens=128))
    caption = out[0].outputs[0].text.strip()
    print(f"  caption: {caption[:200]!r}")
    assert caption, "Qwen3-VL produced an empty caption"


def check_cosmos_embed():
    """sil_wheel CosmosEmbed1 video + text embeddings (downloads ~2.5 GB of weights)."""
    import numpy as np
    from sil_wheel.embeddings.cosmos_embed1 import CosmosEmbed1

    print("  loading CosmosEmbed1('cosmos_embed1_448p') ...")
    model = CosmosEmbed1("cosmos_embed1_448p")
    # cosmos preprocessing_embed1.py expects BTCHW uint8 (B, T, C, H, W)
    batch = _synthetic_frames_thwc(t=8).transpose(0, 3, 1, 2)[None]  # (1, T, C, H, W) uint8
    vemb = model.get_video_embeddings(batch)
    temb = model.get_text_embeddings("a car driving on a city street")
    # cosmos returns the text projection as a bf16 cuda tensor; numpy needs float32
    temb_np = temb.detach().cpu().float().numpy() if hasattr(temb, "detach") else np.asarray(temb)
    print(f"  video emb shape {tuple(vemb.shape)} | text emb shape {tuple(temb_np.shape)}")
    assert vemb.ndim == 2 and vemb.shape[0] == 1 and vemb.shape[1] > 0
    assert temb_np.shape[-1] == vemb.shape[1], "cosmos text/video embedding dims differ"


def check_qwen3vl_embed():
    """sil_wheel Qwen3VLEmbed video/text/image embeddings (downloads the 2B embedder)."""
    from sil_wheel.embeddings.qwen3_vl_embed import Qwen3VLEmbed

    print("  loading Qwen3VLEmbed('qwen3_vl_embed_2b') ...")
    model = Qwen3VLEmbed(model_type="qwen3_vl_embed_2b")
    vid = _synthetic_frames_thwc(t=8).transpose(0, 3, 1, 2)[None]  # (1, T, C, H, W) uint8
    vemb = model.get_video_embeddings(vid)
    temb = model.get_text_embeddings("a car driving on a city street")
    iemb = model.get_image_embeddings(_synthetic_frames_thwc(t=1)[0])
    print(f"  video {tuple(vemb.shape)} | text {tuple(temb.shape)} | image {tuple(iemb.shape)}")
    assert vemb.shape[-1] == temb.shape[-1] == iemb.shape[-1], "embedding dims differ across modalities"
    print(f"  cos(video,text) = {float((vemb[0] * temb[0]).sum()):.4f} (normalized embeddings)")


CHECK_FNS = {
    "imports": check_imports,
    "qwen3vl_caption": check_qwen3vl_caption,
    "cosmos_embed": check_cosmos_embed,
    "qwen3vl_embed": check_qwen3vl_embed,
}


# --------------------------------------------------------------------------- #
# Subprocess isolation helpers
# --------------------------------------------------------------------------- #
def run_one(name: str) -> bool:
    print(f"\n----- CHECK: {name} -----")
    try:
        CHECK_FNS[name]()
        print(f"----- {name}: PASS -----")
        return True
    except Exception as e:
        print(f"----- {name}: FAIL -> {e!r} -----")
        traceback.print_exc()
        return False


def run_in_subprocess(name: str) -> int:
    """Run a single check in a fresh interpreter so its GPU memory is fully released after."""
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.run([sys.executable, __file__, "--_child", name], env=env).returncode


# --------------------------------------------------------------------------- #
# pytest entry points (gated -- need a GPU and big model downloads)
# --------------------------------------------------------------------------- #
_GATE = pytest.mark.skipif(
    not os.environ.get("RUN_ENV_SMOKE_TEST"),
    reason="set RUN_ENV_SMOKE_TEST=1 (in the target env, with a GPU) to run this smoke test",
)


@_GATE
def test_imports():
    assert run_in_subprocess("imports") == 0


@_GATE
def test_qwen3vl_caption():
    assert run_in_subprocess("qwen3vl_caption") == 0


@_GATE
def test_cosmos_embed():
    assert run_in_subprocess("cosmos_embed") == 0


@_GATE
def test_qwen3vl_embed():
    assert run_in_subprocess("qwen3vl_embed") == 0


# --------------------------------------------------------------------------- #
# Script entry point
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checks", nargs="+", choices=ALL_CHECKS, default=ALL_CHECKS,
                    help="Subset of checks to run (default: all).")
    ap.add_argument("--quick", action="store_true", help="Only the imports/versions check (no model downloads).")
    ap.add_argument("--_child", metavar="CHECK", help=argparse.SUPPRESS)  # internal: one check, in-process
    args = ap.parse_args()

    if args._child:
        sys.exit(0 if run_one(args._child) else 1)

    checks = ["imports"] if args.quick else args.checks
    results: dict[str, bool] = {}
    for name in checks:
        # 'imports' is cheap; GPU checks get their own subprocess so VRAM frees between models.
        results[name] = run_one(name) if name == "imports" else (run_in_subprocess(name) == 0)

    print("\n=================== SUMMARY ===================")
    for name in checks:
        print(f"  {name:18s} {'PASS' if results[name] else 'FAIL'}")
    ok = all(results.values())
    print(f"  {'ALL PASSED' if ok else 'SOME CHECKS FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
