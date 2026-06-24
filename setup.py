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

"""Setup for sil_wheel."""

from itertools import dropwhile
from os import path

from setuptools import find_packages, setup


def collect_docstring(lines):
    """Return document docstring if it exists"""
    lines = dropwhile(lambda x: not x.startswith('"""'), lines)
    doc = ""
    for line in lines:
        doc += line
        if doc.endswith('"""\n'):
            break

    return doc[3:-4].replace("\r", "").replace("\n", " ")


def collect_metadata():
    meta = {}
    with open(path.join("sil_wheel", "__init__.py")) as f:
        lines = iter(f)
        meta["description"] = collect_docstring(lines)
        for line in lines:
            if line.startswith("__"):
                key, value = map(lambda x: x.strip(), line.split("="))
                meta[key[2:-2]] = value[1:-1]

    return meta


def get_install_requirements():
    return [
        "boto3",
        "decord",
        "einops",
        "httplib2",
        "mediapy",
        "numpy",
        "ninja",
        "fvcore",
        "google-api-python-client",
        "imageio",
        "librosa",
        "matplotlib",
        "moviepy",
        "openai-clip",
        "Pillow",
        "packaging",
        "pandas",
        "peft",
        "av",
        "openai",
        "opencv-python-headless",
        "oauth2client",
        "orjson",
        "pyarrow",
        "pycountry",
        "pyquaternion",
        "pysimdjson",
        "rangehttpserver",
        "scipy",
        "sentence-transformers",
        "sentencepiece",
        "termcolor",
        "timm",
        "tqdm",
        "soundfile",
        "umap-learn",
        "webdataset",
        "websockets",
    ]


def get_extras_require():
    return {
        # Reference-based caption-quality metrics (nlg + bertscore). Kept
        # optional so the core package installs without the scoring stack.
        "caption-quality": [
            "pycocoevalcap",
            "nltk",
            "rouge-score",
            "bert-score",
        ],
        # EVQAScore only. Pulls Ultralytics YOLO11 (AGPL-3.0) -- isolated here
        # so it is never a transitive dependency of anything else.
        "evqa": [
            "ultralytics",
        ],
    }


def setup_package():
    with open("README.md") as f:
        long_description = f.read()
    meta = collect_metadata()
    setup(
        name="sil_wheel",
        version=meta["version"],
        description=meta["description"],
        long_description=long_description,
        maintainer=meta["maintainer"],
        maintainer_email=meta["email"],
        url=meta["url"],
        license=meta["license"],
        packages=find_packages(exclude=["docs", "tests", "scripts", "benchmarks"]),
        package_data={
            "sil_wheel": [
                "app/static/html/*.html",
                "app/static/js/*.js",
                "app/static/js/bev/*.js",
                "app/static/css/*.css",
                "app/static/images/*",
            ],
        },
        install_requires=get_install_requirements(),
        extras_require=get_extras_require(),
    )


if __name__ == "__main__":
    setup_package()
