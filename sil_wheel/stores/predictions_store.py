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

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sil_wheel.stores.utils import LRUDict


class PredictionsDataStore:
    def __init__(self, path_to_predictions):
        path_to_predictions = Path(path_to_predictions)
        self.clip_to_prediction_paths = defaultdict(dict)

        if path_to_predictions.exists():
            model_dirs = [
                p for p in sorted(path_to_predictions.iterdir()) if p.is_dir()
            ]
            model_names = [mi.name for mi in model_dirs]

            for model in model_names:
                # We assume that within the predictions_viz folder we have 1
                # npz file per clip with its corresponding predictions, which
                # we want to visualize
                path_to_predictions_for_viz = (
                    path_to_predictions / model / "predictions_viz"
                )
                path_to_clip_ids = sorted(
                    path_to_predictions_for_viz.rglob("*.npz")
                )
                for path_to_clip in path_to_clip_ids:
                    clip_id = sys.intern(path_to_clip.stem)
                    self.clip_to_prediction_paths[clip_id].update(
                        {model: path_to_clip}
                    )
        self.cache = LRUDict(size=10)

    def get(self, clip_id):
        if clip_id in self.cache:
            return self.cache[clip_id]

        if clip_id not in self.clip_to_prediction_paths:
            return None

        paths_to_preds = self.clip_to_prediction_paths[clip_id]
        preds = {}
        for model, path in paths_to_preds.items():
            data = np.load(path)
            preds[model] = {
                "pred_txyz": data.get("predicted_txyz", None),
                "gt_txyz": data.get("gt_txyz", None),
                "pred_captions": data.get("predicted_captions", None),
            }
        self.cache[clip_id] = preds

        return self.cache[clip_id]

    def get_models(self, clip_id):
        data = self.get(clip_id)
        if data is None:
            return []
        return sorted(data.keys())

    def get_pred_positions(self, clip_id, model_name):
        data = self.get(clip_id)
        if data is None:
            return []

        ddd = data[model_name]["pred_txyz"]
        if ddd is None:
            return []
        # Number of sliding windows x number of sampled trajectoriex x number
        # of waypoints x txy
        return ddd[:, :, :, :3].tolist()

    def get_gt_positions(self, clip_id, model_name):
        data = self.get(clip_id)
        if data is None:
            return []

        ddd = data[model_name]["gt_txyz"]
        if ddd is None:
            return []
        # Number of frames x txy
        return ddd[:, :3].tolist()[::10]

    def get_pred_captions(self, clip_id, model_name):
        data = self.get(clip_id)
        if data is None:
            return []

        ddd = data[model_name]["pred_captions"]
        if ddd is None:
            return []
        return ddd.tolist()
