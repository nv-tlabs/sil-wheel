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

import json
import math
import pickle
from pathlib import Path
import os
import pandas as pd

import numpy as np
from sil_wheel.stores.search_utils import project_dict
from sil_wheel.stores.utils import LRUDict


def _to_metric_value(v):
    if isinstance(v, str):
        return v
    val = v.item() if hasattr(v, "item") else v
    return "NaN" if isinstance(val, float) and math.isnan(val) else val


class ModelsWithMetricsDataStore:
    def __init__(self, path_to_predictions, index_dir: str | None = None):
        # TODO: Future optimization:
        # We need to create a mapping with clip_ids and index in metrics array per model

        self.searches = {}
        # Maps models to metrics
        self.metrics = {}
        # Maps clip_ids to row index within metrics for each model
        self.clip_id_to_index = {}
        # Maps metric_name to column index within metrics for each model
        self.metric_name_to_index = {}

        self.common_clip_ids = {}

        # Search cache per model
        path_to_predictions = Path(path_to_predictions)
        self.path_to_predictions = path_to_predictions

        self.clips_with_metrics = None
        index_path = Path(index_dir) / "clips_with_metrics_set.pkl" if index_dir else None

        # TODO: we use this to sync metrics origin timestamps to video timestamps
        # in future we should change all metrics to be relative to video start time
        with open(path_to_predictions / "video_timestamp_origin_physical_ai_90kclips_overlap_alpamayo_v2.json", "r") as f:
            self.video_origin_timestamp_map = json.load(f)

        if path_to_predictions.exists():
            model_dirs = [
                p for p in sorted(path_to_predictions.iterdir()) if p.is_dir()
            ]
            model_names = [mi.name for mi in model_dirs]

            for model_name in model_names:

                metric_array_path = (
                    path_to_predictions / f"{model_name}_metrics.npy"
                )
                metric_index_path = (
                    path_to_predictions
                    / f"{model_name}_metric_name_to_index.pkl"
                )
                clip_to_index_path = (
                    path_to_predictions / f"{model_name}_clip_to_index.pkl"
                )

                metric_array = np.load(metric_array_path, allow_pickle=True)
                with open(clip_to_index_path, "rb") as file:
                    clip_to_index = pickle.load(file)

                with open(metric_index_path, "rb") as file:
                    metric_name_to_index = pickle.load(file)

                self.metrics[model_name] = metric_array
                self.clip_id_to_index[model_name] = clip_to_index
                self.metric_name_to_index[model_name] = metric_name_to_index

                self.searches[model_name] = LRUDict(size=10)

            # Load mapping of model -> leaderboard name if present
            self.model_to_leaderboard = {}
            mapping_path = path_to_predictions / "model_to_leaderboard.json"
            try:
                if mapping_path.exists():
                    with open(mapping_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            # Keep only known models, normalize values to strings
                            self.model_to_leaderboard = {
                                m: str(name)
                                for m, name in data.items()
                                if m in self.metrics
                            }
            except Exception:
                models_by_metricset = self.models_by_metricset
                for leaderboard, models in models_by_metricset:
                    for model in models:
                        self.model_to_leaderboard[model] = leaderboard

            # Compute common clip_ids across all models per leaderboard
            for leaderboard, models in self.models_by_leaderboard.items():
                clip_set = []
                for model in models:
                    clip_set.append(set(self.clip_id_to_index[model].keys()))
                common_clips = set.intersection(*clip_set)
                self.common_clip_ids[leaderboard] = common_clips

                print(
                    f"Leaderboard {leaderboard} has {len(self.common_clip_ids[leaderboard])} clips"
                )

        if index_path and index_path.exists():
            try:
                with open(index_path, "rb") as f:
                    self.clips_with_metrics = set(pickle.load(f))
                # intersect with videos that have video_origin_timestamp
                self.clips_with_metrics = self.clips_with_metrics & set(self.video_origin_timestamp_map.keys())
            except Exception as e:
                print(f"Warning: failed to load {index_path}: {e}")
        elif index_path:
            print(f"Warning: metrics index not found at {index_path}. Run scripts/index_available_bev_metrics_files.py to enable With Metrics filter.")


    @property
    def models(self):
        return sorted(self.metrics.keys())

    @property
    def leaderboards(self):
        return sorted(self.models_by_leaderboard.keys())

    @property
    def models_by_leaderboard(self):
        if not self.model_to_leaderboard:
            return self.models_by_metricset

        groups = {}
        for key, value in self.model_to_leaderboard.items():
            groups.setdefault(value, []).append(key)
        return groups

    @property
    def models_by_metricset(self):
        groups = {}
        for model, name_to_idx in self.metric_name_to_index.items():
            key = "|".join(sorted(name_to_idx.keys()))
            groups.setdefault(key, []).append(model)
        return groups

    def search(self, filters, current_results):
        if filters.with_metrics:
            current_results = project_dict(current_results, self.clips_with_metrics)
        return current_results

    def get_metrics_clip_ids(self):
        return self.clips_with_metrics

    def has_metrics_index(self):
        return self.clips_with_metrics is not None

    def get_reduced_metrics(
        self,
        model_name,
        clip_ids,
        hash_key,
        reduction="mean",
        with_same_clips=False,
    ):
        if with_same_clips:
            leaderboard = self.model_to_leaderboard
            clip_ids = list(
                set(clip_ids) & self.common_clip_ids[leaderboard[model_name]]
            )

        if hash_key in self.searches[model_name]:
            return self.searches[model_name][hash_key]

        clip_index_map = self.clip_id_to_index[model_name]
        indices = [
            clip_index_map[clip_id]
            for clip_id in clip_ids
            if clip_id in clip_index_map.keys()
        ]
        if len(indices) == 0:
            return {
                **{
                    metric_name: "N/A"
                    for metric_name in self.metric_name_to_index[model_name]
                },
                "num_clips": 0,
            }

        metric_array = self.metrics[model_name]
        selected_metrics = metric_array[indices]
        if selected_metrics.dtype == object:
            metrics = {}
            for metric_name, idx in self.metric_name_to_index[
                model_name
            ].items():
                try:
                    if reduction == "median":
                        reduced = np.nanmedian(selected_metrics[:, idx])
                    elif reduction == "percentile_90":
                        reduced = np.nanpercentile(selected_metrics[:, idx], 90)
                    elif reduction == "percentile_95":
                        reduced = np.nanpercentile(selected_metrics[:, idx], 95)
                    elif reduction == "max":
                        reduced = np.nanmax(selected_metrics[:, idx])
                    else:
                        reduced = np.nanmean(selected_metrics[:, idx])

                    metrics[metric_name] = _to_metric_value(reduced)
                except TypeError:
                    metrics[metric_name] = "..."
        else:
            if reduction == "median":
                reduced = np.nanmedian(selected_metrics, axis=0)
            elif reduction == "percentile_90":
                reduced = np.nanpercentile(selected_metrics, 90, axis=0)
            elif reduction == "percentile_95":
                reduced = np.nanpercentile(selected_metrics, 95, axis=0)
            elif reduction == "max":
                reduced = np.nanmax(selected_metrics, axis=0)
            else:
                reduced = np.nanmean(selected_metrics, axis=0)

            metrics = {
                metric_name: _to_metric_value(reduced[idx])
                for metric_name, idx in self.metric_name_to_index[
                    model_name
                ].items()
            }
        metrics["num_clips"] = len(indices)

        self.searches[model_name][hash_key] = metrics
        return metrics

    def get_per_clip_metrics(self, model_name, clip_ids, with_same_clips=False):
        # Filter the clip index map to keep only the intersection with the
        # provided clip_ids
        if with_same_clips:
            leaderboard = self.model_to_leaderboard
            clip_ids = list(
                set(clip_ids) & self.common_clip_ids[leaderboard[model_name]]
            )

        clip_index_map = self.clip_id_to_index[model_name]
        valid_clip_indices = {
            clip_id: clip_index_map[clip_id]
            for clip_id in clip_ids
            if clip_id in clip_index_map
        }

        metric_names, metric_indices = zip(
            *self.metric_name_to_index[model_name].items()
        )

        if not valid_clip_indices:
            return {
                "metrics": metric_names,
                "values": [],
                "clips": [],
            }

        metric_array = self.metrics[model_name]
        clip_indices = np.array(list(valid_clip_indices.values()))
        metric_indices = np.array(metric_indices)

        raw_values = metric_array[
            clip_indices[:, None], metric_indices[None]
        ].tolist()
        return {
            "metrics": metric_names,
            "values": [
                [_to_metric_value(v) for v in row] for row in raw_values
            ],
            "clips": list(valid_clip_indices.keys()),
        }
    
    def get_full_clip_metrics(self, model_name, clip_id):
        # HACK: get metrics without any reduction
        assert model_name == "ground_truth"
        full_metrics_path = os.path.join(
            self.path_to_predictions, 
            model_name, "eval", f"{clip_id}.parquet")
        
        try:
            df = pd.read_parquet(full_metrics_path)
        except FileNotFoundError:
            print(f"Metrics file not found for clip {clip_id}")
            return {
                "error": f"Metrics file not found for clip {clip_id}",
                "timestamps": [],
                "metrics": {}
            }

        # convert to dictionary to return as json
        # also convert abs microsecond timestamps to video-relative seconds
        # multi-index on clip_id, start_time, traj_idx
        timestamps = df.index.get_level_values("start_time")
        # HACK for now assume the first timestamp is the start of the video
        # video_rel_timestamp_sec = (timestamps - timestamps[0]) / 1e6
        origin_timestamp = df["origin_time"].iloc[0]
        video_orgin_timestamp = self.video_origin_timestamp_map.get(
            clip_id, origin_timestamp
        )
        video_offset_seconds = (video_orgin_timestamp - origin_timestamp) / 1e6
        video_rel_timestamp_sec = timestamps - video_offset_seconds
        
        # Get unique timestamps (in case there are multiple trajectory indices)
        unique_timestamps = np.unique(video_rel_timestamp_sec)
        
        # For each metric, aggregate across trajectory indices if needed
        metrics_data = {}
        for col in df.columns:
            # Only process numeric columns
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            # Group by timestamp and take mean across trajectory indices
            values = []
            for ts in unique_timestamps:
                mask = video_rel_timestamp_sec == ts
                col_values = df.loc[mask, col].values
                # Handle potential NaN or non-numeric values
                try:
                    mean_val = np.nanmean(col_values)
                    if np.isnan(mean_val):
                        values.append(None)
                    else:
                        values.append(float(mean_val))
                except (TypeError, ValueError):
                    values.append(None)
            
            # Only add metric if it has at least some valid values
            if any(v is not None for v in values):
                metrics_data[col] = values
        
        return {
            "timestamps": unique_timestamps.tolist(),
            "metrics": metrics_data,
            "clip_id": clip_id
        }



