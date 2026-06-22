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

from typing import Dict, List

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    def __init__(
        self,
        max_iter: int = 100000,
        C: int = 100,
        coefficients: np.array = None,
        intercept: np.array = None,
    ):
        self.model = LogisticRegression(
            n_jobs=-1, tol=1e-3, max_iter=max_iter, C=C
        )
        if coefficients is not None and intercept is not None:
            self.model.coef_ = coefficients
            self.model.intercept_ = intercept

    def forward(self, X):
        return self.model.predict_proba(X)

    def train_model(self, X, y, output_directory=None, scenario_name=None):
        if hasattr(X, "detach"):
            X = X.detach().cpu().numpy()
        if hasattr(y, "detach"):
            y = y.detach().cpu().numpy()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self.model.fit(X, y)
        if output_directory is not None and scenario_name is not None:
            np.savez(
                f"{output_directory}/LR_weights.npz",
                coef=self.model.coef_,
                intercept=self.model.intercept_,
            )

    @classmethod
    def from_weights(cls, path_to_weights: str):
        with np.load(path_to_weights) as weights:
            return cls(
                coefficients=weights["coef"], intercept=weights["intercept"]
            )


def load_annotations(
    annotations: Dict[str, Dict[str, List]],
    scenario_name: str,
    clips: np.array,
    features: np.array,
):
    data = annotations[scenario_name]
    # Make sure that we don't have dublicates in our annotated data
    positive = np.unique(data["positive"])
    if len(data["negative"]) > 0:
        negative = np.unique(data["negative"])
    else:
        negative = np.setdiff1d(clips, positive)
        rind = np.random.choice(np.arange(len(negative)), 100)
        negative = negative[rind]
    mask = np.isin(clips, positive)
    X_pos = features[mask]
    y = [1] * len(X_pos)
    mask = np.isin(clips, negative)
    X_neg = features[mask]
    y.extend([0] * len(X_neg))
    X = np.vstack([X_pos, X_neg])
    return X, y


def load_data(
    path_to_annotations: str,
    scenario_name: str,
    clips: np.array,
    features: np.array,
):
    X, y = load_annotations(path_to_annotations,
                            scenario_name, clips, features)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def bootstrap_classifier(
    X_anno: torch.tensor,
    y_anno: torch.tensor,
    X_test: torch.tensor,
    output_directory: str,
    scenario_name: str,
):
    print(
        f"Training classifier with {torch.sum(y_anno==1).item()} positives "
        f"and {torch.sum(y_anno==0).item()} negatives"
    )
    # First train on the labelled data
    linear_probe = LogisticRegressionModel(C=100)
    linear_probe.train_model(X_anno, y_anno, output_directory, scenario_name)
    return linear_probe
    if X_test is None:
        return linear_probe

    y_pred = linear_probe.forward(X_test)
    # From the predictions keep only the "very positive" samples, namely
    # samples with very high probability
    positive_indices = y_pred[:, 1] > 0.95
    # Likewise for the negative predictions keep only the ones that are "very
    # negative"
    negative_indices = y_pred[:, 0] > 0.98
    hard_negative_indices = y_pred[:, 0] > 0.60
    X_pos = X_test[positive_indices]
    X_neg = X_test[negative_indices]
    X_neg_hard = X_test[hard_negative_indices]
    n_pos = len(X_pos.sum(-1))
    X_neg = X_neg[
        np.random.choice(X_neg.shape[0], int(n_pos * 0.3), replace=False)
    ]
    X_neg_hard = X_neg_hard[
        np.random.choice(X_neg_hard.shape[0], int(n_pos * 0.7), replace=False)
    ]

    X_bts = torch.cat(
        [
            torch.from_numpy(X_pos).float(),
            torch.from_numpy(X_neg).float(),
            torch.from_numpy(X_neg_hard).float(),
        ]
    )
    y_bts = [1] * len(X_pos)
    y_bts.extend([0] * len(X_neg))
    y_bts.extend([0] * len(X_neg_hard))
    y_bts = torch.tensor(np.array(y_bts), dtype=torch.float)

    print(
        f"Training classifier with {torch.sum(y_bts==1).item()} positives "
        f"and {torch.sum(y_bts==0).item()} negatives"
    )

    # With the bootstrapped data fit the model again and save the model
    lrg_model = LogisticRegressionModel(C=100)
    lrg_model.train_model(X_bts, y_bts, output_directory, scenario_name)
    return lrg_model
