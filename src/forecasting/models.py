"""The forecasting-ladder models with a common fit/predict interface.

The Floor (seasonal-naive) is parameter-free and handled directly in the harness
(its prediction is the seasonal value already carried in the feature matrix). Only
the two embedding-carrying rungs need a fitted estimator: a penalised linear model
(Ridge) and a gradient-boosted tree ensemble (LightGBM). Both are trained on the
same features and differ only in capacity, so the with/without-embedding delta is
comparable across them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RANDOM_SEED


class RidgeModel:
    """Global penalised linear model. Standardisation matters because the Ridge
    penalty is scale-sensitive and the embedding block is on its own scale."""

    def __init__(self, alpha: float = 10.0):
        self.pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeModel":
        self.pipe.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipe.predict(X)


class GBMModel:
    """Global gradient-boosted trees (the SOTA rung); native handling of many
    features, so the high-dimensional embedding block needs no scaling."""

    def __init__(self, **kwargs):
        params = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
                      min_child_samples=50, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
        params.update(kwargs)
        self.model = LGBMRegressor(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBMModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


FITTED_MODELS = {"ridge": RidgeModel, "gbm": GBMModel}
