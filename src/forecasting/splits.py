"""Rolling-origin fold boundaries and the leave-product-out holdout.

Temporal folds are expanding-window: each fold trains on every quarter up to and
including its cutoff and tests on the next HORIZON quarters. Leave-product-out
additionally removes whole products from the training rows: the model learns
nothing product-specific about them, so the embedding is the only product-level
signal it has for these series. Their test features still carry each series' own
history, so this measures generalisation to products unseen in training, not
forecasting a launch without history.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RANDOM_SEED
from forecasting.config import FOLD_ORIGINS, HORIZON


def fold_cutoffs() -> list[pd.Period]:
    """Last training quarter of each expanding-window fold."""
    return [pd.Period(origin, freq="Q") for origin in FOLD_ORIGINS]


def test_window(cutoff: pd.Period, horizon: int = HORIZON) -> list[pd.Period]:
    """The HORIZON quarters forecast from a fold cutoff."""
    return [cutoff + step for step in range(1, horizon + 1)]


def holdout_products(products: np.ndarray, frac: float = 0.2, seed: int = RANDOM_SEED) -> set[str]:
    """Random subset of products held out of training for the unseen-product test."""
    unique = np.array(sorted(set(products)))
    rng = np.random.default_rng(seed)
    n = int(round(frac * len(unique)))
    return set(rng.choice(unique, size=n, replace=False)) if n else set()
