"""Forecast accuracy metrics on the original (back-transformed) sales scale.

MASE is the only reported accuracy measure: scale-free, its denominator is the in-sample
seasonal-naive MAE of each series, so it stays comparable across series whose volumes
differ by orders of magnitude. It is computed per series and then aggregated unweighted,
volume-weighted and as a median (the heavy tail makes the aggregation choice
outcome-relevant, so all three are reported).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecasting.config import SEASON


def seasonal_naive_scale(insample: np.ndarray, observed: np.ndarray | None = None,
                         season: int = SEASON) -> float:
    """In-sample seasonal-naive MAE for one series = the MASE denominator.

    Missing positions stay in place so seasonal pairs remain 4 quarters apart, and only
    pairs observed at both ends count — imputed values must not contaminate the scale.
    NaN when too few pairs or a flat series leave it undefined, which drops that series
    from MASE rather than reporting a meaningless ratio.
    """
    insample = np.asarray(insample, dtype=float)
    observed = ~np.isnan(insample) if observed is None else np.asarray(observed, dtype=bool)
    if insample.size <= season:
        return np.nan
    both = observed[season:] & observed[:-season]
    if not both.any():
        return np.nan
    scale = np.abs(insample[season:] - insample[:-season])[both].mean()
    return float(scale) if np.isfinite(scale) and scale > 0 else np.nan


def mase(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> float:
    if not np.isfinite(scale):
        return np.nan
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))) / scale)


def weighted_mean(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Mean over series, skipping NaN; volume-weighted when weights are given."""
    values = np.asarray(values, float)
    ok = np.isfinite(values)
    if not ok.any():
        return np.nan
    if weights is None:
        return float(values[ok].mean())
    # negative net volume (returns exceeding sales) is not a meaningful weight
    weights = np.clip(np.asarray(weights, float)[ok], 0.0, None)
    if weights.sum() <= 0:
        return float(values[ok].mean())
    return float(np.average(values[ok], weights=weights))
