"""Direct multi-horizon feature construction for the global forecasting models.

Every example is an (anchor, horizon) pair with all features taken at the forecast
origin, so nothing past the anchor can leak in. Missing lags carry a binary indicator so
an unobserved quarter is not read as zero sales.

Direct rather than recursive, which avoids compounding a one-step model's error across
the horizon. The embedding block is appended per config by the harness, so these lag
features are built once per fold rather than once per configuration.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecasting.config import HORIZON, SEASON
from forecasting.metrics import seasonal_naive_scale


@dataclass
class FoldMatrix:
    X: np.ndarray  # base features (no embedding)
    y: np.ndarray  # signed-log target (NaN for unobserved test rows)
    series_id: np.ndarray
    y_seasonal: np.ndarray  # signed-log value one season before target = the Floor prediction
    feature_names: list[str]
    period: np.ndarray | None = None
    is_imputed: np.ndarray | None = None
    y_true_units: np.ndarray | None = None


def _onehot(values: np.ndarray, vocab: list) -> np.ndarray:
    index = {v: i for i, v in enumerate(vocab)}
    out = np.zeros((len(values), len(vocab)), dtype=np.float32)
    out[np.arange(len(values)), [index[v] for v in values]] = 1.0
    return out


def build_fold(df: pd.DataFrame, cutoff: pd.Period, horizon: int = HORIZON) -> tuple[FoldMatrix, FoldMatrix, dict[str, float]]:
    """Build the train and test matrices for one fold, plus the per-series MASE scale."""
    frame = df.copy()
    frame["period"] = frame["quarter_start"].dt.to_period("Q")
    Y = frame.pivot(index="series_id", columns="period", values="y").sort_index(axis=1)
    U = frame.pivot(index="series_id", columns="period", values="sales_units").reindex(columns=Y.columns)
    I = frame.pivot(index="series_id", columns="period", values="is_imputed").reindex(columns=Y.columns).fillna(True)

    periods = list(Y.columns)
    cpos = periods.index(cutoff)
    sids = Y.index.to_numpy()
    y_obs = Y.to_numpy(dtype=float)  # NaN marks unobserved
    y_fill = np.nan_to_num(y_obs, nan=0.0)  # lag features: unknown -> 0
    imp = I.to_numpy(dtype=bool)
    units = U.to_numpy(dtype=float)

    markets = np.array([s.split(" | ", 1)[1] for s in sids])
    market_vocab = sorted(set(markets))
    quarter_vocab = [1, 2, 3, 4]
    horizon_vocab = list(range(1, horizon + 1))

    def collect(pairs: list[tuple[int, int]], train: bool) -> dict:
        """Build feature rows for the given (target position, horizon) pairs. The
        anchor is always target - horizon, so features never reach past the anchor."""
        cols: dict[str, list] = {k: [] for k in
                                 ("y_last", "y_seasonal", "y_last_missing", "y_seasonal_missing",
                                  "horizon", "quarter", "market", "series_id",
                                  "period", "is_imputed", "y_units", "y")}
        for tp, h in pairs:
            ap = tp - h
            if ap < 0:
                continue
            target = y_obs[:, tp]
            valid = (~np.isnan(target) & ~imp[:, tp]) if train else np.ones(len(sids), bool)
            if not valid.any():
                continue
            idx = np.nonzero(valid)[0]
            sp = tp - SEASON
            # A lag is "missing" if the cell is outside the series span (NaN) or an
            # imputed carry, not just a raw NaN -- an imputed lag is a carried estimate,
            # not a real observation, and the model must be able to tell them apart.
            ap_missing = np.isnan(y_obs[idx, ap]) | imp[idx, ap]
            cols["y_last"].append(y_fill[idx, ap])
            cols["y_seasonal"].append(y_fill[idx, sp] if sp >= 0 else np.zeros(len(idx)))
            cols["y_last_missing"].append(ap_missing.astype(np.float32))
            cols["y_seasonal_missing"].append(
                (np.isnan(y_obs[idx, sp]) | imp[idx, sp]).astype(np.float32)
                if sp >= 0 else np.ones(len(idx), dtype=np.float32))
            cols["horizon"].append(np.full(len(idx), h))
            cols["quarter"].append(np.full(len(idx), periods[tp].quarter))
            cols["market"].append(markets[idx])
            cols["series_id"].append(sids[idx])
            cols["period"].append(np.full(len(idx), str(periods[tp])))
            cols["is_imputed"].append(imp[idx, tp])
            cols["y_units"].append(units[idx, tp])
            cols["y"].append(y_obs[idx, tp])
        return {k: (np.concatenate(v) if v else np.array([])) for k, v in cols.items()}

    def to_matrix(cols: dict, train: bool) -> FoldMatrix:
        numeric = np.column_stack([cols["y_last"], cols["y_seasonal"],
                                   cols["y_last_missing"], cols["y_seasonal_missing"]]).astype(np.float32)
        blocks = [numeric,
                  _onehot(cols["horizon"], horizon_vocab),
                  _onehot(cols["quarter"], quarter_vocab),
                  _onehot(cols["market"], market_vocab)]
        names = (["y_last", "y_seasonal", "y_last_missing", "y_seasonal_missing"]
                 + [f"h_{h}" for h in horizon_vocab]
                 + [f"q_{q}" for q in quarter_vocab]
                 + [f"mkt_{m}" for m in market_vocab])
        return FoldMatrix(
            X=np.hstack(blocks), y=cols["y"].astype(float), series_id=cols["series_id"],
            y_seasonal=cols["y_seasonal"].astype(float), feature_names=names,
            period=None if train else cols["period"],
            is_imputed=None if train else cols["is_imputed"].astype(bool),
            y_true_units=None if train else cols["y_units"].astype(float),
        )

    # Train: every (target, horizon) pair up to the cutoff -> many anchors per series.
    # Test: one row per target, anchored exactly at the cutoff (anchor = cpos + h - h).
    train_pairs = [(tp, h) for tp in range(1, cpos + 1) for h in horizon_vocab if tp - h >= 0]
    test_pairs = [(cpos + h, h) for h in horizon_vocab if cpos + h < len(periods)]
    train = to_matrix(collect(train_pairs, train=True), train=True)
    test = to_matrix(collect(test_pairs, train=False), train=False)

    # Scale/volume/history from observed values only (imputed cells excluded), but on
    # the time-aligned series so seasonal differences stay 4 quarters apart.
    scales, volumes, histories = {}, {}, {}
    for i, sid in enumerate(sids):
        vals = units[i, : cpos + 1]
        observed = (~imp[i, : cpos + 1]) & ~np.isnan(vals)
        scales[sid] = seasonal_naive_scale(vals, observed)
        volumes[sid] = float(vals[observed].sum())
        histories[sid] = int(observed.sum())
    return train, test, scales, volumes, histories
