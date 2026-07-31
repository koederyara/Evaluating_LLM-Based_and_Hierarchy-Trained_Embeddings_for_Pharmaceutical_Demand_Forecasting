"""Load and prepare the modelling cohort for forecasting.

Reads the TEVA product-by-market series, keeps only the single-vector ATC
cohort, attaches each series' resolved ATC code, and applies the signed-log
target transform. Imputed target cells stay flagged so the harness can exclude
them from scoring while still using them as feature inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecasting.config import COHORT, RESOLVED_FILE, SERIES_FILE


def signed_log(x: np.ndarray | pd.Series) -> np.ndarray:
    """Variance-stabilising transform that keeps the sign of negative net sales."""
    return np.sign(x) * np.log1p(np.abs(x))


def inverse_signed_log(z: np.ndarray | pd.Series) -> np.ndarray:
    return np.sign(z) * np.expm1(np.abs(z))


def load_cohort() -> pd.DataFrame:
    """Return the long series table for the single-vector cohort.

    Columns: series_id, product, market, resolved_atc, quarter_start,
    quarter_label, sales_units, is_imputed, y (signed-log target).
    One row per series-quarter, sorted by series then time.
    """
    resolved = pd.read_csv(RESOLVED_FILE, usecols=["product", "cohort", "resolved_atc"])
    keep = resolved[resolved["cohort"] == COHORT][["product", "resolved_atc"]]

    df = pd.read_csv(SERIES_FILE, parse_dates=["quarter_start"])
    df = df.merge(keep, on="product", how="inner")
    df["series_id"] = df["product"].astype(str) + " | " + df["market"].astype(str)
    df["y"] = signed_log(df["sales_units"])
    return df.sort_values(["series_id", "quarter_start"]).reset_index(drop=True)
