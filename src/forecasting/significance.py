"""Paired cluster bootstrap for the forecasting comparisons.

Two complementary, non-interchangeable questions: bootstrap_deltas asks whether an
embedding helps a given learner (baseline = the same learner without one), while
bootstrap_vs_floor asks whether that learner beats the seasonal-naive Floor at all. A
configuration can win the ablation and still lose to the Floor.

Both run unweighted and volume-weighted, because sales are heavy-tailed: a model can help
across many small series while losing the few large ones, which flips the practical
verdict. That has to be tested, not eyeballed.

The resampling unit is the whole series, not the row — a series recurs across folds, so
its rows are dependent. Holm-adjusted p-values accompany the raw CI verdict so neither
choice hides a result.

Reproducibility caveat: each entry point seeds ONE generator and advances it through its
comparisons in order, so an interval depends on how many comparisons ran before it. Point
estimates and p-values are unaffected, but changing the config set, the model set or the
enumeration order shifts every CI. That is why the shipped forecast_config_pair_tests.csv
no longer reproduces exactly — see the reproduction table in README.md.
"""

from __future__ import annotations

import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RANDOM_SEED


def _cluster_bootstrap(num: np.ndarray, den: np.ndarray, n_boot: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Bootstrap the pooled ratio sum(num)/sum(den), resampling series clusters.

    Pre-summarising each series to (num, den) makes a resample a cheap ratio instead of a
    regroup over all rows.
    """
    s = len(num)
    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, s, s)
        out[b] = num[idx].sum() / den[idx].sum()
    return out


def _holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    n = len(p)
    adjusted = np.empty(n)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (n - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def _compare(arm: pd.DataFrame, base: pd.DataFrame, metric: str, weight: str | None,
             n_boot: int, rng: np.random.Generator, alpha: float) -> dict | None:
    """One paired comparison; delta < 0 means the arm has the lower error."""
    merged = arm.merge(base.rename(columns={metric: "_base"}), on=["series_id", "fold"], how="inner")
    merged["_delta"] = merged[metric] - merged["_base"]
    merged = merged.dropna(subset=["_delta"])
    if merged.empty:
        return None

    if weight is None:
        num, den = merged["_delta"], pd.Series(1.0, index=merged.index)
    else:
        w = merged[weight].clip(lower=0.0)  # negative net volume is not a meaningful weight
        num, den = w * merged["_delta"], w
    per_series = merged.assign(_num=num, _den=den).groupby("series_id")[["_num", "_den"]].sum()
    num_s = per_series["_num"].to_numpy()
    den_s = per_series["_den"].to_numpy()
    if den_s.sum() <= 0:
        return None

    boot = _cluster_bootstrap(num_s, den_s, n_boot, rng)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2.0 * min((boot >= 0).mean(), (boot <= 0).mean())
    return dict(delta=num_s.sum() / den_s.sum(), ci_low=float(lo), ci_high=float(hi),
                p_value=float(min(p, 1.0)), n_series=int(len(per_series)), n_pairs=int(len(merged)))


def _finalize(rows: list[dict], delta_name: str) -> pd.DataFrame:
    """Attach Holm-adjusted p-values and both significance verdicts."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={"delta": delta_name})
    df["p_holm"] = _holm(df["p_value"].to_numpy())
    df["significant"] = (df["ci_low"] > 0) | (df["ci_high"] < 0)  # uncorrected, CI-based
    df["significant_holm"] = df["significant"] & (df["p_holm"] < 0.05)
    cols = ["model", "config", delta_name, "ci_low", "ci_high", "p_value", "p_holm",
            "significant", "significant_holm", "n_series", "n_pairs"]
    return df[cols].sort_values(["model", delta_name]).reset_index(drop=True)


def bootstrap_deltas(records: pd.DataFrame, baseline: str = "none", metric: str = "mase",
                     weight: str | None = None, n_boot: int = 2000,
                     seed: int = RANDOM_SEED, alpha: float = 0.05) -> pd.DataFrame:
    """Ablation against the no-embedding control of the same learner."""
    rng = np.random.default_rng(seed)
    keep = ["series_id", "fold", metric] + ([weight] if weight else [])
    rows: list[dict] = []
    fitted = records[records["model"] != "floor"]
    for model, mdf in fitted.groupby("model"):
        base = mdf[mdf["config"] == baseline][["series_id", "fold", metric]]
        for config, cdf in mdf.groupby("config"):
            if config == baseline:
                continue
            out = _compare(cdf[keep], base, metric, weight, n_boot, rng, alpha)
            if out:
                rows.append(dict(model=model, config=config, **out))
    return _finalize(rows, "delta_mase")


def bootstrap_deltas_by_unseen(records: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Ablation within each half of the leave-product-out split.

    The pooled test cannot stand in for this: the seen half carries roughly five times as
    many pairs and dominates it, so an unseen-product verdict must rest on that
    subpopulation. A planned subgroup analysis, not a post-hoc split, so Holm runs within
    each half — that is the family the verdict is read in.
    """
    if "unseen" not in records.columns:
        raise KeyError("records has no 'unseen' column — run with leave_product_out=True")
    return pd.concat(
        [bootstrap_deltas(records[records["unseen"] == flag], **kwargs).assign(unseen=flag)
         for flag in (False, True)],
        ignore_index=True,
    )


def bootstrap_config_pairs(records: pd.DataFrame, pairs: list[tuple[str, str]] | None = None,
                           metric: str = "mase", weight: str | None = None,
                           n_boot: int = 2000, seed: int = RANDOM_SEED,
                           alpha: float = 0.05) -> pd.DataFrame:
    """Direct tests between embedding configurations, which the no-embedding control
    cannot answer: it says whether each helps, not whether they differ from each other.
    """
    rng = np.random.default_rng(seed)
    keep = ["series_id", "fold", metric] + ([weight] if weight else [])
    rows: list[dict] = []
    fitted = records[records["model"] != "floor"]
    for model, mdf in fitted.groupby("model"):
        configs = sorted(c for c in mdf["config"].unique() if c != "none")
        model_pairs = pairs if pairs is not None else list(combinations(configs, 2))
        for config_a, config_b in model_pairs:
            arm = mdf[mdf["config"] == config_a]
            base = mdf[mdf["config"] == config_b]
            if arm.empty or base.empty:
                continue
            out = _compare(arm[keep], base[["series_id", "fold", metric]], metric, weight,
                           n_boot, rng, alpha)
            if out:
                rows.append(dict(model=model, config_a=config_a, config_b=config_b, **out))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={"delta": "delta_mase_a_minus_b"})
    df["p_holm"] = _holm(df["p_value"].to_numpy())
    df["significant"] = (df["ci_low"] > 0) | (df["ci_high"] < 0)
    df["significant_holm"] = df["significant"] & (df["p_holm"] < 0.05)
    cols = ["model", "config_a", "config_b", "delta_mase_a_minus_b", "ci_low", "ci_high",
            "p_value", "p_holm", "significant", "significant_holm", "n_series", "n_pairs"]
    return df[cols].sort_values(["model", "config_a", "config_b"]).reset_index(drop=True)


def bootstrap_vs_floor(records: pd.DataFrame, metric: str = "mase", weight: str | None = None,
                       n_boot: int = 2000, seed: int = RANDOM_SEED,
                       alpha: float = 0.05) -> pd.DataFrame:
    """Benchmark against the seasonal-naive Floor — whether the model earns its complexity."""
    rng = np.random.default_rng(seed)
    keep = ["series_id", "fold", metric] + ([weight] if weight else [])
    floor = records[records["model"] == "floor"][["series_id", "fold", metric]]
    fitted = records[records["model"] != "floor"]
    rows: list[dict] = []
    for (model, config), cdf in fitted.groupby(["model", "config"]):
        out = _compare(cdf[keep], floor, metric, weight, n_boot, rng, alpha)
        if out:
            rows.append(dict(model=model, config=config, **out))
    return _finalize(rows, "delta_mase_vs_floor")
