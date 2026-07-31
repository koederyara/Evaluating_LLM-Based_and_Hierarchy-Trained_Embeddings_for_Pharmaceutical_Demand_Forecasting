"""
Build the modelling series table (stage 3: value-level preparation).

Reads the format-prepared long table (blanks kept as NaN) and makes the value
decisions that stage 1 (`prepare_sales.py`) deliberately left open:

  - merge the granularity down to the forecasting unit (default product x market)
    by summing sales over the collapsed dimensions (brand, dosage, dosage2);
  - resolve empty quarters position-aware:
      * leading / trailing blanks truncate the series (later start / earlier end),
      * internal gaps are imputed locally, causally by default (LOCF: carry the last
        observed value forward). LOCF never uses a future observation, so a lag built
        from an imputed cell cannot leak information across a forecast cutoff. Linear
        interpolation is available but bidirectional (look-ahead) -> not for forecasting;
  - keep negative values (returns / corrections) as net sales.

Every imputed target carries is_imputed=True so the evaluation can exclude it.
Cohort selection (ATC-mapped products, single vs. multi ATC) is a separate,
downstream step and is not done here.

Run:
    python src/data_prep/sales/build_series.py
    python src/data_prep/sales/build_series.py --brand TEVA --max-gap 3 --long-gap zero

Options:
    --brand X        restrict to one brand (focal-brand cohort)
    --max-gap N      longest internal gap still imputed; longer -> --long-gap
    --long-gap zero|drop   over-long gaps: fill 0 or drop the series

Output (data/prepared_data/sales/modeling/), suffix = keys (+ brand):
    series_<suffix>.csv              one row per series-quarter, resolved (+ is_imputed)
    series_<suffix>_imputation.csv   per-series imputation report
    series_<suffix>_gaps.csv         per-series internal-gap stats
    series_<suffix>_gap_distribution.csv   gap-length distribution
    series_<suffix>_gap_guardrail.csv      max-gap trade-off table (thesis artifact)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import SALES_MODELING_DIR, SALES_PREPARED_LONG

TIME_COLUMNS = ["quarter_label", "quarter_start"]
OUTPUT_COLUMNS = TIME_COLUMNS + ["sales_units", "is_imputed", "resolution"]


def load_long(keys: list[str], brand: str | None = None) -> pd.DataFrame:
    if not SALES_PREPARED_LONG.exists():
        raise FileNotFoundError(
            f"Format-prepared table not found: {SALES_PREPARED_LONG}. Run prepare_sales.py first."
        )
    extra = ["brand"] if (brand and "brand" not in keys) else []
    columns = list(dict.fromkeys(keys + TIME_COLUMNS + ["sales_units"] + extra))
    df = pd.read_csv(SALES_PREPARED_LONG, usecols=columns, parse_dates=["quarter_start"])
    if brand:
        df = df[df["brand"].astype("string").str.upper() == brand.upper()]
        if df.empty:
            raise SystemExit(f"No rows for brand={brand!r}.")
        if extra:
            df = df.drop(columns=["brand"])
    return df


def merge_to_series(long_df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Sum sales over the collapsed dimensions. min_count=1 keeps an all-blank
    series-quarter as NaN instead of turning it into a false zero."""
    return (
        long_df.groupby(keys + TIME_COLUMNS, dropna=False, as_index=False)
        .agg(sales_units=("sales_units", lambda values: values.sum(min_count=1)))
        .sort_values(keys + ["quarter_start"])
        .reset_index(drop=True)
    )


def _long_run_mask(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Positions belonging to a True-run longer than max_gap."""
    out = np.zeros_like(mask)
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    diff = np.diff(padded)
    for start, end in zip(np.nonzero(diff == 1)[0], np.nonzero(diff == -1)[0]):
        if (end - start) > max_gap:
            out[start:end] = True
    return out


def resolve_one_series(group: pd.DataFrame, method: str, max_gap: int | None, long_gap: str) -> pd.DataFrame:
    """Truncate leading/trailing blanks and impute internal gaps of one series.
    Internal gaps longer than max_gap are handled by the long_gap policy
    (zero: fill 0; drop: drop the whole series)."""
    group = group.sort_values("quarter_start").reset_index(drop=True)
    observed = group["sales_units"].notna().to_numpy()
    positions = np.nonzero(observed)[0]
    if positions.size == 0:
        return group.iloc[0:0]  # fully empty; should not occur

    group = group.iloc[positions[0] : positions[-1] + 1].copy()
    was_blank = group["sales_units"].isna().to_numpy()

    long_mask = np.zeros_like(was_blank)
    if max_gap is not None and was_blank.any():
        long_mask = _long_run_mask(was_blank, max_gap)
        if long_gap == "drop" and long_mask.any():
            return group.iloc[0:0]  # series has an over-long gap -> drop it

    if method == "locf":
        filled = group["sales_units"].ffill().to_numpy().copy()
    else:
        filled = group["sales_units"].interpolate(method="linear").to_numpy().copy()
    filled[long_mask] = 0.0  # override the ramp/carry across an over-long gap

    group["sales_units"] = filled
    group["is_imputed"] = was_blank
    resolution = np.where(was_blank, method, "observed").astype(object)
    resolution[long_mask] = "long_gap_zero"
    group["resolution"] = resolution
    return group


def resolve_series(series_df: pd.DataFrame, keys: list[str], method: str,
                   max_gap: int | None, long_gap: str) -> pd.DataFrame:
    # The key columns stay inside each group, so they are carried through as-is.
    resolved = [
        group
        for _, raw_group in series_df.groupby(keys, dropna=False, sort=False)
        for group in [resolve_one_series(raw_group, method, max_gap, long_gap)]
        if len(group)
    ]
    return pd.concat(resolved, ignore_index=True)


def _run_lengths(mask: np.ndarray) -> list[int]:
    """Lengths of consecutive True runs in a boolean array."""
    if not mask.any():
        return []
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.nonzero(diff == 1)[0]
    ends = np.nonzero(diff == -1)[0]
    return (ends - starts).tolist()


def analyze_internal_gaps(series_df: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-series internal blank-run stats on the merged (pre-imputation) series.
    An internal gap is a run of blanks between the first and last observed quarter."""
    per_series: list[dict] = []
    gap_lengths: list[int] = []
    for key, group in series_df.groupby(keys, dropna=False, sort=False):
        values = group.sort_values("quarter_start")["sales_units"].to_numpy()
        observed = ~np.isnan(values)
        positions = np.nonzero(observed)[0]
        if positions.size == 0:
            continue
        first, last = positions[0], positions[-1]
        runs = _run_lengths(~observed[first : last + 1])
        gap_lengths.extend(runs)
        key_tuple = key if isinstance(key, tuple) else (key,)
        per_series.append(
            dict(zip(keys, key_tuple))
            | {
                "n_observed": int(positions.size),
                "span": int(last - first + 1),
                "n_internal_gaps": len(runs),
                "longest_internal_gap": max(runs) if runs else 0,
                "total_internal_blanks": int(sum(runs)),
            }
        )
    distribution = (
        pd.Series(gap_lengths, name="gap_length", dtype="int64")
        .value_counts()
        .rename_axis("gap_length")
        .reset_index(name="n_gaps")
        .sort_values("gap_length")
        .reset_index(drop=True)
    )
    return pd.DataFrame(per_series), distribution


def guardrail_table(gap_series: pd.DataFrame, distribution: pd.DataFrame) -> pd.DataFrame:
    """How a max-gap guardrail would act per threshold: how many internal-gap cells
    each threshold still interpolates vs. leaves to the long-gap policy, and how many
    series have a longer gap. Returned (and persisted) so the chosen max_gap is a
    reproducible, citable artifact rather than a one-off console print."""
    lengths = distribution["gap_length"].to_numpy()
    n_gaps = distribution["n_gaps"].to_numpy()
    cells = lengths * n_gaps
    total = int(cells.sum())
    rows = [
        {
            "max_gap": g,
            "cells_filled": int(cells[lengths <= g].sum()),
            "pct_filled": round(cells[lengths <= g].sum() / total, 4) if total else 0.0,
            "cells_in_long_gaps": int(cells[lengths > g].sum()),
            "series_with_longer_gap": int((gap_series["longest_internal_gap"] > g).sum()),
        }
        for g in (1, 2, 3, 4, 6, 8)
    ]
    return pd.DataFrame(rows)


def imputation_report(resolved_df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (
        resolved_df.groupby(keys, dropna=False, as_index=False)
        .agg(
            n_quarters=("sales_units", "size"),
            n_imputed=("is_imputed", "sum"),
            first_quarter=("quarter_label", "first"),
            last_quarter=("quarter_label", "last"),
        )
        .assign(imputed_pct=lambda d: d["n_imputed"] / d["n_quarters"])
        .sort_values("n_imputed", ascending=False)
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the modelling series table (stage 3).")
    parser.add_argument("--keys", nargs="+", default=["product", "market"],
                        help="Series key = forecasting unit. Default: product market.")
    parser.add_argument("--method", choices=["linear", "locf"], default="locf",
                        help="Internal-gap imputation. Default: locf (causal, no look-ahead). "
                             "linear is bidirectional and leaks future info into lag features.")
    parser.add_argument("--brand", help="Restrict to one brand (focal-brand cohort, e.g. TEVA).")
    parser.add_argument("--max-gap", type=int, default=None,
                        help="Longest internal gap still imputed; longer gaps use --long-gap. Default: no limit.")
    parser.add_argument("--long-gap", choices=["zero", "drop"], default="zero",
                        help="Gaps longer than --max-gap: fill 0 or drop the series. Default: zero.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = args.keys

    print(f"Loading {SALES_PREPARED_LONG.name} (keys={keys}{', brand=' + args.brand if args.brand else ''}) ...")
    long_df = load_long(keys, args.brand)
    series_df = merge_to_series(long_df, keys)
    print(f"Merged to {series_df[keys].drop_duplicates().shape[0]:,} series.")

    gap_series, gap_distribution = analyze_internal_gaps(series_df, keys)
    guardrail = guardrail_table(gap_series, gap_distribution)
    total_cells = int((gap_distribution["gap_length"] * gap_distribution["n_gaps"]).sum())
    print(f"\ninternal-gap cells total: {total_cells:,} across {int(gap_distribution['n_gaps'].sum()):,} gaps")
    print(guardrail.to_string(index=False))

    resolved = resolve_series(series_df, keys, args.method, args.max_gap, args.long_gap)
    resolved = resolved[keys + OUTPUT_COLUMNS]
    report = imputation_report(resolved, keys)

    out_dir = SALES_MODELING_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(keys) + (f"_{args.brand.upper().replace(' ', '_')}" if args.brand else "")
    resolved.to_csv(out_dir / f"series_{suffix}.csv", index=False)
    report.to_csv(out_dir / f"series_{suffix}_imputation.csv", index=False)
    gap_series.to_csv(out_dir / f"series_{suffix}_gaps.csv", index=False)
    gap_distribution.to_csv(out_dir / f"series_{suffix}_gap_distribution.csv", index=False)
    guardrail.to_csv(out_dir / f"series_{suffix}_gap_guardrail.csv", index=False)

    n_imputed = int(resolved["is_imputed"].sum())
    print(f"Rows: {len(resolved):,} | imputed target cells: {n_imputed:,} "
          f"({n_imputed / len(resolved):.2%})")
    print(f"Wrote series_{suffix}.csv and series_{suffix}_imputation.csv to {out_dir}")


if __name__ == "__main__":
    main()
