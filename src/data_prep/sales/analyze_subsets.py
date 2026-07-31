"""
Analyze candidate data subsets for their forecasting suitability.

Restricting the forecasting cohort by market or by brand keeps the full product
universe (so the ATC embeddings stay meaningful), while restricting by product
itself would not. This script ranks candidate subsets on data density, the empty-cell
situation, concentration, and ATC coverage, so a dense-but-still-embedding-
relevant cohort can be chosen.

For each subset it aggregates over brand/dosage to the product x market series
(blanks kept as NaN via min_count=1) and reports, per series, how many cells are
empty and how many are empty in a row.

Run:
    python src/data_prep/sales/analyze_subsets.py --by market
    python src/data_prep/sales/analyze_subsets.py --by brand --top-brands 15

Output (data/prepared_data/sales/exploration/subsets/):
    subset_summary_<by>.csv         one row per subset
    subset_series_blanks_<by>.csv   one row per (subset, product, market) series
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import SALES_EXPLORATION_SUBSETS_DIR, SALES_MAPPING_DIR, SALES_PREPARED_LONG
from build_series import _run_lengths, merge_to_series

SERIES_KEYS = ["product", "market"]
LONG_COLUMNS = ["product", "brand", "market", "quarter_label", "quarter_start", "sales_units"]


def load_long() -> pd.DataFrame:
    if not SALES_PREPARED_LONG.exists():
        raise FileNotFoundError(f"Prepared long table not found: {SALES_PREPARED_LONG}. Run prepare_sales.py first.")
    return pd.read_csv(SALES_PREPARED_LONG, usecols=LONG_COLUMNS, parse_dates=["quarter_start"])


def load_mapped_products() -> tuple[set[str], set[str]]:
    """Return (mapped products, single-ATC products) from the ATC mapping, if present."""
    path = SALES_MAPPING_DIR / "product_atc_map.csv"
    if not path.exists():
        return set(), set()
    mapping = pd.read_csv(path)
    mapped = set(mapping.loc[mapping["match_type"] != "none", "product"])
    single = set(mapping.loc[mapping["n_atc_codes"] == 1, "product"])
    return mapped, single


def series_blank_stats(series_df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Per-series blank statistics on the merged (pre-imputation) series."""
    records: list[dict] = []
    groups = series_df.groupby(keys, sort=False)
    for key, group in tqdm(groups, total=groups.ngroups, desc="series", unit="series", leave=False):
        values = group.sort_values("quarter_start")["sales_units"].to_numpy()
        observed = ~np.isnan(values)
        positions = np.nonzero(observed)[0]
        if positions.size == 0:
            continue
        first, last = positions[0], positions[-1]
        inner_blank = ~observed[first : last + 1]
        runs = _run_lengths(inner_blank)
        key_tuple = key if isinstance(key, tuple) else (key,)
        records.append(
            dict(zip(keys, key_tuple))
            | {
                "n_observed": int(positions.size),
                "span": int(last - first + 1),
                "leading_blank": int(first),
                "trailing_blank": int(len(values) - 1 - last),
                "n_internal_blank": int(inner_blank.sum()),
                "longest_internal_run": max(runs) if runs else 0,
                "volume": float(np.nansum(values)),
            }
        )
    return pd.DataFrame(records)


def _gini(values: np.ndarray) -> float:
    values = np.clip(values, 0, None)
    if values.size == 0 or values.sum() == 0:
        return 0.0
    sorted_values = np.sort(values)
    n = sorted_values.size
    index = np.arange(1, n + 1)
    return float((2 * (index * sorted_values).sum()) / (n * sorted_values.sum()) - (n + 1) / n)


def summarize(stats: pd.DataFrame, subset_col: str, mapped: set[str], single: set[str]) -> pd.DataFrame:
    total_volume = stats["volume"].sum()
    rows = []
    for subset, group in stats.groupby(subset_col):
        products = set(group["product"])
        volume = group["volume"].sum()
        rows.append(
            {
                subset_col: subset,
                "n_series": len(group),
                "n_products": len(products),
                "n_atc_mapped": len(products & mapped) if mapped else np.nan,
                "n_single_atc": len(products & single) if single else np.nan,
                "median_history": float(group["n_observed"].median()),
                "pct_history_ge12": float((group["n_observed"] >= 12).mean()),
                "pct_history_ge24": float((group["n_observed"] >= 24).mean()),
                "pct_complete": float(((group["n_internal_blank"] == 0) & (group["span"] == 48)).mean()),
                "mean_internal_blank": float(group["n_internal_blank"].mean()),
                "median_internal_blank": float(group["n_internal_blank"].median()),
                "mean_longest_run": float(group["longest_internal_run"].mean()),
                "p90_longest_run": float(group["longest_internal_run"].quantile(0.90)),
                "pct_longrun_ge4": float((group["longest_internal_run"] >= 4).mean()),
                "gini_volume": _gini(group["volume"].to_numpy()),
                "volume_share": float(volume / total_volume) if total_volume else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("pct_history_ge12", ascending=False).reset_index(drop=True)


def run_market_axis(long_df: pd.DataFrame, mapped: set[str], single: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    series_df = merge_to_series(long_df, SERIES_KEYS)
    stats = series_blank_stats(series_df, SERIES_KEYS)
    return summarize(stats, "market", mapped, single), stats


def run_brand_axis(
    long_df: pd.DataFrame, brands: list[str], mapped: set[str], single: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    for brand in tqdm(brands, desc="brands", unit="brand"):
        subset = long_df[long_df["brand"] == brand]
        if subset.empty:
            continue
        series_df = merge_to_series(subset, SERIES_KEYS)
        stats = series_blank_stats(series_df, SERIES_KEYS)
        stats.insert(0, "brand", brand)
        parts.append(stats)
    all_stats = pd.concat(parts, ignore_index=True)
    return summarize(all_stats, "brand", mapped, single), all_stats


def top_brands(long_df: pd.DataFrame, n: int) -> list[str]:
    volume = long_df.groupby("brand")["sales_units"].sum().sort_values(ascending=False)
    return volume.head(n).index.tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze candidate market/brand subsets for forecasting suitability.")
    parser.add_argument("--by", choices=["market", "brand"], default="market")
    parser.add_argument("--top-brands", type=int, default=15, help="Number of top brands (by volume) when --by brand.")
    parser.add_argument("--brands", nargs="+", help="Explicit brands to analyze (overrides --top-brands).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading {SALES_PREPARED_LONG.name} ...")
    long_df = load_long()
    mapped, single = load_mapped_products()

    if args.by == "market":
        summary, stats = run_market_axis(long_df, mapped, single)
    else:
        brands = args.brands or top_brands(long_df, args.top_brands)
        print(f"Brands: {', '.join(brands)}")
        summary, stats = run_brand_axis(long_df, brands, mapped, single)

    print(f"\n== subset suitability (by {args.by}) ==")
    print(summary.to_string(index=False))

    out_dir = SALES_EXPLORATION_SUBSETS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / f"subset_summary_{args.by}.csv", index=False)
    stats.to_csv(out_dir / f"subset_series_blanks_{args.by}.csv", index=False)
    print(f"\nWrote subset_summary_{args.by}.csv and subset_series_blanks_{args.by}.csv to {out_dir}")


if __name__ == "__main__":
    main()
