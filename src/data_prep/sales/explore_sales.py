"""
Explore the prepared sales dataset.

This script computes dataset-level summary tables for first data checks:
distributions, per-dimension aggregates, time coverage, and raw-import profiles.
It does not change the data; it reads the raw export (for raw-import profiles)
and the prepared long table (for everything else), and writes summary CSVs.

Series-level investigation (blanks, intermittency, seasonality) lives in
``investigate_series.py``. Data preparation lives in ``prepare_sales.py``.

Run:
    python src/data_prep/sales/explore_sales.py

Output (data/sales/exploration/):
    sales_overview.csv, sales_distribution.csv, sales_time_coverage.csv,
    dimension_profile.csv, raw_column_profile.csv,
    raw_quarterly_sales_blank_profile.csv, sales_missing_values.csv,
    top_product_dosage_brand_market_combinations.csv,
    sales_by_{product,dosage,dosage2,brand,market,quarter}.csv,
    sales_dosage_mix.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import SALES_EXPLORATION_DATASET_DIR, SALES_PREPARED_LONG, SALES_RAW_EXPORT
from prepare_sales import (
    DIMENSIONS,
    FIXED_COLUMN_MAP,
    SALES_COLUMNS,
    read_table,
    remove_embedded_metadata_rows,
    validate_fixed_columns,
)


def sample_values(series: pd.Series, max_values: int = 5) -> str:
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""].drop_duplicates().head(max_values)
    return " | ".join(values)


def make_raw_column_profile(raw_df: pd.DataFrame) -> pd.DataFrame:
    expected_names = {**FIXED_COLUMN_MAP, **{col: "quarterly_sales" for col in SALES_COLUMNS}}
    rows = []
    for col in raw_df.columns:
        missing = int(raw_df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "expected_content": expected_names.get(col, ""),
                "dtype": str(raw_df[col].dtype),
                "non_null_rows": int(raw_df[col].notna().sum()),
                "missing_rows": missing,
                "missing_pct": missing / len(raw_df) if len(raw_df) else 0,
                "unique_values": int(raw_df[col].nunique(dropna=True)),
                "example_values": sample_values(raw_df[col]),
            }
        )
    return pd.DataFrame(rows)


def make_raw_sales_blank_profile(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in SALES_COLUMNS:
        raw_text = raw_df[col].astype("string").str.strip()
        blank_mask = raw_df[col].isna() | raw_text.isna() | (raw_text == "")
        rows.append(
            {
                "column": col,
                "blank_sales_cells": int(blank_mask.sum()),
                "blank_sales_pct": float(blank_mask.mean()) if len(raw_df) else 0,
            }
        )
    return pd.DataFrame(rows)


def make_dimension_profile(prepared_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in DIMENSIONS:
        counts = prepared_df.groupby(col, dropna=False)["sales_units"].sum()
        top_value = counts.sort_values(ascending=False).index[0] if len(counts) else pd.NA
        rows.append(
            {
                "dimension": col,
                "unique_values": int(prepared_df[col].nunique(dropna=True)),
                "missing_rows": int(prepared_df[col].isna().sum()),
                "top_value_by_sales": top_value,
                "top_value_sales_units": float(counts.max()) if len(counts) else 0,
            }
        )
    return pd.DataFrame(rows)


def make_sales_distribution(prepared_df: pd.DataFrame) -> pd.DataFrame:
    sales = prepared_df["sales_units"]
    quantiles = sales.quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1])
    rows = [
        ("rows", len(sales)),
        ("blank_sales_rows", int(sales.isna().sum())),
        ("zero_sales_rows", int((sales == 0).sum())),
        ("positive_sales_rows", int((sales > 0).sum())),
        ("negative_sales_rows", int((sales < 0).sum())),
        ("blank_sales_pct", float(sales.isna().mean()) if len(sales) else 0),
        ("zero_sales_pct", float((sales == 0).mean()) if len(sales) else 0),
        ("total_sales_units", float(sales.sum())),
        ("mean_sales_units", float(sales.mean()) if len(sales) else 0),
        ("std_sales_units", float(sales.std()) if len(sales) else 0),
    ]
    rows.extend((f"p{int(q * 100):02d}_sales_units", float(value)) for q, value in quantiles.items())
    return pd.DataFrame(rows, columns=["metric", "value"])


def make_time_coverage(prepared_df: pd.DataFrame) -> pd.DataFrame:
    return (
        prepared_df.groupby(["quarter_label", "quarter_start"], dropna=False, as_index=False)
        .agg(
            sales_units=("sales_units", "sum"),
            n_rows=("sales_units", "size"),
            zero_sales_rows=("sales_units", lambda values: int((values == 0).sum())),
            positive_sales_rows=("sales_units", lambda values: int((values > 0).sum())),
            n_products=("product", "nunique"),
            n_markets=("market", "nunique"),
            n_brands=("brand", "nunique"),
        )
        .assign(zero_sales_pct=lambda df: df["zero_sales_rows"] / df["n_rows"])
        .sort_values("quarter_start")
        .reset_index(drop=True)
    )


def make_top_combinations(prepared_df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    return (
        prepared_df.groupby(DIMENSIONS, dropna=False, as_index=False)
        .agg(
            sales_units=("sales_units", "sum"),
            mean_quarterly_sales=("sales_units", "mean"),
            max_quarterly_sales=("sales_units", "max"),
            zero_sales_quarters=("sales_units", lambda values: int((values == 0).sum())),
            active_quarters=("sales_units", lambda values: int((values > 0).sum())),
            n_quarters=("quarter_label", "nunique"),
        )
        .assign(active_quarter_pct=lambda df: df["active_quarters"] / df["n_quarters"])
        .sort_values("sales_units", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def make_overview(raw_df: pd.DataFrame, prepared_df: pd.DataFrame, metadata_rows_removed: int) -> pd.DataFrame:
    raw_sales_blank_cells = int(make_raw_sales_blank_profile(raw_df)["blank_sales_cells"].sum())
    return pd.DataFrame(
        [
            ("raw_rows", len(raw_df)),
            ("raw_columns", raw_df.shape[1]),
            ("raw_cells", int(raw_df.size)),
            ("embedded_metadata_rows_removed", metadata_rows_removed),
            ("raw_missing_cells", int(raw_df.isna().sum().sum())),
            ("raw_unique_products", raw_df["Column1"].nunique(dropna=True)),
            ("raw_unique_dosages", raw_df["Column2"].nunique(dropna=True)),
            ("raw_unique_dosage2_values", raw_df["Column3"].nunique(dropna=True)),
            ("raw_unique_brands", raw_df["Column4"].nunique(dropna=True)),
            ("raw_unique_markets", raw_df["Column5"].nunique(dropna=True)),
            ("raw_blank_quarterly_sales_cells", raw_sales_blank_cells),
            ("prepared_rows", len(prepared_df)),
            ("long_rows_before_aggregation", int(prepared_df["source_rows"].sum())),
            ("duplicate_key_rows_aggregated", int((prepared_df["source_rows"] > 1).sum())),
            ("missing_sales_rows", int(prepared_df["missing_sales_rows"].sum())),
            ("negative_sales_rows", int((prepared_df["sales_units"] < 0).sum())),
            ("total_sales_units", float(prepared_df["sales_units"].sum())),
            ("n_products", prepared_df["product"].nunique(dropna=True)),
            ("n_dosages", prepared_df["dosage"].nunique(dropna=True)),
            ("n_dosage2_values", prepared_df["dosage2"].nunique(dropna=True)),
            ("n_brands", prepared_df["brand"].nunique(dropna=True)),
            ("n_markets", prepared_df["market"].nunique(dropna=True)),
            ("n_quarters", prepared_df["quarter_label"].nunique(dropna=True)),
        ],
        columns=["metric", "value"],
    )


def summarize_by(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (
        df.groupby(columns, dropna=False, as_index=False)
        .agg(
            sales_units=("sales_units", "sum"),
            n_rows=("sales_units", "size"),
            n_products=("product", "nunique"),
            n_dosages=("dosage", "nunique"),
            n_dosage2_values=("dosage2", "nunique"),
            n_brands=("brand", "nunique"),
            n_markets=("market", "nunique"),
            n_quarters=("quarter_label", "nunique"),
        )
        .sort_values("sales_units", ascending=False)
        .reset_index(drop=True)
    )


def dosage_mix(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["product", "market", "brand"], dropna=False)
        .agg(
            n_dosages=("dosage", "nunique"),
            dosages=("dosage", lambda values: " | ".join(sorted(set(values.dropna().astype(str))))),
            n_dosage2_values=("dosage2", "nunique"),
            dosage2_values=("dosage2", lambda values: " | ".join(sorted(set(values.dropna().astype(str))))),
            sales_units=("sales_units", "sum"),
        )
        .reset_index()
        .sort_values(["n_dosages", "n_dosage2_values", "sales_units"], ascending=[False, False, False])
    )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if not SALES_RAW_EXPORT.exists():
        raise FileNotFoundError(f"Raw export not found: {SALES_RAW_EXPORT}")
    if not SALES_PREPARED_LONG.exists():
        raise FileNotFoundError(
            f"Prepared long table not found: {SALES_PREPARED_LONG}. Run prepare_sales.py first."
        )
    raw_df = read_table(SALES_RAW_EXPORT, None)
    validate_fixed_columns(raw_df)
    raw_df, metadata_rows_removed = remove_embedded_metadata_rows(raw_df)
    prepared_df = pd.read_csv(SALES_PREPARED_LONG)
    return raw_df, prepared_df, metadata_rows_removed


def main() -> None:
    raw_df, prepared_df, metadata_rows_removed = load_inputs()
    out_dir = SALES_EXPLORATION_DATASET_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    make_overview(raw_df, prepared_df, metadata_rows_removed).to_csv(out_dir / "sales_overview.csv", index=False)
    make_raw_column_profile(raw_df).to_csv(out_dir / "raw_column_profile.csv", index=False)
    make_raw_sales_blank_profile(raw_df).to_csv(out_dir / "raw_quarterly_sales_blank_profile.csv", index=False)
    make_dimension_profile(prepared_df).to_csv(out_dir / "dimension_profile.csv", index=False)
    make_sales_distribution(prepared_df).to_csv(out_dir / "sales_distribution.csv", index=False)
    make_time_coverage(prepared_df).to_csv(out_dir / "sales_time_coverage.csv", index=False)
    make_top_combinations(prepared_df).to_csv(
        out_dir / "top_product_dosage_brand_market_combinations.csv", index=False
    )
    missing_values = raw_df.isna().sum().rename("missing_values").reset_index()
    missing_values.columns = ["column", "missing_values"]
    missing_values.to_csv(out_dir / "sales_missing_values.csv", index=False)

    summarize_by(prepared_df, ["product"]).to_csv(out_dir / "sales_by_product.csv", index=False)
    summarize_by(prepared_df, ["dosage"]).to_csv(out_dir / "sales_by_dosage.csv", index=False)
    summarize_by(prepared_df, ["dosage2"]).to_csv(out_dir / "sales_by_dosage2.csv", index=False)
    summarize_by(prepared_df, ["brand"]).to_csv(out_dir / "sales_by_brand.csv", index=False)
    summarize_by(prepared_df, ["market"]).to_csv(out_dir / "sales_by_market.csv", index=False)
    summarize_by(prepared_df, ["quarter_label", "quarter_start"]).sort_values("quarter_start").to_csv(
        out_dir / "sales_by_quarter.csv", index=False
    )
    dosage_mix(prepared_df).to_csv(out_dir / "sales_dosage_mix.csv", index=False)

    print(f"Wrote exploration tables to {out_dir}")


if __name__ == "__main__":
    main()
