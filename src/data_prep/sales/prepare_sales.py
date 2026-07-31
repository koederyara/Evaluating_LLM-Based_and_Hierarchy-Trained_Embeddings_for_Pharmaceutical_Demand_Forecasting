"""
Prepare quarterly sales data.

This script does one thing: it turns the raw IQVIA wide export into a clean long
modelling table (one row per product, dosage, dosage2, brand, market, quarter)
and a matching wide table for manual inspection.

Dataset exploration (distributions, profiles, time coverage) lives in
``explore_sales.py``; series-level investigation (blanks, intermittency,
seasonality) lives in ``investigate_series.py``. Both import the parsing
primitives defined here.

Expected input layout
---------------------
    Column1  = product
    Column2  = dosage
    Column3  = dosage2
    Column4  = brand
    Column5  = market
    Column6  = first quarterly sales column, Q2 2014
    Column53 = last quarterly sales column, Q1 2026

If the first row contains exported labels such as "Molecule List", "Country",
and "Q2 2014_Standard Units", that row is treated as metadata and removed.

This stage only formats and reshapes; it does not decide values. Empty cells in
Column6..Column53 stay NaN. How to interpret them (zero vs. imputation) is a
value decision made in the modeling stage.

Run:
    python src/data_prep/sales/prepare_sales.py
    python src/data_prep/sales/prepare_sales.py --input data/raw/sales.xlsx

Output:
    data/sales/preparation/sales_prepared_long.csv
    data/sales/preparation/sales_prepared_wide.csv
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import SALES_PREPARATION_DIR, SALES_RAW_EXPORT


DEFAULT_INPUT_PATH = SALES_RAW_EXPORT
DEFAULT_OUTPUT_DIR = SALES_PREPARATION_DIR

DIMENSIONS = ["product", "dosage", "dosage2", "brand", "market"]
GROUP_COLUMNS = DIMENSIONS + ["year", "quarter", "quarter_label", "quarter_start"]
NUMBERED_LAYOUT_LAST_QUARTER = "2026Q1"
NUMBERED_LAYOUT_FIRST_SALES_COL = 6
NUMBERED_LAYOUT_LAST_SALES_COL = 53
FIXED_COLUMN_MAP = {
    "Column1": "product",
    "Column2": "dosage",
    "Column3": "dosage2",
    "Column4": "brand",
    "Column5": "market",
}
SALES_COLUMNS = [
    f"Column{index}"
    for index in range(NUMBERED_LAYOUT_FIRST_SALES_COL, NUMBERED_LAYOUT_LAST_SALES_COL + 1)
]

QUARTER_COLUMN_PATTERN = re.compile(
    r"^\s*(?:(?P<year_a>20\d{2}|19\d{2})\s*[-_/ ]?\s*Q(?P<q_a>[1-4])|"
    r"Q(?P<q_b>[1-4])\s*[-_/ ]?\s*(?P<year_b>20\d{2}|19\d{2}))\s*$",
    re.IGNORECASE,
)
QUARTER_LABEL_SEARCH_PATTERN = re.compile(
    r"(?:(?P<year_a>20\d{2}|19\d{2})\s*[-_/ ]?\s*Q(?P<q_a>[1-4])|"
    r"Q(?P<q_b>[1-4])\s*[-_/ ]?\s*(?P<year_b>20\d{2}|19\d{2}))",
    re.IGNORECASE,
)


def read_table(path: Path, sheet_name: str | int | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name if sheet_name is not None else 0)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def clean_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )


def clean_sales(series: pd.Series) -> pd.Series:
    """Parse sales values as numbers. Blanks are kept as NaN (no value decision
    here); interpreting empty cells is deferred to the modeling stage.

    Text is only parsed when it matches an unambiguous number format: US style
    (1,234.56) or EU style (1.234,56). "1.234" fits both and is read as US
    decimal, matching the export locale. Anything else becomes NaN."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    text = series.astype("string").str.strip()
    us = text.str.fullmatch(r"-?(\d{1,3}(,\d{3})*|\d+)(\.\d+)?", na=False)
    eu = text.str.fullmatch(r"-?(\d{1,3}(\.\d{3})*|\d+)(,\d+)?", na=False) & ~us

    numeric = pd.Series(float("nan"), index=series.index, dtype="float64")
    numeric[us] = pd.to_numeric(text[us].str.replace(",", "", regex=False)).astype("float64")
    numeric[eu] = pd.to_numeric(
        text[eu].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    ).astype("float64")
    return numeric


def parse_quarter_value(value: object) -> tuple[int | None, int | None]:
    if pd.isna(value):
        return None, None

    if isinstance(value, pd.Timestamp):
        return int(value.year), int(((value.month - 1) // 3) + 1)

    text = str(value).strip()
    match = QUARTER_COLUMN_PATTERN.match(text)
    if match:
        year = match.group("year_a") or match.group("year_b")
        quarter = match.group("q_a") or match.group("q_b")
        return int(year), int(quarter)

    match = re.match(r"^\s*(?P<year>20\d{2}|19\d{2})[-_/ ](?P<month>\d{1,2})", text)
    if match:
        month = int(match.group("month"))
        if 1 <= month <= 12:
            return int(match.group("year")), ((month - 1) // 3) + 1

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.notna(parsed):
        return int(parsed.year), int(((parsed.month - 1) // 3) + 1)

    return None, None


def parse_quarter_label_from_text(value: object) -> str | None:
    if pd.isna(value):
        return None

    match = QUARTER_LABEL_SEARCH_PATTERN.search(str(value).strip())
    if not match:
        return None

    year = match.group("year_a") or match.group("year_b")
    quarter = match.group("q_a") or match.group("q_b")
    return f"{int(year)}Q{int(quarter)}"


def add_quarter_fields(df: pd.DataFrame, quarter_source: str, year_source: str | None) -> pd.DataFrame:
    df = df.copy()
    if year_source:
        years = pd.to_numeric(df[year_source], errors="coerce")
        quarters = (
            df[quarter_source]
            .astype("string")
            .str.extract(r"([1-4])", expand=False)
            .astype("float")
        )
    else:
        parsed = df[quarter_source].map(parse_quarter_value)
        years = parsed.map(lambda item: item[0])
        quarters = parsed.map(lambda item: item[1])

    df["year"] = pd.to_numeric(years, errors="coerce").astype("Int64")
    df["quarter"] = pd.to_numeric(quarters, errors="coerce").astype("Int64")
    df["quarter_label"] = (
        df["year"].astype("string") + "Q" + df["quarter"].astype("string")
    )
    df.loc[df["year"].isna() | df["quarter"].isna(), "quarter_label"] = pd.NA
    df["quarter_start"] = pd.to_datetime(
        {
            "year": df["year"].astype("float"),
            "month": (df["quarter"].astype("float") - 1) * 3 + 1,
            "day": 1,
        },
        errors="coerce",
    )
    return df


def parse_last_quarter(value: str) -> pd.Period:
    text = str(value).strip()
    match = re.match(r"^(?P<month>\d{1,2})[./-](?P<year>20\d{2}|19\d{2})$", text)
    if match:
        month = int(match.group("month"))
        year = int(match.group("year"))
        return pd.Period(year=year, quarter=((month - 1) // 3) + 1, freq="Q")

    year, quarter = parse_quarter_value(text)
    if year is None or quarter is None:
        raise ValueError(f"Could not parse last quarter value: {value!r}")
    return pd.Period(year=year, quarter=quarter, freq="Q")


def build_quarter_column_map(
    last_quarter_value: str = NUMBERED_LAYOUT_LAST_QUARTER,
) -> dict[str, str]:
    last_period = parse_last_quarter(last_quarter_value)
    first_period = last_period - (len(SALES_COLUMNS) - 1)

    quarter_map = {}
    for offset, col in enumerate(SALES_COLUMNS):
        period = first_period + offset
        quarter_map[col] = f"{period.year}Q{period.quarter}"

    return quarter_map


def extract_quarter_map_from_metadata_rows(df: pd.DataFrame) -> dict[str, str] | None:
    for _, row in df.head(10).iterrows():
        quarter_map = {}
        for col in SALES_COLUMNS:
            label = parse_quarter_label_from_text(row[col])
            if label:
                quarter_map[col] = label

        if len(quarter_map) == len(SALES_COLUMNS):
            return quarter_map

    return None


def is_embedded_metadata_row(row: pd.Series) -> bool:
    column1 = str(row.get("Column1", "")).strip().lower()
    column5 = str(row.get("Column5", "")).strip().lower()
    n_quarter_labels = sum(
        parse_quarter_label_from_text(row.get(col)) is not None for col in SALES_COLUMNS
    )
    return (
        column1 in {"molecule list", "product", "product list"}
        or column5 in {"country", "market"}
        or n_quarter_labels >= 3
    )


def remove_embedded_metadata_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    metadata_mask = df.apply(is_embedded_metadata_row, axis=1)
    cleaned_df = df.loc[~metadata_mask].reset_index(drop=True)
    return cleaned_df, int(metadata_mask.sum())


def validate_fixed_columns(df: pd.DataFrame) -> None:
    required_columns = list(FIXED_COLUMN_MAP) + SALES_COLUMNS
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing expected columns for the fixed sales export layout: "
            + ", ".join(missing)
        )


def to_long_format(
    df: pd.DataFrame,
    last_quarter_value: str,
    quarter_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    validate_fixed_columns(df)
    quarter_map = quarter_map or build_quarter_column_map(last_quarter_value)

    renamed = df.rename(columns={**FIXED_COLUMN_MAP, **quarter_map})
    quarter_columns = list(quarter_map.values())
    long_df = renamed.melt(
        id_vars=DIMENSIONS,
        value_vars=quarter_columns,
        var_name="_quarter_source",
        value_name="sales_units",
    )
    long_df = add_quarter_fields(long_df, "_quarter_source", None)

    for col in DIMENSIONS:
        long_df[col] = clean_text(long_df[col])
    long_df["sales_units"] = clean_sales(long_df["sales_units"])
    return long_df[DIMENSIONS + ["year", "quarter", "quarter_label", "quarter_start", "sales_units"]]


def aggregate_sales(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(GROUP_COLUMNS, dropna=False, as_index=False)
        .agg(
            # min_count=1 keeps an all-blank key as NaN instead of collapsing to 0
            sales_units=("sales_units", lambda values: values.sum(min_count=1)),
            source_rows=("sales_units", "size"),
            missing_sales_rows=("sales_units", lambda values: int(values.isna().sum())),
        )
        .sort_values(["product", "dosage", "dosage2", "brand", "market", "quarter_start"])
        .reset_index(drop=True)
    )


def to_wide_format(prepared_df: pd.DataFrame) -> pd.DataFrame:
    quarter_order = (
        prepared_df[["quarter_label", "quarter_start"]]
        .drop_duplicates()
        .sort_values("quarter_start")["quarter_label"]
        .tolist()
    )

    wide_df = (
        prepared_df.pivot_table(
            index=DIMENSIONS,
            columns="quarter_label",
            values="sales_units",
            aggfunc=lambda values: values.sum(min_count=1),
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    return wide_df[DIMENSIONS + quarter_order]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare quarterly product sales data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input CSV, TSV, XLS, or XLSX file. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sheet-name", default=None, help="Excel sheet name or index. Defaults to first sheet.")
    parser.add_argument(
        "--last-quarter",
        default=NUMBERED_LAYOUT_LAST_QUARTER,
        help="Quarter of Column53. Default: 2026Q1.",
    )
    return parser.parse_args()


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}. "
            "Set SALES_RAW_EXPORT in config.py or pass --input with the file path."
        )

    print(f"Reading input: {args.input}")
    step_started = time.perf_counter()
    df = read_table(args.input, args.sheet_name)
    print(f"Loaded raw data: {len(df)} rows, {df.shape[1]} columns in {time.perf_counter() - step_started:.1f}s")

    validate_fixed_columns(df)
    quarter_map = build_quarter_column_map(args.last_quarter)
    metadata_quarter_map = extract_quarter_map_from_metadata_rows(df)
    if metadata_quarter_map and metadata_quarter_map != quarter_map:
        print("WARNING: Metadata quarter labels differ from the fixed mapping.")

    df, metadata_rows_removed = remove_embedded_metadata_rows(df)
    if metadata_rows_removed:
        print(f"Removed {metadata_rows_removed} embedded metadata/header row(s) before processing")

    print("Converting to long format...")
    step_started = time.perf_counter()
    long_df = to_long_format(df, args.last_quarter, quarter_map)
    print(f"Prepared long data: {len(long_df)} rows in {time.perf_counter() - step_started:.1f}s")

    print("Aggregating duplicate product/dosage/brand/market/quarter combinations...")
    step_started = time.perf_counter()
    prepared_df = aggregate_sales(long_df)
    print(f"Aggregated data: {len(prepared_df)} rows in {time.perf_counter() - step_started:.1f}s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared_df.to_csv(args.output_dir / "sales_prepared_long.csv", index=False)
    to_wide_format(prepared_df).to_csv(args.output_dir / "sales_prepared_wide.csv", index=False)

    print(f"Total runtime: {time.perf_counter() - started_at:.1f}s")
    print(f"Wrote prepared long and wide tables to {args.output_dir}")


if __name__ == "__main__":
    main()
