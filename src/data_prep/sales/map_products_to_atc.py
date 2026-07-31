"""
Map sales products to ATC codes and report mapping coverage.

Chain
-----
    product (sales)  ->  split on "!" into components
                     ->  each component: molecule / ACTIVE_INGREDIENT (map file)
                     ->  ATC_codes link  ->  ATC code
                     ->  embedding (node_index.json / .npz, keyed by ATC code)

Combination products list their substances separated by "!" (e.g.
"BUDESONIDE!FORMOTEROL"). Each component is mapped on its own, so a product can
carry several ATC codes. They are stored pipe-joined in one column; the product
stays a single row (no row explosion).

The map files are read from data/raw/forecasting and are confidential, so this
script writes only the mapped sales file plus aggregate coverage tables to
data/sales. No raw map rows leave that folder.

Tiers (exact > close > related) are applied with fallback per component. By
default only the exact tier is used; pass --tiers to add more.

match_type values: a tier name (single-component product), "combination" (all
components mapped), "combination_partial" (some mapped), or "none".

Examples
--------
    python src/data_prep/sales/map_products_to_atc.py
    python src/data_prep/sales/map_products_to_atc.py --tiers exact close
    python src/data_prep/sales/map_products_to_atc.py --brands TEVA

With --brands, only those brands are mapped and output files get a brand suffix
(e.g. atc_coverage_summary_TEVA.csv), so the full run is not overwritten.

Main outputs (in data/sales/mapping)
------------------------------------
    sales_prepared_long_mapped.csv   (all rows + atc columns)
    atc_coverage_summary.csv
    product_atc_map.csv
    unmapped_top.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import ROOT, SALES_MAPPING_DIR, SALES_PREPARED_LONG, SALES_RAW_FORECAST_DIR

SALES_LONG = SALES_PREPARED_LONG
NODE_INDEX = ROOT / "data" / "embeddings" / "lorentz" / "node_index.json"
OUTPUT_DIR = SALES_MAPPING_DIR
CHUNK_SIZE = 1_000_000

MATCH_FILES: dict[str, Path] = {
    "exact": SALES_RAW_FORECAST_DIR / "IQVIA-Results-Unique-Values-exactMatch.csv",
    "close": SALES_RAW_FORECAST_DIR / "IQVIA-Results-Unique-Values-closeMatch.csv",
    "related": SALES_RAW_FORECAST_DIR / "IQVIA-Results-Unique-Values-related.csv",
}
TIER_ORDER = ["exact", "close", "related"]

KEY_COLUMNS = ["molecule", "ACTIVE_INGREDIENT"]
# Different match files name the code column differently and store either a
# ".../atc/<CODE>" link or a bare code; both forms are handled.
ATC_CODE_COLUMNS = ["ATC_codes", "ATC_CODE"]
ATC_CODE_BODY = r"[A-Za-z]\d{2}[A-Za-z]{0,2}\d{0,2}"
ATC_URL_PATTERN = re.compile(rf"atc/({ATC_CODE_BODY})", re.IGNORECASE)
ATC_BARE_PATTERN = re.compile(rf"(?<![A-Za-z0-9/])({ATC_CODE_BODY})(?![A-Za-z0-9])")


def resolve_column(df: pd.DataFrame, name: str) -> str | None:
    lower = {col.lower(): col for col in df.columns}
    return lower.get(name.lower())


def resolve_first_column(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        col = resolve_column(df, name)
        if col is not None:
            return col
    return None


def normalize_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper().replace({"": pd.NA})


def extract_atc_codes(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value)
    codes = ATC_URL_PATTERN.findall(text) + ATC_BARE_PATTERN.findall(text)
    return sorted({code.upper() for code in codes})


def load_valid_atc_codes() -> set[str]:
    with NODE_INDEX.open(encoding="utf-8") as handle:
        return set(json.load(handle).keys())


def build_substance_to_codes(path: Path, valid_codes: set[str]) -> dict[str, list[str]]:
    """Map normalized substance name -> valid ATC codes for one match file."""
    df = pd.read_csv(path)
    present_keys = [col for col in KEY_COLUMNS if resolve_column(df, col)]
    link_col = resolve_first_column(df, ATC_CODE_COLUMNS)
    if not present_keys or link_col is None:
        raise ValueError(
            f"{path.name}: expected one of {KEY_COLUMNS} and one of {ATC_CODE_COLUMNS}. "
            f"Found columns: {list(df.columns)}"
        )

    codes_per_row = df[link_col].map(extract_atc_codes)

    mapping: dict[str, set[str]] = {}
    for key_col in present_keys:
        keys = normalize_key(df[resolve_column(df, key_col)])
        for key, codes in zip(keys, codes_per_row):
            if pd.isna(key) or not codes:
                continue
            valid = [code for code in codes if code in valid_codes]
            if valid:
                mapping.setdefault(key, set()).update(valid)
    return {key: sorted(codes) for key, codes in mapping.items()}


COMPONENT_SEPARATOR = "!"
MAP_FIELDS = ["match_type", "atc_codes", "n_atc_codes", "n_components", "n_mapped_components"]


def build_resolved_substances(
    tier_maps: dict[str, dict[str, list[str]]], tiers: list[str]
) -> dict[str, tuple[str, list[str]]]:
    """Resolve each substance to its best tier (first tier wins)."""
    resolved: dict[str, tuple[str, list[str]]] = {}
    for tier in tiers:
        for key, codes in tier_maps[tier].items():
            if key not in resolved:
                resolved[key] = (tier, codes)
    return resolved


def resolve_product(product: object, resolved: dict[str, tuple[str, list[str]]]) -> dict:
    """Map every "!"-separated component of a product and union their ATC codes."""
    components = [part.strip().upper() for part in str(product).split(COMPONENT_SEPARATOR)]
    codes: set[str] = set()
    component_tiers: list[str] = []
    for component in components:
        if component in resolved:
            tier, component_codes = resolved[component]
            codes.update(component_codes)
            component_tiers.append(tier)

    n_components = len(components)
    n_mapped = len(component_tiers)
    if n_mapped == 0:
        match_type = "none"
    elif n_components == 1:
        match_type = component_tiers[0]
    elif n_mapped == n_components:
        match_type = "combination"
    else:
        match_type = "combination_partial"

    return {
        "match_type": match_type,
        "atc_codes": "|".join(sorted(codes)),
        "n_atc_codes": len(codes),
        "n_components": n_components,
        "n_mapped_components": n_mapped,
    }


def map_chunk(chunk: pd.DataFrame, cache: dict[str, dict]) -> pd.DataFrame:
    chunk = chunk.copy()
    for field in MAP_FIELDS:
        field_map = {product: fields[field] for product, fields in cache.items()}
        chunk[field] = chunk["product"].map(field_map)
    chunk["match_type"] = chunk["match_type"].fillna("none")
    chunk["atc_codes"] = chunk["atc_codes"].fillna("")
    for field in ["n_atc_codes", "n_components", "n_mapped_components"]:
        chunk[field] = chunk[field].fillna(0).astype(int)
    return chunk


def map_sales_file(
    resolved: dict[str, tuple[str, list[str]]], out_path: Path, brands: set[str] | None = None
) -> tuple[dict[str, float], dict[str, dict]]:
    """Write the fully mapped sales file; return per-product sales and the map cache.

    If brands is given, only rows of those brands are mapped and counted.
    """
    product_sales: dict[str, float] = defaultdict(float)
    cache: dict[str, dict] = {}
    wrote_header = False
    total_rows = 0

    for chunk in pd.read_csv(SALES_LONG, chunksize=CHUNK_SIZE):
        if brands is not None:
            chunk = chunk[normalize_key(chunk["brand"]).isin(brands)]
            if chunk.empty:
                continue
        for product in chunk["product"].dropna().unique():
            if product not in cache:
                cache[product] = resolve_product(product, resolved)

        mapped = map_chunk(chunk, cache)
        mapped.to_csv(out_path, mode="w" if not wrote_header else "a", header=not wrote_header, index=False)
        wrote_header = True
        total_rows += len(mapped)
        for product, sales in mapped.groupby("product")["sales_units"].sum().items():
            product_sales[product] += float(sales)
        print(f"  ...mapped {total_rows} rows")

    return dict(product_sales), cache


def build_product_table(product_sales: dict[str, float], cache: dict[str, dict]) -> pd.DataFrame:
    default = {field: ("none" if field == "match_type" else 0) for field in MAP_FIELDS}
    default["atc_codes"] = ""
    rows = [
        {"product": product, "sales_units": sales, **cache.get(product, default)}
        for product, sales in product_sales.items()
    ]
    return pd.DataFrame(rows).sort_values("sales_units", ascending=False).reset_index(drop=True)


def make_coverage_summary(products: pd.DataFrame, tiers: list[str]) -> pd.DataFrame:
    total_products = len(products)
    total_sales = float(products["sales_units"].sum())
    rows = []

    def block(label: str, mask: pd.Series) -> None:
        n = int(mask.sum())
        sales = float(products.loc[mask, "sales_units"].sum())
        rows.append(
            {
                "scope": label,
                "n_products": n,
                "pct_products": n / total_products if total_products else 0,
                "sales_share": sales / total_sales if total_sales else 0,
            }
        )

    block("all_products", pd.Series(True, index=products.index))
    for tier in tiers:
        block(f"tier_{tier}", products["match_type"] == tier)
    block("combination", products["match_type"] == "combination")
    block("combination_partial", products["match_type"] == "combination_partial")
    block("mapped_total", products["match_type"] != "none")
    block("unmapped", products["match_type"] == "none")
    block("mapped_multi_atc", products["n_atc_codes"] > 1)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Map sales products to ATC codes and report coverage.")
    parser.add_argument("--tiers", nargs="+", choices=TIER_ORDER, default=["exact"])
    parser.add_argument("--brands", nargs="+", default=None, help="Limit mapping to these brands.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--top-unmapped", type=int, default=50)
    args = parser.parse_args()
    tiers = [tier for tier in TIER_ORDER if tier in args.tiers]

    brands = {brand.strip().upper() for brand in args.brands} if args.brands else None
    suffix = "_" + "+".join(sorted(brands)).replace(" ", "-") if brands else ""

    valid_codes = load_valid_atc_codes()
    print(f"Loaded {len(valid_codes)} valid ATC codes with embeddings.")

    tier_maps = {}
    for tier in tiers:
        tier_maps[tier] = build_substance_to_codes(MATCH_FILES[tier], valid_codes)
        print(f"{tier}: {len(tier_maps[tier])} substances mapped to >=1 valid ATC code.")
    resolved = build_resolved_substances(tier_maps, tiers)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapped_path = args.output_dir / f"sales_prepared_long_mapped{suffix}.csv"
    if brands:
        print(f"Restricting to brands: {sorted(brands)}")
    print(f"Mapping sales rows into {mapped_path.name} ...")
    product_sales, cache = map_sales_file(resolved, mapped_path, brands)

    products = build_product_table(product_sales, cache)
    make_coverage_summary(products, tiers).to_csv(
        args.output_dir / f"atc_coverage_summary{suffix}.csv", index=False
    )
    products[["product", "match_type", "atc_codes", "n_atc_codes", "n_components", "n_mapped_components"]].to_csv(
        args.output_dir / f"product_atc_map{suffix}.csv", index=False
    )
    (
        products[products["match_type"] == "none"]
        .sort_values("sales_units", ascending=False)
        .head(args.top_unmapped)[["product", "sales_units"]]
        .to_csv(args.output_dir / f"unmapped_top{suffix}.csv", index=False)
    )

    n_mapped = int((products["match_type"] != "none").sum())
    print(f"Mapped {n_mapped}/{len(products)} products. Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
