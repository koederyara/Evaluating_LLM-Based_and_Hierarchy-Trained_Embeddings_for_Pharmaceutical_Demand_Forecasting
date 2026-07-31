"""
Analyze products that map to several ATC codes.

For each multi-ATC product it finds the deepest ATC level (1-4) at which *all* of
its codes share a common ancestor group:

    level 4  same 5-char group (e.g. C10AA)   -> codes very close
    level 3  same 4-char group (e.g. C10A)
    level 2  same 3-char group (e.g. C10)
    level 1  same 1-letter group (e.g. C)     -> same anatomical main group
    level 0  differ at level 1                -> cross-branch (e.g. A/B/N)

This shows for which products pooling the ATC embeddings is well-founded (codes in
the same sub-tree) versus questionable (codes far apart). It is cross-tabulated
against match_type so combination products and multi-use single substances can be
told apart.

Run:
    python src/data_prep/sales/analyze_multi_atc.py

Without a subset it also *resolves* every product to a single-vector ATC
representation (ancestor-collapse -> LCA node) and assigns a cohort
(A_single_vector / A_crossbranch / B_combo / unmapped).

Output (data/prepared_data/sales/mapping/):
    multi_atc_level_summary_<suffix>.csv   common-level distribution x match_type
    multi_atc_detail_<suffix>.csv          one row per multi-ATC product
    product_atc_resolved.csv               per-product resolved code + cohort (all-products run)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import SALES_EXPLORATION_DATASET_DIR, SALES_MAPPING_DIR, SALES_PREPARED_LONG

PREFIX_LENGTH = {1: 1, 2: 3, 3: 4, 4: 5}  # ATC prefix length per level


def common_level(codes: list[str]) -> int:
    """Deepest ATC level (1-4) at which all codes share a prefix; 0 if none."""
    for level in (4, 3, 2, 1):
        prefixes = {code[: PREFIX_LENGTH[level]] for code in codes}
        if len(prefixes) == 1:
            return level
    return 0


def ancestor_collapse(codes: list[str]) -> list[str]:
    """Drop codes that are an ancestor of another code (A12A ⊃ A12AA -> keep A12AA).
    All codes are valid ATC nodes, so a shorter prefix is a true ancestor."""
    unique = sorted(set(codes))
    return [c for c in unique if not any(d != c and d.startswith(c) and len(d) > len(c) for d in unique)]


def resolve_products(mapping: pd.DataFrame) -> pd.DataFrame:
    """Resolve each product to a single-vector ATC representation where possible.

    Per product: ancestor-collapse the codes, then if one code remains use it, else
    if the codes share an ancestor at level >= 2 use that LCA node, else it is
    cross-branch (no single vector). Cohort split by molecule count (n_components):
    single molecule -> Part A, real combination -> Part B.
    """
    rows: list[dict] = []
    for _, r in mapping.iterrows():
        raw = [c for c in str(r["atc_codes"]).split("|") if c] if pd.notna(r["atc_codes"]) else []
        n_components = int(r["n_components"])
        if not raw or r["match_type"] == "none":
            rows.append({"product": r["product"], "match_type": r["match_type"],
                         "n_components": n_components, "resolved_atc": "", "representation": "none",
                         "common_level": 0, "single_vector_ok": False, "cohort": "unmapped"})
            continue
        collapsed = ancestor_collapse(raw)
        if len(collapsed) == 1:
            resolved, representation, level = collapsed[0], "single", 5
        else:
            level = common_level(collapsed)
            if level >= 2:
                resolved, representation = collapsed[0][: PREFIX_LENGTH[level]], "lca"
            else:
                resolved, representation = "", "crossbranch"
        single_vector_ok = representation in {"single", "lca"}
        cohort = "B_combo" if n_components > 1 else ("A_single_vector" if single_vector_ok else "A_crossbranch")
        rows.append({"product": r["product"], "match_type": r["match_type"],
                     "n_components": n_components, "codes_collapsed": "|".join(collapsed),
                     "n_collapsed": len(collapsed), "common_level": level, "resolved_atc": resolved,
                     "representation": representation, "single_vector_ok": single_vector_ok,
                     "cohort": cohort})
    return pd.DataFrame(rows)


def load_global_volume() -> pd.Series | None:
    path = SALES_EXPLORATION_DATASET_DIR / "sales_by_product.csv"
    if not path.exists():
        return None
    by_product = pd.read_csv(path, usecols=["product", "sales_units"])
    return by_product.set_index("product")["sales_units"]


def load_subset_volume(subset_type: str, name: str) -> pd.Series:
    """Volume per product within one brand/market, from the prepared long table.
    Its index is the set of products that occur in the subset."""
    print(f"Reading products for {subset_type}={name} from {SALES_PREPARED_LONG.name} ...")
    long_df = pd.read_csv(SALES_PREPARED_LONG, usecols=["product", subset_type, "sales_units"])
    match = long_df[subset_type].astype("string").str.upper() == name.upper()
    long_df = long_df[match]
    if long_df.empty:
        raise SystemExit(f"No rows for {subset_type}={name!r}.")
    return long_df.groupby("product")["sales_units"].sum()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze products mapping to several ATC codes, optionally within one brand/market."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--brand", help="Restrict to products sold by this brand.")
    group.add_argument("--market", help="Restrict to products sold in this market.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping_path = SALES_MAPPING_DIR / "product_atc_map.csv"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping not found: {mapping_path}. Run map_products_to_atc.py first.")
    mapping = pd.read_csv(mapping_path)

    subset_type = "brand" if args.brand else "market" if args.market else None
    if subset_type:
        name = args.brand or args.market
        volume = load_subset_volume(subset_type, name)
        mapping = mapping[mapping["product"].isin(volume.index)].copy()
        suffix = f"{subset_type}_{name.upper().replace(' ', '_')}"
        scope = f"{subset_type}={name}"
    else:
        volume = load_global_volume()
        suffix = "all"
        scope = "all products"

    multi = mapping[mapping["n_atc_codes"] > 1].copy()
    multi["codes"] = multi["atc_codes"].str.split("|")
    multi["common_level"] = multi["codes"].apply(common_level)
    multi["common_prefix"] = [
        codes[0][: PREFIX_LENGTH[level]] if level else ""
        for codes, level in zip(multi["codes"], multi["common_level"])
    ]
    if volume is not None:
        multi["volume"] = multi["product"].map(volume).fillna(0.0)

    print(f"Scope: {scope}")
    print(f"Multi-ATC products: {len(multi)} of {len(mapping)} "
          f"({len(multi) / len(mapping):.1%})")

    level_counts = (
        multi["common_level"].value_counts().rename_axis("common_level").reset_index(name="n_products")
        .sort_values("common_level", ascending=False).reset_index(drop=True)
    )
    if volume is not None:
        vol_by_level = multi.groupby("common_level")["volume"].sum()
        total_vol = vol_by_level.sum()
        level_counts["volume_share"] = level_counts["common_level"].map(vol_by_level) / total_vol

    crosstab = pd.crosstab(multi["common_level"], multi["match_type"])

    print("\n== common ATC level across a product's codes ==")
    print(level_counts.to_string(index=False))
    print("\n== common_level x match_type ==")
    print(crosstab.to_string())

    out_dir = SALES_MAPPING_DIR
    summary = crosstab.reset_index()
    summary.to_csv(out_dir / f"multi_atc_level_summary_{suffix}.csv", index=False)
    detail_cols = ["product", "match_type", "n_atc_codes", "atc_codes", "common_level", "common_prefix"]
    if volume is not None:
        detail_cols.append("volume")
    multi[detail_cols].sort_values(["common_level", "n_atc_codes"]).to_csv(
        out_dir / f"multi_atc_detail_{suffix}.csv", index=False
    )
    print(f"\nWrote multi_atc_level_summary_{suffix}.csv and multi_atc_detail_{suffix}.csv to {out_dir}")

    if subset_type is None:  # resolution is a per-product property -> run on all products
        resolved = resolve_products(mapping)
        resolved.to_csv(out_dir / "product_atc_resolved.csv", index=False)
        print("\n== product resolution (cohort) ==")
        print(resolved["cohort"].value_counts().to_string())
        print(f"Wrote product_atc_resolved.csv to {out_dir}")


if __name__ == "__main__":
    main()
