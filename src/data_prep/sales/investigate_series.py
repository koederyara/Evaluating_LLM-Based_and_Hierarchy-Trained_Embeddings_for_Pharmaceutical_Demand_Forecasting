"""
Investigate the sales series at the row (series) level.

The prepared long table converts blank cells to zero, so the blank-vs-zero
distinction is lost there. This script reads the *raw* wide export and keeps the
three states apart per cell: blank (no value), a literal recorded zero, and a
real number. From the resulting per-series matrices it derives:

  - blank states and where blanks sit (leading / trailing / internal gaps),
  - whether a literal 0 ever occurs, and whether any series is fully empty,
  - intermittency of demand (ADI / CV2, Syntetos-Boylan classes),
  - the value level before vs. after internal gaps (impute-vs-zero evidence),
  - the longest run of consecutive blanks per series,
  - the distribution of active history length per series,
  - quarterly seasonality (seasonal indices and lag-4 autocorrelation),
  - sales concentration across series (Gini, top-k shares),
  - where and how the negative entries occur.

A "series" is one raw row = one product-dosage-brand-market combination (the raw
export has no duplicate keys, so one row is one series).

Run:
    python src/data_prep/sales/investigate_series.py

Output is written to data/sales/exploration/.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import SALES_EXPLORATION_SERIES_DIR, SALES_RAW_EXPORT
from prepare_sales import (
    DIMENSIONS,
    FIXED_COLUMN_MAP,
    SALES_COLUMNS,
    build_quarter_column_map,
    read_table,
    remove_embedded_metadata_rows,
    validate_fixed_columns,
)

FIXED_COLUMN_MAP_INV = {value: key for key, value in FIXED_COLUMN_MAP.items()}
SMOOTH_ADI = 1.32  # Syntetos-Boylan cut-offs
SMOOTH_CV2 = 0.49


def parse_sales_keep_blank(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse one raw sales column without collapsing blanks to zero.

    Returns the numeric value (NaN where blank or invalid) and a blank mask.
    Uses the same comma/digit cleaning as the main pipeline so the numbers match.
    """
    text = series.astype("string").str.strip()
    blank_mask = series.isna() | text.isna() | (text == "")
    cleaned = text.str.replace(r"[^\d,\-]", "", regex=True).str.replace(",", "", regex=False)
    value = pd.to_numeric(cleaned, errors="coerce")
    value[blank_mask] = np.nan
    return value, blank_mask


def build_matrices(raw_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (values, blank) matrices of shape (n_series, 48) in quarter order."""
    values = np.empty((len(raw_df), len(SALES_COLUMNS)), dtype="float64")
    blank = np.empty((len(raw_df), len(SALES_COLUMNS)), dtype=bool)
    for col_index, col in enumerate(SALES_COLUMNS):
        value, blank_mask = parse_sales_keep_blank(raw_df[col])
        values[:, col_index] = value.to_numpy()
        blank[:, col_index] = blank_mask.to_numpy()
    return values, blank


# --- cell states (Q1 / Q2) -------------------------------------------------

def build_cell_overview(values: np.ndarray, blank: np.ndarray) -> pd.DataFrame:
    total = values.size
    blank_cells = int(blank.sum())
    recorded = ~blank
    rows = [
        ("series", blank.shape[0]),
        ("quarters", blank.shape[1]),
        ("total_cells", total),
        ("blank_cells", blank_cells),
        ("blank_pct", blank_cells / total),
        ("recorded_zero_cells", int(((values == 0) & recorded).sum())),
        ("positive_cells", int(((values > 0) & recorded).sum())),
        ("negative_cells", int(((values < 0) & recorded).sum())),
        ("fully_empty_series", int(blank.all(axis=1).sum())),
        ("series_first_active_after_start", int((blank[:, 0] & recorded.any(axis=1)).sum())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


# --- where the blanks sit (Q3) ---------------------------------------------

def classify_blank_patterns(blank: np.ndarray) -> np.ndarray:
    present = ~blank
    any_present = present.any(axis=1)
    n_quarters = blank.shape[1]

    first_idx = present.argmax(axis=1)
    last_idx = n_quarters - 1 - present[:, ::-1].argmax(axis=1)
    total_present = present.sum(axis=1)
    span = last_idx - first_idx + 1
    internal_blank = span - total_present
    leading_blank = first_idx
    trailing_blank = n_quarters - 1 - last_idx

    labels = np.full(blank.shape[0], "complete", dtype=object)
    labels[~any_present] = "all_blank"
    has_lead = (leading_blank > 0) & any_present
    has_trail = (trailing_blank > 0) & any_present
    has_internal = (internal_blank > 0) & any_present

    labels[has_internal] = "internal_gaps"
    labels[has_lead & ~has_trail & ~has_internal] = "leading_only"
    labels[has_trail & ~has_lead & ~has_internal] = "trailing_only"
    labels[has_lead & has_trail & ~has_internal] = "leading_and_trailing"
    return labels


def summarize_blank_patterns(labels: np.ndarray) -> pd.DataFrame:
    return (
        pd.Series(labels, name="pattern")
        .value_counts()
        .rename_axis("pattern")
        .reset_index(name="n_series")
        .assign(pct=lambda d: d["n_series"] / len(labels))
    )


def first_active_quarter_distribution(blank: np.ndarray) -> pd.DataFrame:
    present = ~blank
    any_present = present.any(axis=1)
    first_idx = present.argmax(axis=1)
    labels = list(build_quarter_column_map().values())
    first = pd.Series([labels[i] for i in first_idx[any_present]], name="first_active_quarter")
    return (
        first.value_counts()
        .rename_axis("first_active_quarter")
        .reset_index(name="n_series")
        .sort_values("first_active_quarter")
        .reset_index(drop=True)
    )


# --- intermittency (#1) ----------------------------------------------------

def intermittency_summary(values: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """ADI / CV2 over each series' active span, classified Syntetos-Boylan.

    A demand is a positive sale. ADI and CV2 are computed between the first and
    last demand so that late market entry is not mistaken for intermittency.
    """
    positive = values > 0  # NaN compares False, so blanks count as no demand
    n_demands = positive.sum(axis=1)
    has_demand = n_demands > 0
    n_quarters = values.shape[1]

    first = positive.argmax(axis=1)
    last = n_quarters - 1 - positive[:, ::-1].argmax(axis=1)
    span = np.where(has_demand, last - first + 1, 0)

    adi = np.full(values.shape[0], np.nan)
    adi[has_demand] = span[has_demand] / n_demands[has_demand]

    masked = np.where(positive, values, np.nan)
    mean_d = np.full(values.shape[0], np.nan)
    std_d = np.full(values.shape[0], np.nan)
    mean_d[has_demand] = np.nanmean(masked[has_demand], axis=1)
    std_d[has_demand] = np.nanstd(masked[has_demand], axis=1)
    cv2 = (std_d / mean_d) ** 2

    klass = np.full(values.shape[0], "no_demand", dtype=object)
    low_adi = adi < SMOOTH_ADI
    low_cv2 = cv2 < SMOOTH_CV2
    klass[has_demand & low_adi & low_cv2] = "smooth"
    klass[has_demand & low_adi & ~low_cv2] = "erratic"
    klass[has_demand & ~low_adi & low_cv2] = "intermittent"
    klass[has_demand & ~low_adi & ~low_cv2] = "lumpy"

    summary = (
        pd.Series(klass, name="demand_class")
        .value_counts()
        .rename_axis("demand_class")
        .reset_index(name="n_series")
        .assign(pct=lambda d: d["n_series"] / len(klass))
    )
    summary["median_adi"] = [
        float(np.nanmedian(adi[klass == c])) if (klass == c).any() else np.nan
        for c in summary["demand_class"]
    ]
    summary["median_cv2"] = [
        float(np.nanmedian(cv2[klass == c])) if (klass == c).any() else np.nan
        for c in summary["demand_class"]
    ]
    return summary, adi, cv2


# --- level before vs. after internal gaps (#2) -----------------------------

def internal_gap_level_summary(values: np.ndarray, blank: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """For series with internal gaps, compare the recorded value before and
    after each gap. A similar level on both sides suggests the blanks are
    missing data rather than a genuine drop to zero."""
    present = ~blank
    rows = np.nonzero(labels == "internal_gaps")[0]
    pre_vals: list[float] = []
    post_vals: list[float] = []
    for r in rows:
        idx = np.nonzero(present[r])[0]
        gaps = np.nonzero(np.diff(idx) > 1)[0]
        for g in gaps:
            pre_vals.append(values[r, idx[g]])
            post_vals.append(values[r, idx[g + 1]])

    pre = np.array(pre_vals, dtype="float64")
    post = np.array(post_vals, dtype="float64")
    both_positive = (pre > 0) & (post > 0)
    ratio = np.full(pre.shape, np.nan)
    ratio[both_positive] = post[both_positive] / pre[both_positive]
    similar = np.sum((ratio >= 0.5) & (ratio <= 2.0))

    rows_out = [
        ("internal_gaps_total", int(pre.size)),
        ("gaps_both_sides_positive", int(both_positive.sum())),
        ("gaps_post_near_pre_0p5_2x", int(similar)),
        ("gaps_post_near_pre_pct", float(similar / both_positive.sum()) if both_positive.any() else 0.0),
        ("median_pre_value", float(np.nanmedian(pre)) if pre.size else 0.0),
        ("median_post_value", float(np.nanmedian(post)) if post.size else 0.0),
        ("median_post_over_pre_ratio", float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else 0.0),
    ]
    return pd.DataFrame(rows_out, columns=["metric", "value"])


# --- longest blank run (#3) ------------------------------------------------

def longest_blank_run_summary(blank: np.ndarray) -> pd.DataFrame:
    run = np.zeros(blank.shape[0])
    best = np.zeros(blank.shape[0])
    for c in range(blank.shape[1]):
        run = np.where(blank[:, c], run + 1, 0)
        best = np.maximum(best, run)
    present = ~blank
    leading = present.argmax(axis=1).astype(float)
    leading[~present.any(axis=1)] = blank.shape[1]

    quantiles = np.percentile(best, [50, 75, 90, 95, 99, 100])
    rows = [
        ("mean_longest_blank_run", float(best.mean())),
        ("series_with_no_blanks", int((best == 0).sum())),
        ("series_longest_run_ge_4", int((best >= 4).sum())),
        ("series_longest_run_ge_8", int((best >= 8).sum())),
        ("median_leading_blank_run", float(np.median(leading))),
    ]
    rows.extend(
        (f"p{p:02d}_longest_blank_run", float(v))
        for p, v in zip([50, 75, 90, 95, 99, 100], quantiles)
    )
    return pd.DataFrame(rows, columns=["metric", "value"])


# --- active history length (#4) --------------------------------------------

def history_length_summary(values: np.ndarray, blank: np.ndarray) -> pd.DataFrame:
    n_quarters = blank.shape[1]
    recorded = (~blank).sum(axis=1)
    positive = (values > 0).sum(axis=1)
    table = pd.DataFrame({"n_quarters": np.arange(n_quarters + 1)})
    table["n_series_recorded"] = np.bincount(recorded, minlength=n_quarters + 1)
    table["n_series_positive"] = np.bincount(positive, minlength=n_quarters + 1)
    return table


# --- seasonality (#5) ------------------------------------------------------

def seasonality_summary(values: np.ndarray) -> pd.DataFrame:
    labels = list(build_quarter_column_map().values())
    quarter_of_year = np.array([int(label.split("Q")[1]) for label in labels])
    filled = np.nan_to_num(values)  # blanks -> 0 for aggregate volume

    totals = filled.sum(axis=0)
    overall_mean = totals.mean()
    rows = []
    for q in (1, 2, 3, 4):
        q_mean = totals[quarter_of_year == q].mean()
        rows.append((f"seasonal_index_Q{q}", float(q_mean / overall_mean) if overall_mean else 0.0))

    acf1 = row_autocorrelation(filled, lag=1)
    acf4 = row_autocorrelation(filled, lag=4)
    rows.append(("median_series_acf_lag1", float(np.nanmedian(acf1))))
    rows.append(("median_series_acf_lag4", float(np.nanmedian(acf4))))
    rows.append(("series_acf_lag4_gt_0p2_pct", float(np.nanmean(acf4 > 0.2))))
    return pd.DataFrame(rows, columns=["metric", "value"])


def row_autocorrelation(matrix: np.ndarray, lag: int) -> np.ndarray:
    a = matrix[:, :-lag]
    b = matrix[:, lag:]
    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)
    numerator = (a_centered * b_centered).sum(axis=1)
    denominator = np.sqrt((a_centered ** 2).sum(axis=1) * (b_centered ** 2).sum(axis=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.where(denominator > 0, numerator / denominator, np.nan)


# --- concentration (#6) ----------------------------------------------------

def concentration_summary(values: np.ndarray) -> pd.DataFrame:
    totals = np.clip(np.nan_to_num(values).sum(axis=1), 0, None)
    rows = [
        ("n_series", int(totals.size)),
        ("gini_series_sales", float(gini(totals))),
    ]
    sorted_desc = np.sort(totals)[::-1]
    cumulative = np.cumsum(sorted_desc)
    grand_total = cumulative[-1] if cumulative.size else 0.0
    for share in (0.01, 0.05, 0.10, 0.20):
        k = max(1, int(round(share * totals.size)))
        top_share = cumulative[k - 1] / grand_total if grand_total else 0.0
        rows.append((f"top_{int(share * 100)}pct_series_sales_share", float(top_share)))
    return pd.DataFrame(rows, columns=["metric", "value"])


def gini(values: np.ndarray) -> float:
    if values.size == 0 or values.sum() == 0:
        return 0.0
    sorted_values = np.sort(values)
    n = sorted_values.size
    index = np.arange(1, n + 1)
    return float((2 * (index * sorted_values).sum()) / (n * sorted_values.sum()) - (n + 1) / n)


# --- negatives (#7) --------------------------------------------------------

def negatives_summary(raw_df: pd.DataFrame, values: np.ndarray, blank: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    negative = (values < 0) & ~blank
    rows_idx, cols_idx = np.nonzero(negative)
    n_negative = rows_idx.size

    follows_positive = 0
    if n_negative:
        prev_cols = cols_idx - 1
        valid_prev = prev_cols >= 0
        prev_values = values[rows_idx[valid_prev], prev_cols[valid_prev]]
        follows_positive = int(np.nansum(prev_values > 0))

    neg_values = values[rows_idx, cols_idx] if n_negative else np.array([])
    overview = pd.DataFrame(
        [
            ("negative_cells", int(n_negative)),
            ("negative_cells_pct", float(n_negative / values.size)),
            ("series_with_any_negative", int(np.unique(rows_idx).size)),
            ("negatives_following_positive_quarter", follows_positive),
            ("negatives_following_positive_pct", float(follows_positive / n_negative) if n_negative else 0.0),
            ("min_negative_value", float(neg_values.min()) if n_negative else 0.0),
            ("median_negative_value", float(np.median(neg_values)) if n_negative else 0.0),
        ],
        columns=["metric", "value"],
    )

    markets = raw_df[FIXED_COLUMN_MAP_INV["market"]].to_numpy()
    by_market = (
        pd.Series(markets[rows_idx], name="market")
        .value_counts()
        .rename_axis("market")
        .reset_index(name="negative_cells")
        if n_negative
        else pd.DataFrame(columns=["market", "negative_cells"])
    )
    return overview, by_market


# --- blank alignment across brands within a product x market ----------------

def component_blank_alignment(raw_df: pd.DataFrame, blank: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Do the brand/dosage components of a product x market go blank in the *same*
    quarters? A merged blank only survives when ALL components are blank together.
    This quantifies how many surviving merged blanks come from single-component
    groups (merge cannot help) vs. aligned multi-component groups."""
    product_col = FIXED_COLUMN_MAP_INV["product"]
    market_col = FIXED_COLUMN_MAP_INV["market"]
    records = []
    for (product, market), idx in raw_df.groupby([product_col, market_col], sort=False).indices.items():
        sub = blank[idx]
        present_any = (~sub).any(axis=0)
        positions = np.nonzero(present_any)[0]
        if positions.size == 0:
            continue
        first, last = positions[0], positions[-1]
        span = sub[:, first : last + 1]
        all_blank = span.all(axis=0)
        some_blank = span.any(axis=0)
        records.append(
            {
                "product": product,
                "market": market,
                "n_components": len(idx),
                "span": int(last - first + 1),
                "merged_blank_quarters": int(all_blank.sum()),
                "partial_blank_quarters": int((some_blank & ~all_blank).sum()),
            }
        )
    per = pd.DataFrame(records)
    per["single_component"] = per["n_components"] == 1

    single = per["single_component"]
    summary = pd.DataFrame(
        [
            ("product_market_groups", len(per)),
            ("single_component_groups", int(single.sum())),
            ("multi_component_groups", int((~single).sum())),
            ("merged_blank_quarters_total", int(per["merged_blank_quarters"].sum())),
            ("merged_blank_from_single_component", int(per.loc[single, "merged_blank_quarters"].sum())),
            ("merged_blank_from_multi_component", int(per.loc[~single, "merged_blank_quarters"].sum())),
            ("resolved_by_merge_partial_blank_quarters", int(per["partial_blank_quarters"].sum())),
        ],
        columns=["metric", "value"],
    )
    return per, summary


def inspect_product_market(raw_df: pd.DataFrame, product: str, market: str | None) -> None:
    """Print a brand x quarter blank pattern for one product (optionally one
    market), so aligned blanks across brands are visible directly."""
    product_col = FIXED_COLUMN_MAP_INV["product"]
    market_col = FIXED_COLUMN_MAP_INV["market"]
    brand_col = FIXED_COLUMN_MAP_INV["brand"]
    dosage_col = FIXED_COLUMN_MAP_INV["dosage"]

    mask = raw_df[product_col].astype("string").str.upper() == product.upper()
    if market:
        mask &= raw_df[market_col].astype("string").str.upper() == market.upper()
    sub = raw_df.loc[mask].reset_index(drop=True)
    if sub.empty:
        print(f"No rows for product={product!r} market={market!r}")
        return

    _, blank = build_matrices(sub)
    labels = list(build_quarter_column_map().values())
    ruler_chars = [" "] * len(labels)
    for j, label in enumerate(labels):  # place the 2-digit year over its Q1/Q2 columns
        if label.endswith("Q1"):
            ruler_chars[j] = label[2]
            if j + 1 < len(labels):
                ruler_chars[j + 1] = label[3]
    ruler = "".join(ruler_chars)
    print(f"\nproduct={product}  market={market or 'ALL'}  components={len(sub)}  (# = value, . = blank)")
    print(f"{'brand':<22.22} {'dosage':<12.12} | {ruler} | present")
    for i in range(len(sub)):
        pattern = "".join("." if blank[i, j] else "#" for j in range(len(labels)))
        print(f"{str(sub.loc[i, brand_col]):<22.22} {str(sub.loc[i, dosage_col]):<12.12} | {pattern} | {int((~blank[i]).sum())}")
    merged_present = (~blank).any(axis=0)
    merged_pattern = "".join("#" if merged_present[j] else "." for j in range(len(labels)))
    print(f"{'= MERGED':<22} {'':<12} | {merged_pattern} | {int(merged_present.sum())}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Investigate the sales series (blank-aware).")
    parser.add_argument("--inspect-product", help="Drill down: print brand x quarter blanks for this product.")
    parser.add_argument("--inspect-market", help="Restrict the drill-down to this market.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not SALES_RAW_EXPORT.exists():
        raise FileNotFoundError(f"Raw export not found: {SALES_RAW_EXPORT}")

    print(f"Reading raw export: {SALES_RAW_EXPORT}")
    raw_df = read_table(SALES_RAW_EXPORT, None)
    validate_fixed_columns(raw_df)
    raw_df, removed = remove_embedded_metadata_rows(raw_df)
    print(f"Series (rows): {len(raw_df):,}  (removed {removed} metadata row(s))")

    if args.inspect_product:
        inspect_product_market(raw_df, args.inspect_product, args.inspect_market)
        return

    values, blank = build_matrices(raw_df)
    labels = classify_blank_patterns(blank)

    cell_overview = build_cell_overview(values, blank)
    pattern_summary = summarize_blank_patterns(labels)
    first_dist = first_active_quarter_distribution(blank)
    intermittency, _, _ = intermittency_summary(values)
    gap_levels = internal_gap_level_summary(values, blank, labels)
    longest_runs = longest_blank_run_summary(blank)
    history = history_length_summary(values, blank)
    seasonality = seasonality_summary(values)
    concentration = concentration_summary(values)
    negatives, negatives_by_market = negatives_summary(raw_df, values, blank)
    alignment_per, alignment_summary = component_blank_alignment(raw_df, blank)

    print("\n== cell states ==")
    print(cell_overview.to_string(index=False))
    print("\n== blank patterns ==")
    print(pattern_summary.to_string(index=False))
    print("\n== demand intermittency (Syntetos-Boylan) ==")
    print(intermittency.to_string(index=False))
    print("\n== level before vs. after internal gaps ==")
    print(gap_levels.to_string(index=False))
    print("\n== seasonality ==")
    print(seasonality.to_string(index=False))
    print("\n== concentration ==")
    print(concentration.to_string(index=False))
    print("\n== negatives ==")
    print(negatives.to_string(index=False))
    print("\n== brand blank alignment (why merged blanks survive) ==")
    print(alignment_summary.to_string(index=False))

    out_dir = SALES_EXPLORATION_SERIES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_overview.to_csv(out_dir / "blank_cell_overview.csv", index=False)
    pattern_summary.to_csv(out_dir / "blank_pattern_summary.csv", index=False)
    first_dist.to_csv(out_dir / "first_active_quarter.csv", index=False)
    intermittency.to_csv(out_dir / "series_intermittency_summary.csv", index=False)
    gap_levels.to_csv(out_dir / "internal_gap_level_summary.csv", index=False)
    longest_runs.to_csv(out_dir / "longest_blank_run_summary.csv", index=False)
    history.to_csv(out_dir / "history_length_summary.csv", index=False)
    seasonality.to_csv(out_dir / "seasonality_summary.csv", index=False)
    concentration.to_csv(out_dir / "concentration_summary.csv", index=False)
    negatives.to_csv(out_dir / "negatives_summary.csv", index=False)
    negatives_by_market.to_csv(out_dir / "negatives_by_market.csv", index=False)
    alignment_summary.to_csv(out_dir / "brand_blank_alignment_summary.csv", index=False)
    alignment_per.to_csv(out_dir / "brand_blank_alignment_per_group.csv", index=False)
    print(f"\nWrote series investigation tables to {out_dir}")


if __name__ == "__main__":
    main()
