"""CLI: run the full forecasting ablation (RQ2) and write results to disk.

Runs the Floor plus every fitted model x embedding config across all rolling-origin
folds, scores per series, and writes the per-series records and two summaries
(overall and cold-start-stratified) to the forecasting results folder.

Usage:
    python src/forecasting/run.py                    # floor + ridge + gbm, all configs
    python src/forecasting/run.py --models ridge     # linear rung only (fast)
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecasting.config import DIMENSION_CONFIGS, EMBEDDING_CONFIGS, RESULTS_DIR
from forecasting.harness import run, summarize, summarize_strata, summarize_unseen
from forecasting.significance import (
    bootstrap_deltas,
    bootstrap_deltas_by_unseen,
    bootstrap_config_pairs,
    bootstrap_vs_floor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the forecasting ablation (RQ2).")
    parser.add_argument("--models", nargs="+", default=["ridge", "gbm"], choices=["ridge", "gbm"])
    parser.add_argument("--config-set", choices=["main", "dimension"], default="main",
                        help="main = the treatment configs; dimension = native + matched-dim ablation.")
    parser.add_argument("--leave-product-out", action="store_true",
                        help="Hold whole products out of training (unseen-product test).")
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    args.out.mkdir(parents=True, exist_ok=True)
    configs = DIMENSION_CONFIGS if args.config_set == "dimension" else EMBEDDING_CONFIGS
    suffix = ("_dim" if args.config_set == "dimension" else "") + ("_lpo" if args.leave_product_out else "")

    started = time.time()
    records = run(configs=configs, model_names=tuple(args.models),
                  leave_product_out=args.leave_product_out, holdout_frac=args.holdout_frac)
    summary = summarize(records)
    strata = summarize_strata(records)
    # Ablation (does the embedding help this model?) and benchmark (does the model beat the
    # seasonal-naive Floor?), each unweighted and volume-weighted -- the weighting flips the
    # verdict, so both are reported.
    significance = bootstrap_deltas(records)
    significance_w = bootstrap_deltas(records, weight="volume")
    config_pairs = bootstrap_config_pairs(records)
    config_pairs_w = bootstrap_config_pairs(records, weight="volume")
    vs_floor = bootstrap_vs_floor(records)
    vs_floor_w = bootstrap_vs_floor(records, weight="volume")

    records.to_csv(args.out / f"forecast_records{suffix}.csv", index=False)
    summary.to_csv(args.out / f"forecast_summary{suffix}.csv", index=False)
    strata.to_csv(args.out / f"forecast_summary_strata{suffix}.csv", index=False)
    significance.to_csv(args.out / f"forecast_significance{suffix}.csv", index=False)
    significance_w.to_csv(args.out / f"forecast_significance_weighted{suffix}.csv", index=False)
    config_pairs.to_csv(args.out / f"forecast_config_pair_tests{suffix}.csv", index=False)
    config_pairs_w.to_csv(args.out / f"forecast_config_pair_tests_weighted{suffix}.csv", index=False)
    vs_floor.to_csv(args.out / f"forecast_vs_floor{suffix}.csv", index=False)
    vs_floor_w.to_csv(args.out / f"forecast_vs_floor_weighted{suffix}.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\n=== overall (elapsed {time.time() - started:.0f}s) ===")
    print(summary.to_string(index=False))
    print("\n=== cold-start (short history) vs mature ===")
    print(strata.to_string(index=False))
    print("\n=== ablation: delta MASE vs none (95% CI, Holm-adjusted p) ===")
    print(significance.to_string(index=False))
    print("\n=== ablation, volume-weighted ===")
    print(significance_w.to_string(index=False))
    print("\n=== direct embedding-config pair tests ===")
    print(config_pairs.to_string(index=False))
    print("\n=== direct embedding-config pair tests, volume-weighted ===")
    print(config_pairs_w.to_string(index=False))
    print("\n=== benchmark: delta MASE vs seasonal-naive Floor (negative = beats Floor) ===")
    print(vs_floor.to_string(index=False))
    print("\n=== benchmark vs Floor, volume-weighted ===")
    print(vs_floor_w.to_string(index=False))
    if args.leave_product_out:
        unseen = summarize_unseen(records)
        unseen.to_csv(args.out / f"forecast_summary_unseen{suffix}.csv", index=False)
        # The ablation tested within each half of the split. The pooled test above is
        # dominated by the seen half, so the unseen verdict needs its own test.
        by_unseen = bootstrap_deltas_by_unseen(records)
        by_unseen_w = bootstrap_deltas_by_unseen(records, weight="volume")
        by_unseen.to_csv(args.out / f"forecast_significance_by_unseen{suffix}.csv", index=False)
        by_unseen_w.to_csv(
            args.out / f"forecast_significance_weighted_by_unseen{suffix}.csv", index=False)
        print("\n=== unseen product (leave-product-out) vs seen ===")
        print(unseen.to_string(index=False))
        print("\n=== ablation within each half of the split (Holm within half) ===")
        print(by_unseen.to_string(index=False))
        print("\n=== ablation within each half, volume-weighted ===")
        print(by_unseen_w.to_string(index=False))
    print(f"\nWrote records + summaries to {args.out}")


if __name__ == "__main__":
    main()
