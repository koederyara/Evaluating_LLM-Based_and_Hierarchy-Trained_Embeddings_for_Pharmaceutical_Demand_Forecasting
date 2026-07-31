#!/usr/bin/env bash
# Stage 4 of 4 — thesis Chapter 7 "Extrinsic Evaluation" (Goal 3).
#
# NOT REPRODUCIBLE FROM PUBLIC DATA: needs the licensed IQVIA sales export and the Teva
# mapping files. Without them the script says what is missing and exits cleanly — stages
# 1-3 are unaffected and cover the whole intrinsic half of the thesis.
#
# Covers Chapter 7 end to end: the preparation chain of 7.1.2/7.1.3 (raw export -> ATC
# mapping -> cohort resolution -> Teva product x market series), optionally the App. D/E
# dataset tables, then the ablation of 7.2/7.3, which fits, scores and bootstraps in one
# pass:
#
#   main   Floor + Ridge + LightGBM x 5 configs, 4 folds, horizon 4   (Tab. 7.2, 7.3)
#   _lpo   same, with whole products held out of training             (Tab. 7.4)
#   _dim   Q3.4, native vs matched d=10. GBM only: at native width Ridge is
#          ill-conditioned and runs out of memory, so its numbers would not hold.
#
# Usage:
#     bash scripts/04_run_extrinsic.sh                     (~4-5 h + ~3 h for _dim)
#     RUN_EXPLORATION=1 ...                                (+ the App. D/E tables)
#     SKIP_SMOKE=1 ...                                     (skip the ridge-only check)
#     SKIP_PREP=1 ...                                      (series already built)

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

LOG_DIR="logs/extrinsic/$STAMP"
mkdir -p "$LOG_DIR"

RAW_SALES_DIR="data/raw/forecasting"
FORECAST_SERIES="data/prepared_data/sales/modeling/series_product_market_TEVA.csv"

echo "=== [4/4] extrinsic evaluation — forecasting (Ch. 7)  ($STAMP) ==="
echo "python: $PY"
echo

require "data/embeddings/lorentz/embeddings_dim10.npy" "Run: bash scripts/02_build_embeddings.sh"
require "data/embeddings/openai/label_path_embeddings.npz" "Run: bash scripts/02_build_embeddings.sh"

# Each step consumes the previous one, so the chain is strictly ordered.
if [[ "${SKIP_PREP:-0}" == "1" ]]; then
  require "$FORECAST_SERIES" "SKIP_PREP=1 was set, but the series file does not exist yet."
  echo "SKIP_PREP=1 -> reusing the existing modelling series."
elif compgen -G "$RAW_SALES_DIR/*.xlsx" > /dev/null; then
  run "sales_prepare"      "$PY" src/data_prep/sales/prepare_sales.py
  run "sales_map_atc"      "$PY" src/data_prep/sales/map_products_to_atc.py --tiers exact
  run "sales_resolve_atc"  "$PY" src/data_prep/sales/analyze_multi_atc.py
  run "sales_build_series" "$PY" src/data_prep/sales/build_series.py \
    --brand TEVA --keys product market --method locf --max-gap 4 --long-gap zero

  # Appendix D/E tables only; nothing downstream reads them.
  if [[ "${RUN_EXPLORATION:-0}" == "1" ]]; then
    run "explore_dataset" "$PY" src/data_prep/sales/explore_sales.py
    run "explore_series"  "$PY" src/data_prep/sales/investigate_series.py
    run "explore_subsets" "$PY" src/data_prep/sales/analyze_subsets.py --by market
    run "explore_brands"  "$PY" src/data_prep/sales/analyze_subsets.py --by brand
  fi
else
  echo "NOTE: no raw IQVIA export in $RAW_SALES_DIR/."
  echo "      The sales data is licensed and not part of this repository, so Chapter 7"
  echo "      cannot be reproduced here. Chapters 5 and 6 (stages 1-3) are unaffected."
  finish
fi

if [[ ! -f "$FORECAST_SERIES" ]]; then
  echo "ERROR: preparation did not produce $FORECAST_SERIES — see the logs above."
  finish
fi

# Ridge-only first: exercises the whole harness in minutes, so a broken pipeline fails
# before the multi-hour GBM runs. The main run overwrites its output.
if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  run "forecast_smoke_ridge" "$PY" src/forecasting/run.py --models ridge
  if [[ $FAILURES -gt 0 ]]; then
    echo "Smoke run failed -> aborting before the slow runs."
    finish
  fi
fi

run "forecast_main"    "$PY" src/forecasting/run.py
run "forecast_lpo"     "$PY" src/forecasting/run.py --leave-product-out
run "forecast_dim"     "$PY" src/forecasting/run.py --config-set dimension --models gbm
run "forecast_dim_lpo" "$PY" src/forecasting/run.py --config-set dimension --models gbm --leave-product-out

echo "results: data/prepared_data/sales/forecasting/"
finish
