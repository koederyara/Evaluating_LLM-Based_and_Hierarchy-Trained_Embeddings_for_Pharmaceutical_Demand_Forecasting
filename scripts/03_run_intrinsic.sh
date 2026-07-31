#!/usr/bin/env bash
# Stage 3 of 4 — thesis Chapter 6 "Intrinsic Evaluation" (Goals 1 and 2).
#
# Produces every number of Chapter 6 and of Appendices B.2 and C, all from public data.
#
#   Goal 1A  Q1.A1 isotropy, Q1.A2 neighbourhood precision   -> goal1a_results.csv
#   Goal 1B  Q1.B1 norm-rank rho, Q1.B2 HDC, Q1.B3 precision -> goal1b_*.csv
#   Goal 2   Q2.1-Q2.4 over all nine configs                 -> goal2_results.csv,
#                                                               summary_table.csv
#   plus radial_structure.csv, the dimension ablation, the Q2.4 seed robustness check
#   and the edge-direction ablation in results/directional/.
#
# Usage:
#     bash scripts/03_run_intrinsic.sh                     (~6-8 h, the Q2.3 probe dominates)
#     SKIP_DIRECTED=1 bash scripts/03_run_intrinsic.sh     (if stage 2 skipped those models)

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

LOG_DIR="logs/intrinsic/$STAMP"
mkdir -p "$LOG_DIR" results/bidirectional results/directional

echo "=== [3/4] intrinsic evaluation (Ch. 6)  ($STAMP) ==="
echo "python: $PY"
echo

require "data/embeddings/lorentz/embeddings_dim10.npy" "Run: bash scripts/02_build_embeddings.sh"
require "data/embeddings/openai/label_path_embeddings.npz" "Run: bash scripts/02_build_embeddings.sh"

# Listed once so the bidirectional snapshot and the directed restore cannot drift apart.
LORENTZ_CSVS=(goal1b_results.csv goal1b_nn_precision.csv goal1b_radial_by_level.csv goal2_results.csv)

# Q2.2 needs the train90 model; without it the Lorentz row is N/A rather than falling
# back to the full model, which would be leakage.
[[ -f "data/embeddings/lorentz/embeddings_dim10_train90.npy" ]] \
  || echo "WARNING: no train90 model — Q2.2 will report Lorentz as N/A."
run "intrinsic_goals_1_2" "$PY" src/evaluation/run_all.py
cp -f results/goal1a_results.csv results/summary_table.csv "${LORENTZ_CSVS[@]/#/results/}" \
      results/bidirectional/ 2>/dev/null

run "intrinsic_radial_structure" "$PY" src/evaluation/analyze_radial_structure.py --csv

# Materialised first, so the ablation reads the very vectors the forecasting run uses
# instead of re-deriving its own reduction.
run "export_reduced_embeddings" "$PY" scripts/export_reduced_embeddings.py
run "intrinsic_dimension_ablation" "$PY" src/evaluation/dimension_ablation.py
run "intrinsic_stdc_seed_robustness" "$PY" scripts/stratified_tree_distance_seed_robustness.py

# Goal 1A is text-internal and direction-independent, so only 1B and 2 are re-run.
# run_all overwrites the same CSVs, hence the move-aside-and-restore below.
if [[ "${SKIP_DIRECTED:-0}" == "1" || ! -f "data/embeddings/lorentz/embeddings_dim10_directed.npy" ]]; then
  echo "Skipping the directed ablation (no directed model, or SKIP_DIRECTED=1)."
else
  export LORENTZ_VARIANT=_directed
  run "intrinsic_goals_1b_2_directed" "$PY" src/evaluation/run_all.py --goals 1b 2
  unset LORENTZ_VARIANT
  cp -f "${LORENTZ_CSVS[@]/#/results/}" results/directional/ 2>/dev/null
  cp -f "${LORENTZ_CSVS[@]/#/results/bidirectional/}" results/ 2>/dev/null
fi

echo "results: results/  (the flat files are canonical and equal results/bidirectional/)"
echo "next   : bash scripts/04_run_extrinsic.sh"
finish
