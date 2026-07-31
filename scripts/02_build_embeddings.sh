#!/usr/bin/env bash
# Stage 2 of 4 — thesis Chapter 5.2 + 5.3 "Embedding Construction and Implementation".
#
# Builds all nine evaluated representations plus the Lorentz variants the ablations need:
#   5.2  8 text configs = 2 encoders x 4 input formats            (Tab. 5.1)
#   5.3  Lorentz dim 10 (main), dim 10 train-split 0.9 (Q2.2),
#        dim 2 (Fig. B.1), dim 10 directed + train90 (App. C.1)   (Tab. 5.2)
#
# te3 calls the OpenAI API (~6,900 texts per format, needs OPENAI_API_KEY in .env);
# SapBERT runs locally. Idempotent — a re-run overwrites the same files.
#
# Usage:
#     bash scripts/02_build_embeddings.sh                   (text ~30 min, Lorentz ~8-9 h)
#     SKIP_DIRECTED=1 bash scripts/02_build_embeddings.sh   (saves the ~15 h ablation)

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

LOG_DIR="logs/embeddings/$STAMP"
mkdir -p "$LOG_DIR"

echo "=== [2/4] embedding construction (Ch. 5.2 + 5.3)  ($STAMP) ==="
echo "python: $PY"
echo

require "data/prepared_data/data_atc.csv" "Run: bash scripts/01_prepare_data.sh"
[[ -f ".env" ]] || echo "WARNING: no .env found — the openai runs will fail without OPENAI_API_KEY."

# The format axis is RQ1's independent variable, so all four are always built.
for method in openai sapbert; do
  for fmt in preferred_label class_id atc_path label_path; do
    run "text_${method}_${fmt}" "$PY" src/embed.py --method "$method" --format "$fmt"
  done
done

# Defaults follow Nickel & Kiela (2018), including the undirected closure of the paper.
run "lorentz_dim10_full"    "$PY" src/lorentz_training.py --dim 10
run "lorentz_dim10_train90" "$PY" src/lorentz_training.py --dim 10 --train-split 0.9

# Trained separately so the figures show real 2D geometry, not a projection of dim 10.
run "lorentz_dim2" "$PY" src/lorentz_training.py --dim 2

# The _directed suffix keeps the ablation from overwriting the main model.
if [[ "${SKIP_DIRECTED:-0}" == "1" ]]; then
  echo "SKIP_DIRECTED=1 -> skipping the directed ablation (stage 3 will skip it too)."
else
  run "lorentz_dim10_full_directed"    "$PY" src/lorentz_training.py --dim 10 --directed
  run "lorentz_dim10_train90_directed" "$PY" src/lorentz_training.py --dim 10 --train-split 0.9 --directed
fi

# Produced here rather than in stage 3 because they read the dim-2 model.
mkdir -p results/lorentz/visuals
run "figure_poincare_disk"          "$PY" src/visuals/lorentz/lorentz_visual_disk.py --dim 2
run "figure_poincare_disk_by_group" "$PY" src/visuals/lorentz/lorentz_visual_disk_groups.py --dim 2

echo "outputs: data/embeddings/{openai,sapbert,lorentz}/, results/lorentz/visuals/"
echo "next   : bash scripts/03_run_intrinsic.sh  (then 04_run_extrinsic.sh)"
finish
