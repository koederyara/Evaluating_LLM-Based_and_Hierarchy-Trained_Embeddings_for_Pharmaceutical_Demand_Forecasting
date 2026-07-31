"""Q2.4 at three widths, to separate representational quality from representational width.

The arms answer different questions: "native" is each config at full width, "matched"
equalises against Lorentz's d=10, and "forecast_dim" is the operating point of the
extrinsic results — the arm that holds encoder width fixed across RQ1 and RQ3.

Run scripts/export_reduced_embeddings.py first, otherwise the forecast_dim arm silently
re-derives its own reduction instead of reading the vectors the forecasting run used.

Run:
    python src/evaluation/dimension_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.utils import (
    LORENTZ_KEY,
    cosine_distance,
    isotropy_score,
    load_atc_codes,
    load_embeddings,
    lorentz_distance,
    save_csv,
    stratified_tree_distance_correlation,
)

MATCHED_DIM = 10
FORECAST_DIM = 256
NATIVE_CONFIGS = [
    "te3_preferred_label", "te3_atc_path", "te3_label_path",
    "sapbert_atc_path", "sapbert_label_path", LORENTZ_KEY,
]
# Lorentz is natively at MATCHED_DIM, so it has no separate reduced variant.
MATCHED_CONFIGS = [f"{k}_d{MATCHED_DIM}" for k in NATIVE_CONFIGS if k != LORENTZ_KEY]
# The width the forecasting main set runs at, so RQ1<->RQ2 can also be compared at the
# operating point of the extrinsic results, not only at Lorentz's width. Restricted to the
# configs the forecasting main set actually carries -- the others have no exported file.
FORECAST_CONFIGS = [f"{k}_d{FORECAST_DIM}"
                    for k in ("te3_preferred_label", "te3_atc_path", "sapbert_atc_path")]


def evaluate(config_key: str, codes: list[str]) -> dict:
    emb = load_embeddings(config_key)
    dist_fn = lorentz_distance if config_key == LORENTZ_KEY else cosine_distance
    stdc = stratified_tree_distance_correlation(emb, codes, distance_fn=dist_fn)
    # Not meaningful on raw hyperboloid coordinates, so text configs only.
    isotropy = float("nan") if config_key == LORENTZ_KEY else isotropy_score(emb)
    return {"config_key": config_key, "dim": emb.shape[1],
            "stdc_rho": stdc["rho"], "isotropy_rv": isotropy,
            "n_small": stdc["n_small"], "n_medium": stdc["n_medium"],
            "n_large": stdc["n_large"]}


def main() -> None:
    codes = list(load_atc_codes())
    rows = []
    for arm, keys in (("native", NATIVE_CONFIGS), ("forecast_dim", FORECAST_CONFIGS),
                      ("matched", MATCHED_CONFIGS)):
        for key in keys:
            row = evaluate(key, codes) | {"arm": arm}
            rows.append(row)
            print(f"  [{arm:12s}] {key:26s} dim={row['dim']:4d}  STDC rho={row['stdc_rho']:+.4f}  "
                  f"isotropy={row['isotropy_rv']:.4f}")
    df = pd.DataFrame(rows)[["arm", "config_key", "dim", "stdc_rho", "isotropy_rv",
                             "n_small", "n_medium", "n_large"]]
    path = save_csv(df, "dimension_ablation_intrinsic.csv")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
