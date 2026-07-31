"""Run the intrinsic evaluation (thesis Goals 1 and 2) and build the summary table.

Goal 1 validates each embedding family inside its own space and is split into two
independently evaluated parts; Goal 2 compares all nine configurations with rank-based,
space-agnostic metrics:

    1a  goal1a_text_internal    Q1.A1 isotropy, Q1.A2 neighbourhood precision (8 text configs)
    1b  goal1b_lorentz_internal Q1.B1 norm-rank rho, Q1.B2 HDC, Q1.B3 neighbourhood precision
    2   goal2_cross_method      Q2.1 reconstruction, Q2.2 link prediction, Q2.3 probe,
                                Q2.4 stratified tree-distance correlation

Writes goal1a_results.csv, goal1b_results.csv, goal2_results.csv (plus the Lorentz side
tables) and a combined results/summary_table.csv with every config as a row and the
applicable metrics as columns (N/A where a metric does not apply to a config).

Thesis Goal 3 (the extrinsic forecasting ablation) is a separate pipeline and lives in
src/forecasting/ — it is not run from here.

Statistical-significance note (declared, not tested): the intrinsic differences reported
here are descriptive. The only sampled intrinsic metric is Q2.4, whose seed sensitivity is
quantified in scripts/stratified_tree_distance_seed_robustness.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import save_csv
from goal1a_text_internal import run_goal1a
from goal1b_lorentz_internal import run_goal1b
from goal2_cross_method import run_goal2


def build_summary(df1a, df1b, df2) -> pd.DataFrame:
    """Merge the per-goal results into one row per configuration.

    Goal 1A is long-format, so only its level-1 rows enter; the per-level detail stays in
    goal1a_results.csv.
    """
    g1a = df1a[df1a["level"] == 1][
        ["config_key", "isotropy_rv", "precision_at_1"]].rename(
        columns={"precision_at_1": "q1a2_precision_at_1_L1"})
    g1b = df1b[["config_key", "norm_rank_spearman_rho", "hdc_score"]]
    g2 = df2[["config_key", "q21_map", "q21_mean_rank", "q22_map", "q23_delta_f1", "q24_stdc_rho"]]

    summary = g2.merge(g1a, on="config_key", how="left").merge(g1b, on="config_key", how="left")
    return summary.fillna("N/A")


_GOAL_RUNNERS = {
    "1a": ("Goal 1A — text-embedding-internal validation", run_goal1a),
    "1b": ("Goal 1B — Lorentz-internal validation", run_goal1b),
    "2": ("Goal 2 — cross-method comparison", run_goal2),
}


def main(goals: list[str]) -> None:
    results: dict[str, pd.DataFrame] = {}

    for g in ("1a", "1b", "2"):
        if g not in goals:
            continue
        title, runner = _GOAL_RUNNERS[g]
        print("\n", "=" * 60, f"\n{title}\n", "=" * 60, sep="")
        results[g] = runner()

    # Only meaningful when all three ran, since it merges across them.
    if set(results) == {"1a", "1b", "2"}:
        summary = build_summary(results["1a"], results["1b"], results["2"])
        path = save_csv(summary, "summary_table.csv")
        print(f"\nSaved {path}")
        print(summary.to_string(index=False))
    else:
        print(f"\n[note] Ran {sorted(results)}; summary_table.csv is only written when "
              f"1a, 1b and 2 run together. Per-goal CSVs were saved.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the intrinsic evaluation (thesis Goals 1 and 2; all parts or a subset).")
    parser.add_argument("--goals", nargs="+", choices=["1a", "1b", "2"],
                        default=["1a", "1b", "2"], metavar="PART",
                        help="Parts to run, e.g. --goals 1b 2 (default: all). "
                             "summary_table.csv is written only when all three run together.")
    args = parser.parse_args()
    main(args.goals)
