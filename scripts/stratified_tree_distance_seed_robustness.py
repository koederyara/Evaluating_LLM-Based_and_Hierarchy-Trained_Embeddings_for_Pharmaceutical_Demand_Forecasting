"""Seed robustness of the stratified tree-distance correlation (thesis Q2.4).

The metric is computed on a sampled pair set, so its value depends on the seed. This
script quantifies that dependence: if the spread across seeds were comparable to the
gaps between configurations, the reported ranking would be an artefact of sampling.

Run:
    python scripts/stratified_tree_distance_seed_robustness.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "evaluation"))  # utils.py imports metrics.py as a top-level module
from evaluation.utils import (  # noqa: E402
    ALL_BASE_CONFIGS,
    LORENTZ_KEY,
    cosine_distance,
    load_atc_codes,
    load_embeddings,
    lorentz_distance,
    save_csv,
    stratified_tree_distance_correlation,
)

N_SEEDS = 10


def main() -> None:
    warnings.simplefilter("ignore")
    codes = list(load_atc_codes())
    rows = []
    for key in ALL_BASE_CONFIGS:
        emb = load_embeddings(key)
        dist_fn = lorentz_distance if key == LORENTZ_KEY else cosine_distance
        rhos = np.array([
            stratified_tree_distance_correlation(
                emb, codes, distance_fn=dist_fn, rng=np.random.default_rng(seed)
            )["rho"]
            for seed in range(N_SEEDS)
        ])
        rows.append({
            "config_key": key,
            "mean_rho": rhos.mean(),
            "sd_rho": rhos.std(),
            "min_rho": rhos.min(),
            "max_rho": rhos.max(),
            "spread": rhos.max() - rhos.min(),
        })
        print(f"  {key:28s} mean={rhos.mean():+.4f}  sd={rhos.std():.4f}  "
              f"spread={rhos.max() - rhos.min():.4f}")

    df = pd.DataFrame(rows)
    gaps = np.diff(np.sort(df["mean_rho"].to_numpy()))
    print(f"\nLargest sd across configs : {df['sd_rho'].max():.4f}")
    print(f"Smallest gap between configs: {gaps.min():.4f}")
    print(f"Ratio (gap / sd)            : {gaps.min() / df['sd_rho'].max():.1f}x")

    path = save_csv(df, "stratified_tree_distance_seed_robustness.csv")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
