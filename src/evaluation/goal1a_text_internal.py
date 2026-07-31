"""Goal 1, Part A — intra-space validation of the text embeddings (Q1.A1, Q1.A2).

A validation gate only: it establishes that each text space carries non-random grouping
signal at all. Substantive hierarchy preservation is decided in Goal 2, which is also
where the project's single probing classifier lives — probing the text configs a second
time here would answer nothing the cross-method probe does not already answer.

Isotropy is descriptive: it selects no configuration and carries no downstream claim.
But a strongly anisotropic space distorts distance-based metrics, so a low value has to
be declared when reading the Goal 2 distance results.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atc_utils import group_label
from utils import (
    TEXT_CONFIGS,
    chance_precision,
    cosine_distance,
    isotropy_score,
    l2_normalize,
    load_atc_codes,
    load_embeddings,
    nn_precision_at_k,
    pairwise_distance_matrix,
    save_csv,
)

_KS = [1, 5, 10]
# Level 4 has hundreds of classes below _MIN_CLASS and level 5 is all singletons, so
# neither supports a meaningful chance baseline.
_LEVELS = [1, 2, 3]
# Below this, precision@k is dominated by how many same-class codes exist at all rather
# than by neighbourhood quality, which makes the chance baseline unstable.
_MIN_CLASS = 5


def _level_mask_labels(codes: list[str], level: int, min_class: int = _MIN_CLASS):
    """Per-level group labels plus a mask dropping undefined and under-sized classes."""
    labels = [group_label(c, level) for c in codes]
    counts = Counter(l for l in labels if l is not None)
    mask = np.array([l is not None and counts[l] >= min_class for l in labels])
    y = np.array([l if l is not None else "" for l in labels])
    return mask, y


def run_goal1a(config_keys: list[str] | None = None,
               out_name: str = "goal1a_results.csv") -> pd.DataFrame:
    """Run Goal 1A over config_keys (default: all eight native text configs).

    config_keys accepts the "_d<N>" reduced variants; pass a separate out_name so a run
    at the forecasting width never lands in the canonical result file.
    """
    codes = list(load_atc_codes())
    rows = []

    for key in (config_keys or list(TEXT_CONFIGS)):
        emb = load_embeddings(key)
        iso_rv = isotropy_score(emb)

        # Computed once and subset per level: cosine distance between two vectors does
        # not depend on the other rows, so subsetting is exact.
        dist_full = pairwise_distance_matrix(
            l2_normalize(emb).astype("float32"), cosine_distance)

        for level in _LEVELS:
            mask, y = _level_mask_labels(codes, level)
            y_valid = y[mask]

            dist = dist_full[np.ix_(mask, mask)]
            prec_chance = chance_precision(y_valid)
            prec = nn_precision_at_k(dist, y_valid, _KS)

            rows.append({
                "config_key": key,
                "level": level,
                "n_codes": int(mask.sum()),
                "n_classes": int(len(set(y_valid))),
                "isotropy_rv": iso_rv,
                "precision_chance": prec_chance,
                **{f"precision_at_{k}": prec[k] for k in _KS},
            })
            print(f"  {key:24s} L{level}  I_rv={iso_rv:.3f}  "
                  f"P@1={prec[1]:.3f} (chance={prec_chance:.3f})  "
                  f"[{int(mask.sum())} codes, {len(set(y_valid))} classes]")

        # Checkpoint so a kill mid-run keeps the configs already finished.
        save_csv(pd.DataFrame(rows), out_name)

    df = pd.DataFrame(rows)
    save_csv(df, out_name)
    return df


if __name__ == "__main__":
    from cli import parse_config_subset_args

    args = parse_config_subset_args(
        "Goal 1A - text-embedding-internal validation (per ATC level 1-3).",
        default_out="goal1a_results.csv")
    print("Goal 1A - text-embedding-internal validation (per ATC level 1-3)")
    df = run_goal1a(args.configs, args.out)
    print(f"\nSaved results/{args.out} ({len(df)} config-level rows)")
