"""Elastic Net probing classifier with a shuffled-embedding (random-feature) baseline.

The project's only probing classifier: the cross-method probe of Q2.3, run on all nine
configurations (Lorentz first projected into the tangent space at the origin). No PCA is
applied — the text configs are probed at native width.

Target = ATC level-1 therapeutic group (14-class). The baseline trains an identical probe
on row-shuffled embeddings using the SAME cross-validation splits, so real and baseline
Macro-F1 are directly comparable; ΔF1 = real - baseline.

Reproducibility: the saga solver is not run to convergence (max_iter=200, tol=1e-2), so
scores reproduce to within roughly ±0.004 across runs. No reported ordering depends on
differences of that size.

NOTE ON THE BASELINE (cite carefully): this is a shuffled-embedding / random-feature
baseline, NOT Hewitt & Liang's (2019) control task. H&L assign random LABELS and
measure probe capacity (can the probe fit noise?). Here we shuffle the EMBEDDINGS,
breaking the embedding↔label correspondence, which yields a chance-level feature
baseline (does real input beat random input?). Cite H&L only as motivation for the
selectivity idea, not for this implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn import __version__ as _SKLEARN_VERSION
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_SEED = 42
_N_FOLDS = 5
_SKLEARN_MAJOR_MINOR = tuple(int(p) for p in _SKLEARN_VERSION.split(".")[:2])


@dataclass
class ProbeResult:
    real_f1: float
    baseline_f1: float
    delta_f1: float
    flagged: bool  # True when delta_f1 <= 0 (probe adds nothing over shuffled control)


def _make_probe() -> LogisticRegression:
    """Elastic Net logistic regression, saga solver.

    The version guard matters both ways: sklearn >=1.8 warns if `penalty` is passed,
    while older versions silently degrade to plain L2 unless it is.
    """
    kwargs = dict(solver="saga", l1_ratio=0.5, C=1.0, max_iter=200, tol=1e-2)
    if _SKLEARN_MAJOR_MINOR < (1, 8):
        kwargs["penalty"] = "elasticnet"
    return LogisticRegression(**kwargs)


def _cv_macro_f1(x: np.ndarray, y: np.ndarray, splits: list[tuple]) -> float:
    """Mean Macro-F1 over the given fixed CV splits (features standardized per fold)."""
    scores = []
    for train_idx, test_idx in splits:
        scaler = StandardScaler().fit(x[train_idx])
        x_tr, x_te = scaler.transform(x[train_idx]), scaler.transform(x[test_idx])
        clf = _make_probe().fit(x_tr, y[train_idx])
        pred = clf.predict(x_te)
        scores.append(f1_score(y[test_idx], pred, average="macro", zero_division=0))
    return float(np.mean(scores))


def run_probe(x: np.ndarray, y: np.ndarray) -> ProbeResult:
    """Run the probe and its baseline on identical CV splits.

    The baseline permutes the embedding rows, breaking the embedding-label correspondence
    while keeping the label distribution and the splits — so the difference isolates what
    the representation contributes.
    """
    x = np.asarray(x, dtype=np.float64)
    kf = StratifiedKFold(n_splits=_N_FOLDS, shuffle=True, random_state=_SEED)
    splits = list(kf.split(x, y))

    real_f1 = _cv_macro_f1(x, y, splits)

    rng = np.random.default_rng(_SEED)
    x_shuffled = x[rng.permutation(len(x))]
    baseline_f1 = _cv_macro_f1(x_shuffled, y, splits)

    delta = real_f1 - baseline_f1
    return ProbeResult(real_f1, baseline_f1, delta, flagged=delta <= 0)