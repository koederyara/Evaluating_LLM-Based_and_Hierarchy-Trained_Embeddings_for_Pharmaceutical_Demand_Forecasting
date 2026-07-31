"""Run the ablation: Floor + each fitted model x embedding config x fold.

The lag features are built once per fold and only the embedding block is swapped, so the
embedding really is the only thing that differs between a learner's runs. Imputed test
quarters are excluded from scoring: they may inform lag features, but no model is
evaluated against a value that was invented.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecasting.config import EMBEDDING_CONFIGS, EmbeddingConfig
from forecasting.data import inverse_signed_log, load_cohort
from forecasting.embeddings import load_code_vectors
from forecasting.features import FoldMatrix, build_fold
from forecasting.metrics import mase, weighted_mean
from forecasting.models import FITTED_MODELS
from forecasting.splits import fold_cutoffs, holdout_products


def _embedding_block(series_id: np.ndarray, sid_to_atc: dict, code_to_vec: dict, dim: int) -> np.ndarray:
    zero = np.zeros(dim, dtype=np.float32)
    uniq, inv = np.unique(series_id, return_inverse=True)
    per_series = np.vstack([code_to_vec.get(sid_to_atc[s], zero) for s in uniq]).astype(np.float32)
    return per_series[inv]


def _score(test: FoldMatrix, pred_units: np.ndarray, scales: dict, volumes: dict, histories: dict,
           holdout: set, model: str, config: str, fold: str) -> list[dict]:
    records = []
    observed = (~test.is_imputed) & np.isfinite(test.y_true_units)
    for sid in np.unique(test.series_id):
        mask = observed & (test.series_id == sid)
        if not mask.any() or not np.isfinite(scales[sid]):
            continue
        records.append(dict(
            model=model, config=config, fold=fold, series_id=sid,
            mase=mase(test.y_true_units[mask], pred_units[mask], scales[sid]),
            volume=volumes[sid], history=histories[sid],
            unseen=sid.split(" | ", 1)[0] in holdout, n=int(mask.sum()),
        ))
    return records


def run(configs: list[EmbeddingConfig] = EMBEDDING_CONFIGS,
        model_names: tuple[str, ...] = ("ridge", "gbm"),
        cutoffs: list | None = None,
        leave_product_out: bool = False, holdout_frac: float = 0.2,
        progress: bool = True) -> pd.DataFrame:
    df = load_cohort()
    sid_to_atc = df.drop_duplicates("series_id").set_index("series_id")["resolved_atc"].to_dict()
    vecs = {c.name: (None if c.kind == "none" else load_code_vectors(c)) for c in configs}
    dims = {c.name: (0 if c.kind == "none" else len(next(iter(vecs[c.name].values())))) for c in configs}
    cutoffs = cutoffs if cutoffs is not None else fold_cutoffs()
    holdout = holdout_products(df["product"].unique(), frac=holdout_frac) if leave_product_out else set()
    if holdout:
        print(f"leave-product-out: {len(holdout)} products held out of training (scored as unseen)")

    records: list[dict] = []
    bar = tqdm(total=len(cutoffs) * len(configs) * len(model_names),
               desc="fit x config x fold", unit="fit", disable=not progress)
    for cutoff in cutoffs:
        fold = str(cutoff)
        train, test, scales, volumes, histories = build_fold(df, cutoff)
        train_product = np.array([s.split(" | ", 1)[0] for s in train.series_id])
        keep = ~np.isin(train_product, list(holdout))  # all True when no holdout
        tqdm.write(f"[fold {fold}] train {train.X.shape} (fit rows {int(keep.sum())}), test {test.X.shape}")

        floor_pred = inverse_signed_log(test.y_seasonal)
        records += _score(test, floor_pred, scales, volumes, histories, holdout, "floor", "none", fold)

        for cfg in configs:
            if cfg.kind == "none":
                Xtr, Xte = train.X, test.X
            else:
                Etr = _embedding_block(train.series_id, sid_to_atc, vecs[cfg.name], dims[cfg.name])
                Ete = _embedding_block(test.series_id, sid_to_atc, vecs[cfg.name], dims[cfg.name])
                Xtr, Xte = np.hstack([train.X, Etr]), np.hstack([test.X, Ete])
            for name in model_names:
                bar.set_postfix_str(f"{fold} {name} x {cfg.name}")
                model = FITTED_MODELS[name]().fit(Xtr[keep], train.y[keep])
                pred = inverse_signed_log(model.predict(Xte))
                records += _score(test, pred, scales, volumes, histories, holdout, name, cfg.name, fold)
                bar.update(1)
    bar.close()
    return pd.DataFrame(records)


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, config), d in records.groupby(["model", "config"]):
        rows.append(dict(model=model, config=config,
                         mase_mean=weighted_mean(d["mase"].to_numpy()),
                         mase_median=float(np.median(d["mase"].to_numpy())),
                         mase_wmean=weighted_mean(d["mase"].to_numpy(), d["volume"].to_numpy()),
                         n_series_folds=len(d)))
    summary = pd.DataFrame(rows)
    baseline = summary[summary["config"] == "none"].set_index("model")["mase_mean"]
    summary["delta_mase_vs_none"] = summary.apply(
        lambda r: r["mase_mean"] - baseline.get(r["model"], np.nan), axis=1)
    return summary.sort_values(["model", "mase_mean"]).reset_index(drop=True)


def summarize_strata(records: pd.DataFrame, cold_max: int = 8) -> pd.DataFrame:
    """Split by history length — a static covariate should help most where the series'
    own history carries least information."""
    r = records.copy()
    r["stratum"] = np.where(r["history"] <= cold_max, "cold", "mature")
    rows = []
    for (model, config, stratum), d in r.groupby(["model", "config", "stratum"]):
        rows.append(dict(model=model, config=config, stratum=stratum,
                         mase_mean=weighted_mean(d["mase"].to_numpy()),
                         mase_median=float(np.median(d["mase"].to_numpy())),
                         mase_wmean=weighted_mean(d["mase"].to_numpy(), d["volume"].to_numpy()),
                         n_series_folds=len(d)))
    strata = pd.DataFrame(rows)
    base = strata[strata["config"] == "none"].set_index(["model", "stratum"])["mase_mean"]
    strata["delta_mase_vs_none"] = strata.apply(
        lambda r: r["mase_mean"] - base.get((r["model"], r["stratum"]), np.nan), axis=1)
    return strata.sort_values(["stratum", "model", "mase_mean"]).reset_index(drop=True)


def summarize_unseen(records: pd.DataFrame) -> pd.DataFrame:
    """Split by whether the product was held out of training.

    For held-out products the embedding is the only product-level signal the model ever
    saw. Their own history is still in the test features, so this measures generalisation
    to an unseen product, not a launch without history.
    """
    rows = []
    for (model, config, unseen), d in records.groupby(["model", "config", "unseen"]):
        rows.append(dict(model=model, config=config, unseen=bool(unseen),
                         mase_mean=weighted_mean(d["mase"].to_numpy()),
                         mase_median=float(np.median(d["mase"].to_numpy())),
                         mase_wmean=weighted_mean(d["mase"].to_numpy(), d["volume"].to_numpy()),
                         n_series_folds=len(d)))
    unseen = pd.DataFrame(rows)
    base = unseen[unseen["config"] == "none"].set_index(["model", "unseen"])["mase_mean"]
    unseen["delta_mase_vs_none"] = unseen.apply(
        lambda r: r["mase_mean"] - base.get((r["model"], r["unseen"]), np.nan), axis=1)
    return unseen.sort_values(["unseen", "model", "mase_mean"]).reset_index(drop=True)
