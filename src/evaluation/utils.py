"""Shared loading and config registry for the intrinsic evaluation (Goals 1 and 2).

Metric maths lives in metrics.py and is re-exported here, so the goal drivers have a
single import surface and cannot drift onto different implementations.
"""
from __future__ import annotations

import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

_SRC = Path(__file__).resolve().parents[1]  # src/
sys.path.insert(0, str(_SRC))

from config import EMBEDDINGS_DIR, DATA_ATC_PREPARED, COL_CLASS_ID, LORENTZ_DIM, RANDOM_SEED
from atc_utils import get_atc_level, group_label

from metrics import (  # noqa: F401
    lorentz_distance,
    lorentz_distance_matrix,
    compute_mean_rank_and_map,
    isotropy_score,
    hdc_score,
    nn_precision_at_k,
    chance_precision,
    cosine_distance,
    pairwise_distance_matrix,
    stratified_tree_distance_correlation,
)

# The eight text configurations = 2 encoders x 4 input formats, the RQ1 variable.
# Key -> (model_dir, format) under data/embeddings/{model}/{format}_embeddings.npz.
TEXT_CONFIGS: dict[str, tuple[str, str]] = {
    f"{prefix}_{fmt}": (model, fmt)
    for prefix, model in (("te3", "openai"), ("sapbert", "sapbert"))
    for fmt in ("preferred_label", "class_id", "atc_path", "label_path")
}

LORENTZ_KEY = "lorentz"
ALL_BASE_CONFIGS: list[str] = list(TEXT_CONFIGS) + [LORENTZ_KEY]

_LORENTZ_DIR = EMBEDDINGS_DIR / "lorentz"

# Env var rather than an argument, so the directed ablation can be evaluated by
# re-running the same drivers unchanged ("" = bidirectional main, "_directed" = ablation).
LORENTZ_VARIANT = os.environ.get("LORENTZ_VARIANT", "")


@lru_cache(maxsize=1)
def load_atc_codes() -> tuple[str, ...]:
    """The canonical code order every artefact is aligned to (STY/T rows excluded).

    A tuple so it stays hashable for the cache; call sites cast to list.
    """
    df = pd.read_csv(DATA_ATC_PREPARED)
    df = df[~df[COL_CLASS_ID].astype(str).str.startswith("T")]
    df = df.drop_duplicates(subset=[COL_CLASS_ID])
    codes = [c for c in df[COL_CLASS_ID].astype(str) if get_atc_level(c) is not None]
    return tuple(sorted(set(codes)))


def atc_level1_groups(codes: list[str] | None = None) -> np.ndarray:
    """Level-1 group per code, the probing and NN-precision target.

    Group membership rather than tree depth: depth only shows whether the level is
    linearly readable, not whether the hierarchy's structure survived.
    """
    codes = list(load_atc_codes()) if codes is None else codes
    return np.array([group_label(c, 1) for c in codes])


def load_embeddings(config_key: str) -> np.ndarray:
    """Load a config's embeddings, reindexed to the canonical code order.

    Every config is aligned the same way, which is what lets an intrinsic result and a
    forecasting result be compared per configuration at all.
    """
    codes = list(load_atc_codes())

    # "<base>_d<N>" reduces by PCA, not first-k truncation: truncation would assume te3
    # is Matryoshka, which OpenAI does not disclose. A persisted reduction wins, so the
    # intrinsic evaluation reads the very vectors the forecasting run used instead of an
    # independent re-derivation (randomized SVD differs by ~1e-3 per coordinate).
    reduced = re.fullmatch(r"(.+)_d(\d+)", config_key)
    if reduced and reduced.group(1) in TEXT_CONFIGS:
        base, dim = reduced.group(1), int(reduced.group(2))
        model, fmt = TEXT_CONFIGS[base]
        persisted = EMBEDDINGS_DIR / model / f"{fmt}_embeddings_pca{dim}.npz"
        if persisted.exists():
            data = np.load(persisted, allow_pickle=True)
            return _reindex(data["embeddings"], list(map(str, data["class_ids"])), codes)
        emb = load_embeddings(base)
        return l2_normalize(PCA(n_components=dim, random_state=RANDOM_SEED).fit_transform(emb))

    if config_key == LORENTZ_KEY:
        return _load_lorentz(codes)

    if config_key not in TEXT_CONFIGS:
        raise KeyError(f"Unknown config key: {config_key!r}. Valid: {ALL_BASE_CONFIGS}")

    model, fmt = TEXT_CONFIGS[config_key]
    path = EMBEDDINGS_DIR / model / f"{fmt}_embeddings.npz"
    data = np.load(path, allow_pickle=True)
    return _reindex(data["embeddings"], list(map(str, data["class_ids"])), codes)


def _load_lorentz(codes: list[str], suffix: str = "") -> np.ndarray:
    """Load Lorentz embeddings; suffix='_train90' picks the split model Q2.2 requires."""
    emb_path = _LORENTZ_DIR / f"embeddings_dim{LORENTZ_DIM}{suffix}{LORENTZ_VARIANT}.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Lorentz embeddings not found: {emb_path}. "
            f"Train them first: python src/lorentz_training.py --dim {LORENTZ_DIM}"
            + (f" --train-split {suffix.lstrip('_train')}/100" if suffix else "")
        )
    emb = np.load(emb_path)
    node_index = json.loads((_LORENTZ_DIR / "node_index.json").read_text())
    order = np.array([node_index[c] for c in codes])
    return emb[order]


def _reindex(emb: np.ndarray, source_codes: list[str], target_codes: list[str]) -> np.ndarray:
    """Reorder embedding rows from source_codes order to target_codes order."""
    pos = {c: i for i, c in enumerate(source_codes)}
    missing = [c for c in target_codes if c not in pos]
    if missing:
        raise KeyError(f"{len(missing)} canonical codes missing from embedding file, e.g. {missing[:3]}")
    return emb[np.array([pos[c] for c in target_codes])]


def log_map_origin(x: np.ndarray) -> np.ndarray:
    """Project onto the Euclidean tangent space at the origin.

    Needed wherever hyperbolic vectors must be fed to a Euclidean method (the Q2.3 probe,
    the forecasting learners). Raises on non-finite output rather than propagating it.
    """
    x = np.atleast_2d(x)
    x0 = np.clip(x[:, 0], 1.0, None)
    spatial = x[:, 1:]
    sp_norm = np.linalg.norm(spatial, axis=1, keepdims=True)
    safe = sp_norm > 0
    scale = np.where(safe, np.arccosh(x0)[:, None] / np.where(safe, sp_norm, 1.0), 0.0)
    out = spatial * scale
    if not np.all(np.isfinite(out)):
        n_bad = int(np.sum(~np.isfinite(out)))
        raise ValueError(f"log_map_origin produced {n_bad} non-finite values — check Lorentz embeddings.")
    return out


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization. Zero rows are returned unchanged (no division by zero)."""
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms > 0, norms, 1.0)


RESULTS_DIR = _SRC.parent / "results"


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    """Write a results DataFrame to results/<filename> and return the path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    df.to_csv(path, index=False)
    return path
