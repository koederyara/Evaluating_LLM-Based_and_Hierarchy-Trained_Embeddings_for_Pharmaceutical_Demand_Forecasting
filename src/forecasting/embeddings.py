"""Turn a resolved ATC code into a fixed-length static-covariate vector.

PCA is fit unsupervised on all code vectors, so the reduction never sees a forecast
target and cannot leak. Lorentz vectors need the logarithmic map first: raw hyperboloid
coordinates are not a valid Euclidean feature (Ganea et al. 2018).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RANDOM_SEED
from forecasting.config import NODE_INDEX, EmbeddingConfig


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def logmap_origin(points: np.ndarray) -> np.ndarray:
    """Lorentz logarithmic map at the origin.

    The resulting norm is the geodesic distance from the origin, so the projection keeps
    a usable depth proxy rather than discarding hierarchy level entirely.
    """
    x0 = points[:, 0]
    spatial = points[:, 1:]
    radius = np.linalg.norm(spatial, axis=1)
    distance = np.arccosh(np.clip(x0, 1.0, None))
    scale = np.divide(distance, radius, out=np.zeros_like(distance), where=radius > 0)
    return spatial * scale[:, None]


def load_code_vectors(cfg: EmbeddingConfig) -> dict[str, np.ndarray]:
    """Map ATC code -> embedding vector for one config."""
    if cfg.kind == "npz":
        data = np.load(cfg.path, allow_pickle=True)
        codes = [str(c) for c in data["class_ids"]]
        vectors = data["embeddings"]
        if cfg.dim is not None:
            vectors = PCA(n_components=cfg.dim, random_state=RANDOM_SEED).fit_transform(vectors)
            vectors = _l2_normalize(vectors)
        return dict(zip(codes, vectors))
    if cfg.kind == "lorentz":
        index = json.loads(NODE_INDEX.read_text(encoding="utf-8"))
        tangent = logmap_origin(np.load(cfg.path))
        return {code: tangent[row] for code, row in index.items()}
    raise ValueError(f"Cannot load vectors for kind {cfg.kind!r}")


def build_series_matrix(resolved_atc: list[str], cfg: EmbeddingConfig) -> np.ndarray:
    """Embedding matrix (n_series, dim) aligned to the given resolved ATC codes.

    A code without a vector (should not happen for the mapped cohort) becomes a
    zero row so the pipeline never breaks on a lookup miss.
    """
    if cfg.kind == "none":
        return np.zeros((len(resolved_atc), 0), dtype=np.float32)
    code_to_vec = load_code_vectors(cfg)
    dim = len(next(iter(code_to_vec.values())))
    zero = np.zeros(dim, dtype=np.float32)
    return np.vstack([code_to_vec.get(code, zero) for code in resolved_atc]).astype(np.float32)
