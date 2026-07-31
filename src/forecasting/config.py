"""Central configuration for the forecasting ablation. No logic here.

Holding every knob in one file is what makes the ablation an ablation: the treatment axis
below is the only thing that varies between runs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import EMBEDDINGS_DIR, SALES_MAPPING_DIR, SALES_MODELING_DIR

# --- Cohort (the modelling subset) ---
BRAND = "TEVA"
COHORT = "A_single_vector"
SERIES_FILE = SALES_MODELING_DIR / "series_product_market_TEVA.csv"
RESOLVED_FILE = SALES_MAPPING_DIR / "product_atc_resolved.csv"
NODE_INDEX = EMBEDDINGS_DIR / "lorentz" / "node_index.json"

# --- Forecast setup ---
HORIZON = 4  # quarters ahead
SEASON = 4  # seasonal period (quarterly data)
# Last training quarter per expanding-window fold; test = the next HORIZON quarters.
FOLD_ORIGINS = ("2022Q1", "2023Q1", "2024Q1", "2025Q1")

# The treatment axis, carried over from the intrinsic evaluation: a no-embedding control,
# te3 preferred_label as a deliberately weak-intrinsic anchor, atc_path for both encoders,
# and Lorentz as the geometric reference.
@dataclass(frozen=True)
class EmbeddingConfig:
    name: str
    kind: str  # "none" | "npz" | "lorentz"
    path: Path | None = None
    hyperbolic: bool = False  # apply logarithmic map at the origin (Lorentz only)
    dim: int | None = None  # target dimension; None = full. npz reduction is always PCA.


# Native widths (te3 3072, SapBERT 768) leave the linear model ill-conditioned
# (rcond ~1e-9 over ~500k rows), so reducing keeps Ridge trustworthy and both learners on
# the same features. PCA rather than first-k truncation: truncation would assume te3 is
# Matryoshka, which OpenAI does not disclose.
MAIN_DIM = 256
_OPENAI = EMBEDDINGS_DIR / "openai"
_SAPBERT = EMBEDDINGS_DIR / "sapbert"

EMBEDDING_CONFIGS: list[EmbeddingConfig] = [
    EmbeddingConfig("none", "none"),
    EmbeddingConfig("te3_preferred_label", "npz", _OPENAI / "preferred_label_embeddings.npz", dim=MAIN_DIM),
    EmbeddingConfig("te3_atc_path", "npz", _OPENAI / "atc_path_embeddings.npz", dim=MAIN_DIM),
    EmbeddingConfig("sapbert_atc_path", "npz", _SAPBERT / "atc_path_embeddings.npz", dim=MAIN_DIM),
    EmbeddingConfig("lorentz", "lorentz", EMBEDDINGS_DIR / "lorentz" / "embeddings_dim10.npy", hyperbolic=True),
]

# Q3.4: does the benefit survive reduction to Lorentz's width? The native arm is
# ill-conditioned for Ridge (see above), which is why the ablation is run on GBM only.
MATCHED_DIM = 10

DIMENSION_CONFIGS: list[EmbeddingConfig] = [
    EmbeddingConfig("none", "none"),
    # native (te3 = 3072, SapBERT = 768, Lorentz = 10)
    EmbeddingConfig("te3_preferred_label", "npz", _OPENAI / "preferred_label_embeddings.npz"),
    EmbeddingConfig("te3_atc_path", "npz", _OPENAI / "atc_path_embeddings.npz"),
    EmbeddingConfig("sapbert_atc_path", "npz", _SAPBERT / "atc_path_embeddings.npz"),
    EmbeddingConfig("lorentz", "lorentz", EMBEDDINGS_DIR / "lorentz" / "embeddings_dim10.npy", hyperbolic=True),
    # matched dimension = MATCHED_DIM
    EmbeddingConfig("te3_preferred_label_d10", "npz", _OPENAI / "preferred_label_embeddings.npz", dim=MATCHED_DIM),
    EmbeddingConfig("te3_atc_path_d10", "npz", _OPENAI / "atc_path_embeddings.npz", dim=MATCHED_DIM),
    EmbeddingConfig("sapbert_atc_path_d10", "npz", _SAPBERT / "atc_path_embeddings.npz", dim=MATCHED_DIM),
]

# --- Output ---
RESULTS_DIR = SALES_MODELING_DIR.parent / "forecasting"
