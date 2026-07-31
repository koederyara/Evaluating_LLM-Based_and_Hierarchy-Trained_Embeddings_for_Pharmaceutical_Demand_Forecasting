"""Persist the PCA-reduced embeddings that the forecasting main set uses.

The forecasting harness reduces its text configs to MAIN_DIM by PCA at runtime and
never writes them out, so the vectors the extrinsic results actually stand on exist
only inside a run. This script materialises them next to the native .npz files, which
lets the intrinsic evaluation read the exact same vectors instead of re-deriving them
(sklearn picks the randomized SVD solver at this width, and its random projection
depends on row order, so an independent re-derivation differs by ~1e-3 per coordinate).

Reduction is delegated to forecasting.embeddings.load_code_vectors -- the same function
the harness calls -- so the exported vectors are identical by construction rather than
by a reimplementation that could drift.

Only the main set (MAIN_DIM = 256) is exported. The dimension-ablation configs are
deliberately left out: the intrinsic loader prefers a persisted file when one exists,
so exporting d10 files would silently move already-reported matched-dimension numbers.

Run:
    python scripts/export_reduced_embeddings.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))
from forecasting.config import EMBEDDING_CONFIGS, EmbeddingConfig  # noqa: E402
from forecasting.embeddings import load_code_vectors  # noqa: E402


def reduced_path(cfg: EmbeddingConfig) -> Path:
    """Sibling of the native .npz, tagged with the PCA width it was reduced to."""
    return cfg.path.with_name(f"{cfg.path.stem}_pca{cfg.dim}.npz")


def export(cfg: EmbeddingConfig) -> Path:
    code_to_vec = load_code_vectors(cfg)
    codes = sorted(code_to_vec)
    path = reduced_path(cfg)
    np.savez(path, embeddings=np.stack([code_to_vec[c] for c in codes]),
             class_ids=np.array(codes, dtype=object))
    return path


def main() -> None:
    targets = [c for c in EMBEDDING_CONFIGS if c.kind == "npz" and c.dim is not None]
    for cfg in targets:
        path = export(cfg)
        print(f"  {cfg.name:22s} -> {path.relative_to(Path.cwd())}  (dim={cfg.dim})")
    print(f"\nExported {len(targets)} reduced embedding files.")


if __name__ == "__main__":
    main()
