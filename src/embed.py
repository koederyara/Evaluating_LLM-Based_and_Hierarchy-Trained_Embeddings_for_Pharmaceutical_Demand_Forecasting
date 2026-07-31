"""
CLI entry point for generating ATC embeddings.

Usage:
  python src/embed.py --method openai --format preferred_label
  python src/embed.py --method sapbert --format atc_path
  python src/embed.py --method sapbert --format label_path

Formats:
  preferred_label  Preferred Label column          e.g. "atorvastatin"
  class_id         Class ID column                 e.g. "C10AA05"
  atc_path         Full ATC hierarchy path         e.g. "Cardiovascular system drugs (C) > ... > atorvastatin (C10AA05)"
  label_path       Preferred label + hierarchy path e.g. "atorvastatin: Cardiovascular system drugs (C) > ..."

Output: data/embeddings/<method>/<format>_embeddings.npz
  - class_ids  : (N,)   ATC codes — stable row identifier
  - texts      : (N,)   exact text that was embedded
  - embeddings : (N, D) float32 embedding matrix
  - model      : ()     embedding model name
  - format     : ()     format identifier used
"""

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # src/ — config, atc_utils, models.*
from config import DATA_ATC_PREPARED as DATA_PATH, EMBEDDINGS_DIR as OUT_BASE, COL_CLASS_ID, COL_PREFERRED_LABEL
from atc_utils import build_atc_path

import numpy as np
import pandas as pd


def _texts_preferred_label(df: pd.DataFrame, _: dict) -> list[str]:
    return df[COL_PREFERRED_LABEL].astype(str).tolist()


def _texts_class_id(df: pd.DataFrame, _: dict) -> list[str]:
    return df[COL_CLASS_ID].astype(str).tolist()


def _texts_atc_path(df: pd.DataFrame, lookup: dict) -> list[str]:
    return [build_atc_path(c, lookup) for c in df[COL_CLASS_ID]]


def _texts_label_path(df: pd.DataFrame, lookup: dict) -> list[str]:
    paths = [build_atc_path(c, lookup) for c in df[COL_CLASS_ID]]
    labels = df[COL_PREFERRED_LABEL].astype(str).tolist()
    return [f"{lbl}: {path}" for lbl, path in zip(labels, paths)]


_FORMAT_FNS: dict[str, callable] = {
    "preferred_label": _texts_preferred_label,
    "class_id":        _texts_class_id,
    "atc_path":        _texts_atc_path,
    "label_path":      _texts_label_path,
}


def build_texts(df: pd.DataFrame, fmt: str) -> list[str]:
    """Build embedding input strings for the given format."""
    lookup = dict(zip(df[COL_CLASS_ID], df[COL_PREFERRED_LABEL].astype(str)))
    return _FORMAT_FNS[fmt](df, lookup)


def save(path: Path, class_ids: list[str], texts: list[str], embeddings: np.ndarray, model: str, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        class_ids=np.array(class_ids, dtype=object),
        texts=np.array(texts, dtype=object),
        embeddings=embeddings,
        model=np.str_(model),
        format=np.str_(fmt),
    )
    print(f"Saved {path}  shape={embeddings.shape}  model={model}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ATC embeddings.")
    parser.add_argument("--method", required=True, choices=["openai", "sapbert"])
    parser.add_argument("--format", required=True, choices=list(_FORMAT_FNS), dest="fmt",
                        help="Input text format")
    parser.add_argument("--data", default=str(DATA_PATH), metavar="PATH",
                        help=f"Path to input CSV (default: {DATA_PATH})")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows from {args.data}")

    sty_mask = df[COL_CLASS_ID].str.startswith("T")
    if sty_mask.any():
        print(f"Skipping {sty_mask.sum()} STY rows (Class ID prefix 'T')")
        df = df[~sty_mask].reset_index(drop=True)

    class_ids = df[COL_CLASS_ID].tolist()
    texts = build_texts(df, args.fmt)

    # Lazy import so only the required backend's dependencies are loaded
    embedder = importlib.import_module(f"models.embedders.{args.method}")
    embeddings, model_name = embedder.embed(texts)

    # Sort by class_id so evaluation code sees a consistent ordering across all models
    order = sorted(range(len(class_ids)), key=lambda i: class_ids[i])
    class_ids = [class_ids[i] for i in order]
    texts = [texts[i] for i in order]
    embeddings = embeddings[order]

    out_path = OUT_BASE / args.method / f"{args.fmt}_embeddings.npz"
    save(out_path, class_ids, texts, embeddings, model_name, args.fmt)


if __name__ == "__main__":
    main()
