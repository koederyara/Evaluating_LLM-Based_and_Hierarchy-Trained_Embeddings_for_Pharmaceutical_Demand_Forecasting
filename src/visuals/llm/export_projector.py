"""
Export embeddings as TSV files for the TensorFlow Embedding Projector.

Upload the output files at https://projector.tensorflow.org:
  - <format>_tensors.tsv   -> "Load" button under "Vectors"
  - <format>_metadata.tsv  -> "Load" button under "Metadata"

Output is written to data/exports/<embedder>/ where <embedder> is the parent
directory of the input file (e.g. data/embeddings/openai/... -> data/exports/openai/).

Usage:
  python src/visuals/llm/export_projector.py data/embeddings/openai/preferred_label_embeddings.npz
  python src/visuals/llm/export_projector.py data/embeddings/sapbert/atc_path_embeddings.npz
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/
from config import DATA_ATC_PREPARED, ATC_GROUPS, COL_CLASS_ID, COL_ATC_LEVEL, EXPORTS_DIR
from embeddings_utils import load_embeddings

import numpy as np
import pandas as pd


def build_metadata(class_ids: np.ndarray, texts: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"class_id": class_ids, "label": texts})

    df_atc = pd.read_csv(DATA_ATC_PREPARED, usecols=[COL_CLASS_ID, COL_ATC_LEVEL])
    df_atc = df_atc.rename(columns={COL_CLASS_ID: "class_id", COL_ATC_LEVEL: "atc_level"})
    df_atc["atc_level"] = df_atc["atc_level"].fillna(0).astype(int)
    df = df.merge(df_atc, on="class_id", how="left")

    df["atc_group"] = df["class_id"].str[0].str.upper().map(ATC_GROUPS).fillna("Unknown")

    return df[["class_id", "label", "atc_level", "atc_group"]]


def export(npz_path: str, outdir: Path) -> None:
    print(f"Loading {npz_path} ...")
    d = load_embeddings(npz_path)
    n, dim = d["embeddings"].shape
    print(f"  {n} points, {dim} dims, model={d['model']}, format={d['format']}")

    outdir.mkdir(parents=True, exist_ok=True)
    prefix = Path(npz_path).stem.removesuffix("_embeddings")

    tensors_path = outdir / f"{prefix}_tensors.tsv"
    np.savetxt(tensors_path, d["embeddings"].astype(np.float32), delimiter="\t", fmt="%.6f")
    print(f"  Written: {tensors_path}")

    metadata_path = outdir / f"{prefix}_metadata.tsv"
    build_metadata(d["class_ids"], d["texts"]).to_csv(metadata_path, sep="\t", index=False)
    print(f"  Written: {metadata_path}")

    print(f"\nDone. Upload both files at https://projector.tensorflow.org")
    print(f"  Vectors  -> {tensors_path}")
    print(f"  Metadata -> {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export embeddings for projector.tensorflow.org.")
    parser.add_argument("file", help="Path to .npz embeddings file")
    parser.add_argument(
        "--base-outdir", default=str(EXPORTS_DIR),
        help=f"Base output directory (default: {EXPORTS_DIR.name}/); embedder name is appended",
    )
    args = parser.parse_args()

    embedder = Path(args.file).parent.name
    outdir = Path(args.base_outdir) / embedder
    export(args.file, outdir)


if __name__ == "__main__":
    main()
