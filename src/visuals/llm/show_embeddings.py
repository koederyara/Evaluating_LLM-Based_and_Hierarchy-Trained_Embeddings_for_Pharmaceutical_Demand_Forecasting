"""
Inspect a saved embeddings .npz file.

Usage:
  python src/visuals/llm/show_embeddings.py data/embeddings/openai/preferred_label_embeddings.npz
  python src/visuals/llm/show_embeddings.py data/embeddings/openai/atc_path_embeddings.npz --head 5
  python src/visuals/llm/show_embeddings.py data/embeddings/openai/preferred_label_embeddings.npz --code D11AX06
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/
from embeddings_utils import load_embeddings, print_metadata


def print_row(idx: int, class_id: str, text: str, embedding: np.ndarray) -> None:
    preview = ", ".join(f"{v:.4f}" for v in embedding[:8])
    print(f"[{idx}] {class_id}")
    print(f"  text      : {text}")
    print(f"  embedding : [{preview}, ...]  (dim={len(embedding)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a saved embeddings .npz file.")
    parser.add_argument("file", help="Path to .npz embeddings file")
    parser.add_argument("--head", type=int, metavar="N", help="Show first N embeddings")
    parser.add_argument("--code", metavar="ATC_CODE", help="Look up a specific ATC code")
    args = parser.parse_args()

    d = load_embeddings(args.file)
    print_metadata(d)

    if args.head:
        print(f"\n--- First {args.head} embeddings ---")
        for i in range(min(args.head, len(d["class_ids"]))):
            print_row(i, d["class_ids"][i], d["texts"][i], d["embeddings"][i])

    if args.code:
        matches = np.where(d["class_ids"] == args.code)[0]
        if len(matches) == 0:
            print(f"\nCode '{args.code}' not found.")
        else:
            print(f"\n--- ATC code: {args.code} ---")
            for i in matches:
                print_row(i, d["class_ids"][i], d["texts"][i], d["embeddings"][i])


if __name__ == "__main__":
    main()
