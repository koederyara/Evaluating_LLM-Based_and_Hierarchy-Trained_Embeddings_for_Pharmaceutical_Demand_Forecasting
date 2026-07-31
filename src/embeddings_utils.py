"""
Utilities for loading and inspecting embedding files.
"""
import numpy as np

def load_embeddings(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "class_ids": data["class_ids"],
        "texts": data["texts"],
        "embeddings": data["embeddings"],
        "model": str(data["model"]),
        "format": str(data["format"]) if "format" in data else "unknown",
    }

def print_metadata(d: dict) -> None:
    n, dim = d["embeddings"].shape
    print(f"Model    : {d['model']}")
    print(f"Format   : {d['format']}")
    print(f"Rows     : {n}")
    print(f"Dim      : {dim}")
