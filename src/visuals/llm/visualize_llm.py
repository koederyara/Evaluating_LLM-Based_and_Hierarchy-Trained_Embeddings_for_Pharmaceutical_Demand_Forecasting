"""
Interactive 2D/3D visualization of LLM-based embedding files (.npz) using Plotly.

Usage:
  python src/visuals/llm/visualize_llm.py data/embeddings/openai/preferred_label_embeddings.npz
  python src/visuals/llm/visualize_llm.py data/embeddings/sapbert/atc_path_embeddings.npz --method tsne
  python src/visuals/llm/visualize_llm.py data/embeddings/openai/preferred_label_embeddings.npz --dims 2
  python src/visuals/llm/visualize_llm.py data/embeddings/openai/preferred_label_embeddings.npz --save plot.html
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RANDOM_SEED
from embeddings_utils import load_embeddings

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce(embeddings: np.ndarray, method: str, dims: int) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=dims).fit_transform(embeddings)
    # PCA first to 50 dims speeds up t-SNE and UMAP significantly
    pre = PCA(n_components=min(50, embeddings.shape[1])).fit_transform(embeddings)
    if method == "tsne":
        return TSNE(n_components=dims, random_state=RANDOM_SEED, init="pca").fit_transform(pre)
    if method == "umap":
        import umap

        return umap.UMAP(n_components=dims, random_state=RANDOM_SEED).fit_transform(pre)
    raise ValueError(f"Unknown reduction method: {method!r}. Choose 'pca', 'tsne', or 'umap'.")


def plot(
    coords: np.ndarray,
    class_ids: np.ndarray,
    texts: np.ndarray,
    model: str,
    method: str,
    save_path: str | None,
) -> None:
    dims = coords.shape[1]
    color_groups = [str(cid)[0].upper() if cid else "?" for cid in class_ids]
    unique_groups = sorted(set(color_groups))

    hover = [f"<b>{cid}</b><br>{txt}" for cid, txt in zip(class_ids, texts)]

    traces = []
    for group in unique_groups:
        mask = [g == group for g in color_groups]
        group_text = [h for h, m in zip(hover, mask) if m]
        if dims == 3:
            traces.append(
                go.Scatter3d(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    z=coords[mask, 2],
                    mode="markers",
                    name=f"Group {group}",
                    text=group_text,
                    hovertemplate="%{text}<extra></extra>",
                    marker=dict(size=3, opacity=0.7),
                )
            )
        else:
            traces.append(
                go.Scatter(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    mode="markers",
                    name=f"Group {group}",
                    text=group_text,
                    hovertemplate="%{text}<extra></extra>",
                    marker=dict(size=5, opacity=0.7),
                )
            )

    fig = go.Figure(traces)
    layout = dict(
        title=f"Embeddings — {model} | {method.upper()} ({dims}D)",
        legend_title="ATC group",
        margin=dict(l=0, r=0, b=0, t=40),
    )
    if dims == 3:
        layout["scene"] = dict(
            xaxis_title=f"{method.upper()}-1",
            yaxis_title=f"{method.upper()}-2",
            zaxis_title=f"{method.upper()}-3",
        )
    else:
        layout["xaxis_title"] = f"{method.upper()}-1"
        layout["yaxis_title"] = f"{method.upper()}-2"
    fig.update_layout(**layout)

    if save_path:
        fig.write_html(save_path)
        print(f"Saved to {save_path}")
    else:
        fig.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize LLM embeddings in 2D/3D with Plotly.")
    parser.add_argument("file", help="Path to .npz embeddings file")
    parser.add_argument(
        "--method",
        choices=["pca", "tsne", "umap"],
        default="pca",
        help="Dimensionality reduction method (default: pca)",
    )
    parser.add_argument(
        "--dims",
        type=int,
        choices=[2, 3],
        default=3,
        help="Number of dimensions to project to (default: 3)",
    )
    parser.add_argument("--save", metavar="PATH", help="Save plot as HTML instead of opening browser")
    args = parser.parse_args()

    print(f"Loading {args.file} ...")
    d = load_embeddings(args.file)
    n, dim = d["embeddings"].shape
    print(f"  {n} points, {dim} dims, model={d['model']}")

    print(f"Reducing to {args.dims}D via {args.method.upper()} ...")
    coords = reduce(d["embeddings"].astype(np.float32), args.method, args.dims)

    plot(coords, d["class_ids"], d["texts"], d["model"], args.method, args.save)


if __name__ == "__main__":
    main()
