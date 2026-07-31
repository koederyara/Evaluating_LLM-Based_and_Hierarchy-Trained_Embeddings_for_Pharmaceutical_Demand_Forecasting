"""
Poincaré disk visualization of Lorentz embeddings, colored by ATC hierarchy level.

For Lorentz dim=2 the mapping is exact (Eq. 11, Nickel & Kiela 2018).
For dim>2 the Poincaré ball coordinates are reduced to 2D via PCA or t-SNE —
the plot is then a visual approximation and labelled accordingly.

Usage:
  python src/visuals/lorentz/lorentz_visual_disk.py --dim 2
  python src/visuals/lorentz/lorentz_visual_disk.py --dim 6 --reduce pca
  python src/visuals/lorentz/lorentz_visual_disk.py --dim 6 --reduce tsne
  python src/visuals/lorentz/lorentz_visual_disk.py --dim 2 --show-root --show-edges --max-level 3

Flags:
  --dim N           Lorentz embedding dimension to load (default: 2).
  --reduce {pca,tsne}  Reduction method for dim>2 (default: pca). Ignored for dim=2.
  --show-root       Also render the virtual ROOT node.
  --show-edges      Draw parent→child edges between nodes.
  --max-level N     Draw edges down to ATC level N (default: 3; level 5 ≈ 10 000 lines).

Output:
  results/lorentz/visuals/poincare_disk_dim{N}.png
  results/lorentz/visuals/poincare_disk_dim{N}_{reduce}.png   (for dim>2)
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import ROOT
from atc_utils import get_atc_level, build_atc_edges, VIRTUAL_ROOT
from lorentz_proj import project_to_disk

_EMBEDDINGS_DIR = ROOT / "data" / "embeddings" / "lorentz"
_RESULTS_DIR = ROOT / "results" / "lorentz" / "visuals"

_EDGE_STYLE: dict[int, tuple[float, float]] = {
    1: (1.2, 0.55),  # ROOT → L1
    2: (0.8, 0.35),  # L1  → L2
    3: (0.5, 0.20),  # L2  → L3
    4: (0.3, 0.12),  # L3  → L4
    5: (0.2, 0.07),  # L4  → L5
}
_LEVEL_COLORS = {1: "#e41a1c", 2: "#377eb8", 3: "#4daf4a", 4: "#984ea3", 5: "#ff7f00"}


def _draw_edges(
    ax,
    edges: list[tuple[str, str]],
    poincare: np.ndarray,
    node_index: dict[str, int],
    max_level: int,
) -> None:
    """Draw parent→child edges on the Poincaré disk, colored by child's ATC level."""
    for parent, child in edges:
        child_level = get_atc_level(child)
        if child_level is None or child_level > max_level:
            continue
        if parent not in node_index or child not in node_index:
            continue
        xy_p = poincare[node_index[parent]]
        xy_c = poincare[node_index[child]]
        color = _LEVEL_COLORS.get(child_level, "gray")
        lw, alpha = _EDGE_STYLE.get(child_level, (0.2, 0.07))
        ax.plot([xy_p[0], xy_c[0]], [xy_p[1], xy_c[1]],
                color=color, lw=lw, alpha=alpha, zorder=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Poincaré disk visualization colored by level.")
    parser.add_argument("--dim", type=int, default=2,
                        help="Lorentz embedding dimension to load (default: 2).")
    parser.add_argument("--reduce", choices=["pca", "tsne"], default="pca",
                        help="Reduction method for dim>2 embeddings (default: pca). Ignored for dim=2.")
    parser.add_argument("--show-root", action="store_true", help="Also render the virtual ROOT node.")
    parser.add_argument("--show-edges", action="store_true", help="Draw parent→child hierarchy edges.")
    parser.add_argument("--max-level", type=int, default=3,
                        help="Draw edges down to this ATC level (default: 3).")
    args = parser.parse_args()

    emb_path = _EMBEDDINGS_DIR / f"embeddings_dim{args.dim}.npy"
    if not emb_path.exists():
        sys.exit(f"Error: no embedding file found at {emb_path}")

    emb = np.load(emb_path)
    with open(_EMBEDDINGS_DIR / "node_index.json") as f:
        node_index: dict[str, int] = json.load(f)

    # Load root early so it's included in the dimensionality reduction step.
    # This is critical for t-SNE, which cannot transform unseen points after fitting.
    if args.show_root:
        root_v = np.load(_EMBEDDINGS_DIR / f"virtual_root_emb_dim{args.dim}.npy")
        emb = np.vstack([emb, root_v[np.newaxis, :]])
        node_index = {**node_index, VIRTUAL_ROOT: len(emb) - 1}

    poincare, proj_label, is_approx = project_to_disk(emb, method=args.reduce)

    # Build level bins (VIRTUAL_ROOT skipped: get_atc_level returns None)
    level_sizes  = {1: 120, 2: 40, 3: 15, 4: 6, 5: 3}
    level_zorder = {1: 5,   2: 4,  3: 3,  4: 2, 5: 1}
    by_level = {lvl: {"xy": [], "codes": []} for lvl in range(1, 6)}
    for code, idx in node_index.items():
        lvl = get_atc_level(code)
        if lvl:
            by_level[lvl]["xy"].append(poincare[int(idx)])
            by_level[lvl]["codes"].append(code)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_patch(plt.Circle((0, 0), 1.0, color="lightgray", fill=False,
                             linewidth=1.5, linestyle="--"))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- edges ---
    if args.show_edges:
        atc_codes = [c for c in node_index if c != VIRTUAL_ROOT]
        edges = build_atc_edges(atc_codes)
        if args.show_root:
            root_edges = [(VIRTUAL_ROOT, c) for c in sorted(atc_codes) if get_atc_level(c) == 1]
            edges = root_edges + edges
        _draw_edges(ax, edges, poincare, node_index, args.max_level)

    # --- scatter nodes ---
    for lvl in [5, 4, 3, 2, 1]:
        xy = np.array(by_level[lvl]["xy"])
        if len(xy) == 0:
            continue
        ax.scatter(xy[:, 0], xy[:, 1],
                   s=level_sizes[lvl], c=_LEVEL_COLORS[lvl],
                   alpha=0.7, zorder=level_zorder[lvl],
                   label=f"Level {lvl} (n={len(xy)})")

    for code, xy in zip(by_level[1]["codes"], by_level[1]["xy"]):
        ax.annotate(code, xy=(xy[0], xy[1]), fontsize=9, fontweight="bold",
                    ha="center", va="bottom", xytext=(0, 6), textcoords="offset points")

    # --- root ---
    if args.show_root:
        root_xy = poincare[node_index[VIRTUAL_ROOT]]
        ax.scatter(root_xy[0], root_xy[1], s=200, c="black", marker="D", zorder=10, label="Virtual ROOT")
        ax.annotate("ROOT", xy=(root_xy[0], root_xy[1]), fontsize=9, fontweight="bold",
                    ha="center", va="bottom", xytext=(0, 7), textcoords="offset points")

    ax.legend(loc="upper right", framealpha=0.9)
    edge_tag = f" · edges ≤ level {args.max_level}" if args.show_edges else ""
    subtitle = f"\n⚠ {proj_label}" if is_approx else ""
    ax.set_title(
        f"ATC Hierarchy — Poincaré Disk{edge_tag} (dim={args.dim}){subtitle}",
        fontsize=12, pad=15,
    )

    dim_tag = f"dim{args.dim}" if not is_approx else f"dim{args.dim}_{args.reduce}"
    out = _RESULTS_DIR / f"poincare_disk_{dim_tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
