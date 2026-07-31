"""
Poincaré disk visualization of Lorentz embeddings, colored by Level-1 ATC group.

For Lorentz dim=2 the mapping is exact (Eq. 11, Nickel & Kiela 2018).
For dim>2 the Poincaré ball coordinates are reduced to 2D via PCA or t-SNE —
the plot is then a visual approximation and labelled accordingly.

Usage:
  python src/visuals/lorentz/lorentz_visual_disk_groups.py --dim 2
  python src/visuals/lorentz/lorentz_visual_disk_groups.py --dim 6 --reduce pca
  python src/visuals/lorentz/lorentz_visual_disk_groups.py --dim 6 --reduce tsne
  python src/visuals/lorentz/lorentz_visual_disk_groups.py --dim 2 --show-root --show-edges --max-level 3

Flags:
  --dim N           Lorentz embedding dimension to load (default: 2).
  --reduce {pca,tsne}  Reduction method for dim>2 (default: pca). Ignored for dim=2.
  --show-root       Also render the virtual ROOT node.
  --show-edges      Draw parent→child edges between nodes.
  --max-level N     Draw edges down to ATC level N (default: 3; level 5 ≈ 10 000 lines).

Output:
  results/lorentz/visuals/poincare_disk_by_group_dim{N}.png
  results/lorentz/visuals/poincare_disk_by_group_dim{N}_{reduce}.png   (for dim>2)
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import ROOT, ATC_GROUPS
from atc_utils import get_atc_level, build_atc_edges, VIRTUAL_ROOT
from lorentz_proj import project_to_disk

_EMBEDDINGS_DIR = ROOT / "data" / "embeddings" / "lorentz"
_RESULTS_DIR = ROOT / "results" / "lorentz" / "visuals"

COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3",
]
group_color = {g: COLORS[i] for i, g in enumerate(ATC_GROUPS)}

_EDGE_STYLE: dict[int, tuple[float, float]] = {
    1: (1.2, 0.55),  # ROOT → L1
    2: (0.8, 0.35),  # L1  → L2
    3: (0.5, 0.20),  # L2  → L3
    4: (0.3, 0.12),  # L3  → L4
    5: (0.2, 0.07),  # L4  → L5
}


def _draw_edges(
    ax,
    edges: list[tuple[str, str]],
    poincare: np.ndarray,
    node_index: dict[str, int],
    max_level: int,
) -> None:
    """Draw parent→child edges on the Poincaré disk, colored by child's ATC group."""
    for parent, child in edges:
        child_level = get_atc_level(child)
        if child_level is None or child_level > max_level:
            continue
        if parent not in node_index or child not in node_index:
            continue
        xy_p = poincare[node_index[parent]]
        xy_c = poincare[node_index[child]]
        g = child[0].upper()
        color = group_color.get(g, "gray")
        lw, alpha = _EDGE_STYLE.get(child_level, (0.2, 0.07))
        ax.plot([xy_p[0], xy_c[0]], [xy_p[1], xy_c[1]],
                color=color, lw=lw, alpha=alpha, zorder=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Poincaré disk visualization colored by group.")
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

    # Build group bins (VIRTUAL_ROOT skipped: first char not in ATC_GROUPS)
    by_group = {g: {"xy": [], "codes": []} for g in ATC_GROUPS}
    roots = {"xy": [], "codes": [], "groups": []}

    for code, idx in node_index.items():
        g = str(code).strip().upper()[0] if code else None
        if g not in ATC_GROUPS:
            continue
        lvl = get_atc_level(code)
        xy = poincare[int(idx)]
        if lvl == 1:
            roots["xy"].append(xy)
            roots["codes"].append(code)
            roots["groups"].append(g)
        else:
            by_group[g]["xy"].append(xy)
            by_group[g]["codes"].append(code)

    fig, ax = plt.subplots(figsize=(13, 13))
    ax.add_patch(plt.Circle((0, 0), 1.0, color="lightgray",
                             fill=False, linewidth=1.5, linestyle="--"))
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
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
    for g in ATC_GROUPS:
        xy = np.array(by_group[g]["xy"])
        if len(xy) == 0:
            continue
        ax.scatter(xy[:, 0], xy[:, 1], s=8, color=group_color[g], alpha=0.45, zorder=2)

    for xy, code, g in zip(roots["xy"], roots["codes"], roots["groups"]):
        ax.scatter(xy[0], xy[1], s=220, color=group_color[g],
                   edgecolors="black", linewidths=1.2, zorder=5)
        ax.annotate(code, xy=(xy[0], xy[1]), fontsize=10, fontweight="bold",
                    ha="center", va="bottom", xytext=(0, 8),
                    textcoords="offset points", zorder=6)

    # --- root ---
    if args.show_root:
        root_xy = poincare[node_index[VIRTUAL_ROOT]]
        ax.scatter(root_xy[0], root_xy[1], s=300, c="black", marker="D", zorder=10)
        ax.annotate("ROOT", xy=(root_xy[0], root_xy[1]), fontsize=10, fontweight="bold",
                    ha="center", va="bottom", xytext=(0, 8), textcoords="offset points", zorder=11)

    legend_handles = [
        mlines.Line2D([], [], marker="o", color="w",
                      markerfacecolor=group_color[g], markeredgecolor="black",
                      markersize=10, label=f"{g} — {ATC_GROUPS[g]}")
        for g in ATC_GROUPS
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=9, framealpha=0.9, title="ATC Level-1 Groups", title_fontsize=10)

    edge_tag = f" · edges ≤ level {args.max_level}" if args.show_edges else ""
    subtitle = f"\n⚠ {proj_label}" if is_approx else ""
    ax.set_title(
        f"ATC Hierarchy — Poincaré Disk by Group{edge_tag} (dim={args.dim}){subtitle}",
        fontsize=12, pad=15,
    )

    dim_tag = f"dim{args.dim}" if not is_approx else f"dim{args.dim}_{args.reduce}"
    out = _RESULTS_DIR / f"poincare_disk_by_group_{dim_tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
