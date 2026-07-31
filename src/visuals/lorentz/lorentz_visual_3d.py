"""
Interactive 3D visualization of Lorentz embeddings on the hyperboloid.

Only Lorentz dim=2 embeddings are supported: those points live natively in ℝ³
on the upper sheet of x₀² − x₁² − x₂² = 1, so the 3D plot shows the true
geometry without any information loss.  For higher Lorentz dimensions the
hyperboloid is a manifold in a higher-dimensional space and cannot be embedded
in ℝ³ without projection — the resulting plot would be geometrically misleading.

Usage:
  python src/visuals/lorentz/lorentz_visual_3d.py
  python src/visuals/lorentz/lorentz_visual_3d.py --dim 2
  python src/visuals/lorentz/lorentz_visual_3d.py --color group
  python src/visuals/lorentz/lorentz_visual_3d.py --show-root

Flags:
  --dim N     Lorentz embedding dimension to load (default: 2; only 2 is valid).
  --color     Color by 'level' (default) or 'group'.
  --show-root Also render the virtual ROOT node.

Output:
  results/lorentz/visuals/hyperboloid_dim{N}_level.html
  results/lorentz/visuals/hyperboloid_dim{N}_group.html
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import ROOT, ATC_GROUPS
from atc_utils import get_atc_level

_EMBEDDINGS_DIR = ROOT / "data" / "embeddings" / "lorentz"
_RESULTS_DIR = ROOT / "results" / "lorentz" / "visuals"

_LEVEL_COLORS = {1: "#e41a1c", 2: "#377eb8", 3: "#4daf4a", 4: "#984ea3", 5: "#ff7f00"}
_LEVEL_SIZES  = {1: 12,        2: 7,         3: 4,         4: 3,         5: 2}

_GROUP_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3",
]
_GROUP_COLOR = {g: _GROUP_COLORS[i] for i, g in enumerate(ATC_GROUPS)}


def _hyperboloid_surface() -> go.Surface:
    r = np.linspace(-3, 3, 60)
    x1, x2 = np.meshgrid(r, r)
    x0 = np.sqrt(1 + x1**2 + x2**2)
    return go.Surface(
        x=x1, y=x2, z=x0,
        colorscale=[[0, "lightgray"], [1, "lightgray"]],
        opacity=0.15, showscale=False, hoverinfo="skip", name="Hyperboloid",
    )


def _traces_by_level(emb: np.ndarray, node_index: dict) -> list:
    by_level: dict = {lvl: {"x0": [], "x1": [], "x2": [], "codes": []} for lvl in range(1, 6)}
    for code, idx in node_index.items():
        lvl = get_atc_level(code)
        if lvl:
            p = emb[int(idx)]
            by_level[lvl]["x0"].append(float(p[0]))
            by_level[lvl]["x1"].append(float(p[1]))
            by_level[lvl]["x2"].append(float(p[2]))
            by_level[lvl]["codes"].append(code)

    traces = []
    for lvl in [5, 4, 3, 2, 1]:
        d = by_level[lvl]
        if not d["x0"]:
            continue
        traces.append(go.Scatter3d(
            x=d["x1"], y=d["x2"], z=d["x0"],
            mode="markers",
            name=f"Level {lvl} (n={len(d['x0'])})",
            text=d["codes"],
            hovertemplate="<b>%{text}</b><extra></extra>",
            marker=dict(size=_LEVEL_SIZES[lvl], color=_LEVEL_COLORS[lvl], opacity=0.8),
        ))
    return traces


def _traces_by_group(emb: np.ndarray, node_index: dict) -> list:
    by_group: dict = {g: {"x0": [], "x1": [], "x2": [], "codes": []} for g in ATC_GROUPS}
    roots: dict = {"x0": [], "x1": [], "x2": [], "codes": [], "colors": []}

    for code, idx in node_index.items():
        g = str(code).strip().upper()[0] if code else None
        if g not in ATC_GROUPS:
            continue
        p = emb[int(idx)]
        coords = (float(p[0]), float(p[1]), float(p[2]))
        if get_atc_level(code) == 1:
            roots["x0"].append(coords[0])
            roots["x1"].append(coords[1])
            roots["x2"].append(coords[2])
            roots["codes"].append(code)
            roots["colors"].append(_GROUP_COLOR[g])
        else:
            by_group[g]["x0"].append(coords[0])
            by_group[g]["x1"].append(coords[1])
            by_group[g]["x2"].append(coords[2])
            by_group[g]["codes"].append(code)

    traces = []
    for g in ATC_GROUPS:
        d = by_group[g]
        if not d["x0"]:
            continue
        traces.append(go.Scatter3d(
            x=d["x1"], y=d["x2"], z=d["x0"],
            mode="markers",
            name=f"{g} — {ATC_GROUPS[g]}",
            text=d["codes"],
            hovertemplate="<b>%{text}</b><extra></extra>",
            marker=dict(size=3, color=_GROUP_COLOR[g], opacity=0.5),
        ))

    # Level-1 roots on top with larger markers
    if roots["x0"]:
        traces.append(go.Scatter3d(
            x=roots["x1"], y=roots["x2"], z=roots["x0"],
            mode="markers+text",
            name="Level-1 roots",
            text=roots["codes"],
            textposition="top center",
            hovertemplate="<b>%{text}</b><extra></extra>",
            marker=dict(size=10, color=roots["colors"], opacity=1.0,
                        line=dict(color="black", width=1)),
            showlegend=False,
        ))
    return traces


def _root_trace(root_emb: np.ndarray) -> go.Scatter3d:
    p = root_emb
    return go.Scatter3d(
        x=[float(p[1])], y=[float(p[2])], z=[float(p[0])],
        mode="markers+text",
        name="Virtual ROOT",
        text=["ROOT"],
        textposition="top center",
        hovertemplate="<b>ROOT</b><extra></extra>",
        marker=dict(size=14, color="black", opacity=1.0,
                    symbol="diamond", line=dict(color="white", width=1)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive 3D Lorentz hyperboloid visualization.")
    parser.add_argument(
        "--dim", type=int, default=2,
        help="Lorentz embedding dimension to visualize (default: 2; only 2 is valid for this plot).",
    )
    parser.add_argument(
        "--color", choices=["level", "group"], default="level",
        help="Color points by hierarchy depth (default) or Level-1 ATC group.",
    )
    parser.add_argument(
        "--show-root", action="store_true",
        help="Also render the virtual ROOT node.",
    )
    args = parser.parse_args()

    emb_path = _EMBEDDINGS_DIR / f"embeddings_dim{args.dim}.npy"
    if not emb_path.exists():
        sys.exit(f"Error: no embedding file found at {emb_path}")

    emb = np.load(emb_path)
    lorentz_dim = emb.shape[1] - 1  # ambient dim − 1 time coordinate
    if lorentz_dim != 2:
        sys.exit(
            f"Error: Lorentz dim={lorentz_dim} is not supported for the 3D hyperboloid plot.\n"
            "The hyperboloid surface x₀² − x₁² − x₂² = 1 lives in ℝ³ only for dim=2.\n"
            "For higher dimensions the manifold cannot be embedded in ℝ³ without projection,\n"
            "which would make the plot geometrically misleading.  Use --dim 2."
        )

    with open(_EMBEDDINGS_DIR / "node_index.json") as f:
        node_index = json.load(f)

    surface = _hyperboloid_surface()

    if args.color == "group":
        traces = _traces_by_group(emb, node_index)
        title = f"ATC Hierarchy — Lorentz Hyperboloid dim={args.dim} (colored by Level-1 group)"
        out = _RESULTS_DIR / f"hyperboloid_dim{args.dim}_group.html"
    else:
        traces = _traces_by_level(emb, node_index)
        title = f"ATC Hierarchy — Lorentz Hyperboloid dim={args.dim} (colored by level)"
        out = _RESULTS_DIR / f"hyperboloid_dim{args.dim}_level.html"

    if args.show_root:
        root_emb = np.load(_EMBEDDINGS_DIR / f"virtual_root_emb_dim{args.dim}.npy")
        traces.append(_root_trace(root_emb))

    fig = go.Figure([surface] + traces)
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x₁", yaxis_title="x₂", zaxis_title="x₀ (time)"),
        legend_title="ATC Level" if args.color == "level" else "ATC Group",
        margin=dict(l=0, r=0, b=0, t=40),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out))
    fig.show()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
