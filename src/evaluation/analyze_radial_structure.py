"""
What does the Lorentz norm x0 actually encode — tree depth or subtree breadth?

Goal 1B (Q1.B1) finds a near-zero norm-vs-depth correlation (rho ~ 0.09). The Nickel &
Kiela (2018) WordNet property (general concepts sit near the origin) does not transfer
to ATC. Hypothesis: x0 tracks a node's connectivity/breadth in the transitive closure
(general = related to many), which in WordNet coincides with depth but in ATC's shallow
5-level tree does not.

This script quantifies that: Spearman rho of x0 against depth, normalized rank, and
subtree size — plus the within-level rho of x0 vs. breadth, which isolates breadth from
depth. A high within-level rho is the decisive evidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atc_utils import build_atc_edges, get_atc_level
from utils import LORENTZ_KEY, load_atc_codes, load_embeddings, save_csv


def _normalized_ranks(codes: list[str], g: nx.DiGraph) -> np.ndarray:
    """rank(c) = sp(c) / (sp(c) + lp(c)) per Nickel & Kiela (2018) — same as Goal 1B."""
    root = "__root__"
    for n in [n for n in g.nodes if g.in_degree(n) == 0]:
        g.add_edge(root, n)
    sp = nx.single_source_shortest_path_length(g, root)
    lp: dict[str, int] = {n: 0 for n in g.nodes}
    for n in reversed(list(nx.topological_sort(g))):
        for child in g.successors(n):
            lp[n] = max(lp[n], lp[child] + 1)
    return np.array([sp[c] / (sp[c] + lp[c]) for c in codes], dtype=float)


def build_table(codes: list[str]) -> pd.DataFrame:
    emb = load_embeddings(LORENTZ_KEY)  # R^{dim+1}, x0 at column 0
    x0 = emb[:, 0]

    g = nx.DiGraph()
    g.add_edges_from(build_atc_edges(codes))
    tc = nx.transitive_closure(g)
    subtree_size = np.array([tc.out_degree(c) for c in codes], dtype=float)  # # descendants
    depth = np.array([get_atc_level(c) for c in codes], dtype=float)
    rank = _normalized_ranks(codes, g)

    return pd.DataFrame({"code": codes, "x0": x0, "depth": depth,
                         "norm_rank": rank, "subtree_size": subtree_size})


def report(df: pd.DataFrame) -> None:
    def rho(a, b):
        r, p = stats.spearmanr(df[a], df[b])
        return r, p

    print("Spearman rho of x0 against:")
    for col, label in [("depth", "depth (ATC level)"),
                       ("norm_rank", "normalized rank sp/(sp+lp)  [reproduces Q1.B1]"),
                       ("subtree_size", "subtree size (# descendants)")]:
        r, p = rho("x0", col)
        print(f"  {label:48s} rho={r:+.4f}  (p={p:.2e})")

    print("\nWithin-level rho(x0, subtree_size)  [isolates breadth from depth]:")
    for lvl in sorted(df["depth"].unique()):
        sub = df[df["depth"] == lvl]
        if len(sub) < 5 or sub["subtree_size"].nunique() < 2:
            print(f"  level {int(lvl)}: n={len(sub):5d}  (no spread -- leaves)")
            continue
        r, p = stats.spearmanr(sub["x0"], sub["subtree_size"])
        print(f"  level {int(lvl)}: n={len(sub):5d}  rho={r:+.4f}  (p={p:.2e})")

    print("\nMean x0 per level (the radial compression):")
    g = df.groupby("depth")["x0"].agg(["mean", "median", "min", "max", "count"])
    print(g.to_string(float_format=lambda v: f"{v:.1f}"))


def main(write_csv: bool) -> None:
    codes = list(load_atc_codes())
    df = build_table(codes)
    report(df)
    if write_csv:
        path = save_csv(df, "radial_structure.csv")
        print(f"\nSaved {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze what the Lorentz norm x0 encodes.")
    parser.add_argument("--csv", action="store_true", help="Also dump per-node table to results/")
    args = parser.parse_args()
    main(args.csv)
