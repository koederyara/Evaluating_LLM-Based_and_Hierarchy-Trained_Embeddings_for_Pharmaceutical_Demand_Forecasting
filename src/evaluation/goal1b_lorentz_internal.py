"""Goal 1, Part B — intra-space validation of the Lorentz embedding (Q1.B1-Q1.B3).

These metrics are structurally biased toward the Lorentz model by design, so they are
diagnostic within the hyperbolic space only and are never read against Part A's values.
Q1.B3 mirrors Part A's neighbourhood precision (same >=5-member filter) so the two
families are at least measured on the same clustering criterion.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atc_utils import build_atc_edges, group_label, parent_id, get_atc_level
from utils import (
    LORENTZ_KEY,
    chance_precision,
    hdc_score,
    load_atc_codes,
    load_embeddings,
    lorentz_distance,
    nn_precision_at_k,
    pairwise_distance_matrix,
    save_csv,
)

_KS = [1, 5, 10]
_LEVELS = [1, 2, 3]
_MIN_CLASS = 5  # match Part A (goal1a_text_internal.py) for comparable subsets


def _normalized_ranks(codes: list[str]) -> np.ndarray:
    """Normalised rank sp/(sp+lp) per Nickel & Kiela (2018).

    The virtual root gives the 14 level-1 codes a common reference, without which their
    shortest path is undefined.
    """
    g = nx.DiGraph()
    g.add_edges_from(build_atc_edges(codes))
    root = "__root__"
    for n in [n for n in g.nodes if g.in_degree(n) == 0]:
        g.add_edge(root, n)

    sp = nx.single_source_shortest_path_length(g, root)
    lp: dict[str, int] = {n: 0 for n in g.nodes}
    for n in reversed(list(nx.topological_sort(g))):
        for child in g.successors(n):
            lp[n] = max(lp[n], lp[child] + 1)

    return np.array([sp[c] / (sp[c] + lp[c]) for c in codes], dtype=float)


def _nn_precision_by_level(emb: np.ndarray, codes: list[str]) -> pd.DataFrame:
    """Precision@k in the native Lorentz geometry, filtered as in Part A."""
    dist_full = pairwise_distance_matrix(emb, lorentz_distance)
    rows = []
    for level in _LEVELS:
        labels = [group_label(c, level) for c in codes]
        counts = Counter(l for l in labels if l is not None)
        mask = np.array([l is not None and counts[l] >= _MIN_CLASS for l in labels])
        y = np.array([l if l is not None else "" for l in labels])[mask]
        dist = dist_full[np.ix_(mask, mask)]
        prec = nn_precision_at_k(dist, y, _KS)
        rows.append({
            "config_key": LORENTZ_KEY,
            "level": level,
            "n_codes": int(mask.sum()),
            "n_classes": int(len(set(y))),
            "precision_chance": chance_precision(y),
            **{f"precision_at_{k}": prec[k] for k in _KS},
        })
        print(f"  Q1.B3 L{level}  P@1={prec[1]:.3f} P@5={prec[5]:.3f} P@10={prec[10]:.3f} "
              f"(chance={chance_precision(y):.3f})")
    return pd.DataFrame(rows)


def _radial_by_level(emb: np.ndarray, codes: list[str]) -> pd.DataFrame:
    """Breakdown that explains the weak aggregates: the radial ordering is correct in the
    upper hierarchy and saturates at the leaves, where 81 % of the pairs sit.
    """
    c2i = {c: i for i, c in enumerate(codes)}
    norms = np.sqrt(np.maximum(emb[:, 0] ** 2 - 1.0, 0.0))
    rows = []
    for level in range(1, 6):
        idx = [i for i, c in enumerate(codes) if get_atc_level(c) == level]
        sat = tot = 0
        for c in codes:
            p = parent_id(c)
            if p is None or p not in c2i or get_atc_level(p) != level:
                continue
            tot += 1
            sat += norms[c2i[p]] < norms[c2i[c]]
        rows.append({
            "level": level,
            "n_codes": len(idx),
            "mean_norm": float(norms[idx].mean()) if idx else float("nan"),
            "std_norm": float(norms[idx].std()) if idx else float("nan"),
            "hdc_parent_at_level": float(sat / tot) if tot else float("nan"),
            "n_child_pairs": tot,
        })
        print(f"  radial L{level}: mean||x||={rows[-1]['mean_norm']:.2f} "
              f"HDC(parent@L{level})={rows[-1]['hdc_parent_at_level']:.3f} (n={tot})")
    return pd.DataFrame(rows)


def run_goal1b() -> pd.DataFrame:
    codes = list(load_atc_codes())
    emb = load_embeddings(LORENTZ_KEY)  # R^{dim+1}, time coordinate at column 0

    # Q1.B1 — ||x||_L = sqrt(x0^2 - 1) (spatial norm of a valid hyperboloid point)
    norms = np.sqrt(np.maximum(emb[:, 0] ** 2 - 1.0, 0.0))
    ranks = _normalized_ranks(codes)
    rho, p_value = stats.spearmanr(norms, ranks)

    # Q1.B2 — HDC
    hdc = hdc_score(emb, codes)

    df = pd.DataFrame([{
        "config_key": LORENTZ_KEY,
        "norm_rank_spearman_rho": float(rho),
        "norm_rank_p_value": float(p_value),
        "hdc_score": hdc["hdc"],
        "hdc_n_pairs": hdc["n_pairs"],
        "note": "Lorentz-internal; structurally biased toward the Lorentz model by design.",
    }])
    save_csv(df, "goal1b_results.csv")
    print(f"  Q1.B1 norm–rank ρ={rho:.4f} (p={p_value:.2e})")
    print(f"  Q1.B2 HDC={hdc['hdc']:.4f} over {hdc['n_pairs']} parent–child pairs")

    # Q1.B3 — native neighbourhood precision (clustering), mirrors Part A
    nn_df = _nn_precision_by_level(emb, codes)
    save_csv(nn_df, "goal1b_nn_precision.csv")

    # Radial breakdown backing the "saturates at the leaves" reading of Q1.B1/Q1.B2
    save_csv(_radial_by_level(emb, codes), "goal1b_radial_by_level.csv")
    return df


if __name__ == "__main__":
    print("Goal 1B — Lorentz-internal validation")
    df = run_goal1b()
    print("\nSaved results/goal1b_results.csv")