"""Goal 2 — cross-method comparison over the nine configurations (Q2.1-Q2.4).

Every metric here is rank-based: Euclidean and Lorentz distances share no numerical
scale, so only orderings within each space may be compared. Text uses cosine, Lorentz
its native geodesic distance.

Two asymmetries are structural and must be declared when reporting, not fixed:

  - Q2.1 and Q2.2 measure exactly the Lorentz training objective, while the text configs
    are frozen and have never seen the hierarchy. A Lorentz win there is trained-versus-
    zero-shot, not evidence about geometry. Q2.4 is the most task-neutral signal.
  - In Q2.2 the Lorentz model is retrained on the 90 % split, whereas frozen embeddings
    can only be evaluated on the held-out relations — recorded per row in
    split_semantics so the two are never read as the same test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import EMBEDDINGS_DIR, LORENTZ_DIM, RANDOM_SEED
from atc_utils import build_atc_edges, tc_link_prediction_split
from utils import (
    ALL_BASE_CONFIGS,
    LORENTZ_KEY,
    LORENTZ_VARIANT,
    atc_level1_groups,
    compute_mean_rank_and_map,
    cosine_distance,
    l2_normalize,
    load_atc_codes,
    load_embeddings,
    log_map_origin,
    lorentz_distance,
    save_csv,
    stratified_tree_distance_correlation,
)
from utils import _load_lorentz  # split-model loader
from probing import run_probe

_TRAIN_FRACTION = 0.9
_NA_LORENTZ_SPLIT = "N/A — model not trained on split"


# ---------------------------------------------------------------------------
# Edge sets
# ---------------------------------------------------------------------------

def _tc_edges(codes: list[str]) -> list[tuple[int, int]]:
    """Transitive-closure edges as index pairs, symmetrised."""
    g = nx.DiGraph()
    g.add_edges_from(build_atc_edges(codes))
    tc = nx.transitive_closure(g)
    idx = {c: i for i, c in enumerate(codes)}
    directed = [(idx[u], idx[v]) for u, v in tc.edges()]
    return directed + [(v, u) for u, v in directed]


def _link_prediction_test_edges(codes: list[str]) -> list[tuple[int, int]]:
    """The held-out relations of the Q2.2 split.

    The file written by lorentz_training.py is authoritative, so the Lorentz split model
    and the frozen text configs are scored on identical links; the seed-matched
    recomputation is only a fallback.
    """
    idx = {c: i for i, c in enumerate(codes)}

    saved = EMBEDDINGS_DIR / "lorentz" / f"test_edges_train{int(_TRAIN_FRACTION * 100)}{LORENTZ_VARIANT}.json"
    if saved.exists():
        test_pairs = [tuple(e) for e in json.loads(saved.read_text())]
    else:
        _, test_pairs = tc_link_prediction_split(codes, _TRAIN_FRACTION, RANDOM_SEED)

    return [(idx[u], idx[v]) for u, v in test_pairs if u in idx and v in idx]


# ---------------------------------------------------------------------------
# Q2.1 — reconstruction
# ---------------------------------------------------------------------------

def q21_reconstruction(codes: list[str], config_keys: list[str]) -> dict[str, dict]:
    tc = _tc_edges(codes)
    results: dict[str, dict] = {}
    for key in config_keys:
        emb = load_embeddings(key)
        dist_fn = lorentz_distance if key == LORENTZ_KEY else cosine_distance
        m = compute_mean_rank_and_map(emb, tc, tc, distance_fn=dist_fn)
        results[key] = {"q21_mean_rank": m["mean_rank"], "q21_map": m["map"]}
        print(f"  Q2.1 {key:28s} MR={m['mean_rank']:.2f}  MAP={m['map']:.4f}")
    return results


# ---------------------------------------------------------------------------
# Q2.2 — link prediction
# ---------------------------------------------------------------------------

def q22_link_prediction(codes: list[str], config_keys: list[str]) -> dict[str, dict]:
    tc = _tc_edges(codes)  # masks true neighbours, so correct unobserved edges cost nothing
    test_edges = _link_prediction_test_edges(codes)
    print(f"  Q2.2 test edges (root/leaf excluded): {len(test_edges)}")
    results: dict[str, dict] = {}

    for key in [k for k in config_keys if k != LORENTZ_KEY]:
        emb = load_embeddings(key)
        m = compute_mean_rank_and_map(emb, test_edges, tc, distance_fn=cosine_distance)
        results[key] = {
            "q22_mean_rank": m["mean_rank"],
            "q22_map": m["map"],
            "split_semantics": "frozen_embeddings_eval_on_test_split",
        }
        print(f"  Q2.2 {key:28s} MR={m['mean_rank']:.2f}  MAP={m['map']:.4f}")

    if LORENTZ_KEY not in config_keys:
        return results

    # Must come from the split model: the full-TC model has seen the test relations.
    try:
        emb = _load_lorentz(codes, suffix=f"_train{int(_TRAIN_FRACTION * 100)}")
        m = compute_mean_rank_and_map(emb, test_edges, tc, distance_fn=lorentz_distance)
        results[LORENTZ_KEY] = {
            "q22_mean_rank": m["mean_rank"],
            "q22_map": m["map"],
            "split_semantics": "retrained_on_train_split",
        }
        print(f"  Q2.2 {LORENTZ_KEY:28s} MR={m['mean_rank']:.2f}  MAP={m['map']:.4f}")
    except FileNotFoundError:
        results[LORENTZ_KEY] = {
            "q22_mean_rank": _NA_LORENTZ_SPLIT,
            "q22_map": _NA_LORENTZ_SPLIT,
            "split_semantics": "retrained_on_train_split",
        }
        print(f"  Q2.2 {LORENTZ_KEY:28s} {_NA_LORENTZ_SPLIT} "
              f"(train embeddings_dim{LORENTZ_DIM}_train90.npy first)")
    return results


# ---------------------------------------------------------------------------
# Q2.3 — cross-method probing (Lorentz projected to tangent space)
# ---------------------------------------------------------------------------

def q23_probing(codes: list[str], config_keys: list[str]) -> dict[str, dict]:
    """Linear accessibility of the level-1 group, the project's only probing classifier.

    For Lorentz this measures accessibility in the tangent space at the origin, not in the
    native geometry: the log map discards global curvature. That is a deliberate cost of
    making the nine configurations commensurable, and the reason Q2.4 is reported beside it.
    """
    groups = atc_level1_groups(codes)
    results: dict[str, dict] = {}
    for key in config_keys:
        emb = load_embeddings(key)
        x = l2_normalize(log_map_origin(emb)) if key == LORENTZ_KEY else l2_normalize(emb)
        probe = run_probe(x, groups)
        results[key] = {
            "q23_macro_f1": probe.real_f1,
            "q23_baseline_f1": probe.baseline_f1,
            "q23_delta_f1": probe.delta_f1,
            "q23_flagged": probe.flagged,
        }
        print(f"  Q2.3 {key:28s} F1={probe.real_f1:.3f}  ΔF1={probe.delta_f1:+.3f}")
    return results


# ---------------------------------------------------------------------------
# Q2.4 — stratified tree-distance correlation (STDC, space-agnostic)
# ---------------------------------------------------------------------------

def q24_stratified_tree_distance_correlation(codes: list[str],
                                             config_keys: list[str]) -> dict[str, dict]:
    """Distance-balanced Spearman ρ between tree and embedding distance.

    Depends on no learned task, which is what makes it the decisive cross-method metric
    rather than Q2.1/Q2.2.
    """
    results: dict[str, dict] = {}
    for key in config_keys:
        emb = load_embeddings(key)
        dist_fn = lorentz_distance if key == LORENTZ_KEY else cosine_distance
        out = stratified_tree_distance_correlation(emb, codes, distance_fn=dist_fn)
        results[key] = {
            "q24_stdc_rho": out["rho"],
            "q24_n_small": out["n_small"],
            "q24_n_medium": out["n_medium"],
            "q24_n_large": out["n_large"],
        }
        print(f"  Q2.4 {key:28s} STDC ρ={out['rho']:+.4f}  "
              f"(pairs {out['n_small']}/{out['n_medium']}/{out['n_large']})")
    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_goal2(config_keys: list[str] | None = None,
              out_name: str = "goal2_results.csv") -> pd.DataFrame:
    """Run Goal 2 over config_keys (default: all nine base configs).

    config_keys accepts the "_d<N>" reduced variants; pass a separate out_name so a run
    at the forecasting width never lands in the canonical result file.
    """
    keys = config_keys or list(ALL_BASE_CONFIGS)
    codes = list(load_atc_codes())
    print("Q2.1 — hierarchy reconstruction")
    q21 = q21_reconstruction(codes, keys)
    print("Q2.2 — link prediction")
    q22 = q22_link_prediction(codes, keys)
    print("Q2.3 — cross-method probing")
    q23 = q23_probing(codes, keys)
    print("Q2.4 — stratified tree-distance correlation (STDC)")
    q24 = q24_stratified_tree_distance_correlation(codes, keys)

    rows = []
    for key in keys:
        rows.append({"config_key": key, **q21[key], **q22.get(key, {}), **q23[key], **q24[key]})
    df = pd.DataFrame(rows)
    save_csv(df, out_name)
    return df


if __name__ == "__main__":
    from cli import parse_config_subset_args

    args = parse_config_subset_args("Goal 2 - cross-method comparison.",
                                    default_out="goal2_results.csv")
    print("Goal 2 — cross-method comparison\n")
    df = run_goal2(args.configs, args.out)
    print(f"\nSaved results/{args.out}")
