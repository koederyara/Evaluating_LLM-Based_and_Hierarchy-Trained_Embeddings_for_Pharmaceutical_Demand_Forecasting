"""Evaluation metrics for ATC embedding spaces.

Only what the thesis reports. The rank-based ones (nn_precision_at_k, chance_precision,
stratified_tree_distance_correlation, compute_mean_rank_and_map) are cross-method
comparable; isotropy_score, hdc_score and compute_generality_correlation are internal to
one geometry and must not be read across paradigms.
"""
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_distances, pairwise_distances

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atc_utils import _PREFIX_LEN, get_atc_level, parent_id

_SEED = 42


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 1.0 - float(np.dot(a, b) / denom) if denom > 0 else 1.0


def pairwise_distance_matrix(embeddings: np.ndarray, distance_fn) -> np.ndarray:
    """Full pairwise distance matrix, via a vectorised path where one exists."""
    if distance_fn is cosine_distance:
        return cosine_distances(embeddings).astype(np.float32)
    if distance_fn is lorentz_distance:
        return lorentz_distance_matrix(embeddings)
    mat = pairwise_distances(embeddings, metric=distance_fn).astype(np.float32)
    np.fill_diagonal(mat, 0.0)  # float32 rounding leaves tiny non-zero self-distances
    finite = mat[np.isfinite(mat)]
    if finite.size > 0:
        fallback = float(finite.max()) * 10.0
        mat = np.nan_to_num(mat, posinf=fallback)
    return mat


def nn_precision_at_k(
    dist_matrix: np.ndarray,
    lbls: np.ndarray,
    ks: list[int],
) -> dict[int, float]:
    """Local neighbourhood purity: 1.0 means every neighbour shares the code's group.

    Only meaningful against chance_precision, which is what random vectors would score.
    """
    n = len(lbls)
    dist_m = dist_matrix.copy()
    np.fill_diagonal(dist_m, np.inf)

    result: dict[int, float] = {}
    for k in ks:
        if n < k + 1:
            result[k] = float("nan")
            continue
        precisions = [
            int(np.sum(lbls[np.argpartition(dist_m[i], k)[:k]] == lbls[i])) / k
            for i in range(n)
        ]
        result[k] = float(np.mean(precisions))
    return result


def chance_precision(lbls: np.ndarray) -> float:
    """Probability that two randomly drawn codes share a group = the chance baseline."""
    n = len(lbls)
    if n < 2:
        return float("nan")
    _, counts = np.unique(lbls, return_counts=True)
    return float(np.sum(counts * (counts - 1)) / (n * (n - 1)))


_STDC_BUCKETS = (("small", 1, 3), ("medium", 4, 6), ("large", 7, 10**6))
_STDC_BATCH = 200_000
_STDC_MAX_ROUNDS = 200
_STDC_N_PER_BUCKET = 20_000  # sd(rho) across seeds ~0.002; at 2_000 it was ~0.011


def _tree_distance_encoder(codes: list[str]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Prefix-group ids per level, so tree distances can be taken in bulk."""
    levels = np.array([get_atc_level(c) or 0 for c in codes], dtype=np.int16)
    prefix_ids = {
        lvl: np.unique([c[: _PREFIX_LEN[lvl]] for c in codes], return_inverse=True)[1]
        for lvl in _PREFIX_LEN
    }
    return levels, prefix_ids


def _pair_tree_distances(levels, prefix_ids, aa: np.ndarray, bb: np.ndarray) -> np.ndarray:
    """d_T(a,b) = level(a) + level(b) - 2 * level(LCA(a,b)), vectorised over index arrays.

    Taking the maximum shared prefix level rather than the last match keeps the result
    independent of the iteration order over levels.
    """
    la, lb = levels[aa], levels[bb]
    lca = np.zeros(len(aa), dtype=np.int16)
    for lvl, ids in prefix_ids.items():
        shared = (ids[aa] == ids[bb]) & (la >= lvl) & (lb >= lvl)
        lca = np.maximum(lca, np.where(shared, lvl, 0).astype(np.int16))
    return la + lb - 2 * lca


def stratified_tree_distance_correlation(
    embeddings: np.ndarray,
    codes: list[str],
    distance_fn=cosine_distance,
    n_per_bucket: int = _STDC_N_PER_BUCKET,
    rng: np.random.Generator | None = None,
) -> dict:
    """Spearman ρ between ATC tree distance and embedding distance (Q2.4).

    Stratified into near/medium/distant buckets because only ~0.4 % of uniformly drawn
    ATC pairs are within tree distance 3 — unstratified, the sample would be almost
    entirely distant pairs and say nothing about fine-grained structure. Positive ρ means
    the hierarchy is preserved. The realised bucket sizes are returned so a sample that
    failed to balance stays visible.
    """
    rng = rng if rng is not None else np.random.default_rng(_SEED)
    n = len(codes)
    levels, prefix_ids = _tree_distance_encoder(codes)
    collected: dict[str, dict[tuple[int, int], int]] = {n_: {} for n_, _, _ in _STDC_BUCKETS}

    # Rejection sampling in batches: the small bucket needs ~10^6 candidates to fill.
    for _ in range(_STDC_MAX_ROUNDS):
        if all(len(collected[n_]) >= n_per_bucket for n_, _, _ in _STDC_BUCKETS):
            break
        aa = rng.integers(0, n, size=_STDC_BATCH)
        bb = rng.integers(0, n, size=_STDC_BATCH)
        keep = (aa != bb) & (levels[aa] > 0) & (levels[bb] > 0)
        aa, bb = aa[keep], bb[keep]
        td = _pair_tree_distances(levels, prefix_ids, aa, bb)
        for name, lo, hi in _STDC_BUCKETS:
            store = collected[name]
            if len(store) >= n_per_bucket:
                continue
            for i in np.flatnonzero((td >= lo) & (td <= hi)):
                a, b = int(aa[i]), int(bb[i])
                store.setdefault((a, b) if a < b else (b, a), int(td[i]))
                if len(store) >= n_per_bucket:
                    break

    sizes = {name: len(store) for name, store in collected.items()}
    for name, size in sizes.items():
        if size < n_per_bucket:
            warnings.warn(
                f"stratified_tree_distance_correlation: '{name}' bucket reached only "
                f"{size}/{n_per_bucket} distinct pairs — the sample is not fully balanced."
            )

    pairs = [(a, b, td) for store in collected.values() for (a, b), td in store.items()]
    if len(pairs) < 10:
        return {"rho": float("nan"), **{f"n_{k}": v for k, v in sizes.items()}}

    tree_dists = [td for _, _, td in pairs]
    emb_dists = [distance_fn(embeddings[a], embeddings[b]) for a, b, _ in pairs]
    rho, _ = stats.spearmanr(tree_dists, emb_dists)
    return {"rho": float(rho), **{f"n_{k}": v for k, v in sizes.items()}}


def isotropy_score(embeddings: np.ndarray) -> float:
    """Effective dimensionality after Roy & Vetterli (2007); range (0, 1], 1 = isotropic.

    Preferred over the λ_min/λ_max ratio because a single degenerate direction would
    dominate that ratio at these dimensionalities, whereas the entropy form is smooth.
    """
    centered = embeddings - embeddings.mean(axis=0)
    _, sv, _ = np.linalg.svd(centered, full_matrices=False)
    sv_sq = sv ** 2
    total = sv_sq.sum()
    if total == 0:
        return float("nan")
    p = sv_sq / total
    entropy = -float(np.sum(p[p > 0] * np.log(p[p > 0])))
    return float(np.exp(entropy) / embeddings.shape[1])


def lorentz_distance(u: np.ndarray, v: np.ndarray) -> float:
    """arccosh(-<u,v>_L) with the Minkowski inner product.

    The clip guards against rounding pushing the arccosh argument below 1.
    """
    inner = -u[0] * v[0] + float(np.dot(u[1:], v[1:]))
    return float(np.arccosh(np.clip(-inner, 1.0, None)))


def lorentz_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Vectorised, because the generic per-pair path is unusable beyond n ~ 1000."""
    inner = (
        -np.outer(embeddings[:, 0], embeddings[:, 0])
        + embeddings[:, 1:] @ embeddings[:, 1:].T
    )
    return np.arccosh(np.clip(-inner, 1.0, None)).astype(np.float32)


def compute_mean_rank_and_map(
    embeddings: np.ndarray,
    test_edges: list[tuple[int, int]],
    all_edges: list[tuple[int, int]],
    distance_fn=lorentz_distance,
) -> dict:
    """Mean rank and MAP over held-out edges (Q2.1 / Q2.2).

    Neighbours are masked from the candidate list using all_edges, not just the training
    split, so a model is not penalised for ranking a correct but unobserved edge highly.

    Each query has exactly one correct target, so AP = 1/rank and MAP is numerically MRR;
    the name follows the Nickel & Kiela (2018) protocol.
    """
    dm = pairwise_distance_matrix(embeddings, distance_fn)

    neighbors: dict[int, set[int]] = {}
    for u, v in all_edges:
        neighbors.setdefault(u, set()).add(v)

    ranks: list[int] = []
    aps: list[float] = []
    for u_idx, v_idx in test_edges:
        row = dm[u_idx].copy()
        row[u_idx] = np.inf
        for nbr in neighbors.get(u_idx, set()) - {v_idx}:
            row[nbr] = np.inf
        rank = int(np.sum(row <= dm[u_idx, v_idx]))
        ranks.append(rank)
        aps.append(1.0 / rank if rank > 0 else 0.0)

    return {
        "mean_rank": float(np.mean(ranks)) if ranks else float("nan"),
        "map": float(np.mean(aps)) if aps else float("nan"),
    }


def compute_generality_correlation(
    generality_scores: np.ndarray,
    node_index: dict[str, int],
    original_edges: list[tuple[str, str]],
) -> float:
    """Spearman ρ against the normalised rank sp/(sp+lp) of Nickel & Kiela (2018) §4.1.

    The virtual root gives the 14 level-1 groups a common reference point, without which
    sp is undefined for them. ρ should be positive: x0 grows with specificity.
    """
    g = nx.DiGraph()
    g.add_edges_from(original_edges)

    virtual_root = "__root__"
    for node in list(g.nodes):
        if g.in_degree(node) == 0:
            g.add_edge(virtual_root, node)

    sp_lengths = nx.single_source_shortest_path_length(g, virtual_root)

    # Bottom-up in topological order, so each child's height is final when read.
    lp_lengths: dict[str, int] = {n: 0 for n in g.nodes}
    for n in reversed(list(nx.topological_sort(g))):
        for child in g.successors(n):
            lp_lengths[n] = max(lp_lengths[n], lp_lengths[child] + 1)

    norm_ranks: list[float] = []
    scores: list[float] = []
    for code, idx in node_index.items():
        sp = sp_lengths.get(code)
        lp = lp_lengths.get(code)
        if sp is None or lp is None or sp + lp == 0:
            continue
        norm_ranks.append(sp / (sp + lp))
        scores.append(float(generality_scores[idx]))

    if len(norm_ranks) < 2:
        return float("nan")

    rho, _ = stats.spearmanr(scores, norm_ranks)
    return float(rho)


def hdc_score(embeddings: np.ndarray, codes: list[str]) -> dict:
    """Fraction of direct parent-child pairs with the parent closer to the origin (Q1.B2).

    After Alshargi et al. (2019). Strict comparison, so ties count as violations. Lorentz
    only — the origin has no special role in Euclidean space. Complementary to
    stratified_tree_distance_correlation, not a substitute: this tests the origin-relative
    property Lorentz training is designed to produce, that one general distance fidelity.
    """
    code_to_idx = {c: i for i, c in enumerate(codes)}
    # Clip absorbs float error that would otherwise take sqrt of a small negative.
    norms = np.sqrt(np.maximum(embeddings[:, 0] ** 2 - 1.0, 0.0))

    satisfied = 0
    total = 0
    for i, code in enumerate(codes):
        par = parent_id(code)
        if par is None or par not in code_to_idx:
            continue
        pi = code_to_idx[par]
        total += 1
        if norms[pi] < norms[i]:
            satisfied += 1

    if total == 0:
        return {"hdc": float("nan"), "n_pairs": 0}
    return {"hdc": float(satisfied / total), "n_pairs": total}
