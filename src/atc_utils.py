"""ATC code utilities. A code's level is its length and its ancestors are its valid
prefixes, which is why every function here is pure string slicing.
"""
import numpy as np

_LEVEL_BY_LEN: dict[int, int] = {1: 1, 3: 2, 4: 3, 5: 4, 7: 5}
_PREFIX_LEN: dict[int, int] = {1: 1, 2: 3, 3: 4, 4: 5, 5: 7}

def get_atc_level(class_id: str) -> int | None:
    return _LEVEL_BY_LEN.get(len(class_id))

def group_label(class_id: str, level: int) -> str | None:
    prefix_len = _PREFIX_LEN[level]
    return class_id[:prefix_len] if len(class_id) >= prefix_len else None

def parent_id(class_id: str) -> str | None:
    level = get_atc_level(class_id)
    if level is None or level == 1:
        return None
    return class_id[: _PREFIX_LEN[level - 1]]

def build_atc_edges(codes: list[str]) -> list[tuple[str, str]]:
    """Build direct parent→child edges for codes present in the given list."""
    code_set = set(codes)
    return [
        (par, code)
        for code in codes
        if (par := parent_id(code)) is not None and par in code_set
    ]


VIRTUAL_ROOT = "ROOT"  # synthetic root node connecting all 14 Level-1 ATC groups


def add_virtual_root(edges: list[tuple[str, str]],
                     codes: list[str]) -> list[tuple[str, str]]:
    """Give the 14 level-1 groups a common ancestor.

    Without one they become negative samples for each other and are pushed away from the
    origin instead of toward it.
    """
    level1_codes = [c for c in codes if get_atc_level(c) == 1]
    root_edges = [(VIRTUAL_ROOT, c) for c in sorted(level1_codes)]
    return root_edges + list(edges)


def _normalize_label(label: str) -> str:
    """Sentence-case the ALL-CAPS L1–L3 labels and drop the " in ATC" export suffix.

    Mixed-case labels are returned untouched, so L4/L5 substance names are not mangled.
    """
    s = label.strip()
    if s.upper().endswith(" IN ATC"):
        s = s[:-7].strip()  # len(" in ATC") == 7
    return s.lower().capitalize() if s == s.upper() else s


def build_atc_path(code: str, label_lookup: dict[str, str]) -> str:
    """Hierarchy path from level 1 down to the code, e.g. for C10AA05:

        "Cardiovascular system (C) > Lipid modifying agents (C10) > ... > atorvastatin (C10AA05)"

    Missing labels fall back to the code itself rather than dropping a level.
    """
    chain: list[str] = []
    cur: str | None = code
    while cur is not None:
        chain.append(cur)
        cur = parent_id(cur)
    chain.reverse()  # Level 1 first

    return " > ".join(
        f"{_normalize_label(label_lookup.get(c, c))} ({c})"
        for c in chain
    )


def atc_tree_distance(cid_a: str, cid_b: str) -> int | None:
    la = get_atc_level(cid_a)
    lb = get_atc_level(cid_b)
    if la is None or lb is None:
        return None
    lca = 0
    for lvl in range(1, min(la, lb) + 1):
        if cid_a[: _PREFIX_LEN[lvl]] == cid_b[: _PREFIX_LEN[lvl]]:
            lca = lvl
        else:
            break
    return la + lb - 2 * lca


def tc_link_prediction_split(
    codes: list[str],
    train_fraction: float,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Hold out transitive-closure links for link prediction, after Nickel & Kiela (2017).

    Transitive-closure relations, not direct edges: removing a direct edge in a strict
    tree detaches the child's whole subtree and makes the link unrecoverable in
    principle, whereas a held-out TC relation leaves each node anchored by its remaining
    links. Root and leaf links are excluded as trivially or impossibly predictable.
    """
    code_set = set(codes)
    links: list[tuple[str, str]] = []
    has_descendant: set[str] = set()
    for c in codes:
        p = parent_id(c)
        while p is not None:
            if p in code_set:
                links.append((p, c))
                has_descendant.add(p)
            p = parent_id(p)

    leaves = code_set - has_descendant
    eligible = [(u, v) for u, v in links if get_atc_level(u) != 1 and v not in leaves]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(eligible))
    n_test = int(round(len(eligible) * (1.0 - train_fraction)))
    test_pairs = [eligible[i] for i in sorted(perm[:n_test].tolist())]

    test_set = set(test_pairs)
    train_pairs = [(u, v) for u, v in links if (u, v) not in test_set]
    return train_pairs, test_pairs
