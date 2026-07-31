"""
Train Lorentz (hyperboloid) embeddings on the ATC hierarchy.

Protocol follows the Nickel & Kiela (2018) reference implementation (train-nouns.sh):
all transitive-closure edges are used for training (no train/val/test split),
burn-in runs for 20 epochs at lr * 0.01, evaluation is Reconstruction on all TC
edges excluding those involving the virtual root.

Usage:
  python src/lorentz_training.py
  python src/lorentz_training.py --dim 20 --epochs 1000

Output:
  data/embeddings/lorentz/embeddings_dim{dim}.npy   - embedding matrix [n_nodes, dim+1]
  data/embeddings/lorentz/node_index.json           - {atc_code: row_index} mapping
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import geoopt.optim
import networkx as nx
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

_SRC = Path(__file__).resolve().parent  # src/
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "evaluation"))
sys.path.insert(0, str(_SRC / "models"))

from config import (
    DATA_ATC_PREPARED,
    COL_CLASS_ID,
    ROOT,
    LORENTZ_DIM,
    LORENTZ_N_NEGATIVES,
    LORENTZ_LR,
    LORENTZ_EPOCHS,
    LORENTZ_BATCH_SIZE,
    LORENTZ_SEED,
    LORENTZ_BURN_IN_EPOCHS,
    LORENTZ_BURN_IN_MULTIPLIER,
)
from atc_utils import (
    build_atc_edges,
    add_virtual_root,
    VIRTUAL_ROOT,
    get_atc_level,
    tc_link_prediction_split,
)
from metrics import compute_mean_rank_and_map, compute_generality_correlation
from lorentz import LorentzEmbedding

_RESULTS_DIR = ROOT / "data" / "embeddings" / "lorentz"



class NegativeSamplingDataset(Dataset):
    """
    Positive v is placed at candidate index 0; n_negatives random non-neighbors follow.
    Negatives are drawn from the full transitive closure so true pairs are never mislabeled.

    Valid negative candidates are pre-computed per anchor node at construction time.
    Edges whose anchor has zero valid negatives (e.g. the virtual root, which is
    connected to every node) are silently dropped - they carry no gradient signal.
    Anchors with fewer than n_negatives candidates sample with replacement so that
    every item returns a tensor of fixed size [1 + n_negatives], which is required
    for DataLoader collation.
    """

    def __init__(
        self,
        positive_edges: list[tuple[int, int]],
        all_neighbors: dict[int, set[int]],
        n_nodes: int,
        n_negatives: int,
    ):
        self.n_negatives = n_negatives

        all_indices = np.arange(n_nodes, dtype=np.int64)
        self._valid_negatives: dict[int, np.ndarray] = {}
        for u, _ in positive_edges:
            if u in self._valid_negatives:
                continue
            exclude = all_neighbors.get(u, set()) | {u}
            mask = np.ones(n_nodes, dtype=bool)
            for ex in exclude:
                mask[ex] = False
            self._valid_negatives[u] = all_indices[mask]

        # Drop edges where anchor has no valid negatives (e.g. virtual root).
        self.positive_edges = [
            (u, v) for u, v in positive_edges if len(self._valid_negatives[u]) > 0
        ]
        n_dropped = len(positive_edges) - len(self.positive_edges)
        if n_dropped:
            print(f"  Dataset: dropped {n_dropped} edges whose anchor has no valid negatives")

    def __len__(self) -> int:
        return len(self.positive_edges)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        u, v = self.positive_edges[idx]
        valid = self._valid_negatives[u]
        replace = len(valid) < self.n_negatives
        chosen = valid[np.random.choice(len(valid), size=self.n_negatives, replace=replace)]
        candidates = torch.tensor([v, *chosen], dtype=torch.long)
        return torch.tensor(u, dtype=torch.long), candidates



def _load_codes() -> list[str]:
    """Load unique, valid ATC codes from data_atc.csv, excluding STY rows."""
    df = pd.read_csv(DATA_ATC_PREPARED)
    df = df[~df[COL_CLASS_ID].str.startswith("T")]
    df = df.drop_duplicates(subset=[COL_CLASS_ID])
    codes = [c for c in df[COL_CLASS_ID].tolist() if get_atc_level(str(c)) is not None]
    return sorted(set(codes))


def _build_transitive_closure(
    codes: list[str],
    node_index: dict[str, int],
) -> list[tuple[int, int]]:
    """
    Directed transitive closure of the ATC hierarchy as (u_idx, v_idx) pairs.
    Returns one directed edge per ancestor-descendant pair; reverses are added
    downstream to form the full bidirectional training set.
    """
    edges = build_atc_edges(codes)
    edges = add_virtual_root(edges, codes)
    g = nx.DiGraph()
    g.add_edges_from(edges)
    tc = nx.transitive_closure(g)
    return [
        (node_index[u], node_index[v])
        for u, v in tc.edges()
        if u in node_index and v in node_index
    ]



def _prepare_data(codes: list[str], symmetrize: bool = True) -> tuple[list, dict[int, set[int]], list]:
    """
    Returns:
        all_edges     - training TC edges (bidirectional, or ancestor->descendant if symmetrize=False)
        all_neighbors - {u: {neighbors of u}} for negative sampling
        eval_edges    - TC edges without VR involvement (reconstruction evaluation)

    symmetrize=True (default, N&K-adapted) adds reverse edges so children also rank
    parents close. symmetrize=False keeps the directed ancestor->descendant objective
    (closer to the N&K reference); used for the directed-vs-bidirectional ablation.
    """
    node_index = {code: i for i, code in enumerate(codes)}
    directed = _build_transitive_closure(codes, node_index)
    all_edges = directed + [(v, u) for u, v in directed] if symmetrize else list(directed)
    mode = "bidirectional" if symmetrize else "directed (ancestor->descendant)"
    print(f"Transitive closure: {len(directed)} directed pairs -> {len(all_edges)} {mode} training edges")

    all_neighbors: dict[int, set[int]] = {}
    for u, v in all_edges:
        all_neighbors.setdefault(u, set()).add(v)

    vr_idx = node_index[VIRTUAL_ROOT]
    eval_edges = [
        (u, v) for u, v in all_edges
        if u != vr_idx and v != vr_idx
    ]
    print(f"Eval edges (excl. VR): {len(eval_edges)}")

    return all_edges, all_neighbors, eval_edges


def _prepare_data_split(
    codes: list[str],
    train_fraction: float,
    seed: int,
    symmetrize: bool = True,
) -> tuple[list, dict[int, set[int]], list, list[tuple[str, str]]]:
    """
    Hold out (1-train_fraction) of the transitive-closure links for link prediction,
    following Nickel & Kiela (2017): individual TC relations are held out (root/leaf
    excluded), NOT direct edges. Holding out a direct edge in a strict tree would
    disconnect the child's whole subtree; holding out TC links keeps each node anchored
    by its remaining relations so the held-out pair stays predictable.

    Returns:
        train_edges      - bidirectional training links (TC minus held-out, incl. virtual root)
        all_neighbors    - {u: {neighbors}} for negative sampling (training links only)
        test_edges       - directed test links as index pairs (ancestor->descendant)
        test_edge_codes  - test links as (ancestor_code, descendant_code) string pairs (for saving)
    """
    real_codes = codes[1:]  # strip virtual root; split is on the real hierarchy only
    node_index = {code: i for i, code in enumerate(codes)}

    train_pairs, test_pairs = tc_link_prediction_split(real_codes, train_fraction, seed)
    print(f"TC link split (seed={seed}): {len(train_pairs)} train / {len(test_pairs)} test links "
          f"(N&K 2017: hold out TC relations, root/leaf excluded)")

    # Virtual root is an ancestor of every real node and is never a test candidate.
    directed_train = train_pairs + [(VIRTUAL_ROOT, c) for c in real_codes]
    train_directed_idx = [(node_index[u], node_index[v]) for u, v in directed_train]
    train_edges = (
        train_directed_idx + [(v, u) for u, v in train_directed_idx]
        if symmetrize else list(train_directed_idx)
    )
    mode = "bidirectional" if symmetrize else "directed"
    print(f"Train edges ({mode}, incl. virtual root): {len(train_edges)}")

    all_neighbors: dict[int, set[int]] = {}
    for u, v in train_edges:
        all_neighbors.setdefault(u, set()).add(v)

    # Directed test links (ancestor->descendant) — used for in-training MR logging.
    test_edges = [(node_index[u], node_index[v]) for u, v in test_pairs]

    return train_edges, all_neighbors, test_edges, test_pairs


def _init_training(
    args: argparse.Namespace,
    codes: list[str],
    all_edges: list,
    all_neighbors: dict[int, set[int]],
) -> tuple:
    """Create model, Riemannian optimizer, and data loader."""
    n_nodes = len(codes)
    model = LorentzEmbedding(n_nodes=n_nodes, dim=args.dim)
    optimizer = geoopt.optim.RiemannianSGD(model.parameters(), lr=args.lr)
    dataset = NegativeSamplingDataset(
        positive_edges=all_edges,
        all_neighbors=all_neighbors,
        n_nodes=n_nodes,
        n_negatives=args.n_negatives,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    return model, optimizer, loader


def _set_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _write_embeddings(emb: np.ndarray, codes: list[str], dim: int, suffix: str) -> Path:
    """Strip the virtual root and persist embeddings + root + node index.

    Shared by the periodic checkpoint and the final save, so an interrupted run leaves a
    directly usable embeddings_dim{dim}{suffix}.npy in the same format as a completed run.
    """
    real_codes = codes[1:]
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = _RESULTS_DIR / f"embeddings_dim{dim}{suffix}.npy"
    np.save(emb_path, emb[1:])
    np.save(_RESULTS_DIR / f"virtual_root_emb_dim{dim}{suffix}.npy", emb[0])
    (_RESULTS_DIR / "node_index.json").write_text(
        json.dumps({c: i for i, c in enumerate(real_codes)})
    )
    return emb_path


def _run_training_loop(
    args: argparse.Namespace,
    model: LorentzEmbedding,
    optimizer,
    loader: DataLoader,
    eval_edges: list,
    all_edges: list,
    codes: list[str],
    suffix: str,
) -> None:
    """
    Burn-in: epochs 1–burn_in use lr * burn_in_multiplier; full lr thereafter.
    Follows the Nickel & Kiela (2018) Lorentz reference implementation.
    Logs reconstruction mean rank every 25 epochs and checkpoints the embeddings every
    250 epochs, so a crash/standby during the multi-hour run does not lose progress.
    """
    burn_in_lr = args.lr * args.burn_in_multiplier
    print(f"Burn-in: epochs 1–{args.burn_in} at lr={burn_in_lr:.4g}, "
          f"then full lr={args.lr:.4g}")

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training epochs", unit="epoch"):
        _set_lr(optimizer, burn_in_lr if epoch <= args.burn_in else args.lr)
        model.train()
        total_loss = 0.0
        for u_idx, candidates_idx in tqdm(loader, desc=f"Epoch {epoch}", leave=False, unit="batch"):
            optimizer.zero_grad()
            distances = model(u_idx, candidates_idx)
            targets = torch.zeros(len(u_idx), dtype=torch.long)
            loss = F.cross_entropy(-distances, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 25 == 0:
            avg_loss = total_loss / len(loader)
            model.eval()
            with torch.no_grad():
                emb = model.get_all_embeddings()
            recon_rank = float("nan")
            if eval_edges:
                recon_rank = compute_mean_rank_and_map(emb, eval_edges, all_edges)["mean_rank"]
            print(f"Epoch {epoch:4d} | loss={avg_loss:.4f} | recon_mean_rank={recon_rank:.2f}")

        # Periodic checkpoint; the final epoch is persisted by _save_outputs.
        if epoch % 250 == 0 and epoch < args.epochs:
            with torch.no_grad():
                _write_embeddings(model.get_all_embeddings(), codes, args.dim, suffix)
            print(f"  [checkpoint] saved embeddings at epoch {epoch}")


def _save_outputs(
    args: argparse.Namespace,
    model: LorentzEmbedding,
    codes: list[str],
    eval_edges: list,
    all_edges: list,
    suffix: str = "",
    test_edge_codes: list[tuple[str, str]] | None = None,
) -> None:
    """Run final evaluation, compute rho, and save embeddings + node index.

    suffix: appended to output filenames, e.g. '_train90' for split runs.
    test_edge_codes: if provided, saved as test_edges{suffix}.json for use by the Q2.2 evaluation.
    """
    model.eval()
    with torch.no_grad():
        emb = model.get_all_embeddings()

    if eval_edges:
        recon_metrics = compute_mean_rank_and_map(emb, eval_edges, all_edges)
        label = "Link-pred test" if suffix else "Reconstruction"
        print(
            f"{label}  MR: {recon_metrics['mean_rank']:.2f}  "
            f"MAP: {recon_metrics['map']:.4f}"
        )

    real_codes = codes[1:]
    real_node_index = {c: i for i, c in enumerate(real_codes)}

    original_edges = build_atc_edges(real_codes)
    generality_scores = emb[1:, 0]
    rho = compute_generality_correlation(generality_scores, real_node_index, original_edges)
    print(f"Generality rho: {rho:.4f}")

    emb_path = _write_embeddings(emb, codes, args.dim, suffix)
    print(f"\nSaved embeddings -> {emb_path}  shape={emb[1:].shape}  ({len(real_codes)} codes)")

    if test_edge_codes is not None:
        test_path = _RESULTS_DIR / f"test_edges{suffix}.json"
        with open(test_path, "w") as f:
            json.dump(test_edge_codes, f)
        print(f"Saved test edges -> {test_path}  ({len(test_edge_codes)} pairs)")



def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # negative sampling uses np.random.choice

    codes = [VIRTUAL_ROOT] + _load_codes()
    print(f"Loaded {len(codes) - 1} ATC codes + virtual root")

    symmetrize = not args.directed
    edge_suffix = "_directed" if args.directed else ""

    if args.train_split is not None:
        pct = int(args.train_split * 100)
        suffix = f"_train{pct}{edge_suffix}"
        train_edges, all_neighbors, test_edges, test_edge_codes = _prepare_data_split(
            codes, args.train_split, args.seed, symmetrize
        )
        model, optimizer, loader = _init_training(args, codes, train_edges, all_neighbors)
        _run_training_loop(args, model, optimizer, loader, test_edges, train_edges, codes, suffix)
        _save_outputs(args, model, codes, test_edges, train_edges,
                      suffix=suffix, test_edge_codes=test_edge_codes)
    else:
        all_edges, all_neighbors, eval_edges = _prepare_data(codes, symmetrize)
        model, optimizer, loader = _init_training(args, codes, all_edges, all_neighbors)
        _run_training_loop(args, model, optimizer, loader, eval_edges, all_edges, codes, edge_suffix)
        _save_outputs(args, model, codes, eval_edges, all_edges, suffix=edge_suffix)



def main() -> None:
    parser = argparse.ArgumentParser(description="Train Lorentz embeddings on ATC hierarchy.")
    parser.add_argument("--dim", type=int, default=LORENTZ_DIM,
                        help=f"Embedding dimension (default: {LORENTZ_DIM})")
    parser.add_argument("--n-negatives", type=int, default=LORENTZ_N_NEGATIVES,
                        dest="n_negatives",
                        help=f"Negatives per positive (default: {LORENTZ_N_NEGATIVES})")
    parser.add_argument("--lr", type=float, default=LORENTZ_LR,
                        help=f"Riemannian SGD learning rate (default: {LORENTZ_LR})")
    parser.add_argument("--epochs", type=int, default=LORENTZ_EPOCHS,
                        help=f"Training epochs (default: {LORENTZ_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=LORENTZ_BATCH_SIZE,
                        dest="batch_size",
                        help=f"Batch size (default: {LORENTZ_BATCH_SIZE})")
    parser.add_argument("--seed", type=int, default=LORENTZ_SEED,
                        help=f"Random seed (default: {LORENTZ_SEED})")
    parser.add_argument("--burn-in", type=int, default=LORENTZ_BURN_IN_EPOCHS,
                        dest="burn_in",
                        help=f"Burn-in epochs (default: {LORENTZ_BURN_IN_EPOCHS})")
    parser.add_argument("--burn-in-multiplier", type=float, default=LORENTZ_BURN_IN_MULTIPLIER,
                        dest="burn_in_multiplier",
                        help=f"lr multiplier during burn-in (default: {LORENTZ_BURN_IN_MULTIPLIER})")
    parser.add_argument("--train-split", type=float, default=None,
                        dest="train_split",
                        help="If set, train on this fraction of transitive-closure links (e.g. 0.9). "
                             "Remaining TC links are held out as test set (Q2.2 link prediction). "
                             "Saves embeddings_dim{dim}_train{pct}.npy and test_edges_train{pct}.json.")
    parser.add_argument("--directed", action="store_true",
                        help="Train only on directed ancestor->descendant TC edges (no reverse). "
                             "Default is bidirectional (N&K-adapted). Used for the directed-vs-"
                             "bidirectional ablation; writes a '_directed' suffix so it never "
                             "overwrites the default embeddings. Compare generality rho across modes.")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
