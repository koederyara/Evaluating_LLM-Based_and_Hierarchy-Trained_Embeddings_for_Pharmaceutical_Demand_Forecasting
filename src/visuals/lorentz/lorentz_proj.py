"""Projection utilities: Lorentz (hyperboloid) → 2D Poincaré disk coordinates.

For Lorentz dim=2 the mapping is exact (Eq. 11, Nickel & Kiela 2018).
For dim>2 an additional PCA or t-SNE step reduces the Poincaré ball to 2D —
this is an approximation and the result is labelled accordingly.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def lorentz_to_poincare(emb: np.ndarray) -> np.ndarray:
    """Map Lorentz embeddings to the Poincaré ball (Eq. 11, Nickel & Kiela 2018)."""
    return emb[:, 1:] / (emb[:, 0:1] + 1)


def lorentz_to_poincare_single(v: np.ndarray) -> np.ndarray:
    """Map a single Lorentz point to Poincaré ball coordinates."""
    return v[1:] / (v[0] + 1)


def _normalize_to_disk(coords: np.ndarray, margin: float = 0.95) -> np.ndarray:
    """PCA and t-SNE output arbitrary scales, so rescale to keep every point inside the
    unit-circle boundary the disk scripts draw."""
    max_norm = np.linalg.norm(coords, axis=1).max()
    return coords / max_norm * margin if max_norm > 0 else coords


def project_to_disk(
    emb: np.ndarray,
    method: str = "pca",
) -> tuple[np.ndarray, str, bool]:
    """Project to 2D disk coordinates; returns (coords, label, is_approx).

    Exact for dim=2, an approximation above it — hence is_approx, which the plots print
    so no reader mistakes a reduced view for the trained geometry.

    Pass every point that appears in the plot at once: t-SNE cannot transform new points
    after fitting.
    """
    lorentz_dim = emb.shape[1] - 1
    ball = lorentz_to_poincare(emb)

    if lorentz_dim == 2:
        return ball, "Eq. 11, Nickel & Kiela (2018)", False

    if method == "pca":
        reducer = PCA(n_components=2, random_state=0)
        coords = reducer.fit_transform(ball)
        var = reducer.explained_variance_ratio_.sum()
        label = (
            f"Poincaré ball (dim={lorentz_dim}) + PCA → 2D "
            f"[{var:.0%} variance explained] (approximation)"
        )
        return _normalize_to_disk(coords), label, True

    if method == "tsne":
        # Pre-reduce with PCA, otherwise t-SNE is impractically slow at high ball dim.
        pre = ball if ball.shape[1] <= 10 else PCA(n_components=10, random_state=0).fit_transform(ball)
        coords = TSNE(n_components=2, random_state=0, init="pca", perplexity=30).fit_transform(pre)
        label = f"Poincaré ball (dim={lorentz_dim}) + t-SNE → 2D (approximation)"
        return _normalize_to_disk(coords), label, True

    raise ValueError(f"Unknown method {method!r}. Choose 'pca' or 'tsne'.")
