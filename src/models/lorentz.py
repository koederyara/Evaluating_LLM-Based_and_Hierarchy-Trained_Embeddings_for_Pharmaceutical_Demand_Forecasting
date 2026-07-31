"""
Lorentz (hyperboloid) embedding model following Nickel & Kiela (2018).

The hyperboloid model represents hyperbolic space as the upper sheet of a two-sheeted
hyperboloid: H^n = {x ∈ R^(n+1) : <x, x>_L = -1, x_0 > 0} with Minkowski inner
product <u, v>_L = -u_0 v_0 + u_1 v_1 + ... + u_n v_n.
"""
import numpy as np
import torch
import torch.nn as nn
import geoopt


class LorentzEmbedding(nn.Module):
    def __init__(self, n_nodes: int, dim: int):
        super().__init__()
        self.manifold = geoopt.manifolds.Lorentz()

        # Eq. 6 of Nickel & Kiela (2018): sample spatial coords near origin so
        # all nodes start close together — avoids large initial gradients.
        spatial = torch.zeros(n_nodes, dim).uniform_(-0.001, 0.001)
        x0 = torch.sqrt(1.0 + (spatial ** 2).sum(dim=-1, keepdim=True))
        init = torch.cat([x0, spatial], dim=-1)  # [n_nodes, dim+1]

        self.embeddings = geoopt.ManifoldParameter(init, manifold=self.manifold)

    def forward(
        self,
        u_idx: torch.Tensor,
        candidates_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Lorentz distances from each anchor [batch] to candidates [batch, 1+n_neg]."""
        u = self.embeddings[u_idx]                         # [batch, dim+1]
        candidates = self.embeddings[candidates_idx]       # [batch, 1+n_neg, dim+1]
        u_expanded = u.unsqueeze(1).expand_as(candidates)  # [batch, 1+n_neg, dim+1]
        return self.manifold.dist(u_expanded, candidates)  # [batch, 1+n_neg]

    def get_all_embeddings(self) -> np.ndarray:
        """Return full embedding matrix as a detached numpy array, shape [n_nodes, dim+1]."""
        return self.embeddings.detach().cpu().numpy()
