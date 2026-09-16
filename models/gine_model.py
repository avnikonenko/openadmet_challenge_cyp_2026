"""GINE graph encoder built on PyTorch Geometric's ``GINEConv``.

This is a genuinely different message-passing family from the project's directed
D-MPNN: node states are updated by an injective sum aggregation of edge-conditioned
neighbour states, rather than by directed bond messages. It consumes the same
`src.features` atom/bond featurization and the same `GraphBatch` produced by
`batch_graphs`, so every fold, checkpoint, transfer, and evaluation path is shared
with the D-MPNN.
"""

from __future__ import annotations

import torch
from torch import nn

from src.features import ATOM_FDIM, BOND_FDIM

from .multitask_gnn import GraphBatch, build_head, reset_module_parameters


def require_torch_geometric():
    try:
        import torch_geometric  # noqa: F401
        from torch_geometric.nn import GINEConv, global_add_pool, global_max_pool, global_mean_pool
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "model.type=gine requires torch-geometric. Create the modelling environment "
            "from environment-modeling.yml (CPU) or environment-modeling-cuda.yml (GPU)."
        ) from exc
    return GINEConv, {"mean": global_mean_pool, "sum": global_add_pool, "max": global_max_pool}


NORMALIZATIONS = ("layernorm", "batchnorm", "none")
POOLINGS = ("mean", "sum", "max")


class GINEEncoder(nn.Module):
    """Residual GINE stack with optional normalization and global pooling."""

    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 5,
        dropout: float = 0.1,
        residual: bool = True,
        normalization: str = "layernorm",
        pooling: str = "mean",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("model.num_layers must be >=1")
        if normalization not in NORMALIZATIONS:
            raise ValueError(f"model.normalization must be one of {NORMALIZATIONS}")
        if pooling not in POOLINGS:
            raise ValueError(f"model.pooling must be one of {POOLINGS}")
        convolution, pools = require_torch_geometric()
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        self.num_layers = num_layers
        self.residual = bool(residual)
        self.normalization = normalization
        self.pooling = pooling
        self._pool = pools[pooling]
        self.atom_encoder = nn.Linear(ATOM_FDIM, hidden_dim)
        self.convolutions = nn.ModuleList(
            [
                convolution(
                    nn.Sequential(
                        nn.Linear(hidden_dim, 2 * hidden_dim),
                        nn.ReLU(),
                        nn.Linear(2 * hidden_dim, hidden_dim),
                    ),
                    train_eps=True,
                    edge_dim=BOND_FDIM,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [self._normalization(hidden_dim) for _ in range(num_layers)]
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def _normalization(self, dim: int) -> nn.Module:
        if self.normalization == "layernorm":
            return nn.LayerNorm(dim)
        if self.normalization == "batchnorm":
            return nn.BatchNorm1d(dim)
        return nn.Identity()

    @staticmethod
    def _batch_index(graph: GraphBatch) -> torch.Tensor:
        lengths = torch.tensor(
            [length for _start, length in graph.atom_scopes],
            device=graph.atoms.device, dtype=torch.long,
        )
        return torch.repeat_interleave(
            torch.arange(len(graph.atom_scopes), device=graph.atoms.device), lengths
        )

    def forward(self, graph: GraphBatch) -> torch.Tensor:
        edge_index = torch.stack([graph.sources, graph.destinations], dim=0)
        edge_attr = graph.bonds
        hidden = self.activation(self.atom_encoder(graph.atoms))
        for convolution, norm in zip(self.convolutions, self.norms):
            updated = convolution(hidden, edge_index, edge_attr)
            updated = self.dropout(self.activation(norm(updated)))
            hidden = hidden + updated if self.residual else updated
        return self._pool(hidden, self._batch_index(graph), size=len(graph.atom_scopes))


class CYPGINE(nn.Module):
    """GINE encoder with one prediction head per CYP, mirroring `CYPDMPNN`."""

    def __init__(
        self,
        task_names: list[str] | tuple[str, ...],
        hidden_dim: int = 512,
        num_layers: int = 5,
        dropout: float = 0.1,
        residual: bool = True,
        normalization: str = "layernorm",
        pooling: str = "mean",
        ffn_hidden_dim: int = 512,
        ffn_num_layers: int = 3,
        ffn_dropout: float = 0.1,
        head_type: str = "mlp",
        head_layer_norm: bool = False,
    ):
        super().__init__()
        self.task_names = tuple(task_names)
        self.encoder = GINEEncoder(
            hidden_dim, num_layers, dropout, residual, normalization, pooling
        )
        self.heads = nn.ModuleDict(
            {
                task: build_head(
                    hidden_dim, ffn_hidden_dim, ffn_num_layers, ffn_dropout,
                    head_type, head_layer_norm,
                )
                for task in self.task_names
            }
        )

    def forward(self, graph: GraphBatch) -> torch.Tensor:
        embedding = self.encoder(graph)
        return torch.cat([self.heads[task](embedding) for task in self.task_names], dim=1)

    def reset_heads(self) -> None:
        reset_module_parameters(self.heads)
