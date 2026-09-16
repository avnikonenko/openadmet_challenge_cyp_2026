from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.features import ATOM_FDIM, BOND_FDIM, MoleculeGraph, molecule_graph


@dataclass
class GraphBatch:
    atoms: torch.Tensor
    bonds: torch.Tensor
    sources: torch.Tensor
    destinations: torch.Tensor
    reverses: torch.Tensor
    atom_scopes: list[tuple[int, int]]

    def to(self, device: torch.device | str) -> "GraphBatch":
        return GraphBatch(
            self.atoms.to(device), self.bonds.to(device), self.sources.to(device),
            self.destinations.to(device), self.reverses.to(device), self.atom_scopes,
        )


def batch_graphs(smiles: list[str]) -> GraphBatch:
    atoms: list[np.ndarray] = []
    bonds: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    destinations: list[np.ndarray] = []
    reverses: list[np.ndarray] = []
    scopes: list[tuple[int, int]] = []
    atom_offset = bond_offset = 0
    for text in smiles:
        graph: MoleculeGraph = molecule_graph(text)
        atom_count = len(graph.atom_features)
        bond_count = len(graph.bond_features)
        scopes.append((atom_offset, atom_count))
        atoms.append(graph.atom_features)
        bonds.append(graph.bond_features)
        sources.append(graph.bond_sources + atom_offset)
        destinations.append(graph.bond_destinations + atom_offset)
        reverses.append(graph.reverse_bonds + bond_offset)
        atom_offset += atom_count
        bond_offset += bond_count
    return GraphBatch(
        atoms=torch.from_numpy(np.concatenate(atoms, axis=0)),
        bonds=torch.from_numpy(np.concatenate(bonds, axis=0) if bonds else np.empty((0, BOND_FDIM), np.float32)),
        sources=torch.from_numpy(np.concatenate(sources) if sources else np.empty(0, np.int64)),
        destinations=torch.from_numpy(np.concatenate(destinations) if destinations else np.empty(0, np.int64)),
        reverses=torch.from_numpy(np.concatenate(reverses) if reverses else np.empty(0, np.int64)),
        atom_scopes=scopes,
    )


class DMPNNEncoder(nn.Module):
    """Directed message-passing encoder following the Chemprop D-MPNN update."""

    def __init__(self, hidden_dim: int = 300, depth: int = 3, dropout: float = 0.1, aggregation: str = "mean"):
        super().__init__()
        if depth < 1 or aggregation not in {"mean", "sum"}:
            raise ValueError("depth must be >=1 and aggregation must be mean or sum")
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.aggregation = aggregation
        self.input_layer = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_dim)
        self.message_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.atom_layer = nn.Linear(ATOM_FDIM + hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: GraphBatch) -> torch.Tensor:
        atom_count = graph.atoms.shape[0]
        if graph.bonds.shape[0]:
            initial = self.activation(
                self.input_layer(torch.cat([graph.atoms[graph.sources], graph.bonds], dim=1))
            )
            messages = initial
            for _ in range(self.depth - 1):
                incoming = torch.zeros(
                    (atom_count, self.hidden_dim), device=messages.device, dtype=messages.dtype
                )
                incoming.index_add_(0, graph.destinations, messages)
                directed = incoming[graph.sources] - messages[graph.reverses]
                messages = self.dropout(self.activation(initial + self.message_layer(directed)))
            atom_messages = torch.zeros(
                (atom_count, self.hidden_dim), device=messages.device, dtype=messages.dtype
            )
            atom_messages.index_add_(0, graph.destinations, messages)
        else:
            atom_messages = torch.zeros(
                (atom_count, self.hidden_dim), device=graph.atoms.device, dtype=graph.atoms.dtype
            )
        atom_hidden = self.dropout(
            self.activation(self.atom_layer(torch.cat([graph.atoms, atom_messages], dim=1)))
        )
        molecules = []
        for start, length in graph.atom_scopes:
            values = atom_hidden[start : start + length]
            molecules.append(values.mean(dim=0) if self.aggregation == "mean" else values.sum(dim=0))
        return torch.stack(molecules)


class ResidualBlock(nn.Module):
    """x -> Linear -> activation -> dropout -> Linear -> activation -> dropout -> x + block(x).

    The optional LayerNorm is applied before the block (pre-norm) so the identity
    path stays unnormalized and the block remains a pure residual correction.
    """

    def __init__(self, dim: int, dropout: float, layer_norm: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if layer_norm else nn.Identity()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.block(self.norm(values))


def build_head(
    input_dim: int, hidden_dim: int, layers: int, dropout: float,
    head_type: str = "mlp", layer_norm: bool = False,
) -> nn.Module:
    """Build one scalar prediction head.

    ``mlp`` reproduces the original plain feed-forward head exactly, so existing
    checkpoints keep loading. ``residual_mlp`` projects into the hidden width once
    and then applies ``max(1, layers - 1)`` residual blocks before the output layer.
    """
    if head_type == "mlp":
        if layer_norm:
            raise ValueError(
                "model.head_layer_norm applies only to head_type=residual_mlp; "
                "the mlp head is kept bit-identical to the original architecture"
            )
        modules: list[nn.Module] = []
        current = input_dim
        for _ in range(max(0, layers - 1)):
            modules.extend([nn.Linear(current, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            current = hidden_dim
        modules.append(nn.Linear(current, 1))
        return nn.Sequential(*modules)
    if head_type != "residual_mlp":
        raise ValueError(f"model.head_type must be mlp or residual_mlp, not {head_type!r}")
    blocks = [
        ResidualBlock(hidden_dim, dropout, layer_norm) for _ in range(max(1, layers - 1))
    ]
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        *blocks, nn.Linear(hidden_dim, 1),
    )


def reset_module_parameters(module: nn.Module) -> None:
    """Reinitialize every parameterized submodule.

    The root module is skipped: a container that defines its own ``reset_parameters``
    in terms of this helper would otherwise recurse into itself forever.
    """
    for child in module.modules():
        if child is module:
            continue
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()


class CYPDMPNN(nn.Module):
    def __init__(
        self,
        task_names: list[str] | tuple[str, ...],
        message_hidden_dim: int = 300,
        message_passing_depth: int = 3,
        dropout: float = 0.1,
        aggregation: str = "mean",
        ffn_hidden_dim: int = 300,
        ffn_num_layers: int = 2,
        ffn_dropout: float = 0.1,
        head_type: str = "mlp",
        head_layer_norm: bool = False,
    ):
        super().__init__()
        self.task_names = tuple(task_names)
        self.encoder = DMPNNEncoder(
            message_hidden_dim, message_passing_depth, dropout, aggregation
        )
        self.heads = nn.ModuleDict(
            {
                task: build_head(
                    message_hidden_dim, ffn_hidden_dim, ffn_num_layers, ffn_dropout,
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
