"""Target-conditioned and joint multi-assay prediction heads.

Instead of four independent CYP heads, these predictors learn one shared function
``f(molecular_embedding, CYP_embedding[, assay_embedding]) -> pIC50``. The encoder is
unchanged and interchangeable (D-MPNN or GINE), so this is an architecture comparison
against `cyp_specific_heads`, not a replacement for it.

Task names carry the conditioning: ``"CYP2D6"`` for a single-assay model and
``"CYP2D6|single_concentration"`` for the joint multi-assay model. The embedding
tables are sized from fixed vocabularies rather than from the active task list, so a
checkpoint trained on all tasks can be reloaded when refining a single endpoint.
"""

from __future__ import annotations

import torch
from torch import nn

from src.constants import CYPS

from .multitask_gnn import GraphBatch, build_head, reset_module_parameters


ASSAYS = ("direct_pic50", "single_concentration")
TASK_SEPARATOR = "|"


def split_task_name(task: str) -> tuple[str, str | None]:
    """Split ``CYP`` or ``CYP|assay`` into its conditioning components."""
    if TASK_SEPARATOR not in task:
        return task, None
    cyp, assay = task.split(TASK_SEPARATOR, 1)
    return cyp, assay


class TargetConditionedHeads(nn.Module):
    """Learned CYP (and optionally assay) embeddings plus one shared predictor."""

    def __init__(
        self,
        task_names: list[str] | tuple[str, ...],
        input_dim: int,
        cyp_embedding_dim: int = 32,
        predictor_hidden_dim: int = 600,
        predictor_layers: int = 3,
        dropout: float = 0.1,
        head_type: str = "mlp",
        head_layer_norm: bool = False,
        assay_conditioned: bool = False,
        assay_embedding_dim: int | None = None,
    ):
        super().__init__()
        self.task_names = tuple(task_names)
        self.assay_conditioned = bool(assay_conditioned)
        self.cyp_vocabulary = tuple(CYPS)
        self.assay_vocabulary = tuple(ASSAYS)
        cyp_indices, assay_indices = [], []
        for task in self.task_names:
            cyp, assay = split_task_name(task)
            if cyp not in self.cyp_vocabulary:
                raise ValueError(f"Unknown CYP in task name {task!r}")
            if self.assay_conditioned:
                if assay is None:
                    raise ValueError(
                        f"Joint multi-assay tasks must be named CYP{TASK_SEPARATOR}assay, got {task!r}"
                    )
                if assay not in self.assay_vocabulary:
                    raise ValueError(f"Unknown assay in task name {task!r}")
                assay_indices.append(self.assay_vocabulary.index(assay))
            cyp_indices.append(self.cyp_vocabulary.index(cyp))
        # Non-persistent: these are derived from task_names at construction, so they must
        # stay out of state_dict. Persisting them would size the checkpoint by the active
        # task count and break reloading a full multitask checkpoint for a single-endpoint
        # refinement, which is exactly what the fixed embedding vocabularies exist to allow.
        self.register_buffer(
            "cyp_indices", torch.tensor(cyp_indices, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "assay_indices",
            torch.tensor(assay_indices or [0] * len(self.task_names), dtype=torch.long),
            persistent=False,
        )
        self.cyp_embedding = nn.Embedding(len(self.cyp_vocabulary), cyp_embedding_dim)
        conditioning_dim = cyp_embedding_dim
        if self.assay_conditioned:
            assay_dim = int(assay_embedding_dim or cyp_embedding_dim)
            self.assay_embedding = nn.Embedding(len(self.assay_vocabulary), assay_dim)
            conditioning_dim += assay_dim
        else:
            self.assay_embedding = None
        self.predictor = build_head(
            input_dim + conditioning_dim, predictor_hidden_dim, predictor_layers,
            dropout, head_type, head_layer_norm,
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        rows = embedding.shape[0]
        outputs = []
        for position in range(len(self.task_names)):
            conditioning = [self.cyp_embedding(self.cyp_indices[position])]
            if self.assay_embedding is not None:
                conditioning.append(self.assay_embedding(self.assay_indices[position]))
            context = torch.cat(conditioning, dim=-1).unsqueeze(0).expand(rows, -1)
            outputs.append(self.predictor(torch.cat([embedding, context], dim=1)))
        return torch.cat(outputs, dim=1)

    def reset_parameters(self) -> None:
        reset_module_parameters(self)


class CYPTargetConditioned(nn.Module):
    """Any molecular encoder combined with a target-conditioned predictor."""

    def __init__(
        self, encoder: nn.Module, heads: TargetConditionedHeads,
        task_names: list[str] | tuple[str, ...],
    ):
        super().__init__()
        self.task_names = tuple(task_names)
        self.encoder = encoder
        self.heads = heads

    def forward(self, graph: GraphBatch) -> torch.Tensor:
        return self.heads(self.encoder(graph))

    def reset_heads(self) -> None:
        reset_module_parameters(self.heads)
