"""Chemprop-style D-MPNN public model surface.

The implementation is self-contained so cluster runs do not depend on a changing
external Chemprop API. Checkpoints expose the encoder separately for transfer.
"""

from .multitask_gnn import CYPDMPNN, DMPNNEncoder, GraphBatch, batch_graphs

__all__ = ["CYPDMPNN", "DMPNNEncoder", "GraphBatch", "batch_graphs"]
