from __future__ import annotations

import unittest

import numpy as np
import torch

from models.multitask_gnn import CYPDMPNN, batch_graphs
from src.training import masked_loss


class TorchPipelineTests(unittest.TestCase):
    def test_dmpnn_forward_and_separate_heads(self):
        graph = batch_graphs(["CCO", "c1ccccc1"])
        model = CYPDMPNN(("CYP1A2", "CYP2D6"), message_hidden_dim=32, ffn_hidden_dim=16)
        output = model(graph)
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertIsNot(model.heads["CYP1A2"], model.heads["CYP2D6"])

    def test_masked_and_interval_losses(self):
        prediction = torch.tensor([[0.0, 9.0], [2.0, 4.0]], requires_grad=True)
        target = torch.tensor([[0.0, float("nan")], [1.0, 4.0]])
        lower = torch.tensor([[-0.2, float("nan")], [0.8, 3.8]])
        upper = torch.tensor([[0.2, float("nan")], [1.2, 4.2]])
        weights = torch.ones_like(target)
        standard = masked_loss(prediction, target, lower, upper, weights, "standard")
        interval = masked_loss(prediction, target, lower, upper, weights, "interval_aware")
        self.assertTrue(np.isfinite(float(standard.detach())))
        self.assertLess(float(interval.detach()), float(standard.detach()))
        interval.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())


if __name__ == "__main__":
    unittest.main()
