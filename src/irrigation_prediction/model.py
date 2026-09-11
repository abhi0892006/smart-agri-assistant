"""
Baseline deep learning model for Irrigation Prediction.

Binary classification (irrigate: yes/no), same MLP architecture family as
the other tabular modules (Section III-D), sized for a small dataset.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class IrrigationPredictionMLP(nn.Module):
    """Feed-forward network for binary irrigation classification.

    Input is [scaled numeric features | one-hot crop type].
    Outputs 2 logits (No, Yes) rather than a single sigmoid unit so it
    shares the same CrossEntropyLoss-based training loop as the other
    modules in this project.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [32, 16]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
