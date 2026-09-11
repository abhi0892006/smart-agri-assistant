"""
Baseline deep learning model for Fertilizer Recommendation.

Same architecture family as the crop recommendation baseline (per
Section III-D: baseline MLPs for each tabular task), sized smaller here
to match the much smaller fertilizer dataset and reduce overfitting risk.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FertilizerRecommendationMLP(nn.Module):
    """Feed-forward network for multi-class fertilizer classification.

    Input is [scaled numeric features | one-hot soil type | one-hot crop type].
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
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
