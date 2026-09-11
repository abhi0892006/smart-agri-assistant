"""
Baseline deep learning model for Crop Recommendation.

Per Section III-D of the paper: "baseline architectures for each tabular
task (e.g., multilayer perceptrons) will be compared against improved or
integrated deep learning architectures." This is the baseline MLP.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CropRecommendationMLP(nn.Module):
    """A simple feed-forward network for multi-class crop classification.

    Architecture: input -> [Linear -> BatchNorm -> ReLU -> Dropout]* -> output logits.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]

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
