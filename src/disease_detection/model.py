"""
Baseline CNN for multi-crop plant disease detection.

Per Section III-A/III-D of the paper: a baseline CNN architecture,
to be compared against deeper/pretrained architectures (VGG, ResNet,
DenseNet) in future work. This is that baseline — a compact
from-scratch CNN sized for a small (1200-image) dataset trained on CPU.
Transfer learning with a pretrained backbone (e.g. ResNet18) is the
natural next step once more data/compute is available; the input
normalization already used here (ImageNet mean/std) is chosen so
swapping in a pretrained backbone later is a drop-in change.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DiseaseDetectionCNN(nn.Module):
    def __init__(self, num_classes: int, image_size: int = 96):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> image_size/2

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> image_size/4

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> image_size/8
        )

        reduced = image_size // 8
        flattened_dim = 64 * reduced * reduced

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
