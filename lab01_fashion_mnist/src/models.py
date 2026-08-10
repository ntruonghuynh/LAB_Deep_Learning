"""Model architectures for FashionMNIST classification.

Two intentionally simple, explainable architectures:
- FashionMLP: fully-connected network on flattened pixels.
- FashionCNN: small conv/pool stack followed by a classifier head.
"""

from typing import Optional

import torch
from torch import nn


class FashionMLP(nn.Module):
    """Simple fully-connected network for 28x28 grayscale images.

    Flatten -> [Linear -> ReLU -> Dropout] x N -> Linear(10)
    """

    def __init__(self, hidden_sizes: Optional[list[int]] = None, dropout: float = 0.2):
        super().__init__()
        hidden_sizes = hidden_sizes if hidden_sizes is not None else [256, 128]
        layers: list[nn.Module] = [nn.Flatten()]
        in_features = 28 * 28
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FashionCNN(nn.Module):
    """Small CNN: two Conv-ReLU-Pool blocks followed by a fully-connected classifier.

    Input: (N, 1, 28, 28) -> Conv block 1 -> Conv block 2 -> Flatten -> FC -> 10 classes.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model(name: str, config: dict) -> nn.Module:
    """Factory that builds a model from a config dict (as loaded from YAML).

    Args:
        name: "mlp" or "cnn".
        config: the "model" section of a run's config (e.g. hidden_sizes, dropout).
    """
    if name == "mlp":
        return FashionMLP(
            hidden_sizes=config.get("hidden_sizes", [256, 128]),
            dropout=config.get("dropout", 0.2),
        )
    if name == "cnn":
        return FashionCNN(dropout=config.get("dropout", 0.3))
    raise ValueError(f"Unknown model name: {name!r}. Expected 'mlp' or 'cnn'.")
