"""Model architectures for FashionMNIST classification.

Includes two simple from-scratch architectures and selected torchvision pretrained
architectures for transfer learning experiments.
"""

import torch
from torch import nn
from torchvision import models
from typing import Optional


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


def _freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _unfreeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def _apply_resnet_transfer_learning(model: nn.Module, transfer_config: dict) -> None:
    freeze_backbone = transfer_config.get("freeze_backbone", False)
    unfreeze_layers = transfer_config.get("unfreeze_layers", [])

    if freeze_backbone or unfreeze_layers:
        _freeze_all(model)

    if unfreeze_layers:
        for layer_name in unfreeze_layers:
            if not hasattr(model, layer_name):
                raise ValueError(f"ResNet18 has no layer named {layer_name!r}.")
            _unfreeze_module(getattr(model, layer_name))
    elif freeze_backbone:
        _unfreeze_module(model.fc)


def _apply_mobilenet_transfer_learning(model: nn.Module, transfer_config: dict) -> None:
    freeze_backbone = transfer_config.get("freeze_backbone", False)
    unfreeze_layers = transfer_config.get("unfreeze_layers", [])

    if freeze_backbone or unfreeze_layers:
        _freeze_all(model)

    if unfreeze_layers:
        for layer_name in unfreeze_layers:
            if layer_name == "classifier":
                _unfreeze_module(model.classifier)
            elif layer_name == "features":
                _unfreeze_module(model.features)
            else:
                raise ValueError(f"MobileNetV2 supports unfreezing 'features' or 'classifier', got {layer_name!r}.")
    elif freeze_backbone:
        _unfreeze_module(model.classifier)


def build_resnet18(config: dict, transfer_config: Optional[dict] = None) -> nn.Module:
    """Build ResNet18 and replace the ImageNet classifier with a FashionMNIST head."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if config.get("pretrained", False) else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, config.get("num_classes", 10))
    _apply_resnet_transfer_learning(model, transfer_config or {})
    return model


def build_mobilenet_v2(config: dict, transfer_config: Optional[dict] = None) -> nn.Module:
    """Build MobileNetV2 and replace the ImageNet classifier with a FashionMNIST head."""
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if config.get("pretrained", False) else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, config.get("num_classes", 10))
    _apply_mobilenet_transfer_learning(model, transfer_config or {})
    return model


def build_model(name: str, config: dict, transfer_config: Optional[dict] = None) -> nn.Module:
    """Factory that builds a model from a config dict (as loaded from YAML).

    Args:
        name: "mlp", "cnn", "resnet18", or "mobilenet_v2".
        config: the "model" section of a run's config (e.g. hidden_sizes, dropout).
        transfer_config: optional transfer learning section for pretrained models.
    """
    if name == "mlp":
        return FashionMLP(
            hidden_sizes=config.get("hidden_sizes", [256, 128]),
            dropout=config.get("dropout", 0.2),
        )
    if name == "cnn":
        return FashionCNN(dropout=config.get("dropout", 0.3))
    if name == "resnet18":
        return build_resnet18(config, transfer_config)
    if name == "mobilenet_v2":
        return build_mobilenet_v2(config, transfer_config)
    raise ValueError(f"Unknown model name: {name!r}. Expected 'mlp', 'cnn', 'resnet18', or 'mobilenet_v2'.")
