"""Small pure helper functions shared by engine.py and evaluation.py."""

import torch


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy for a batch given raw model logits."""
    predictions = logits.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    return correct / labels.size(0)


class RunningAverage:
    """Accumulates a weighted running average (e.g. loss/accuracy across batches of varying size)."""

    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, weight: int = 1) -> None:
        self.total += value * weight
        self.count += weight

    @property
    def average(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0
