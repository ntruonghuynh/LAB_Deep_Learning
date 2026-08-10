"""Reusable training loop logic: one training epoch, one validation epoch, and the full loop.

Keeps the canonical PyTorch pattern explicit and easy to explain:
    model.train() -> zero_grad() -> forward -> loss -> backward() -> optimizer.step()
    model.eval()  -> torch.no_grad() -> forward -> loss (no backward)
"""

from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.metrics import RunningAverage, accuracy_from_logits


def build_optimizer(name: str, params: Iterable, learning_rate: float) -> torch.optim.Optimizer:
    """Build an optimizer by name. Kept simple: only the two optimizers this lab needs."""
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=learning_rate)
    if name == "sgd":
        return torch.optim.SGD(params, lr=learning_rate, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name!r}. Expected 'adam' or 'sgd'.")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one full pass over the training data. Returns (train_loss, train_accuracy)."""
    model.train()
    loss_avg = RunningAverage()
    acc_avg = RunningAverage()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        loss_avg.update(loss.item(), batch_size)
        acc_avg.update(accuracy_from_logits(logits, labels), batch_size)

    return loss_avg.average, acc_avg.average


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one full pass over the validation data with no gradient tracking.

    Returns (val_loss, val_accuracy).
    """
    model.eval()
    loss_avg = RunningAverage()
    acc_avg = RunningAverage()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            loss_avg.update(loss.item(), batch_size)
            acc_avg.update(accuracy_from_logits(logits, labels), batch_size)

    return loss_avg.average, acc_avg.average


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    optimizer_name: str = "adam",
    verbose: bool = True,
) -> dict:
    """Run the full training loop for `epochs` epochs and track history every epoch.

    Returns a dict with:
        - history: list of per-epoch dicts (epoch, train_loss, val_loss, train_accuracy,
          val_accuracy, learning_rate)
        - best_state_dict: model state_dict at the epoch with the lowest val_loss
        - best_epoch: which epoch achieved best_state_dict
        - best_val_loss / best_val_accuracy: metrics at that best epoch
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(optimizer_name, model.parameters(), learning_rate)

    history = []
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = -1
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = validate_one_epoch(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "learning_rate": current_lr,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(
                f"Epoch {epoch}/{epochs} - "
                f"train_loss: {train_loss:.4f} train_acc: {train_accuracy:.4f} - "
                f"val_loss: {val_loss:.4f} val_acc: {val_accuracy:.4f}"
            )

    return {
        "history": history,
        "best_state_dict": best_state_dict,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
    }
