"""Reusable plotting functions for exploration, training monitoring, and error analysis.

Each function takes data in and either returns a matplotlib Figure or saves directly to a
given path - callers decide whether to show, save, or both.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_sample_images(images: np.ndarray, labels: np.ndarray, class_names: list[str], n: int = 10) -> plt.Figure:
    """Plot a grid of sample images with their class name as the title."""
    n = min(n, len(images))
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap="gray")
        axes[i].set_title(class_names[labels[i]], fontsize=10)
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Sample FashionMNIST Images", fontsize=13)
    fig.tight_layout()
    return fig


def plot_class_distribution(labels: np.ndarray, class_names: list[str], title: str = "Class Distribution") -> plt.Figure:
    """Bar chart of how many samples belong to each class."""
    counts = np.bincount(labels, minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(class_names, counts, color="steelblue")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of samples")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_loss_curve(history: pd.DataFrame, title: str = "Training vs Validation Loss") -> plt.Figure:
    """Plot train_loss and val_loss against epoch."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train Loss", marker="o")
    ax.plot(history["epoch"], history["val_loss"], label="Validation Loss", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_accuracy_curve(history: pd.DataFrame, title: str = "Training vs Validation Accuracy") -> plt.Figure:
    """Plot train_accuracy and val_accuracy against epoch."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history["epoch"], history["train_accuracy"], label="Train Accuracy", marker="o")
    ax.plot(history["epoch"], history["val_accuracy"], label="Validation Accuracy", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion_matrix(conf_matrix: np.ndarray, class_names: list[str], title: str = "Confusion Matrix") -> plt.Figure:
    """Heatmap of a confusion matrix with class name ticks and per-cell counts."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(conf_matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    max_count = conf_matrix.max()
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            color = "white" if conf_matrix[i, j] > max_count / 2 else "black"
            ax.text(j, i, str(conf_matrix[i, j]), ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_prediction_examples(
    images: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: list[str],
    n: int = 10,
) -> plt.Figure:
    """Plot a grid of images with 'true / predicted' captions (green if correct, red if wrong)."""
    n = min(n, len(images))
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5))
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        correct = true_labels[i] == predicted_labels[i]
        axes[i].imshow(images[i].squeeze(), cmap="gray")
        axes[i].set_title(
            f"true: {class_names[true_labels[i]]}\npred: {class_names[predicted_labels[i]]}",
            fontsize=8,
            color="green" if correct else "red",
        )
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Prediction Examples", fontsize=13)
    fig.tight_layout()
    return fig


def plot_misclassified_grid(
    images: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    confidences: np.ndarray,
    class_names: list[str],
    n: int = 10,
) -> plt.Figure:
    """Plot a grid of misclassified images with true/predicted labels and model confidence."""
    n = min(n, len(images))
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6))
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap="gray")
        axes[i].set_title(
            f"true: {class_names[true_labels[i]]}\n"
            f"pred: {class_names[predicted_labels[i]]} ({confidences[i]:.2f})",
            fontsize=8,
            color="red",
        )
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Misclassified Samples", fontsize=13)
    fig.tight_layout()
    return fig


def plot_confusion_pair_examples(
    images: np.ndarray,
    class_names: list[str],
    true_label: str,
    predicted_label: str,
    n: int = 8,
) -> plt.Figure:
    """Plot example images for one specific confusion pair (true_label mistaken as predicted_label)."""
    n = min(n, len(images))
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap="gray")
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"Confusion pair: true='{true_label}' predicted='{predicted_label}'", fontsize=12)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save a figure to disk, creating the parent directory if needed, and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
