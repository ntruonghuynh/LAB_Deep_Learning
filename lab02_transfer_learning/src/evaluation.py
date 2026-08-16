"""Evaluation utilities: overall accuracy, confusion matrix, per-class metrics, and
prediction/confidence collection for error analysis.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Run the model over `loader` and collect labels, predictions, and confidences.

    Returns a dict with numpy arrays: "labels", "predictions", "confidences" (softmax
    probability of the predicted class), and "probabilities" (full softmax distribution).
    """
    model.eval()
    model.to(device)

    all_labels = []
    all_predictions = []
    all_confidences = []
    all_probabilities = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        confidences, predictions = probabilities.max(dim=1)

        all_labels.append(labels.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())
        all_confidences.append(confidences.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())

    return {
        "labels": np.concatenate(all_labels),
        "predictions": np.concatenate(all_predictions),
        "confidences": np.concatenate(all_confidences),
        "probabilities": np.concatenate(all_probabilities),
    }


def overall_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Fraction of correctly classified samples."""
    return float((labels == predictions).mean())


def compute_confusion_matrix(labels: np.ndarray, predictions: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Return the (num_classes x num_classes) confusion matrix, rows=true, cols=predicted."""
    return confusion_matrix(labels, predictions, labels=list(range(num_classes)))


def per_class_metrics(labels: np.ndarray, predictions: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    """Return a DataFrame with per-class precision, recall, F1, and support."""
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(len(class_names))), zero_division=0
    )
    return pd.DataFrame({
        "class": class_names,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "support": support,
    })


def find_top_confusion_pairs(conf_matrix: np.ndarray, class_names: list[str], top_k: int = 5) -> pd.DataFrame:
    """Identify the top-k off-diagonal (true, predicted) confusion pairs by count.

    This is computed from the actual confusion matrix - no assumptions about which
    classes are confused are hard-coded.
    """
    pairs = []
    num_classes = conf_matrix.shape[0]
    for true_idx in range(num_classes):
        for pred_idx in range(num_classes):
            if true_idx == pred_idx:
                continue
            count = int(conf_matrix[true_idx, pred_idx])
            if count > 0:
                pairs.append({
                    "true_label": class_names[true_idx],
                    "predicted_label": class_names[pred_idx],
                    "count": count,
                })

    df = pd.DataFrame(pairs).sort_values("count", ascending=False).reset_index(drop=True)
    return df.head(top_k)


def build_misclassified_table(
    labels: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
) -> pd.DataFrame:
    """Build a DataFrame of misclassified samples: sample_index, true_label, predicted_label, confidence."""
    mismatch_mask = labels != predictions
    sample_indices = np.nonzero(mismatch_mask)[0]

    return pd.DataFrame({
        "sample_index": sample_indices,
        "true_label": labels[mismatch_mask],
        "predicted_label": predictions[mismatch_mask],
        "confidence": confidences[mismatch_mask],
    })
