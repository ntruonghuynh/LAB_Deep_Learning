"""
Evaluation metrics and analysis utilities.

Functions for computing classification metrics, confusion matrices, and reports.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import json
import os


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy for multiclass ('weighted', 'macro', 'micro')

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Confusion matrix as numpy array
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
) -> None:
    """
    Print detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional names for classes
    """
    print(f"\n{'='*60}")
    print("Classification Report")
    print(f"{'='*60}\n")

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4
    )
    print(report)


def save_metrics(
    metrics: Dict[str, Any],
    output_path: str = './outputs/exercise2/metrics.json'
) -> None:
    """
    Save metrics to JSON file.

    Args:
        metrics: Dictionary of metrics
        output_path: Path to save metrics
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Metrics saved to {output_path}")


def explain_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None
) -> None:
    """
    Explain confusion matrix in detail.

    Args:
        cm: Confusion matrix
        class_names: Optional class names
    """
    num_classes = cm.shape[0]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    print(f"\n{'='*60}")
    print("Confusion Matrix Explanation")
    print(f"{'='*60}\n")

    print("Structure:")
    print("              Predicted")
    print("          ", end="")
    for name in class_names:
        print(f"  {name:8s}", end="")
    print()

    for i, name in enumerate(class_names):
        if i == 0:
            print(f"Actual  {name:8s}", end="")
        else:
            print(f"        {name:8s}", end="")

        for j in range(num_classes):
            print(f"  {cm[i, j]:8d}", end="")
        print()

    print("\nInterpretation:")

    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        print(f"  True Negatives (TN):  {tn:5d} - Correctly predicted {class_names[0]}")
        print(f"  False Positives (FP): {fp:5d} - Wrongly predicted {class_names[1]} (Type I error)")
        print(f"  False Negatives (FN): {fn:5d} - Wrongly predicted {class_names[0]} (Type II error)")
        print(f"  True Positives (TP):  {tp:5d} - Correctly predicted {class_names[1]}")

        print(f"\nDiagonal (TN + TP):     {tn + tp:5d} - Correct predictions")
        print(f"Off-diagonal (FP + FN): {fp + fn:5d} - Errors")

        total = tn + fp + fn + tp
        print(f"\nAccuracy: {(tn + tp) / total:.4f}")
        print(f"Error rate: {(fp + fn) / total:.4f}")

    else:
        # Multiclass
        total = cm.sum()
        correct = np.trace(cm)
        errors = total - correct

        print(f"  Diagonal: {correct} - Correct predictions")
        print(f"  Off-diagonal: {errors} - Errors")
        print(f"  Total: {total}")
        print(f"  Accuracy: {correct / total:.4f}")

        print("\nPer-class breakdown:")
        for i, name in enumerate(class_names):
            total_actual = cm[i, :].sum()
            correct_class = cm[i, i]
            errors_class = total_actual - correct_class

            print(f"  {name}:")
            print(f"    Correct: {correct_class}/{total_actual} ({correct_class/total_actual:.2%})")
            print(f"    Errors: {errors_class}")

    print(f"{'='*60}\n")


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each class separately.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names

    Returns:
        Dictionary mapping class names to their metrics
    """
    num_classes = len(np.unique(y_true))

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    per_class_metrics = {}

    for i, name in enumerate(class_names):
        # Binary mask for this class
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)

        per_class_metrics[name] = {
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
            'support': np.sum(y_true == i)
        }

    return per_class_metrics


def print_per_class_metrics(
    per_class_metrics: Dict[str, Dict[str, float]]
) -> None:
    """
    Print per-class metrics in a formatted table.

    Args:
        per_class_metrics: Output from compute_per_class_metrics
    """
    print(f"\n{'='*60}")
    print("Per-Class Metrics")
    print(f"{'='*60}")
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print(f"{'-'*60}")

    for class_name, metrics in per_class_metrics.items():
        print(f"{class_name:<15} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f} {metrics['support']:>10d}")

    print(f"{'='*60}\n")


def analyze_prediction_confidence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray
) -> Dict[str, Any]:
    """
    Analyze prediction confidence for correct and incorrect predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities (shape: [n_samples, n_classes])

    Returns:
        Dictionary of confidence statistics
    """
    # Get confidence (max probability) for each prediction
    confidences = np.max(y_probs, axis=1)

    # Separate correct and incorrect predictions
    correct_mask = (y_true == y_pred)
    correct_confidences = confidences[correct_mask]
    incorrect_confidences = confidences[~correct_mask]

    analysis = {
        'correct_predictions': {
            'count': len(correct_confidences),
            'mean_confidence': float(np.mean(correct_confidences)),
            'std_confidence': float(np.std(correct_confidences)),
            'min_confidence': float(np.min(correct_confidences)) if len(correct_confidences) > 0 else None,
            'max_confidence': float(np.max(correct_confidences)) if len(correct_confidences) > 0 else None,
        },
        'incorrect_predictions': {
            'count': len(incorrect_confidences),
            'mean_confidence': float(np.mean(incorrect_confidences)) if len(incorrect_confidences) > 0 else None,
            'std_confidence': float(np.std(incorrect_confidences)) if len(incorrect_confidences) > 0 else None,
            'min_confidence': float(np.min(incorrect_confidences)) if len(incorrect_confidences) > 0 else None,
            'max_confidence': float(np.max(incorrect_confidences)) if len(incorrect_confidences) > 0 else None,
        }
    }

    return analysis


def print_confidence_analysis(
    confidence_analysis: Dict[str, Any]
) -> None:
    """
    Print confidence analysis results.

    Args:
        confidence_analysis: Output from analyze_prediction_confidence
    """
    print(f"\n{'='*60}")
    print("Prediction Confidence Analysis")
    print(f"{'='*60}")

    correct = confidence_analysis['correct_predictions']
    print(f"\nCorrect Predictions ({correct['count']}):")
    print(f"  Mean confidence: {correct['mean_confidence']:.4f}")
    print(f"  Std confidence:  {correct['std_confidence']:.4f}")
    print(f"  Min confidence:  {correct['min_confidence']:.4f}")
    print(f"  Max confidence:  {correct['max_confidence']:.4f}")

    incorrect = confidence_analysis['incorrect_predictions']
    if incorrect['count'] > 0:
        print(f"\nIncorrect Predictions ({incorrect['count']}):")
        print(f"  Mean confidence: {incorrect['mean_confidence']:.4f}")
        print(f"  Std confidence:  {incorrect['std_confidence']:.4f}")
        print(f"  Min confidence:  {incorrect['min_confidence']:.4f}")
        print(f"  Max confidence:  {incorrect['max_confidence']:.4f}")

        print(f"\nObservation:")
        diff = correct['mean_confidence'] - incorrect['mean_confidence']
        if diff > 0.1:
            print(f"  Model is more confident on correct predictions (+{diff:.4f})")
        elif diff < -0.1:
            print(f"  Model is MORE confident on incorrect predictions ({diff:.4f}) - concerning!")
        else:
            print(f"  Similar confidence on correct and incorrect predictions ({diff:+.4f})")
    else:
        print(f"\nNo incorrect predictions - perfect accuracy!")

    print(f"{'='*60}\n")
