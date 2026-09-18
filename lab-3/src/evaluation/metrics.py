"""
Evaluation utilities: metrics computation and confusion matrix analysis.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from typing import Tuple, Dict, List
import json
from pathlib import Path


def evaluate_model(model: nn.Module,
                   test_loader: DataLoader,
                   device: torch.device,
                   class_names: List[str] = None) -> Dict:
    """
    Evaluate model on test set and compute all metrics.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        class_names: List of class names

    Returns:
        Dictionary with all metrics
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Prepare results
    results = {
        'accuracy': float(accuracy),
        'per_class': {}
    }

    num_classes = len(np.unique(all_labels))
    for i in range(num_classes):
        class_name = class_names[i] if class_names else f"Class_{i}"
        results['per_class'][class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }

    return results, cm, all_preds, all_labels


def save_test_results(results: Dict,
                     cm: np.ndarray,
                     output_dir: str):
    """
    Save test results to output directory.

    Args:
        results: Results dictionary from evaluate_model
        cm: Confusion matrix
        output_dir: Output directory
    """
    output_path = Path(output_dir) / 'metrics'
    output_path.mkdir(parents=True, exist_ok=True)

    # Save metrics as JSON
    json_path = output_path / 'test_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Test metrics saved to {json_path}")

    # Save confusion matrix as numpy array
    cm_path = output_path / 'confusion_matrix.npy'
    np.save(cm_path, cm)
    print(f"Confusion matrix saved to {cm_path}")


def analyze_confusion_matrix(cm: np.ndarray,
                             class_names: List[str] = None,
                             top_k: int = 5) -> Dict:
    """
    Analyze confusion matrix to find most confused class pairs.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        top_k: Number of top confused pairs to return

    Returns:
        Dictionary with confusion analysis
    """
    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    # Find off-diagonal elements (misclassifications)
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    'true_class': class_names[i],
                    'predicted_class': class_names[j],
                    'count': int(cm[i, j]),
                    'true_class_idx': i,
                    'predicted_class_idx': j
                })

    # Sort by count
    confused_pairs.sort(key=lambda x: x['count'], reverse=True)

    analysis = {
        'top_confused_pairs': confused_pairs[:top_k],
        'total_misclassifications': int(cm.sum() - np.trace(cm)),
        'total_correct': int(np.trace(cm))
    }

    return analysis


def print_confusion_matrix_explanation():
    """Print detailed explanation of confusion matrix."""
    explanation = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║           CONFUSION MATRIX EXPLANATION                        ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║                                                               ║
    ║  Structure:                                                   ║
    ║      - ROWS: True labels (actual class)                       ║
    ║      - COLUMNS: Predicted labels (model's prediction)         ║
    ║      - CELL [i, j]: Number of samples with true class i       ║
    ║                     that were predicted as class j            ║
    ║                                                               ║
    ║  Interpretation:                                              ║
    ║      - DIAGONAL: Correct predictions ✓                        ║
    ║      - OFF-DIAGONAL: Misclassifications ✗                     ║
    ║                                                               ║
    ║  Example:                                                     ║
    ║               Predicted                                       ║
    ║           Cat  Dog  Bird                                      ║
    ║      Cat [ 45   3    2  ] ← 45 cats correctly classified      ║
    ║  True Dog [  2  38   10 ]   3 cats confused as dogs          ║
    ║      Bird[  1   5   44 ]   2 cats confused as birds          ║
    ║                                                               ║
    ║  From this, we derive:                                        ║
    ║      - Precision (Cat) = 45/(45+2+1) = 0.938                  ║
    ║        "Of all predicted cats, 93.8% are correct"             ║
    ║                                                               ║
    ║      - Recall (Cat) = 45/(45+3+2) = 0.900                     ║
    ║        "Of all actual cats, 90% were found"                   ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(explanation)


def compare_models_metrics(cnn_results: Dict,
                          mlp_results: Dict,
                          class_names: List[str] = None) -> Dict:
    """
    Compare CNN and MLP performance metrics.

    Args:
        cnn_results: CNN test results
        mlp_results: MLP test results
        class_names: List of class names

    Returns:
        Comparison dictionary
    """
    comparison = {
        'overall': {
            'cnn_accuracy': cnn_results['accuracy'],
            'mlp_accuracy': mlp_results['accuracy'],
            'accuracy_diff': cnn_results['accuracy'] - mlp_results['accuracy']
        },
        'per_class': {}
    }

    for class_name in cnn_results['per_class'].keys():
        cnn_f1 = cnn_results['per_class'][class_name]['f1_score']
        mlp_f1 = mlp_results['per_class'][class_name]['f1_score']

        comparison['per_class'][class_name] = {
            'cnn_f1': cnn_f1,
            'mlp_f1': mlp_f1,
            'f1_diff': cnn_f1 - mlp_f1
        }

    return comparison
