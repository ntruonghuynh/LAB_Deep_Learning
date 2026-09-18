"""
Visualization utilities for training and evaluation.

Functions for plotting training curves, confusion matrices, and error distributions.
"""

from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def plot_training_history(
    history: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
) -> None:
    """
    Plot training and validation loss/metrics over epochs.

    Args:
        history: Training history from trainer.state.log_history
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    # Separate train and eval logs
    train_logs = [log for log in history if 'loss' in log and 'eval_loss' not in log]
    eval_logs = [log for log in history if 'eval_loss' in log]

    if not eval_logs:
        print("No evaluation logs found in history")
        return

    # Extract data
    train_steps = [log.get('step', 0) for log in train_logs]
    train_losses = [log.get('loss') for log in train_logs]

    eval_epochs = [log.get('epoch', i+1) for i, log in enumerate(eval_logs)]
    eval_losses = [log.get('eval_loss') for log in eval_logs]

    # Check for metrics
    has_accuracy = 'eval_accuracy' in eval_logs[0]
    has_f1 = 'eval_f1' in eval_logs[0]

    # Determine number of subplots
    num_plots = 1 + (1 if has_accuracy or has_f1 else 0)

    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    if num_plots == 1:
        axes = [axes]

    # Plot 1: Loss
    axes[0].plot(train_steps, train_losses, label='Train Loss', alpha=0.6)
    axes[0].plot(eval_epochs, eval_losses, label='Validation Loss', marker='o', linewidth=2)
    axes[0].set_xlabel('Steps / Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Metrics (if available)
    if num_plots > 1:
        if has_accuracy:
            accuracies = [log.get('eval_accuracy') for log in eval_logs]
            axes[1].plot(eval_epochs, accuracies, label='Accuracy', marker='o', linewidth=2)

        if has_f1:
            f1_scores = [log.get('eval_f1') for log in eval_logs]
            axes[1].plot(eval_epochs, f1_scores, label='F1-Score', marker='s', linewidth=2)

        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Evaluation Metrics')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to {save_path}")

    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
    cmap: str = 'Blues',
    normalize: bool = False
) -> None:
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix
        class_names: Optional class names
        save_path: Optional path to save the plot
        figsize: Figure size
        cmap: Colormap
        normalize: Whether to normalize by row (show percentages)
    """
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix'

    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage' if normalize else 'Count'}
    )

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")

    plt.show()


def plot_error_distribution(
    errors: List[Dict[str, Any]],
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5)
) -> None:
    """
    Plot distribution of errors by confusion pairs.

    Args:
        errors: List of error dictionaries
        class_names: Optional class names
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    if not errors:
        print("No errors to plot")
        return

    # Count confusion pairs
    confusion_pairs = {}
    for error in errors:
        pair = (error['true_label'], error['predicted_label'])
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

    # Sort by count
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)

    # Create labels
    if class_names:
        labels = [f"{class_names[pair[0]]} → {class_names[pair[1]]}" for pair, _ in sorted_pairs]
    else:
        labels = [f"{pair[0]} → {pair[1]}" for pair, _ in sorted_pairs]

    counts = [count for _, count in sorted_pairs]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Bar chart
    ax1.barh(range(len(labels)), counts, color='coral')
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('Number of Errors')
    ax1.set_title('Error Distribution by Confusion Pair')
    ax1.grid(True, alpha=0.3, axis='x')

    # Pie chart
    ax2.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Error Proportion')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Error distribution saved to {save_path}")

    plt.show()


def plot_text_length_distribution(
    dataset,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    text_key: str = 'text',
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5)
) -> None:
    """
    Plot text length distribution, optionally comparing correct vs incorrect predictions.

    Args:
        dataset: Dataset with text samples
        y_true: Optional true labels
        y_pred: Optional predicted labels
        text_key: Key for text field
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    # Calculate text lengths
    lengths = []
    for item in dataset:
        text = item[text_key] if hasattr(item, '__getitem__') else item
        lengths.append(len(str(text).split()))

    if y_true is not None and y_pred is not None:
        # Separate correct and incorrect
        correct_mask = (y_true == y_pred)
        correct_lengths = [lengths[i] for i in range(len(lengths)) if correct_mask[i]]
        incorrect_lengths = [lengths[i] for i in range(len(lengths)) if not correct_mask[i]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Overall distribution
        ax1.hist(lengths, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.1f}')
        ax1.set_xlabel('Text Length (words)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Overall Text Length Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Comparison
        ax2.hist(correct_lengths, bins=30, alpha=0.6, label='Correct', color='green', edgecolor='black')
        ax2.hist(incorrect_lengths, bins=30, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
        ax2.axvline(np.mean(correct_lengths), color='darkgreen', linestyle='--', linewidth=2)
        ax2.axvline(np.mean(incorrect_lengths), color='darkred', linestyle='--', linewidth=2)
        ax2.set_xlabel('Text Length (words)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Text Length: Correct vs Incorrect Predictions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    else:
        # Just overall distribution
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        plt.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.1f}')
        plt.xlabel('Text Length (words)')
        plt.ylabel('Frequency')
        plt.title('Text Length Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Text length distribution saved to {save_path}")

    plt.show()


def plot_confidence_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5)
) -> None:
    """
    Plot prediction confidence distribution for correct vs incorrect predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    # Get max confidence for each prediction
    confidences = np.max(y_probs, axis=1)

    # Separate correct and incorrect
    correct_mask = (y_true == y_pred)
    correct_confidences = confidences[correct_mask]
    incorrect_confidences = confidences[~correct_mask]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax1.hist(correct_confidences, bins=20, alpha=0.6, label='Correct', color='green', edgecolor='black')
    if len(incorrect_confidences) > 0:
        ax1.hist(incorrect_confidences, bins=20, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
    ax1.axvline(np.mean(correct_confidences), color='darkgreen', linestyle='--', linewidth=2)
    if len(incorrect_confidences) > 0:
        ax1.axvline(np.mean(incorrect_confidences), color='darkred', linestyle='--', linewidth=2)
    ax1.set_xlabel('Prediction Confidence')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Confidence Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    data_to_plot = [correct_confidences]
    labels = ['Correct']
    colors = ['lightgreen']

    if len(incorrect_confidences) > 0:
        data_to_plot.append(incorrect_confidences)
        labels.append('Incorrect')
        colors.append('lightcoral')

    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax2.set_ylabel('Prediction Confidence')
    ax2.set_title('Confidence Comparison')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confidence distribution saved to {save_path}")

    plt.show()


def plot_per_class_metrics(
    per_class_metrics: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot per-class metrics as a grouped bar chart.

    Args:
        per_class_metrics: Output from compute_per_class_metrics
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    classes = list(per_class_metrics.keys())
    precision = [metrics['precision'] for metrics in per_class_metrics.values()]
    recall = [metrics['recall'] for metrics in per_class_metrics.values()]
    f1 = [metrics['f1'] for metrics in per_class_metrics.values()]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(x - width, precision, width, label='Precision', color='steelblue')
    ax.bar(x, recall, width, label='Recall', color='coral')
    ax.bar(x + width, f1, width, label='F1-Score', color='mediumseagreen')

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Per-class metrics saved to {save_path}")

    plt.show()
