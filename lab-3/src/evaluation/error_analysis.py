"""
Error analysis utilities for understanding model failures.

Functions for identifying and analyzing misclassified examples.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


def analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset,
    text_key: str = 'text'
) -> List[Dict[str, Any]]:
    """
    Identify and collect misclassified examples.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        dataset: Dataset with text samples
        text_key: Key for text field in dataset

    Returns:
        List of error dictionaries
    """
    errors = []

    for idx, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label != pred_label:
            error = {
                'index': idx,
                'text': dataset[idx][text_key] if hasattr(dataset[idx], '__getitem__') else dataset[idx],
                'true_label': int(true_label),
                'predicted_label': int(pred_label),
            }
            errors.append(error)

    return errors


def show_error_examples(
    errors: List[Dict[str, Any]],
    num_examples: int = 10,
    class_names: Optional[List[str]] = None
) -> None:
    """
    Display misclassified examples.

    Args:
        errors: List of error dictionaries from analyze_errors
        num_examples: Number of examples to show
        class_names: Optional class names for display
    """
    print(f"\n{'='*60}")
    print(f"Error Examples (showing {min(num_examples, len(errors))} of {len(errors)})")
    print(f"{'='*60}\n")

    for i, error in enumerate(errors[:num_examples], 1):
        true_label = error['true_label']
        pred_label = error['predicted_label']

        if class_names:
            true_str = f"{true_label} ({class_names[true_label]})"
            pred_str = f"{pred_label} ({class_names[pred_label]})"
        else:
            true_str = str(true_label)
            pred_str = str(pred_label)

        text = error['text']
        if isinstance(text, str):
            display_text = text[:200] + '...' if len(text) > 200 else text
        else:
            display_text = str(text)[:200]

        print(f"Error {i} (Index {error['index']}):")
        print(f"  True label:      {true_str}")
        print(f"  Predicted label: {pred_str}")
        print(f"  Text: {display_text}")
        print()


def analyze_error_patterns(
    errors: List[Dict[str, Any]],
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze patterns in misclassified examples.

    Args:
        errors: List of error dictionaries
        class_names: Optional class names

    Returns:
        Dictionary of error patterns
    """
    if not errors:
        return {
            'total_errors': 0,
            'confusion_pairs': {},
            'error_rate': 0.0
        }

    # Count confusion pairs (true_label -> predicted_label)
    confusion_pairs = {}
    for error in errors:
        pair = (error['true_label'], error['predicted_label'])
        if pair not in confusion_pairs:
            confusion_pairs[pair] = []
        confusion_pairs[pair].append(error)

    # Text length analysis
    text_lengths = [len(str(error['text']).split()) for error in errors]

    patterns = {
        'total_errors': len(errors),
        'confusion_pairs': {
            f"{pair[0]}->{pair[1]}": len(errors_list)
            for pair, errors_list in confusion_pairs.items()
        },
        'text_length_stats': {
            'mean': float(np.mean(text_lengths)),
            'std': float(np.std(text_lengths)),
            'min': int(np.min(text_lengths)),
            'max': int(np.max(text_lengths))
        }
    }

    return patterns


def print_error_patterns(
    patterns: Dict[str, Any],
    class_names: Optional[List[str]] = None
) -> None:
    """
    Print error pattern analysis.

    Args:
        patterns: Output from analyze_error_patterns
        class_names: Optional class names
    """
    print(f"\n{'='*60}")
    print("Error Pattern Analysis")
    print(f"{'='*60}")

    print(f"\nTotal errors: {patterns['total_errors']}")

    print(f"\nMost common confusion pairs:")
    sorted_pairs = sorted(
        patterns['confusion_pairs'].items(),
        key=lambda x: x[1],
        reverse=True
    )

    for pair_str, count in sorted_pairs[:5]:
        true_label, pred_label = map(int, pair_str.split('->'))

        if class_names:
            pair_display = f"{class_names[true_label]} -> {class_names[pred_label]}"
        else:
            pair_display = pair_str

        percentage = (count / patterns['total_errors']) * 100
        print(f"  {pair_display}: {count} ({percentage:.1f}%)")

    print(f"\nText length of misclassified examples:")
    print(f"  Mean:   {patterns['text_length_stats']['mean']:.2f} words")
    print(f"  Std:    {patterns['text_length_stats']['std']:.2f}")
    print(f"  Range:  {patterns['text_length_stats']['min']}-{patterns['text_length_stats']['max']}")

    print(f"{'='*60}\n")


def compare_error_lengths_with_correct(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset,
    text_key: str = 'text'
) -> Dict[str, Any]:
    """
    Compare text lengths between correct and incorrect predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        dataset: Dataset with text samples
        text_key: Key for text field

    Returns:
        Dictionary with length comparison statistics
    """
    correct_mask = (y_true == y_pred)

    correct_lengths = []
    incorrect_lengths = []

    for idx, is_correct in enumerate(correct_mask):
        text = dataset[idx][text_key] if hasattr(dataset[idx], '__getitem__') else dataset[idx]
        length = len(str(text).split())

        if is_correct:
            correct_lengths.append(length)
        else:
            incorrect_lengths.append(length)

    comparison = {
        'correct': {
            'count': len(correct_lengths),
            'mean': float(np.mean(correct_lengths)),
            'std': float(np.std(correct_lengths)),
            'median': float(np.median(correct_lengths))
        },
        'incorrect': {
            'count': len(incorrect_lengths),
            'mean': float(np.mean(incorrect_lengths)) if incorrect_lengths else 0,
            'std': float(np.std(incorrect_lengths)) if incorrect_lengths else 0,
            'median': float(np.median(incorrect_lengths)) if incorrect_lengths else 0
        }
    }

    return comparison


def print_length_comparison(comparison: Dict[str, Any]) -> None:
    """
    Print text length comparison between correct and incorrect predictions.

    Args:
        comparison: Output from compare_error_lengths_with_correct
    """
    print(f"\n{'='*60}")
    print("Text Length Comparison")
    print(f"{'='*60}")

    correct = comparison['correct']
    incorrect = comparison['incorrect']

    print(f"\nCorrect predictions ({correct['count']}):")
    print(f"  Mean length:   {correct['mean']:.2f} words")
    print(f"  Std:           {correct['std']:.2f}")
    print(f"  Median:        {correct['median']:.2f}")

    if incorrect['count'] > 0:
        print(f"\nIncorrect predictions ({incorrect['count']}):")
        print(f"  Mean length:   {incorrect['mean']:.2f} words")
        print(f"  Std:           {incorrect['std']:.2f}")
        print(f"  Median:        {incorrect['median']:.2f}")

        diff = incorrect['mean'] - correct['mean']
        print(f"\nDifference: {diff:+.2f} words")

        if abs(diff) > 5:
            if diff > 0:
                print("  ⚠️  Model struggles with LONGER texts")
            else:
                print("  ⚠️  Model struggles with SHORTER texts")
        else:
            print("  ✓ No significant length bias")

    print(f"{'='*60}\n")


def find_high_confidence_errors(
    errors: List[Dict[str, Any]],
    y_probs: np.ndarray,
    threshold: float = 0.8,
    num_examples: int = 10
) -> List[Dict[str, Any]]:
    """
    Find misclassified examples where model was very confident.

    Args:
        errors: List of error dictionaries
        y_probs: Prediction probabilities
        threshold: Confidence threshold
        num_examples: Max examples to return

    Returns:
        List of high-confidence errors
    """
    high_conf_errors = []

    for error in errors:
        idx = error['index']
        confidence = float(np.max(y_probs[idx]))

        if confidence >= threshold:
            error_with_conf = error.copy()
            error_with_conf['confidence'] = confidence
            high_conf_errors.append(error_with_conf)

    # Sort by confidence (highest first)
    high_conf_errors.sort(key=lambda x: x['confidence'], reverse=True)

    return high_conf_errors[:num_examples]


def print_high_confidence_errors(
    high_conf_errors: List[Dict[str, Any]],
    class_names: Optional[List[str]] = None
) -> None:
    """
    Print high-confidence errors.

    Args:
        high_conf_errors: Output from find_high_confidence_errors
        class_names: Optional class names
    """
    if not high_conf_errors:
        print("\n✓ No high-confidence errors found")
        return

    print(f"\n{'='*60}")
    print(f"High-Confidence Errors ({len(high_conf_errors)})")
    print(f"{'='*60}")
    print("These are cases where the model was very confident but wrong.\n")

    for i, error in enumerate(high_conf_errors, 1):
        true_label = error['true_label']
        pred_label = error['predicted_label']
        confidence = error['confidence']

        if class_names:
            true_str = f"{class_names[true_label]}"
            pred_str = f"{class_names[pred_label]}"
        else:
            true_str = f"Class {true_label}"
            pred_str = f"Class {pred_label}"

        text = str(error['text'])[:150] + '...' if len(str(error['text'])) > 150 else str(error['text'])

        print(f"{i}. Confidence: {confidence:.4f}")
        print(f"   True: {true_str}, Predicted: {pred_str}")
        print(f"   Text: {text}")
        print()

    print(f"{'='*60}\n")


def save_error_analysis(
    errors: List[Dict[str, Any]],
    patterns: Dict[str, Any],
    output_path: str = './outputs/exercise2/error_analysis.json'
) -> None:
    """
    Save error analysis to JSON file.

    Args:
        errors: List of errors
        patterns: Error patterns
        output_path: Path to save analysis
    """
    import json
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    analysis = {
        'total_errors': len(errors),
        'patterns': patterns,
        'sample_errors': errors[:20]  # Save first 20 errors as examples
    }

    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"✓ Error analysis saved to {output_path}")
