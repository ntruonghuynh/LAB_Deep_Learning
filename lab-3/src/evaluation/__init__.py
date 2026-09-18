"""Evaluation utilities module."""

from .eval_metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    print_classification_report,
    save_metrics,
    explain_confusion_matrix,
    compute_per_class_metrics,
    print_per_class_metrics,
    analyze_prediction_confidence,
    print_confidence_analysis
)

from .error_analysis import (
    analyze_errors,
    show_error_examples,
    analyze_error_patterns,
    print_error_patterns,
    compare_error_lengths_with_correct,
    print_length_comparison,
    find_high_confidence_errors,
    print_high_confidence_errors,
    save_error_analysis
)

__all__ = [
    'compute_classification_metrics',
    'compute_confusion_matrix',
    'print_classification_report',
    'save_metrics',
    'explain_confusion_matrix',
    'compute_per_class_metrics',
    'print_per_class_metrics',
    'analyze_prediction_confidence',
    'print_confidence_analysis',
    'analyze_errors',
    'show_error_examples',
    'analyze_error_patterns',
    'print_error_patterns',
    'compare_error_lengths_with_correct',
    'print_length_comparison',
    'find_high_confidence_errors',
    'print_high_confidence_errors',
    'save_error_analysis'
]
