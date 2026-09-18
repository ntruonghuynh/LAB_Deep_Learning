"""
Dataset utilities for Hugging Face NLP tasks.

Functions for loading, splitting, and preprocessing text datasets.
"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from collections import Counter


def save_splits(
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    output_dir: str = 'splits/'
) -> None:
    """
    Save train/val/test split indices to JSON files.

    Args:
        train_indices: List of training set indices
        val_indices: List of validation set indices
        test_indices: List of test set indices
        output_dir: Directory to save split files
    """
    os.makedirs(output_dir, exist_ok=True)

    splits = {
        'train_indices.json': train_indices,
        'val_indices.json': val_indices,
        'test_indices.json': test_indices
    }

    for filename, indices in splits.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(indices, f, indent=2)

    print(f"✓ Saved splits to {output_dir}")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val: {len(val_indices)} samples")
    print(f"  Test: {len(test_indices)} samples")


def load_splits(
    split_dir: str = 'splits/'
) -> Tuple[List[int], List[int], List[int]]:
    """
    Load train/val/test split indices from JSON files.

    Args:
        split_dir: Directory containing split files

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    train_path = os.path.join(split_dir, 'train_indices.json')
    val_path = os.path.join(split_dir, 'val_indices.json')
    test_path = os.path.join(split_dir, 'test_indices.json')

    with open(train_path, 'r') as f:
        train_indices = json.load(f)
    with open(val_path, 'r') as f:
        val_indices = json.load(f)
    with open(test_path, 'r') as f:
        test_indices = json.load(f)

    print(f"✓ Loaded splits from {split_dir}")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val: {len(val_indices)} samples")
    print(f"  Test: {len(test_indices)} samples")

    return train_indices, val_indices, test_indices


def create_stratified_splits(
    dataset,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    label_key: str = 'label'
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create stratified train/val/test splits.

    Args:
        dataset: Hugging Face dataset object
        train_size: Proportion for training (default 0.7)
        val_size: Proportion for validation (default 0.15)
        test_size: Proportion for test (default 0.15)
        random_state: Random seed for reproducibility
        label_key: Key name for labels in dataset

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    from sklearn.model_selection import train_test_split

    # Ensure proportions sum to 1
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "train_size + val_size + test_size must equal 1.0"

    # Get labels
    labels = [item[label_key] for item in dataset]
    indices = list(range(len(dataset)))

    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(val_size + test_size),
        random_state=random_state,
        stratify=[labels[i] for i in indices]
    )

    # Second split: val vs test
    temp_labels = [labels[i] for i in temp_idx]
    val_ratio = val_size / (val_size + test_size)

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_ratio),
        random_state=random_state,
        stratify=temp_labels
    )

    return train_idx, val_idx, test_idx


def get_dataset_statistics(
    dataset,
    label_key: str = 'label',
    text_key: str = 'text'
) -> Dict[str, Any]:
    """
    Compute statistics about the dataset.

    Args:
        dataset: Hugging Face dataset object
        label_key: Key name for labels
        text_key: Key name for text

    Returns:
        Dictionary of statistics
    """
    labels = [item[label_key] for item in dataset]
    texts = [item[text_key] for item in dataset]

    # Label distribution
    label_counts = Counter(labels)

    # Text length statistics
    text_lengths = [len(text.split()) for text in texts]

    stats = {
        'total_samples': len(dataset),
        'label_distribution': dict(label_counts),
        'num_classes': len(set(labels)),
        'text_length': {
            'mean': np.mean(text_lengths),
            'std': np.std(text_lengths),
            'min': np.min(text_lengths),
            'max': np.max(text_lengths),
            'median': np.median(text_lengths)
        }
    }

    return stats


def print_dataset_statistics(
    dataset,
    split_name: str = 'Dataset',
    label_key: str = 'label',
    text_key: str = 'text'
) -> None:
    """
    Print formatted dataset statistics.

    Args:
        dataset: Hugging Face dataset object
        split_name: Name of the split (for display)
        label_key: Key name for labels
        text_key: Key name for text
    """
    stats = get_dataset_statistics(dataset, label_key, text_key)

    print(f"\n{'='*60}")
    print(f"{split_name} Statistics")
    print(f"{'='*60}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Number of classes: {stats['num_classes']}")
    print(f"\nLabel distribution:")
    for label, count in sorted(stats['label_distribution'].items()):
        percentage = (count / stats['total_samples']) * 100
        print(f"  Class {label}: {count:5d} samples ({percentage:5.2f}%)")

    print(f"\nText length (words):")
    print(f"  Mean:   {stats['text_length']['mean']:.2f}")
    print(f"  Std:    {stats['text_length']['std']:.2f}")
    print(f"  Min:    {stats['text_length']['min']}")
    print(f"  Max:    {stats['text_length']['max']}")
    print(f"  Median: {stats['text_length']['median']:.2f}")
    print(f"{'='*60}\n")


def show_examples(
    dataset,
    num_examples: int = 5,
    label_key: str = 'label',
    text_key: str = 'text',
    label_names: Optional[List[str]] = None
) -> None:
    """
    Display random examples from the dataset.

    Args:
        dataset: Hugging Face dataset object
        num_examples: Number of examples to show
        label_key: Key name for labels
        text_key: Key name for text
        label_names: Optional list of label names for display
    """
    import random

    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))

    print(f"\n{'='*60}")
    print(f"Random Examples from Dataset")
    print(f"{'='*60}\n")

    for i, idx in enumerate(indices, 1):
        item = dataset[idx]
        label = item[label_key]
        text = item[text_key]

        if label_names:
            label_str = f"{label} ({label_names[label]})"
        else:
            label_str = str(label)

        print(f"Example {i}:")
        print(f"  Label: {label_str}")
        print(f"  Text: {text[:200]}{'...' if len(text) > 200 else ''}")
        print()


def tokenize_dataset(
    dataset,
    tokenizer,
    text_key: str = 'text',
    max_length: int = 512,
    padding: str = 'max_length',
    truncation: bool = True
):
    """
    Tokenize a Hugging Face dataset.

    Args:
        dataset: Hugging Face dataset object
        tokenizer: Hugging Face tokenizer
        text_key: Key name for text field
        max_length: Maximum sequence length
        padding: Padding strategy
        truncation: Whether to truncate

    Returns:
        Tokenized dataset
    """
    def tokenize_function(examples):
        return tokenizer(
            examples[text_key],
            padding=padding,
            truncation=truncation,
            max_length=max_length
        )

    tokenized = dataset.map(tokenize_function, batched=True)
    return tokenized


def prepare_dataset_for_training(
    dataset,
    tokenizer,
    text_key: str = 'text',
    label_key: str = 'label',
    max_length: int = 512
):
    """
    Prepare dataset for training with proper formatting.

    Args:
        dataset: Hugging Face dataset object
        tokenizer: Hugging Face tokenizer
        text_key: Key name for text field
        label_key: Key name for label field
        max_length: Maximum sequence length

    Returns:
        Prepared dataset ready for Trainer
    """
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples[text_key],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )

    tokenized = dataset.map(tokenize_function, batched=True)

    # Rename label column if needed
    if label_key != 'labels':
        tokenized = tokenized.rename_column(label_key, 'labels')

    # Set format for PyTorch
    tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    return tokenized
