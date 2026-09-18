"""
Dataset utilities for Lab 3: CNN vs MLP
Handles data loading, preprocessing, and split management.
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from torch.utils.data import Dataset, Subset, DataLoader
import torch


def load_splits(splits_dir: str = 'splits') -> Tuple[List[int], List[int], List[int]]:
    """
    Load pre-saved train/val/test split indices.

    Args:
        splits_dir: Directory containing split JSON files

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    splits_path = Path(splits_dir)

    with open(splits_path / 'train_indices.json', 'r') as f:
        train_idx = json.load(f)

    with open(splits_path / 'val_indices.json', 'r') as f:
        val_idx = json.load(f)

    with open(splits_path / 'test_indices.json', 'r') as f:
        test_idx = json.load(f)

    return train_idx, val_idx, test_idx


def save_splits(train_idx: List[int], val_idx: List[int], test_idx: List[int],
                splits_dir: str = 'splits'):
    """
    Save train/val/test split indices to JSON files.

    Args:
        train_idx: Training set indices
        val_idx: Validation set indices
        test_idx: Test set indices
        splits_dir: Directory to save splits
    """
    splits_path = Path(splits_dir)
    splits_path.mkdir(exist_ok=True)

    with open(splits_path / 'train_indices.json', 'w') as f:
        json.dump(train_idx, f)

    with open(splits_path / 'val_indices.json', 'w') as f:
        json.dump(val_idx, f)

    with open(splits_path / 'test_indices.json', 'w') as f:
        json.dump(test_idx, f)

    print(f"Splits saved to {splits_dir}/")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val:   {len(val_idx)} samples")
    print(f"  Test:  {len(test_idx)} samples")


def create_dataloaders(dataset: Dataset,
                       train_idx: List[int],
                       val_idx: List[int],
                       test_idx: List[int],
                       batch_size: int = 32,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train/val/test splits.

    Args:
        dataset: Full dataset
        train_idx: Training indices
        val_idx: Validation indices
        test_idx: Test indices
        batch_size: Batch size for training
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


class ImageDataset(Dataset):
    """
    Generic image dataset wrapper.
    Override for specific datasets (CIFAR, MNIST, custom, etc.)
    """

    def __init__(self, data, labels, transform=None):
        """
        Args:
            data: Image data (numpy array or tensor)
            labels: Labels (numpy array or tensor)
            transform: Optional transforms to apply
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
