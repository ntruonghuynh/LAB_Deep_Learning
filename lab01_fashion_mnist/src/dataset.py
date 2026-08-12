"""FashionMNIST data pipeline: download, transforms, train/val/test split, DataLoaders.

Design notes:
- The official FashionMNIST test set is only ever returned for final evaluation.
- The official training set is split once into train/validation using a fixed seed, and the
  resulting indices are cached under data/splits/ so every experiment (MLP, CNN, CNN experiment)
  trains and validates on exactly the same data.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Paths are derived from this file's location - no hard-coded absolute machine paths.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# Standard FashionMNIST grayscale mean/std.
FASHION_MNIST_MEAN = (0.2860,)
FASHION_MNIST_STD = (0.3530,)

# Torchvision's default FashionMNIST mirror can be unavailable on some networks.
datasets.FashionMNIST.mirrors = [
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
]


def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms(augmentation: bool = False) -> tuple[transforms.Compose, transforms.Compose]:
    """Build (train_transform, eval_transform).

    The eval transform (used for validation and test) never includes augmentation, so
    augmentation - when enabled - only ever affects the training split.
    """
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
    ])

    if augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
        ])
    else:
        train_transform = eval_transform

    return train_transform, eval_transform


def _split_file_path(seed: int) -> Path:
    return SPLITS_DIR / f"split_seed{seed}.json"


def get_or_create_split(seed: int, validation_ratio: float, train_size: int) -> dict:
    """Load a saved train/val index split, creating and saving it if it doesn't exist yet.

    Reusing the same saved split across experiments is what guarantees MLP and CNN runs are
    compared under equivalent data conditions.
    """
    split_path = _split_file_path(seed)
    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as f:
            return json.load(f)

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(train_size, generator=generator).tolist()
    num_val = int(train_size * validation_ratio)
    val_indices = sorted(permutation[:num_val])
    train_indices = sorted(permutation[num_val:])

    split = {
        "seed": seed,
        "validation_ratio": validation_ratio,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f)

    return split


def get_dataloaders(
    batch_size: int = 64,
    validation_ratio: float = 0.1,
    augmentation: bool = False,
    seed: int = 42,
    num_workers: int = 0,
    data_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/validation/test DataLoaders for FashionMNIST.

    Downloads FashionMNIST into `data_dir` (defaults to `<project_root>/data`) if not already
    present, applies the configured transforms, and reuses the saved train/val split for the
    given seed. Returns (train_loader, val_loader, test_loader).
    """
    data_dir = data_dir or DATA_DIR
    train_transform, eval_transform = build_transforms(augmentation)

    # Used only to know how many training examples exist, so the split can be created.
    reference_train_set = datasets.FashionMNIST(root=str(data_dir), train=True, download=True)
    split = get_or_create_split(seed, validation_ratio, len(reference_train_set))

    train_dataset = datasets.FashionMNIST(root=str(data_dir), train=True, download=False, transform=train_transform)
    val_dataset = datasets.FashionMNIST(root=str(data_dir), train=True, download=False, transform=eval_transform)
    test_dataset = datasets.FashionMNIST(root=str(data_dir), train=False, download=True, transform=eval_transform)

    train_subset = Subset(train_dataset, split["train_indices"])
    val_subset = Subset(val_dataset, split["val_indices"])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
