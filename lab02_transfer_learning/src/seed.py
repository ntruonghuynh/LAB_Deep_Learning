"""Single utility for deterministic/reproducible runs.

Seeds Python's `random`, NumPy, and PyTorch (CPU + CUDA/MPS) with one call.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all relevant random number generators for reproducibility.

    Args:
        seed: Integer seed value. The same seed should be recorded in the
            run's config.yaml so results can be reproduced later.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
