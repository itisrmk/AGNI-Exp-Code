"""
Reproducibility utilities.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # For Apple Silicon MPS
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # For CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior (may impact performance)
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_deterministic(enabled: bool = True):
    """
    Enable/disable deterministic operations.

    Note: This may significantly impact performance.

    Args:
        enabled: Whether to enable deterministic mode
    """
    if enabled:
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
