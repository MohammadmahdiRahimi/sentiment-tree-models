"""
Utility functions for training and evaluation.

This module provides helper functions for device management, random seed setting,
and other common utilities used across the project.
"""

import torch
import random
import math


def get_device():
    """
    Get the appropriate device for PyTorch operations.
    
    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=456):
    """
    Set random seeds for reproducibility.
    
    Sets seeds for Python's random module, NumPy, and PyTorch. Also configures
    PyTorch's CUDA backend for deterministic behavior if CUDA is available.
    
    Args:
        seed: Random seed value (default: 456)
    """
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
