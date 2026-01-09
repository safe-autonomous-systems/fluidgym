"""General helper functions."""

import random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Apply manual seeding for numpy and pyTorch.

    Parameters
    ----------
    seed: int
        The random number generator seed to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
