"""Helpers for generating non-uniform structured grid spacings."""

from typing import Literal

import numpy as np


def make_weights_simple_grading(
    res: int, grading: float, refinement: Literal["START", "END", "BOTH"]
) -> list[float]:
    """
    Create cell weights using OpenFOAM-style simpleGrading cell expansion ratio.

    Parameters
    ----------
    res: int
        Number of cells.
    grading: float
        Cell expansion ratio (last cell size / first cell size).
        > 1: refinement at START (or walls for BOTH)
        < 1: refinement at END (or center for BOTH)
    refinement: str
        "START" : first cell is smallest
        "END"   : last cell is smallest
        "BOTH"  : symmetric, smallest cells at both ends (walls)
    """

    def geometric_sizes(n: int, r: float) -> list[float]:
        """
        Generate n cell sizes with geometric progression and give expansion ratio r.

        Parameters
        ----------
        n: int
            Number of cells.

        r: float
            Expansion ratio (last cell size / first cell size). For example, r=2 means
            the last cell is twice as large as the first cell.

        Returns
        -------
        list[float]
            List of n cell sizes summing to 1.
        """
        if abs(r - 1.0) < 1e-10:
            return [1.0 / n] * n

        # For geometric series: sizes[i] = a * q^i
        # r = sizes[n-1] / sizes[0] = q^(n-1)  =>  q = r^(1/(n-1))

        q = r ** (1.0 / (n - 1))
        a = 1.0 / sum(q**i for i in range(n))

        return [a * q**i for i in range(n)]

    if refinement == "START":
        # grading > 1 means last/first > 1, so first cell is smallest
        sizes = geometric_sizes(res, grading)

    elif refinement == "END":
        # reverse: last cell is smallest
        sizes = geometric_sizes(res, grading)[::-1]

    elif refinement == "BOTH":
        # Symmetric: each half has refinement toward the walls (ends of full block)
        # First half: small -> large (grading > 1 means wall-refined)
        half = res // 2
        left_sizes = geometric_sizes(half, grading)  # small at left wall
        right_sizes = geometric_sizes(res - half, grading)[::-1]  # small at right wall
        sizes = left_sizes + right_sizes

    else:
        raise ValueError(
            f"Unknown refinement '{refinement}'. Use 'START', 'END', or 'BOTH'."
        )

    # Normalize (already sums to 1 per half, but re-normalize for safety)
    total = sum(sizes)
    sizes = [s / total for s in sizes]

    weights = [0.0] + list(np.cumsum(sizes))
    return weights


def make_weights_chebyshev_identity(
    res: int, gamma: float, refinement: Literal["START", "END", "BOTH"]
) -> list[float]:
    """
    Create cell weights using a linear combination of the Chebyshev and identity
    transforms.

    Parameters
    ----------
    res: int
        Number of cells.
    gamma: float
        Weighting factor for the Chebyshev transformation (0 <= gamma <= 1).
        gamma = 0 corresponds to a uniform grid (identity).
        gamma = 1 corresponds to a pure Chebyshev (sine) spacing.
    refinement: str
        "START" : first cell is smallest
        "END"   : last cell is smallest
        "BOTH"  : symmetric, smallest cells at both ends (walls)
    """
    if gamma < 0 or gamma > 1:
        raise ValueError("Gamma must be in the range [0, 1].")

    if refinement == "START":
        # Fine resolution at -1, mapping [-1, 0] to weights [0, 1]
        xi = np.linspace(-1, 0, res + 1)
        weights = (gamma * np.sin(np.pi / 2 * xi) + (1 - gamma) * xi) + 1.0

    elif refinement == "END":
        # Fine resolution at 1, mapping [0, 1] to weights [0, 1]
        xi = np.linspace(0, 1, res + 1)
        weights = gamma * np.sin(np.pi / 2 * xi) + (1 - gamma) * xi

    elif refinement == "BOTH":
        # Fine resolution at -1 and 1, mapping [-1, 1] to weights [0, 1]
        xi = np.linspace(-1, 1, res + 1)
        weights = 0.5 * (gamma * np.sin(np.pi / 2 * xi) + (1 - gamma) * xi) + 0.5

    else:
        raise ValueError(
            f"Unknown refinement '{refinement}'. Use 'START', 'END', or 'BOTH'."
        )

    # Force exact 0.0 and 1.0 at bounds to avoid floating point inaccuracies
    weights[0] = 0.0
    weights[-1] = 1.0

    return weights.tolist()


def make_weights_tanh(
    res: int, A: float, refinement: Literal["START", "END", "BOTH"]
) -> list[float]:
    """
    Create cell weights using the classical tanh transformation.

    Parameters
    ----------
    res: int
        Number of cells.
    A: float
        Stretching parameter (A > 0). Higher values cluster cells closer to the walls.
    refinement: str
        "START" : first cell is smallest
        "END"   : last cell is smallest
        "BOTH"  : symmetric, smallest cells at both ends (walls)
    """
    # Fallback to uniform grading if stretching parameter is practically zero
    if abs(A) < 1e-10:
        return np.linspace(0.0, 1.0, res + 1).tolist()

    if refinement == "START":
        # Fine resolution at -1, mapping [-1, 0] to weights [0, 1]
        xi = np.linspace(-1, 0, res + 1)
        weights = (np.tanh(A * xi) / np.tanh(A)) + 1.0

    elif refinement == "END":
        # Fine resolution at 1, mapping [0, 1] to weights [0, 1]
        xi = np.linspace(0, 1, res + 1)
        weights = np.tanh(A * xi) / np.tanh(A)

    elif refinement == "BOTH":
        # Fine resolution at -1 and 1, mapping [-1, 1] to weights [0, 1]
        xi = np.linspace(-1, 1, res + 1)
        weights = 0.5 * (np.tanh(A * xi) / np.tanh(A)) + 0.5

    else:
        raise ValueError(
            f"Unknown refinement '{refinement}'. Use 'START', 'END', or 'BOTH'."
        )

    # Force exact 0.0 and 1.0 at bounds to avoid floating point inaccuracies
    weights[0] = 0.0
    weights[-1] = 1.0

    return weights.tolist()


def make_weights(
    grading_type: Literal["simple", "chebyshev_identity", "tanh"],
    res: int,
    grading: float,
    refinement: Literal["START", "END", "BOTH"],
) -> list[float]:
    """Create cell weights using the requested grading scheme."""
    if grading_type == "simple":
        return make_weights_simple_grading(res, grading, refinement)
    elif grading_type == "chebyshev_identity":
        return make_weights_chebyshev_identity(res, grading, refinement)
    elif grading_type == "tanh":
        return make_weights_tanh(res, grading, refinement)
    else:
        raise ValueError(
            f"Unknown grading type '{grading_type}'. Use 'simple', "
            "'chebyshev_identity', or 'tanh'."
        )
