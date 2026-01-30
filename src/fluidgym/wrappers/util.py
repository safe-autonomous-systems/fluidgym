"""Utility functions for space manipulation."""

import numpy as np
from gymnasium import spaces


def flatten_box_space(space: spaces.Box) -> spaces.Box:
    """Flattens a Gymnasium Box space into a 1D Box space.

    Parameters
    ----------
    space: spaces.Box
        The Box space to flatten.

    Returns
    -------
    spaces.Box
        The flattened Box space.
    """
    low = space.low.flatten()
    high = space.high.flatten()
    return spaces.Box(low=low, high=high, dtype=space.dtype)  # type: ignore


def flatten_dict_space(space: spaces.Dict, keys: list[str] | None = None) -> spaces.Box:
    """Flattens a Gymnasium Dict space into a single Box space.

    Parameters
    ----------
    space: spaces.Dict
        The Dict space to flatten.

    keys: list[str] | None
        The keys to include in the flattening. If None, all keys are included.

    Returns
    -------
    spaces.Box
        The flattened Box space.
    """
    if not isinstance(space, spaces.Dict):
        raise TypeError(f"Expected spaces.Dict, got {type(space)}")

    if keys is not None:
        for k in keys:
            if k not in space.spaces:
                raise KeyError(f"Key '{k}' not found in the Dict space.")
        items = [(k, space.spaces[k]) for k in keys]
    else:
        items = list(space.spaces.items())

    lows = []
    highs = []
    dtypes = []

    for k, s in items:
        if not isinstance(s, spaces.Box):
            raise TypeError(f"Only Box subspaces supported, but key '{k}' is {type(s)}")

        # Ensure bounds are arrays and flatten
        low = np.asarray(s.low).reshape(-1)
        high = np.asarray(s.high).reshape(-1)

        lows.append(low)
        highs.append(high)
        dtypes.append(s.dtype)

    if not lows:
        raise ValueError("Dict space contains no Box subspaces to flatten.")

    low_cat = np.concatenate(lows, axis=0)
    high_cat = np.concatenate(highs, axis=0)

    # Choose a dtype that can represent everything (common choice: float32)
    out_dtype = np.result_type(*dtypes, np.float32)

    # Cast bounds to match dtype (Gymnasium likes consistent dtypes)
    low_cat = low_cat.astype(out_dtype, copy=False)
    high_cat = high_cat.astype(out_dtype, copy=False)

    return spaces.Box(low=low_cat, high=high_cat, dtype=out_dtype)  # type: ignore
