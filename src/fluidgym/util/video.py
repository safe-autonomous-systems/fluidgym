"""Utility functions for saving rendered frames as videos."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

# Episodes longer than this (e.g. TCF) are subsampled to keep the gifs small
MAX_GIF_FRAMES = 500
GIF_FRAME_STRIDE = 5


def save_gif(
    frames: Sequence[np.ndarray] | np.ndarray, path: Path | str, fps: int = 24
) -> Path:
    """Save a sequence of rendered frames as a looping GIF file.

    Parameters
    ----------
    frames: Sequence[np.ndarray] | np.ndarray
        The rendered frames ``[H, W, 3]``, e.g. as returned by :meth:`render`, or
        a stacked array of shape ``[T, H, W, 3]``.

    path: Path | str
        The path of the GIF file. The ``.gif`` suffix is appended if missing, and
        the parent directory is created if it does not exist.

    fps: int
        The frames per second of the GIF. Defaults to 24.

    Returns
    -------
    Path
        The path of the saved GIF file.
    """
    from PIL import Image

    if len(frames) == 0:
        raise ValueError("No frames to save as GIF.")

    path = Path(path)
    if path.suffix != ".gif":
        path = path.with_name(path.name + ".gif")
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(frames) > MAX_GIF_FRAMES:
        frames = frames[::GIF_FRAME_STRIDE]

    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=1000 / fps,  # per frame, in ms
        loop=0,
    )
    return path
