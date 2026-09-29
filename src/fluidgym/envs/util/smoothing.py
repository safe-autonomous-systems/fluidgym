"""Spatial smoothing of piecewise-constant actuation profiles."""

from __future__ import annotations

from pathlib import Path

import torch


def smooth_segment_profile(
    values: torch.Tensor, segment_width: int, alpha: float = 0.1
) -> torch.Tensor:
    """Expand per-segment values to a smoothly blended, periodic cell profile.

    Each segment value is repeated ``segment_width`` times along the last dimension,
    and the steps between neighbouring segments are replaced by a cubic
    (smoothstep) blend, following https://doi.org/10.1063/5.0153181. The first
    cell of a segment holds the midpoint of the two neighbouring values, and the
    blend spans about ``round(alpha * segment_width)`` cells around each
    interface. The profile wraps around, i.e., the last segment blends into the
    first one.

    Parameters
    ----------
    values: torch.Tensor
        Per-segment values of shape ``(..., n_segments)``.

    segment_width: int
        Number of cells per segment.

    alpha: float
        Fraction of ``segment_width`` to blend over. Defaults to 0.1.

    Returns
    -------
    torch.Tensor
        The smoothed profile of shape ``(..., n_segments * segment_width)``.
    """
    blend_width = min(round(segment_width * alpha), segment_width)
    if blend_width == 0:
        # Nothing to blend, and the blend parameters would divide by zero
        return values.repeat_interleave(segment_width, dim=-1)

    def cubic_blend(t: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        s = t * t * (3 - 2 * t)
        return (1 - s) * A + s * B

    # Segment indexing
    n_cells = values.size(-1) * segment_width
    cell_idx = torch.arange(n_cells, device=values.device, dtype=torch.long)
    seg_id = cell_idx // segment_width
    pos = cell_idx % segment_width

    # Previous, own and next segment value of each cell
    prev = torch.roll(values, shifts=1, dims=-1)[..., seg_id]
    cur = values[..., seg_id]
    nxt = torch.roll(values, shifts=-1, dims=-1)[..., seg_id]

    # Zones
    left_zone = pos < blend_width
    right_zone = pos >= segment_width - blend_width

    # Blend parameters, the right one mirrors the left one about the interface
    pos_f = pos.to(values.dtype)
    t_left = (pos_f / blend_width + 0.5).clamp(0.0, 1.0)
    t_right = (0.5 - (segment_width - pos_f) / blend_width).clamp(0.0, 1.0)

    blend_left = cubic_blend(t_left, prev, cur)
    blend_right = cubic_blend(t_right, cur, nxt)

    return torch.where(left_zone, blend_left, torch.where(right_zone, blend_right, cur))


def plot_segment_profile(
    values: torch.Tensor,
    smoothed: torch.Tensor,
    segment_width: int,
    path: str | Path,
    xlabel: str = "z index",
) -> None:
    """Plot a piecewise-constant profile and its smoothed version as line plots.

    Parameters
    ----------
    values: torch.Tensor
        Per-segment values of shape ``(n_segments,)``.

    smoothed: torch.Tensor
        The smoothed profile of shape ``(n_segments * segment_width,)``.

    segment_width: int
        Number of cells per segment.

    path: str | Path
        Where to save the figure.

    xlabel: str
        Label of the x-axis. Defaults to "z index".
    """
    import matplotlib.pyplot as plt

    stepped = values.detach().repeat_interleave(segment_width).cpu()
    smoothed = smoothed.detach().cpu()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(stepped, drawstyle="steps-mid", color="b", label="per segment")
    ax.plot(smoothed, marker=".", linestyle="--", color="r", label="smoothed")
    for boundary in range(0, stepped.numel() + 1, segment_width):
        ax.axvline(boundary - 0.5, color="gray", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Actuation")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
