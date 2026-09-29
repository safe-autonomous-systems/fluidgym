"""Utility functions for generating flow profiles."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection


def get_jet_profile(h: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Generate a parabolic jet profile tensor.

    Parameters
    ----------
    h: int
        Height of the jet profile.

    dtype: torch.dtype
        Data type of the tensor.

    device: torch.device
        Device on which to create the tensor.

    Returns
    -------
    torch.Tensor
        The jet profile tensor.
    """
    y = torch.linspace(-h / 2, h / 2, h, dtype=dtype, device=device)

    profile = 6 * (h / 2 - y) * (h / 2 + y) / h**2

    # We ensure a max of 1.0 for the profile
    profile /= torch.max(profile)

    return profile


def get_inflow_profile(
    h: float,
    res_y: int,
    n_dims: int,
    dtype: torch.dtype,
    device: torch.device,
    res_z: int | None = None,
) -> torch.Tensor:
    """Generate a parabolic inflow profile tensor.

    Parameters
    ----------
    h: float
        Height of the inflow profile.

    res_y: int
        Number of points in the y-direction.

    n_dims: int
        Number of spatial dimensions (2 or 3).

    dtype: torch.dtype
        Data type of the tensor.

    device: torch.device
        Device on which to create the tensor.

    res_z: int | None, optional
        Number of points in the z-direction (required if n_dims is 3).

    Returns
    -------
    torch.Tensor
        The inflow profile tensor.
    """
    y = torch.linspace(-h / 2, h / 2, res_y, dtype=dtype, device=device)

    profile = 6 * (h / 2 - y) * (h / 2 + y) / h**2

    # We ensure a mean of 1.0 for the profile
    profile = profile / profile.mean()

    if n_dims == 2:
        inflow = torch.zeros((1, 2, res_y, 1), device=device, dtype=dtype)
        inflow[:, 0, :, :] = profile[None, :, None]
    else:
        if res_z is None:
            raise ValueError("res_z must be provided for 3D inflow profile.")

        inflow = torch.zeros((1, 3, 1, res_y, 1), device=device, dtype=dtype)
        inflow[:, 0, :, :] = profile[None, None, :, None]
        inflow = inflow.repeat(1, 1, res_z, 1, 1)

    inflow = inflow.contiguous()

    return inflow


def get_parabolic_profile_3d(
    x_coords: torch.Tensor,
    y_coords: torch.Tensor,
    face_areas_xy: torch.Tensor,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> torch.Tensor:
    """Generate a true 3D parabolic (bi-parabolic) inflow profile for a duct.

    The profile is the product of two 1D parabolas in x and y, corresponding to
    the cross-sectional shape of fully-developed Poiseuille flow in a rectangular duct.

    Parameters
    ----------
    x_coords: torch.Tensor
        1-D x cell-center coordinates, shape ``(res_x,)``.

    y_coords: torch.Tensor
        1-D y cell-center coordinates, shape ``(res_y,)``.

    face_areas_xy: torch.Tensor
        Face-areas, shape ``(res_x, res_y)``. Used for volume-averaged normalization.

    x_min, x_max: float
        Extents of the duct in x.

    y_min, y_max: float
        Extents of the duct in y.

    Returns
    -------
    torch.Tensor
        The inflow profile tensor of shape ``(res_x, res_y)`` with mean 1.0.
    """
    x_norm = 2.0 * (x_coords - x_min) / (x_max - x_min) - 1.0  # (res_x,) in [-1, 1]
    y_norm = 2.0 * (y_coords - y_min) / (y_max - y_min) - 1.0  # (res_y,) in [-1, 1]

    profile_x = 1.0 - x_norm**2  # (res_x,)
    profile_y = 1.0 - y_norm**2  # (res_y,)

    profile_2d = profile_x[:, None] * profile_y[None, :]  # (res_x, res_y)

    profile_2d = profile_2d / (profile_2d * face_areas_xy).sum() * face_areas_xy.sum()

    return profile_2d.T


def get_outflow_profile(
    z_min: float,
    z_max: float,
    x_min: float,
    x_max: float,
    z_centers: torch.Tensor,
    x_centers: torch.Tensor,
    cell_sizes: torch.Tensor,
) -> torch.Tensor:
    """Generate a 2D parabolic outflow profile with volume-averaged mean 1.0.

    The profile is the pointwise product of a parabola in z and a parabola in x,
    each centered in their respective domain.

    Parameters
    ----------
    z_min, z_max: float
        Z extent of the outflow region.

    x_min, x_max: float
        X extent of the outflow region.

    z_centers: torch.Tensor
        1-D z cell-center coordinates, shape ``(nz,)``.

    x_centers: torch.Tensor
        1-D x cell-center coordinates, shape ``(nx,)``.

    cell_sizes: torch.Tensor
        Face-area per cell (dz_i * dx_j), shape ``(nz, nx)``.

    Returns
    -------
    torch.Tensor
        Shape ``(nz, nx)`` with volume-averaged mean 1.0.
    """
    mid_z = (z_min + z_max) / 2.0
    half_z = (z_max - z_min) / 2.0
    profile_z = (half_z - (z_centers - mid_z)) * (half_z + (z_centers - mid_z))  # (nz,)

    mid_x = (x_min + x_max) / 2.0
    half_x = (x_max - x_min) / 2.0
    profile_x = (half_x - (x_centers - mid_x)) * (half_x + (x_centers - mid_x))  # (nx,)

    profile_2d = profile_z[:, None] * profile_x[None, :]  # (nz, nx)

    vol_mean = (profile_2d * cell_sizes).sum() / cell_sizes.sum()
    return profile_2d / vol_mean


def _local_spacing(centers: torch.Tensor, x: float) -> float:
    """Cell spacing of the graded ``centers`` grid nearest to coordinate ``x``."""
    idx = int(torch.argmin(torch.abs(centers - x)))
    lo = max(idx - 1, 0)
    hi = min(idx + 1, centers.shape[0] - 1)
    return float((centers[hi] - centers[lo]) / max(hi - lo, 1))


def get_neighboring_inflow_profiles_3d(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    profile_fraction: float,
    x_centers: torch.Tensor,
    y_centers: torch.Tensor,
    face_areas_xy: torch.Tensor,
) -> torch.Tensor:
    """Generates two neighboring 2D parabolic inflow profiles summing to mean 1.0.

    In the x-direction two parabolic sub-profiles are placed at the two sides of
    the inflow region, each occupying a fraction ``profile_fraction < 0.5`` of the
    total x-width, with zero velocity outside each sub-region. Profile 0 occupies
    ``[x_min, x_min + profile_fraction * width]``; profile 1 occupies
    ``[x_max - profile_fraction * width, x_max]``. The gap between them of width
    ``1 - 2 * profile_fraction`` is empty (zero).
    In the y-direction a single parabola spans ``[y_min, y_max]``. The
    final profile for each sub-region is the pointwise product of the x sub-profile
    and the y profile, normalized so the two profiles together have a mean of 1 over
    the full inflow face. By the symmetry of the two sub-regions each profile then
    has a face-averaged mean of ``1/2``, which is what keeps the bulk velocity
    invariant under an antisymmetric split between the two.

    Parameters
    ----------
    x_min: float
        Minimum x extent of the full inflow region.

    x_max: float
        Maximum x extent of the full inflow region.

    profile_fraction: float
        Fraction of the total x-width occupied by each side profile. Must be
        ``< 0.5`` so the two sub-regions do not overlap; the central gap of width
        ``1 - 2 * profile_fraction`` stays empty.

    x_centers: torch.Tensor
        1-D x cell-center coordinates, shape ``(nx,)``.

    y_centers: torch.Tensor
        1-D y cell-center coordinates, shape ``(ny,)``.

    face_areas_xy: torch.Tensor
        Face-area per cell, shape ``(nx, ny)``. Cell volumes are equally valid as
        long as they stay proportional to the face areas over the inflow face.

    Returns
    -------
    torch.Tensor
        Shape ``(2, ny, nx)`` — two inflow profiles with a face-averaged mean of
        ``1/2`` each, summing to a combined volume-averaged mean of 1.
    """
    dtype = y_centers.dtype
    device = y_centers.device
    nx = x_centers.shape[0]
    ny = y_centers.shape[0]

    # x parabola spanning [y_max, y_max]
    mid_y = (y_min + y_max) / 2.0
    half_y = (y_max - y_min) / 2.0
    profile_y = (half_y - (y_centers - mid_y)) * (half_y + (y_centers - mid_y))  # (nz,)

    width_x = x_max - x_min
    profile_width = profile_fraction * width_x
    left = (x_min, x_min + profile_width)
    right = (x_max - profile_width, x_max)

    profiles = torch.zeros((2, ny, nx), dtype=dtype, device=device)

    for i, (p_min, p_max) in enumerate([left, right]):
        mask_x = (x_centers >= p_min) & (x_centers <= p_max)  # (nx,)
        if not mask_x.any():
            continue

        centers_x_p = x_centers[mask_x]
        sizes_x_p = face_areas_xy[mask_x, 0]  # dx per column in this sub-region
        sizes_z = face_areas_xy[0, :]  # dz per row

        mid_x = (p_min + p_max) / 2.0
        half_x = (p_max - p_min) / 2.0
        profile_x = (half_x - (centers_x_p - mid_x)) * (
            half_x + (centers_x_p - mid_x)
        )  # (nx_sub,)

        # The parabola meets the empty gap with nonzero slope, and that kink is a
        # grid-locked seed for 2-cell oscillations once the cell Reynolds number
        # is well above 2. Ramp the inner edge to zero slope over `blend_cells`
        # The outer edge sits on a wall, so its shear is left as-is
        # inner_edge = p_max if i == 0 else p_min
        # delta = min(blend_cells * _local_spacing(x_centers, inner_edge), half_x)
        # dist = (inner_edge - centers_x_p) if i == 0 else (centers_x_p - inner_edge)
        # s = torch.clamp(dist / delta, 0.0, 1.0)
        # profile_x = profile_x * (s * s * (3.0 - 2.0 * s))

        # 2D product: (ny, nx_sub)
        profile_2d = profile_y[:, None] * profile_x[None, :]

        # Normalize to mean 1 over this sub-region, then apply alpha weighting
        weights = sizes_z[:, None] * sizes_x_p[None, :]
        vol_mean = (profile_2d * weights).sum() / weights.sum()
        profiles[i, :, mask_x] = profile_2d / vol_mean

    # Normalize so the *combined* profile has a volume-averaged mean of 1 over
    # the full inflow face (empty gap included). Each sub-profile currently has
    # mean 1 over its own sub-region of area ``profile_fraction * A_full``, so
    # the combined full-face mean is ``2 * profile_fraction``; this rescales both
    # by ``1 / (2 * profile_fraction)`` to full-face mean 1/2 each. The symmetric
    # jet/base split in ``_apply_action`` then keeps the combined bulk at U for
    # every action
    weights = face_areas_xy.T  # (ny, nx) — cell face-areas over the full face
    combined_mean = (profiles.sum(dim=0) * weights).sum() / weights.sum()
    profiles = profiles / combined_mean

    return profiles


def get_neighboring_inflow_profiles_2d(
    x_min: float,
    x_max: float,
    profile_fraction: float,
    x_centers: torch.Tensor,
    cell_sizes: torch.Tensor,
) -> torch.Tensor:
    """Generate two neighboring 1D parabolic inflow profiles summing to mean 1.0.

    The 1D analogue of :func:`get_neighboring_inflow_profiles_3d` for a 2D
    domain, whose inflow face has a single cross-section direction. Two
    parabolas of width ``profile_fraction * (x_max - x_min)`` sit at the two ends
    of the span -- profile 0 on ``[x_min, x_min + w]``, profile 1 on
    ``[x_max - w, x_max]`` -- separated by an empty gap of width
    ``1 - 2 * profile_fraction``, exactly as the 3D version lays them out along
    its x direction.

    Each profile is normalized so the **pair together** has a span-averaged mean
    of 1, giving each a mean of 1/2 over the full span. That is the invariant
    ``_apply_action`` relies on: it scales the two antisymmetrically, so equal
    halves keep the combined bulk at ``U`` for every action. Normalizing each to
    mean 1 over its own sub-region instead would leave the two carrying unequal
    shares of the flux, and the bulk velocity would then drift with the action.

    Parameters
    ----------
    x_min, x_max: float
        Extent of the full inflow region.

    profile_fraction: float
        Fraction of the total width occupied by *each* profile. Must be
        ``< 0.5``, or the two would overlap.

    x_centers: torch.Tensor
        1-D cell-center coordinates, shape ``(res,)``.

    cell_sizes: torch.Tensor
        Cell size per center, shape ``(res,)``. Used for volume-averaged
        normalization.

    Returns
    -------
    torch.Tensor
        Shape ``(2, res)`` — the two profiles, each with a span-averaged mean of
        1/2, so their sum has mean 1.
    """
    if not 0.0 < profile_fraction < 0.5:
        raise ValueError(
            f"profile_fraction must be in (0, 0.5) so the two profiles do not "
            f"overlap, got {profile_fraction}."
        )

    dtype = x_centers.dtype
    device = x_centers.device
    res = x_centers.shape[0]

    width = x_max - x_min
    profile_width = profile_fraction * width
    left = (x_min, x_min + profile_width)
    right = (x_max - profile_width, x_max)

    profiles = torch.zeros((2, res), dtype=dtype, device=device)

    for i, (p_min, p_max) in enumerate([left, right]):
        mask = (x_centers >= p_min) & (x_centers <= p_max)
        if not mask.any():
            continue

        centers_p = x_centers[mask]
        sizes_p = cell_sizes[mask]

        mid = (p_min + p_max) / 2.0
        half = (p_max - p_min) / 2.0
        profile = (half - (centers_p - mid)) * (half + (centers_p - mid))

        vol_mean = (profile * sizes_p).sum() / sizes_p.sum()
        profiles[i, mask] = profile / vol_mean

    # Rescale so the combined profile averages 1 over the whole span, gap
    # included -- each sub-profile currently averages 1 over its own
    # `profile_fraction` of the span, so the pair averages `2 * profile_fraction`
    combined_mean = (profiles.sum(dim=0) * cell_sizes).sum() / cell_sizes.sum()
    return profiles / combined_mean


def plot_inflow_profile_3d(
    profile: torch.Tensor,
    x_centers: torch.Tensor,
    y_centers: torch.Tensor,
    ax: Axes | Axes3D | None = None,
    cmap: str = "rainbow",
    render_shape: tuple[int, int] = (256, 256),
    xlabel: str = r"$x$",
    ylabel: str = r"$y$",
    x_ticks: list[float] | None = None,
    y_ticks: list[float] | None = None,
    title: str = "Inflow profile",
) -> Axes3D:
    """Plot a 2D inflow profile as a 3D surface.

    Parameters
    ----------
    profile: torch.Tensor
        Profile of shape ``(nz, nx)`` or ``(n_profiles, nz, nx)``.
        If 3-D, the profiles are summed along the first dimension before plotting.

    z_centers: torch.Tensor
        1-D z cell-center coordinates, shape ``(nz,)``.

    x_centers: torch.Tensor
        1-D x cell-center coordinates, shape ``(nx,)``.

    ax: Axes3D | None
        Existing 3D axes. If ``None``, a new figure is created.

    cmap, render_shape, xlabel, ylabel, zlabel, title:
        Forwarded to ``plot_surface`` / axis decoration.

    Returns
    -------
    Axes3D
        The 3D axes containing the surface.
    """
    X_coords = x_centers.cpu().numpy()  # (nx,)
    Y_coords = y_centers.cpu().numpy()  # (ny,)
    U = profile.cpu().numpy()  # (nx, ny)

    X, Y = np.meshgrid(X_coords, Y_coords, indexing="ij")

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")  # type: ignore[assignment]

    surf = ax.plot_surface(  # type: ignore[union-attr]
        X,
        Y,
        U,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        rcount=render_shape[0],
        ccount=render_shape[1],
    )
    ax.get_figure().colorbar(surf, ax=ax, shrink=0.5, pad=0.1)  # type: ignore[union-attr]

    ax.set_xlabel(xlabel)  # type: ignore[union-attr]
    ax.set_ylabel(ylabel)  # type: ignore[union-attr]
    ax.set_zlabel("Velocity", rotation=0)  # type: ignore[union-attr]
    ax.set_title(title)  # type: ignore[union-attr]

    if x_ticks is not None:
        ax.set_xticks(x_ticks)  # type: ignore[union-attr]
    if y_ticks is not None:
        ax.set_yticks(y_ticks)  # type: ignore[union-attr]

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # type: ignore[union-attr]
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # type: ignore[union-attr]
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # type: ignore[union-attr]
    ax.view_init(elev=30, azim=-135)  # type: ignore[union-attr]

    x_range = float(X_coords.max() - X_coords.min())
    y_range = float(Y_coords.max() - Y_coords.min())
    z_range = max(x_range, y_range)
    aspect = [x_range, y_range, z_range * 0.5]
    ax.set_box_aspect(aspect)  # type: ignore[union-attr,arg-type]

    return ax  # type: ignore[return-value]
