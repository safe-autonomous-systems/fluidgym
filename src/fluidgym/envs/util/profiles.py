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
