"""Utility functions for generating flow profiles."""

import torch


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
