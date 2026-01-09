"""Grid and domain setup for turbulent channel flow simulations."""

from typing import Any

import numpy as np
import torch

import fluidgym.simulation.pict.data.shapes as shapes
import fluidgym.simulation.pict.PISOtorch_simulation as PISOtorch_simulation
from fluidgym.simulation.extensions import (
    PISOtorch,  # type: ignore[import-untyped,import-not-found]
)


def _make_y_weights(N: int = 1, ny_half: int = 48) -> list[float]:
    ny = 2 * (ny_half // N)

    r = 1.2 ** (N / 2)
    h0 = 0.5 * (1 - r) / (1 - r ** (ny / 2))  # cell y-size at boundary
    h = 0  # current distance from boundary
    y = [0.0] * ny
    for i in range((ny - 2) // 2):
        h += h0 * (r**i)
        y[i] = h
        y[ny - i - 2] = 1 - h
    y[ny // 2 - 1] = 0.5
    y[ny - 1] = 1.0

    y = [0] + y

    return y


def _make_grid(
    H: float,
    L: float,
    D: float,
    x: int,
    y_half: int,
    yN: int,
    z: int,
    dims: int = 3,
    dtype: torch.dtype = torch.float32,
    global_scale: float | None = None,
    _use_cos_y: bool = False,
) -> torch.Tensor:
    assert (x % 4) == 0
    assert dims in [2, 3]

    delta = H / 2

    if global_scale is not None:
        y_weights = shapes.make_weights_exp_global(y_half * 2, global_scale, "BOTH")
    elif _use_cos_y:
        y_weights = shapes.make_weights_cos(y_half * 2, "BOTH")
    else:
        y_weights = _make_y_weights(ny_half=y_half * yN, N=yN)
    y_sizes = np.asarray(y_weights)
    y_sizes = y_sizes[1:] - y_sizes[:-1]

    corners = [(-L / 2, -delta), (L / 2, -delta), (-L / 2, delta), (L / 2, delta)]

    y = len(y_weights) - 1

    grid = shapes.generate_grid_vertices_2D(
        [y + 1, x + 1], corners, None, x_weights=y_weights, dtype=dtype
    )
    if dims == 3:
        grid = shapes.extrude_grid_z(grid, z, start_z=-D / 2, end_z=D / 2)  # type: ignore
    grid = grid.cuda().contiguous()

    return grid


def _get_viscosity_wall_distance(block, domain, u_wall, cuda_devie: torch.device):
    # assumes channel flow centered on 0 with delta=1
    pos_y = block.getCellCoordinates()[:, 1:2]
    wall_distance = (1 - torch.abs(pos_y)) * u_wall / domain.viscosity.to(cuda_devie)

    return wall_distance  # NCDHW with C=1


def _make_reichardt_profile(domain, u_wall: float, cuda_device):
    k = 0.41
    k_inv = 1 / k

    def reichardt_profile(y_wall):
        y11 = y_wall / 11.0
        return k_inv * torch.log(1 + k * y_wall) + 7.8 * (
            1 - torch.exp(-y11) - y11 * torch.exp(-y_wall / 3)
        )

    pos_y = domain.getBlock(0).getCellCoordinates()[0, 1, 0, :, 0]  # NCDHW -> H
    wall_distance = (1 - torch.abs(pos_y)) * u_wall / domain.viscosity.to(cuda_device)

    u = reichardt_profile(wall_distance)

    return u * u_wall


def get_van_driest_sqr(
    block: PISOtorch.Block,
    domain: PISOtorch.Domain,
    u_wall: float,
    cuda_device: torch.device,
) -> torch.Tensor:
    """Compute the squared Van Driest damping function based on wall distance.

    Parameters
    ----------
    block: PISOtorch.Block
        The PISOtorch block object.

    domain: PISOtorch.Domain
        The PISOtorch domain object.

    u_wall: float
        The wall velocity tensor.

    cuda_device: torch.device
        The CUDA device to use for computations.
    """
    wall_distance = _get_viscosity_wall_distance(block, domain, u_wall, cuda_device)
    van_driest_scale = 1 - torch.exp(-wall_distance * (1.0 / 25.0))
    return van_driest_scale * van_driest_scale


def set_dynamic_forcing(
    ndims: int, domain: PISOtorch.Domain, prep_fn: dict[str, Any]
) -> None:
    """Set up dynamic forcing based on wall shear stress.

    Parameters
    ----------
    ndims: int
        Number of spatial dimensions (2 or 3).

    domain: PISOtorch.Domain
        The PISOtorch domain object.

    prep_fn: dict[str, Any]
        The preparation function dictionary to which the forcing function will be added.
    """
    pos_y = torch.mean(domain.getBlock(0).getCellCoordinates()[0, 1], dim=(0, 2))
    d_y = (1 + pos_y[0].cpu().numpy(), 1 - pos_y[-1].cpu().numpy())

    def pfn_set_forcing(domain, **kwargs):
        block = domain.getBlock(0)
        viscosity = domain.viscosity.to(domain.getDevice())

        mean_vel_u = torch.mean(block.velocity[0, 0], dim=(0, 2))
        tau_wall_n = viscosity * mean_vel_u[0] / d_y[0]
        tau_wall_p = viscosity * mean_vel_u[-1] / d_y[-1]

        forcing = (tau_wall_n + tau_wall_p) * 0.5
        G = torch.tensor(
            [[forcing] + [0] * (ndims - 1)],
            dtype=domain.getDtype(),
            device=domain.getDevice(),
        )  # NC, static velocity source
        block.setVelocitySource(G)

    PISOtorch_simulation.append_prep_fn(prep_fn, "PRE", pfn_set_forcing)


def make_channel_flow_domain(
    H: float,
    L: float,
    D: float,
    x: int,
    y: int,
    z: int,
    refinement_strength: int,
    n_dims: int,
    u_wall: float,
    viscosity: torch.Tensor,
    cuda_device: torch.device,
    init_with_noise: bool = True,
    dtype: torch.dtype = torch.float32,
):
    """Create the grid and domain for the channel flow simulation.

    Parameters
    ----------
    x: int
        Number of grid points in the streamwise (x) direction. Must be a multiple of 4.

    y: int
        Number of grid points in the wall-normal (y) direction. Must be an even number.

    z: int
        Number of grid points in the spanwise (z) direction. Must be a multiple of 2.

    n_dims: int
        Number of spatial dimensions (2 or 3).

    refinement_strength: int
        Refinement strength for the wall-normal direction. Higher values lead to more
        refined grids near the walls. Default is None.

    init_with_noise: bool
        Whether to initialize the velocity field with added noise.

    cuda_device: torch.device
        The CUDA device to use for computations.

    cpu_device: torch.device
        The CPU device to use for computations.
    """
    y_half = y // 2
    grid_N = refinement_strength

    grid = _make_grid(
        H=H,
        L=L,
        D=D,
        x=x,
        y_half=y_half,
        yN=grid_N,
        z=z,
        dtype=dtype,
        global_scale=None,
    )
    y = grid.size(-2) - 1
    z = grid.size(-3) - 1

    vel_init = None

    # create domain
    domain = PISOtorch.Domain(
        n_dims,
        viscosity,
        passiveScalarChannels=0,
        name="ChannelDomain",
        device=cuda_device,
        dtype=dtype,
    )

    # create block from the mesh on the domain (missing settings, fields and
    # transformation metrics are created automatically)
    block = domain.CreateBlock(
        velocity=vel_init, vertexCoordinates=grid, name="ChannelBlock"
    )
    block.CloseBoundary("-y")

    vel_init = _make_reichardt_profile(domain, u_wall=u_wall, cuda_device=cuda_device)
    vel_init = vel_init.view(1, 1, 1, y, 1).expand(-1, -1, z, -1, x)
    vel_init = torch.cat(
        [vel_init] + [torch.zeros_like(vel_init)] * (n_dims - 1), dim=1
    )

    if init_with_noise:
        from fluidgym.simulation.extensions import (
            SimplexNoiseVariations,  # type: ignore[import-untyped,import-not-found]
        )

        curl_noise = SimplexNoiseVariations.GenerateSimplexNoiseVariation(
            [x, y, z],
            cuda_device,
            [2 / x, 2 / y, 2 / z],
            [0] * 3,
            SimplexNoiseVariations.NoiseVariation.CURL,
        )
        curl_mag = torch.linalg.vector_norm(curl_noise, dim=1)
        curl_mag_max = torch.max(curl_mag)
        curl_noise *= 0.5 * vel_init / curl_mag_max
        vel_init += curl_noise

    vel_init = vel_init.to(dtype).contiguous()
    block.setVelocity(vel_init)

    return domain
