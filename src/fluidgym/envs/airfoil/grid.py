"""Utilities for creating the airfoil grid and domain."""

import numpy as np
import torch

import fluidgym.simulation.pict.data.shapes as shapes
from fluidgym.envs.airfoil.coords import NACA12_SHARP_COORDS_LIST
from fluidgym.envs.util.profiles import get_inflow_profile
from fluidgym.simulation.extensions import (
    PISOtorch,  # type: ignore[import-untyped,import-not-found]
)
from fluidgym.simulation.pict.PISOtorch_simulation import balance_boundary_fluxes

JET_CENTERS = [0.2, 0.4, 0.6]
JET_WIDTH = 0.08


def get_jet_locations(domain: PISOtorch.Domain) -> list[list[int]]:
    """Get the jet locations on the airfoil surface.

    Parameters
    ----------
    domain: PISOtorch.Domain
        The PISOtorch Domain.

    Returns
    -------
    list[list[int]]
        List of [start_idx, end_idx] for each jet on the top surface.
    """
    grids = domain.getVertexCoordinates()
    airfoil_coords_top = grids[2]  # AirfoilTop block

    if domain.getSpatialDims() == 3:
        airfoil_coords_top = airfoil_coords_top[:, :, 0, :, :]

    jet_indices_top = []
    for jet_center in JET_CENTERS:
        jet_min, jet_max = jet_center - JET_WIDTH / 2, jet_center + JET_WIDTH / 2
        diff_min_top = torch.abs(airfoil_coords_top[0, 0, 0, :] - jet_min)
        diff_max_top = torch.abs(airfoil_coords_top[0, 0, 0, :] - jet_max)

        _, idx_min_top = torch.min(diff_min_top, dim=-1)
        _, idx_max_top = torch.min(diff_max_top, dim=-1)

        jet_indices_top += [[int(idx_min_top.item()), int(idx_max_top.item())]]

    return jet_indices_top


def read_airfoil(
    attack_angle_deg: float,
    cpu_device: torch.device,
    dtype=torch.float32,
) -> tuple[str, torch.Tensor]:
    """Read airfoil coordinates from a file.

    Parameters
    ----------
    attack_angle_deg: float
        Attack angle of the airfoil in degrees.

    cpu_device: torch.device
        CPU device.

    dtype: torch.dtype, optional
        Data type for tensors. Defaults to torch.float32.

    Returns
    -------
    tuple[str, torch.Tensor]
        Name of the airfoil and the coordinates as a tensor of shape (1, 2, 1, N).
    """
    coords = torch.tensor(
        NACA12_SHARP_COORDS_LIST, device=cpu_device, dtype=dtype
    )  # WC
    coords = torch.movedim(coords, 1, 0)  # CW
    coords = torch.reshape(coords, (1, 2, 1, -1))  # NCHW

    if attack_angle_deg != 0.0:
        # deg to rad
        attack_angle = -attack_angle_deg * np.pi / 180.0

        alpha = torch.tensor(attack_angle, device=cpu_device, dtype=dtype)
        cos_alpha = torch.cos(alpha)
        sin_alpha = torch.sin(alpha)
        rotation_matrix = torch.tensor(
            [[cos_alpha, -sin_alpha], [sin_alpha, cos_alpha]], device=cpu_device
        )

        # Bring coords to shape (num_points, 2)
        _airfoil_coords = coords.squeeze(0).squeeze(1).permute(1, 0)
        rotated_coords = rotated_coords = torch.matmul(
            _airfoil_coords, rotation_matrix.T
        )
        coords = rotated_coords.permute(1, 0).unsqueeze(0).unsqueeze(2)

    return "NACA 0012", coords


def _distance_to_point(
    o: torch.Tensor, d: torch.Tensor, p: torch.Tensor
) -> torch.Tensor:
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    # o, d: origin and direction of line, NCHW
    # p: point, NCHW
    p1 = o
    p2 = o + d

    d1 = p2[:, 0] - p1[:, 0]
    d2 = p2[:, 1] - p1[:, 1]

    a = torch.abs(d1 * (p1[:, 1] - p[:, 1]) - (p1[:, 0] - p[:, 0]) * d2)
    b = torch.sqrt(d1 * d1 + d2 * d2)

    distance = (a / b).reshape((1, 1, o.size(-2), o.size(-1)))

    return distance


def _ray_circle_intersection(
    o: torch.Tensor, d: torch.Tensor, r: torch.Tensor
) -> torch.Tensor:
    # https://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm
    # assumes circle centered on origin, origin of ray in circle
    # o, d: origin and direction of line, NCHW
    # r: radius of circle, centered on (0,0)
    # f = o

    # r /= 2

    a = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]  # dot(d, d)
    b = 2 * (o[:, 0] * d[:, 0] + o[:, 1] * d[:, 1])  # 2*dot(f, d)
    c = (o[:, 0] * o[:, 0] + o[:, 1] * o[:, 1]) - r * r  # dot(f, f) - r*r

    discriminant = b * b - 4 * a * c

    discriminant = torch.sqrt(discriminant)

    t2 = (-b + discriminant) / (2 * a)  # NHW

    t2 = t2.reshape((1, 1, o.size(-2), o.size(-1)))

    intersection = o + d * t2

    return intersection


def _ray_rectangle_intersection(
    origin: torch.Tensor,
    normals: torch.Tensor,
    half_height: float,
    width_left: float,
    attack_angle_def: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normal_angles = torch.atan2(normals[:, 1], normals[:, 0])

    # We ensure that angles are in [-90, 90] and zero is pointing left
    normal_angles = 180 - normal_angles * 180.0 / np.pi
    normal_angles = torch.where(
        normal_angles < 180, normal_angles, -360 + normal_angles
    )
    normal_angles -= attack_angle_def

    # We find the upper and lower angle closest to the corner angle
    corner_angle = np.rad2deg(np.atan2(half_height, width_left))

    upper_idxs = (normal_angles > 0).squeeze()
    angles_upper = normal_angles[0, 0, upper_idxs]
    angles_lower = normal_angles[0, 0, ~upper_idxs]

    closest_idx_top = torch.argmin(torch.abs(angles_upper - corner_angle), dim=-1)
    closest_idx_bot = torch.argmin(torch.abs(angles_lower + corner_angle), dim=-1)

    # Number of points to fill with vertical intersection at the end
    n_fill_x_top = angles_upper.shape[-1] - closest_idx_top - 1
    x_intersect_upper = torch.concatenate(
        [
            torch.linspace(
                0,
                -width_left,
                int(angles_upper.shape[-1] - n_fill_x_top + 1),
                device=origin.device,
            )[1:-1],
            torch.tensor([-width_left] * (n_fill_x_top + 1), device=origin.device),
        ],
        dim=0,
    )

    n_fill_x_bot = closest_idx_bot
    x_intersect_lower = torch.concatenate(
        [
            torch.tensor([-width_left] * (n_fill_x_bot + 1), device=origin.device),
            torch.linspace(
                -width_left,
                0,
                int(angles_lower.shape[-1] - closest_idx_bot + 1),
                device=origin.device,
            )[1:-1],
        ],
        dim=0,
    )

    x_fill_y_top = angles_upper.shape[-1] - n_fill_x_top
    y_intersect_upper = torch.concatenate(
        [
            torch.tensor([half_height] * x_fill_y_top, device=origin.device),
            torch.linspace(
                half_height,
                0,
                int(angles_upper.shape[-1] - x_fill_y_top + 2),
                device=origin.device,
            )[1:-1],
        ],
        dim=0,
    )

    y_fill_y_bot = n_fill_x_bot
    y_intersect_lower = torch.concatenate(
        [
            torch.linspace(
                0, -half_height, int(y_fill_y_bot + 1), device=origin.device
            )[:-1],
            torch.tensor(
                [-half_height] * (angles_lower.shape[-1] - y_fill_y_bot),
                device=origin.device,
            ),
        ],
        dim=0,
    )

    intersection = (
        torch.concatenate(
            [
                torch.stack([x_intersect_upper, y_intersect_upper], dim=0),
                torch.stack([x_intersect_lower, y_intersect_lower], dim=0),
            ],
            dim=-1,
        )
        .unsqueeze(0)
        .unsqueeze(2)
    )

    return intersection, closest_idx_top, x_intersect_upper.shape[-1] + closest_idx_bot


def make_airfoil_domain(
    n_dims: int,
    res_z: int,
    H: float,
    L: float,
    vel_in: float,
    attack_angle_deg: float,
    viscosity: torch.Tensor,
    resolution_div: int,
    tail_grow_mul: float,
    cpu_device: torch.device,
    cuda_device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> PISOtorch.Domain:
    """Create the PISOtorch Domain for the airfoil environment.

    Parameters
    ----------
    n_dims: int
        Number of spatial dimensions (2 or 3).

    res_z: int
        Resolution in the spanwise direction (only for 3D).

    H: float
        Height of the domain.

    L: float
        Length of the domain.

    airfoil_path: Path
        Path to the airfoil coordinates file.

    vel_in: float
        Inflow velocity.

    attack_angle_deg: float
        Attack angle of the airfoil in degrees.

    viscosity: torch.Tensor
        Kinematic viscosity of the fluid.

    resolution_div: int
        Resolution divisor for the airfoil grid.

    tail_grow_mul: float
        Growth multiplier for the tail grid spacing.

    cpu_device: torch.device
        CPU device.

    cuda_device: torch.device
        CUDA device.

    dtype: torch.dtype, optional
        Data type for tensors. Defaults to torch.float32.

    Returns
    -------
    PISOtorch.Domain
        The created PISOtorch Domain.
    """
    if resolution_div not in [1, 2, 4]:
        raise ValueError("resolution_div must be 1, 2, or 4.")

    offset_left = 1.5
    front_x_width = 0.5
    grid_half_height = H / 2

    normal_res = 96 // resolution_div
    normal_base = 0.97
    normal_weights = shapes.make_weights_exp(
        res=normal_res - 1, base=normal_base, refinement="START"
    )
    normal_weights_reversed = shapes.make_weights_exp(
        res=normal_res - 1, base=normal_base, refinement="END"
    )

    ### MAKE AIRFOIL GRID ###
    _, airfoil_coords = read_airfoil(
        attack_angle_deg=attack_angle_deg,
        cpu_device=cpu_device,
        dtype=dtype,
    )  # NCHW

    airfoil_res = airfoil_coords.size(-1)
    airfoil_len_x = torch.max(airfoil_coords[:, 0])

    airfoil_end = airfoil_coords[:, :, :, :1]

    point_start_top = torch.tensor(
        [0, grid_half_height], device=cpu_device, dtype=dtype
    ).reshape((1, 2, 1, 1))
    point_end_top = torch.tensor(
        [airfoil_len_x, grid_half_height], device=cpu_device, dtype=dtype
    ).reshape((1, 2, 1, 1))
    point_start_bot = torch.tensor(
        [0, -grid_half_height], device=cpu_device, dtype=dtype
    ).reshape((1, 2, 1, 1))
    point_end_bot = torch.tensor(
        [airfoil_len_x, -grid_half_height], device=cpu_device, dtype=dtype
    ).reshape((1, 2, 1, 1))

    if resolution_div > 1:
        div = int(resolution_div)
        airfoil_coords = torch.nn.AvgPool2d([1, div])(airfoil_coords)  # type: ignore
        airfoil_coords = torch.cat([airfoil_end, airfoil_coords, airfoil_end], dim=-1)
        airfoil_res = airfoil_coords.size(-1)

    airfoil_end_spacing = torch.linalg.vector_norm(
        airfoil_coords[0, :, 0, 1] - airfoil_coords[0, :, 0, 0]
    )
    airfoil_end_extend = airfoil_end + torch.tensor(
        [airfoil_end_spacing, 0], device=cpu_device, dtype=dtype
    ).reshape((1, 2, 1, 1))
    airfoil_coords_extended = torch.cat(
        [airfoil_end_extend, airfoil_coords, airfoil_end_extend], dim=-1
    )

    # extrude normal
    airfoil_spacing = (
        airfoil_coords_extended[0, :, 0, 2:] - airfoil_coords_extended[0, :, 0, :-2]
    )  # CW
    airfoil_normals = torch.flip(airfoil_spacing, dims=(0,)) * torch.tensor(
        [1, -1], device=cpu_device, dtype=dtype
    ).reshape((2, 1))
    airfoil_normals /= torch.linalg.vector_norm(airfoil_normals, dim=0)
    airfoil_normals = torch.reshape(airfoil_normals, (1, 2, 1, -1))

    airfoil_spacing = (
        airfoil_coords_extended[0, :, 0, 1:] - airfoil_coords_extended[0, :, 0, :-1]
    )  # CW
    airfoil_spacing = torch.linalg.vector_norm(airfoil_spacing, dim=0)  # W
    min_size = torch.min(airfoil_spacing).numpy().tolist()

    tail_sizes = [min_size]
    tail_dist = min_size
    while tail_dist < grid_half_height:
        size = tail_sizes[-1] * tail_grow_mul
        tail_sizes.append(size)
        tail_dist = tail_dist + size
    tail_weights = [0] + (np.cumsum(tail_sizes) / tail_dist).tolist()
    tail_res_x = len(tail_weights)

    distances_top = _distance_to_point(
        airfoil_coords[..., : airfoil_res // 2],
        airfoil_normals[..., : airfoil_res // 2],
        point_start_top,
    )
    _, min_d_top_idx = torch.min(distances_top[0, 0, 0, :], dim=-1)
    distances_bot = _distance_to_point(
        airfoil_coords[..., airfoil_res // 2 :],
        airfoil_normals[..., airfoil_res // 2 :],
        point_start_bot,
    )
    _, min_d_bot_idx = torch.min(distances_bot[0, 0, 0, :], dim=-1)
    min_d_bot_idx += airfoil_res // 2

    # upper outer boundary
    dist_top = point_start_top - point_end_top
    steps_top = min_d_top_idx
    step_top = dist_top / steps_top
    coords_top = torch.cat(
        [point_end_top + step_top * i for i in range(steps_top + 1)], dim=-1
    )

    # lower outer boundary
    dist_bot = point_end_bot - point_start_bot
    steps_bot = (airfoil_res - 1) - min_d_bot_idx
    step_bot = dist_bot / steps_bot
    coords_bot = torch.cat(
        [point_start_bot + step_bot * i for i in range(steps_bot + 1)], dim=-1
    )

    # front outer boundary
    coords_front, upper_corner_idx, lower_corner_idx = _ray_rectangle_intersection(
        origin=airfoil_coords[..., min_d_top_idx + 1 : min_d_bot_idx],
        normals=airfoil_normals[..., min_d_top_idx + 1 : min_d_bot_idx],
        half_height=grid_half_height,
        width_left=front_x_width,
        attack_angle_def=attack_angle_deg,
    )
    coords_outer = torch.cat([coords_top, coords_front, coords_bot], dim=-1)

    if resolution_div == 2:
        upper_corner_idx += 2
        lower_corner_idx += 2
    elif resolution_div == 1:
        upper_corner_idx += 7
        lower_corner_idx += 7

    top_slice = slice(0, coords_bot.shape[-1] + upper_corner_idx + 3)
    front_slice = slice(
        coords_bot.shape[-1] + upper_corner_idx + 2,
        coords_bot.shape[-1] + lower_corner_idx + 3,
    )
    bot_slice = slice(coords_bot.shape[-1] + lower_corner_idx + 2, None)

    # Top originally was right to left, we flip to left to right
    airfoil_coords_top = airfoil_coords[..., top_slice]
    coords_outer_top = coords_outer[..., top_slice]
    airfoil_res_top = coords_outer_top.shape[-1]

    airfoil_coords_top = torch.flip(airfoil_coords_top, dims=(-1,))
    coords_outer_top = torch.flip(coords_outer_top, dims=(-1,))

    # Front originally was right to left, so we first swap x and y
    # to get bottom to top, then flip to top to bottom
    airfoil_coords_front = airfoil_coords[..., front_slice]
    coords_outer_front = coords_outer[..., front_slice]
    airfoil_res_front = coords_outer_front.shape[-1]

    airfoil_coords_front = torch.movedim(airfoil_coords_front, -1, -2)
    airfoil_coords_front = torch.flip(airfoil_coords_front, dims=(-2,))
    coords_outer_front = torch.movedim(coords_outer_front, -2, -1)
    coords_outer_front = torch.flip(coords_outer_front, dims=(-2,))

    # Bottom is already left to right
    airfoil_coords_bot = airfoil_coords[..., bot_slice]
    coords_outer_bot = coords_outer[..., bot_slice]
    airfoil_res_bot = coords_outer_bot.shape[-1]

    point_start_top_x = airfoil_coords_top[0, 0, 0, 0].item()
    point_start_top_y = airfoil_coords_top[0, 1, 0, 0].item()
    point_start_bot_x = airfoil_coords_bot[0, 0, 0, 0].item()
    point_start_bot_y = airfoil_coords_bot[0, 1, 0, 0].item()

    point_end_top_x = airfoil_coords_top[0, 0, 0, -1].item()
    point_end_top_y = airfoil_coords_top[0, 1, 0, -1].item()
    point_end_bot_x = airfoil_coords_bot[0, 0, 0, -1].item()
    point_end_bot_y = airfoil_coords_bot[0, 1, 0, -1].item()

    left_grid_corners = [
        (-offset_left, -grid_half_height),
        (-front_x_width, -grid_half_height),
        (-offset_left, grid_half_height),
        (-front_x_width, grid_half_height),
    ]
    airfoil_grid_corners_top = [
        (point_start_top_x, point_start_top_y),
        (point_end_top_x, point_end_top_y),
        (-front_x_width, grid_half_height),
        (point_end_top_x, grid_half_height),
    ]
    airfoil_grid_corners_front = [
        (-front_x_width, -grid_half_height),
        (point_start_bot_x, point_start_bot_y),
        (-front_x_width, grid_half_height),
        (point_start_top_x, point_start_top_y),
    ]
    airfoil_grid_corners_bot = [
        (-front_x_width, -grid_half_height),
        (point_end_bot_x, -grid_half_height),
        (point_start_bot_x, point_start_bot_y),
        (point_end_bot_x, point_end_bot_y),
    ]

    def make_border(t: torch.Tensor, shift_x: float | None = None) -> list[list[float]]:
        if shift_x is not None:
            t = t.clone()
            t[0, :] += shift_x
        return torch.movedim(t, 0, 1).cpu().clone().numpy().tolist()

    grid_left = shapes.generate_grid_vertices_2D(
        [airfoil_res_front, int(0.75 * normal_res)],
        left_grid_corners,
        [
            None,
            None,  # make_border(coords_outer_front[0, :, :, 0]),
            None,
            None,
        ],
        dtype=dtype,
    )
    airfoil_grid_top = shapes.generate_grid_vertices_2D(
        [normal_res, airfoil_res_top],
        airfoil_grid_corners_top,
        [None, None, make_border(airfoil_coords_top[0, :, 0, :]), None],
        x_weights=normal_weights_reversed,
        dtype=dtype,
    )

    airfoil_grid_front = shapes.generate_grid_vertices_2D(
        [airfoil_res_front, normal_res],
        airfoil_grid_corners_front,
        [
            None,  # make_border(coords_outer_front[0, :, :, 0]),
            make_border(airfoil_coords_front[0, :, :, 0]),
            None,
            None,
        ],
        y_weights=normal_weights,
        dtype=dtype,
    )

    airfoil_grid_bot = shapes.generate_grid_vertices_2D(
        [normal_res, airfoil_res_bot],
        airfoil_grid_corners_bot,
        [
            None,
            None,
            None,
            make_border(airfoil_coords_bot[0, :, 0, :]),
        ],
        x_weights=normal_weights,
        dtype=dtype,
    )

    corners_tail_upper = [
        (point_end_top_x, point_end_top_y),
        (L, point_end_top_y),
        (point_end_top_x, grid_half_height),
        (L, grid_half_height),
    ]
    corners_tail_lower = [
        (point_end_bot_x, -grid_half_height),
        (L, -grid_half_height),
        (point_end_bot_x, point_end_bot_y),
        (L, point_end_bot_y),
    ]

    coords_tail_upper = shapes.generate_grid_vertices_2D(
        [normal_res, tail_res_x],
        corners_tail_upper,
        None,
        x_weights=normal_weights_reversed,
        y_weights=tail_weights,
        dtype=dtype,
    )
    coords_tail_lower = shapes.generate_grid_vertices_2D(
        [normal_res, tail_res_x],
        corners_tail_lower,
        None,
        x_weights=normal_weights,
        y_weights=tail_weights,
        dtype=dtype,
    )

    grid_left = grid_left.to(cuda_device)
    airfoil_grid_top = airfoil_grid_top.to(cuda_device)
    airfoil_grid_front = airfoil_grid_front.to(cuda_device)
    airfoil_grid_bot = airfoil_grid_bot.to(cuda_device)
    coords_tail_upper = coords_tail_upper.to(cuda_device)
    coords_tail_lower = coords_tail_lower.to(cuda_device)

    grids = [
        grid_left,
        airfoil_grid_top,
        airfoil_grid_front,
        airfoil_grid_bot,
        coords_tail_upper,
        coords_tail_lower,
    ]

    if n_dims == 3:
        [
            grid_left,
            airfoil_grid_top,
            airfoil_grid_front,
            airfoil_grid_bot,
            coords_tail_upper,
            coords_tail_lower,
        ] = [
            shapes.extrude_grid_z(
                g,
                res_z=res_z,
                start_z=-H / 2,  # type: ignore
                end_z=H / 2,  # type: ignore
            )
            for g in grids
        ]

    ### MAKE DOMAIN AND BLOCKS ###
    domain = PISOtorch.Domain(
        n_dims,
        viscosity,
        name="AirfoilDomain",
        dtype=dtype,
        device=cuda_device,
        passiveScalarChannels=0,
    )

    block_left = domain.CreateBlock(vertexCoordinates=grid_left, name="LeftBlock")
    block_airfoil_front = domain.CreateBlock(
        vertexCoordinates=airfoil_grid_front, name="AirfoilFront"
    )
    block_airfoil_top = domain.CreateBlock(
        vertexCoordinates=airfoil_grid_top, name="AirfoilTop"
    )
    block_airfoil_bot = domain.CreateBlock(
        vertexCoordinates=airfoil_grid_bot, name="AirfoilBot"
    )

    block_tail_upper = domain.CreateBlock(
        vertexCoordinates=coords_tail_upper, name="TailUpper"
    )
    block_tail_lower = domain.CreateBlock(
        vertexCoordinates=coords_tail_lower, name="TailLower"
    )

    inflow = get_inflow_profile(
        h=H,
        res_y=airfoil_res_front - 1,
        n_dims=n_dims,
        dtype=dtype,
        device=cuda_device,
        res_z=res_z if n_dims == 3 else None,
    )
    inflow *= vel_in

    block_left.CloseBoundary("-x", inflow)

    # Walls
    block_left.CloseBoundary("+y")
    block_left.CloseBoundary("-y")
    block_airfoil_top.CloseBoundary("+y")
    block_tail_upper.CloseBoundary("+y")
    block_tail_lower.CloseBoundary("-y")

    # Airfoil
    block_airfoil_front.CloseBoundary("+x")
    block_airfoil_top.CloseBoundary("-y")
    block_airfoil_bot.CloseBoundary("+y")

    # Outflow
    if n_dims == 2:
        outflow_vel = torch.tensor([[vel_in, 0]], device=cuda_device, dtype=dtype)
    else:
        outflow_vel = torch.tensor([[vel_in, 0, 0]], device=cuda_device, dtype=dtype)

    block_tail_upper.CloseBoundary("+x", outflow_vel)
    block_tail_upper.getBoundary("+x").makeVelocityVarying()
    block_tail_lower.CloseBoundary("+x", outflow_vel)
    block_tail_lower.getBoundary("+x").makeVelocityVarying()

    if n_dims == 3:
        block_left.MakePeriodic("z")
        block_airfoil_front.MakePeriodic("z")
        block_airfoil_top.MakePeriodic("z")
        block_airfoil_bot.MakePeriodic("z")
        block_tail_upper.MakePeriodic("z")
        block_tail_lower.MakePeriodic("z")

    if n_dims == 2:
        block_left.ConnectBlock("+x", block_airfoil_front, "-x", "-y")

        block_airfoil_front.ConnectBlock("+y", block_airfoil_top, "-x", "+y")
        block_airfoil_front.ConnectBlock("-y", block_airfoil_bot, "-x", "-y")

        block_airfoil_top.ConnectBlock("+x", block_tail_upper, "-x", "-y")
        block_airfoil_bot.ConnectBlock("+x", block_tail_lower, "-x", "-y")
        block_tail_upper.ConnectBlock("-y", block_tail_lower, "+y", "-x")
    else:
        block_left.ConnectBlock("+x", block_airfoil_front, "-x", "-y", "-z")

        block_airfoil_front.ConnectBlock("+y", block_airfoil_top, "-x", "-z", "+y")
        block_airfoil_front.ConnectBlock("-y", block_airfoil_bot, "-x", "-z", "-y")

        block_airfoil_top.ConnectBlock("+x", block_tail_upper, "-x", "-y", "-z")
        block_airfoil_bot.ConnectBlock("+x", block_tail_lower, "-x", "-y", "-z")
        block_tail_upper.ConnectBlock("-y", block_tail_lower, "+y", "-z", "-x")

    out_bounds = [
        block_tail_upper.getBoundary("+x"),
        block_tail_lower.getBoundary("+x"),
    ]

    balance_boundary_fluxes(domain, out_bounds)

    return domain
