"""Utility functions for computing forces on wall boundaries in FluidGym."""

from typing import Any

import torch

from fluidgym.simulation.extensions import (
    PISOtorch,  # type: ignore[import-untyped,import-not-found]
)


def wall_distance_from_vertices(
    vc: torch.Tensor, centers: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute wall distances and normals from vertex coordinates and cell centers."""
    v0 = vc[:, :-1]
    v1 = vc[:, 1:]
    e = v1 - v0

    eps = 1e-20

    # Unit tangent along each edge
    e_norm = torch.linalg.norm(e, dim=0, keepdim=True) + eps
    t = e / e_norm  # (2, N)

    n = torch.stack([t[1], -t[0]], dim=0)  # (2, N)

    # We use edge midpoints for the wall reference point
    m = 0.5 * (v0 + v1)  # (2, N)

    d_raw = ((centers - m) * n).sum(dim=0)  # (N,)

    d = d_raw.abs()
    d = torch.clamp(d, min=eps)

    # Ensure normals point into the fluid
    n *= -1

    return d, n


def collect_boundary_coords(
    domain: PISOtorch.Domain,
    block_idxs: list[int],
    boundary_cell_slices: list[tuple[Any, ...]],
    flip_dims: list[list[int]],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """Collect boundary coordinates and cell centers from multiple blocks in the domain.

    Parameters
    ----------
    domain: PISOtorch.Domain
        The fluid domain containing the blocks.

    block_idxs: list[int]
        List of block indices to collect data from.

    boundary_cell_slices: list[tuple[Any, ...]]
        List of slices to extract boundary cells for each block.

    flip_dims: list[list[int]]
        List of dimensions to flip for each block.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing concatenated boundary coordinates and cell centers from the
        specified boundary blocks.
    """
    all_boundary_coords_list = []
    all_cell_centers_list = []

    for block_idx, boundary_slice, flip in zip(
        block_idxs, boundary_cell_slices, flip_dims, strict=True
    ):
        block = domain.getBlock(block_idx)

        # Cell boundary coordinates
        vertex_coords: torch.Tensor = domain.getVertexCoordinates()[block_idx]
        vertex_coords = vertex_coords.squeeze()
        boundary_coords = vertex_coords[boundary_slice]

        # Cell centers
        cell_centers: torch.Tensor = block.getCellCoordinates()
        cell_centers = cell_centers.squeeze()
        cell_centers = cell_centers[boundary_slice]

        if flip:
            # reverse order of x-coords
            boundary_coords = torch.flip(boundary_coords, dims=flip)
            cell_centers = torch.flip(cell_centers, dims=flip)

        # Do avoid duplicates we remove the last coords
        if block_idx != block_idxs[-1]:
            boundary_coords = boundary_coords[..., :-1]

        all_boundary_coords_list.append(boundary_coords)
        all_cell_centers_list.append(cell_centers)

    # Concatenate all blocks
    all_boundary_coords = torch.cat(all_boundary_coords_list, dim=-1)
    all_cell_centers = torch.cat(all_cell_centers_list, dim=-1)

    return all_boundary_coords, all_cell_centers


def collect_boundary_fields(
    domain: PISOtorch.Domain,
    block_idxs: list[int],
    boundary_faces: list[str],
    boundary_cell_slices: list[tuple[Any, ...]],
    flip_dims: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect boundary field data from multiple blocks in the domain.

    Parameters
    ----------
    domain: PISOtorch.Domain
        The fluid domain containing the blocks.

    block_idxs: list[int]
        List of block indices to collect data from.

    boundary_faces: list[str]
        List of boundary face names corresponding to each block.

    boundary_cell_slices: list[tuple[Any, ...]]
        List of slices to extract boundary cells for each block.

    flip_dims: list[list[int]]
        List of dimensions to flip for each block.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing concatenated cell velocities, boundary velocities, and
        cell pressures from the specified boundary blocks.
    """
    all_u_cell_list = []
    all_u_boundary_list = []
    all_p_cell_list = []

    for block_idx, cell_slice, flip, face in zip(
        block_idxs, boundary_cell_slices, flip_dims, boundary_faces, strict=True
    ):
        block = domain.getBlock(block_idx)

        # Cell velocity
        u_cell: torch.Tensor = block.getVelocity(False)
        u_cell = u_cell.squeeze()
        u_cell = u_cell[cell_slice]

        # Boundary velocity
        u_boundary: torch.Tensor = block.getBoundary(face).velocity
        u_boundary = u_boundary.squeeze()

        # If the boundary velocity is not varying, it has shape [N_DIMS,],
        # we need to expand it to match the cell velocity shape
        if u_cell.ndim != u_boundary.ndim:
            if domain.getSpatialDims() == 2:
                u_boundary = u_boundary[:, None].repeat(1, u_cell.shape[1])
            else:
                u_boundary = u_boundary[:, None, None].repeat(
                    1, u_cell.shape[1], u_cell.shape[2]
                )

        # Pressure
        p_wall: torch.Tensor = block.pressure
        p_wall = p_wall.squeeze()
        p_wall = p_wall[cell_slice]

        if flip:
            # reverse order of x-coords
            u_cell = torch.flip(u_cell, dims=flip)
            u_boundary = torch.flip(u_boundary, dims=flip)
            p_wall = torch.flip(p_wall, dims=flip)

        all_u_cell_list.append(u_cell)
        all_u_boundary_list.append(u_boundary)
        all_p_cell_list.append(p_wall)

    # Concatenate all blocks
    all_u_cell = torch.cat(all_u_cell_list, dim=-1)
    all_u_boundary = torch.cat(all_u_boundary_list, dim=-1)
    all_p_cell = torch.cat(all_p_cell_list, dim=-1)

    return all_u_cell, all_u_boundary, all_p_cell


def compute_forces_2d(
    u_cell: torch.Tensor,
    u_boundary: torch.Tensor,
    p_cell: torch.Tensor,
    wall_normals: torch.Tensor,
    tangent_lengths: torch.Tensor,
    wall_distances: torch.Tensor,
    wall_face_lengths: torch.Tensor,
    viscosity: torch.Tensor,
) -> torch.Tensor:
    """Compute forces along the x- and y-axis on 2D wall boundaries.

    Parameters
    ----------
    u_cell: torch.Tensor
        Cell velocities at the wall boundary. Shape: [2, N].

    u_boundary: torch.Tensor
        Boundary velocities at the wall boundary. Shape: [2, N].

    p_cell: torch.Tensor
        Cell pressures at the wall boundary. Shape: [N,].

    wall_normals: torch.Tensor
        Wall normal vectors. Shape: [2, N].

    tangent_lengths: torch.Tensor
        Lengths of the wall tangents. Shape: [N,].

    wall_distances: torch.Tensor
        Distances from cell centers to the wall. Shape: [N,].

    wall_face_lengths: torch.Tensor
        Lengths of the wall faces. Shape: [N,].

    viscosity: torch.Tensor
        Fluid viscosity.

    Returns
    -------
    torch.Tensor
        Total forces on the wall boundaries. Shape: [2,].
    """
    wall_tangents = torch.stack([wall_normals[1, :], -wall_normals[0, :]], dim=0)

    u_cell_left = torch.roll(u_cell, shifts=-1, dims=1)
    u_cell_right = torch.roll(u_cell, shifts=1, dims=1)

    # Normal derivatives
    du_dn = (u_cell[0, :] - u_boundary[0, :]) / wall_distances
    dv_dn = (u_cell[1, :] - u_boundary[1, :]) / wall_distances

    # Tangential derivatives
    du_dt = (u_cell_right[0, :] - u_cell_left[0, :]) / (2 * tangent_lengths)
    dv_dt = (u_cell_right[1, :] - u_cell_left[1, :]) / (2 * tangent_lengths)

    # Compose full velocity gradients
    du_dx = du_dn * wall_normals[0, :] + du_dt * wall_tangents[0, :]
    du_dy = du_dn * wall_normals[1, :] + du_dt * wall_tangents[1, :]
    dv_dx = dv_dn * wall_normals[0, :] + dv_dt * wall_tangents[0, :]
    dv_dy = dv_dn * wall_normals[1, :] + dv_dt * wall_tangents[1, :]

    G = torch.stack(
        [
            torch.stack([du_dx, du_dy], dim=1),
            torch.stack([dv_dx, dv_dy], dim=1),
        ],
        dim=1,
    )  # [N, 2, 2]

    S = 0.5 * (G + G.transpose(1, 2))  # [N, 2, 2]
    viscous_stress = 2 * viscosity * S
    pressure_stress = -p_cell[..., None, None] * torch.eye(2, device=S.device)
    total_stress = viscous_stress + pressure_stress

    normals = wall_normals.transpose(1, 0).unsqueeze(2)

    traction = torch.bmm(total_stress, normals).squeeze()  # [N,2]
    force = traction * wall_face_lengths.unsqueeze(1)  # [N,2]

    total_force = torch.sum(force, dim=0)

    return total_force


def compute_forces_3d(
    u_cell: torch.Tensor,
    u_boundary: torch.Tensor,
    p_cell: torch.Tensor,
    wall_normals: torch.Tensor,
    tangent_lengths: torch.Tensor,
    wall_distances: torch.Tensor,
    wall_face_areas: torch.Tensor,
    viscosity: torch.Tensor,
) -> torch.Tensor:
    """Compute forces along the x- and y-axis on 2D wall boundaries in a 3D domain.

    Parameters
    ----------
    u_cell: torch.Tensor
        Cell velocities at the wall boundary. Shape: [3, N, N_Z].

    u_boundary: torch.Tensor
        Boundary velocities at the wall boundary. Shape: [3, N, N_Z].

    p_cell: torch.Tensor
        Cell pressures at the wall boundary. Shape: [N, N_Z].

    wall_normals: torch.Tensor
        Wall normal vectors. Shape: [2, N, 1].

    tangent_lengths: torch.Tensor
        Lengths of the wall tangents. Shape: [N,].

    wall_distances: torch.Tensor
        Distances from cell centers to the wall. Shape: [N,].

    wall_face_areas: torch.Tensor
        Areas of the wall faces. Shape: [N, N_Z].

    viscosity: torch.Tensor
        Fluid viscosity.

    Returns
    -------
    torch.Tensor
        Total forces on the wall boundaries. Shape: [2, N_Z].
    """
    wall_tangents = torch.stack([wall_normals[1, :], -wall_normals[0, :]], dim=0)

    u_cell = u_cell.permute(0, 2, 1)  # [3, N, N_Z]
    u_boundary = u_boundary.permute(0, 2, 1)  # [3, N, N_Z]
    p_cell = p_cell.permute(1, 0)  # [N, N_Z]

    u_cell_left = torch.roll(u_cell, shifts=-1, dims=1)
    u_cell_right = torch.roll(u_cell, shifts=1, dims=1)

    # Normal derivatives
    du_dn = (u_cell[0, :, :] - u_boundary[0, :, :]) / wall_distances
    dv_dn = (u_cell[1, :, :] - u_boundary[1, :, :]) / wall_distances

    # Tangential derivatives
    du_dt = (u_cell_right[0, :, :] - u_cell_left[0, :, :]) / (2 * tangent_lengths)
    dv_dt = (u_cell_right[1, :, :] - u_cell_left[1, :, :]) / (2 * tangent_lengths)

    # Compose full velocity gradients
    du_dx = du_dn * wall_normals[0, :] + du_dt * wall_tangents[0, :]
    du_dy = du_dn * wall_normals[1, :] + du_dt * wall_tangents[1, :]
    dv_dx = dv_dn * wall_normals[0, :] + dv_dt * wall_tangents[0, :]
    dv_dy = dv_dn * wall_normals[1, :] + dv_dt * wall_tangents[1, :]

    G = torch.stack(
        [
            torch.stack([du_dx, du_dy], dim=1),
            torch.stack([dv_dx, dv_dy], dim=1),
        ],
        dim=1,
    )  # [N, 2, 2, N_Z]

    S = 0.5 * (G + G.transpose(1, 2))  # [N, 2, 2, N_Z]
    viscous_stress = 2 * viscosity * S
    pressure_stress = (
        -p_cell.unsqueeze(-2).unsqueeze(-2) * torch.eye(2, device=S.device)[..., None]
    )  # [N, 2, 2, N_Z]
    total_stress = viscous_stress + pressure_stress

    total_stress = total_stress.permute(0, 3, 1, 2)  # [N, N_Z, 2, 2]

    # [2, N, 1] -> [N, 2, 1]
    wall_normals = wall_normals.permute(1, 0, 2)
    wall_normals = wall_normals.squeeze(-1)  # [N, 2]
    wall_normals = wall_normals[:, None, :, None]  # [N, 1, 2, 1]
    wall_normals = wall_normals.expand(
        -1, total_stress.size(1), -1, -1
    )  # [N, N_Z, 2, 1]

    traction = torch.matmul(total_stress, wall_normals).squeeze(-1)  # [N, N_Z, 2]

    forces = traction * wall_face_areas[..., None, None]  # [N, N_Z, 2]

    forces = forces.permute(0, 2, 1)  # [N, 2, N_Z]

    forces_total = torch.sum(forces, dim=0)

    return forces_total  # [2, N_Z]
