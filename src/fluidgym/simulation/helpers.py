"""Simulation helper functions for FluidGym using PISOtorch."""

import torch

from fluidgym.simulation.extensions import (
    PISOtorch,  # type: ignore[import-untyped,import-not-found]
)


def get_cell_size(block: PISOtorch.Block) -> torch.Tensor:
    """Get cell volumes from a block."""
    if block.hasTransform():
        dims = block.getSpatialDims()
        p = [0, dims + 1] + list(range(1, dims + 1))
        return torch.permute(block.transform[..., -1:], p)
    else:
        return torch.tensor(
            1.0, dtype=block.velocity.dtype, device=block.velocity.device
        )


def get_cell_centers(vertex_coords: torch.Tensor) -> torch.Tensor:
    """Compute cell centers from vertex coordinates."""
    vertex_coords = vertex_coords.squeeze(0)

    if vertex_coords.size(0) == 2:
        if vertex_coords.dim() != 3:  # [2, H, W]
            raise ValueError("Expected vertex_coords to have 4 dimensions for 2D data.")
        c0 = 0.25 * (
            vertex_coords[:, :-1, :-1]
            + vertex_coords[:, 1:, :-1]
            + vertex_coords[:, :-1, 1:]
            + vertex_coords[:, 1:, 1:]
        )
    else:
        if vertex_coords.dim() != 4:  # [3, D, H, W]
            raise ValueError("Expected vertex_coords to have 5 dimensions for 3D data.")
        c0 = (
            1.0
            / 8.0
            * (
                vertex_coords[:, :-1, :-1, :-1]
                + vertex_coords[:, 1:, :-1, :-1]
                + vertex_coords[:, :-1, 1:, :-1]
                + vertex_coords[:, 1:, 1:, :-1]
                + vertex_coords[:, :-1, :-1, 1:]
                + vertex_coords[:, 1:, :-1, 1:]
                + vertex_coords[:, :-1, 1:, 1:]
                + vertex_coords[:, 1:, 1:, 1:]
            )
        )

    return c0.squeeze()


def get_transform_matrices(
    transforms: torch.Tensor, n_dims: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get transformation matrices from a block."""
    # Extract M matrix (reshape to [1, D, H, W, n_dims, n_dims])
    M = transforms[..., : n_dims * n_dims].reshape(
        *transforms.shape[:-1], n_dims, n_dims
    )

    # Extract Minv matrix
    Minv = transforms[..., n_dims * n_dims : 2 * n_dims * n_dims].reshape(
        *transforms.shape[:-1], n_dims, n_dims
    )

    # Extract determinant
    det = transforms[..., -1]

    return M, Minv, det


def transform_to_computional_domain(
    coords: torch.Tensor, Minv: torch.Tensor, ndims: int
) -> torch.Tensor:
    """Transform coordinates to computational domain."""
    if ndims == 2:
        # From [1, 2, H, W] to [1, H, W, 2]
        coords = coords.permute(0, 2, 3, 1)
    else:
        # From [1, 3, D, H, W] to [1, D, H, W, 3]
        coords = coords.permute(0, 2, 3, 4, 1)

    # Apply transformation: computational_pos = Minv @ physical_pos
    # Using torch.einsum for batched matrix-vector multiplication
    coords_computational = torch.einsum("...ij,...j->...i", Minv, coords)

    if ndims == 2:
        # From [1, H, W, 2] to [1, 2, H, W]
        coords_computational = coords_computational.permute(0, 3, 1, 2)
    else:
        # From [1, D, H, W, 3] to [1, 3, D, H, W]
        coords_computational = coords_computational.permute(0, 4, 1, 2, 3)

    return coords_computational
