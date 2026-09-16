# Copyright 2025 Aleksandra Franz, Nils Thürey
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch

from fluidgym.simulation.extensions import PISOtorch  # type: ignore[import-untyped]
from fluidgym.simulation.pict.data.shapes import (
    ortho_transform_to_coords,
    coords_to_center_coords,
)


def make_matrix_translation(t):
    dims = t.size()[0]
    mat = torch.zeros([dims + 1] * 2, dtype=t.dtype)
    for d in range(dims):
        mat[d, -1] = t[d]
        mat[d, d] = 1
    mat[-1, -1] = 1
    return mat


def make_matrix_scaling(s):
    dims = s.size()[0]
    mat = torch.zeros([dims + 1] * 2, dtype=s.dtype)
    for d in range(dims):
        mat[d, d] = s[d]
    mat[-1, -1] = 1
    return mat


def make_meshgrid_AABB(vertex_coords, cells_per_unit, dims, dtype=torch.float32):
    vertex_coords = vertex_coords.view(dims, -1)
    lower, _ = vertex_coords.min(dim=-1)
    upper, _ = vertex_coords.max(dim=-1)
    size = upper - lower
    resolution = np.ceil(cells_per_unit * size.cpu().numpy()).astype(int)

    grid = torch.meshgrid(
        *[
            torch.linspace(lower[dim], upper[dim], steps=resolution[dim])
            for dim in range(dims)
        ],
        indexing="xy",
    )
    grid = torch.stack(grid)
    grid = torch.unsqueeze(grid, 0)

    print("meshgrid_AABB:", grid.size())
    print("meshgrid_AABB:", grid)
    return grid


def make_uniform_transform_AABB_outer(
    vertex_coords, out_shape, dims, dtype=torch.float32
):
    vertex_coords = vertex_coords.view(dims, -1)
    lower, _ = vertex_coords.min(dim=-1)
    upper, _ = vertex_coords.max(dim=-1)
    size = upper - lower
    center = lower + size * 0.5

    # transform matrix:
    # lower maps to -0.5 (lower border of cell)
    # higher maps to out_shape-0.5
    # scaling and translation, no rotation
    out_shape_float = (
        out_shape.to(dtype)
        if isinstance(out_shape, torch.Tensor)
        else torch.tensor(out_shape, dtype=dtype)
    )
    scale = torch.tensor([torch.max(size.cpu() / out_shape_float)] * dims)
    translation_1 = make_matrix_translation(
        -out_shape_float * 0.5 + 0.5
    )  # center on origin #torch.tensor([0.5]*dims, dtype=data_list[0].dtype))
    scaling = make_matrix_scaling(scale)
    translation_2 = make_matrix_translation(
        center.cpu()
    )  # center on bounding box center

    mat = torch.matmul(translation_2, torch.matmul(scaling, translation_1))
    mat = torch.reshape(mat, (1, dims + 1, dims + 1))

    return mat


def make_uniform_transform_AABB_inner(
    vertex_coords, out_shape, dims, dtype=torch.float32
):
    vertex_coords = vertex_coords.view(dims, -1)
    lower, _ = vertex_coords.min(dim=-1)
    upper, _ = vertex_coords.max(dim=-1)
    size = upper - lower
    center = lower + size * 0.5

    # transform matrix:
    # lower maps to -0.5 (lower border of cell)
    # higher maps to out_shape-0.5
    # scaling and translation, no rotation
    out_shape_float = torch.tensor(out_shape, dtype=dtype)
    scale = torch.tensor([torch.min(size.cpu() / out_shape_float)] * dims)
    translation_1 = make_matrix_translation(
        -out_shape_float * 0.5 + 0.5
    )  # center on origin #torch.tensor([0.5]*dims, dtype=data_list[0].dtype))
    scaling = make_matrix_scaling(scale)
    translation_2 = make_matrix_translation(
        center.cpu()
    )  # center on bounding box center

    mat = torch.matmul(translation_2, torch.matmul(scaling, translation_1))
    mat = torch.reshape(mat, (1, dims + 1, dims + 1))

    return mat


def get_uniform_transform(transform, vertex_coords, shape, dims, dtype=torch.float32):
    if isinstance(transform, torch.Tensor):
        if not (
            transform.size(0) == 1
            and transform.size(1) == dims + 1
            and transform.size(2) == dims + 1
        ):
            raise ValueError(
                "Invalid transform matrix shape. must be (1,%d,%d), is %s"
                % (dims + 1, dims + 1, transform.size())
            )
        return transform
    elif isinstance(transform, np.ndarray):
        transform = torch.tensor(transform, dtype=dtype)
        if not (
            transform.size(0) == 1
            and transform.size(1) == dims + 1
            and transform.size(2) == dims + 1
        ):
            raise ValueError(
                "Invalid transform matrix shape. must be (1,%d,%d), is %s"
                % (dims + 1, dims + 1, transform.size())
            )
        return transform
    elif transform == "AABB_OUTER":
        return make_uniform_transform_AABB_outer(
            vertex_coords, shape, dims, dtype=dtype
        )
    elif transform == "AABB_INNER":
        return make_uniform_transform_AABB_inner(
            vertex_coords, shape, dims, dtype=dtype
        )
    else:
        raise ValueError("Unknown transform parameter.")


def get_output_shape(out_shape, dims):
    if isinstance(out_shape, torch.Tensor):
        # must have shape [dims] and integer type
        if not (out_shape.dim() == 1 and out_shape.size(0) == dims):
            raise ValueError("Resampling output shape does not match dimensions.")
        if not (out_shape.dtype == torch.int32):
            raise TypeError("Resampling output shape must have dtype torch.int32.")
        return out_shape
    elif isinstance(out_shape, np.ndarray):
        # must have shape [dims] and integer type
        return out_shape
    elif isinstance(out_shape, (list, tuple)):
        if not len(out_shape) == dims:
            raise ValueError("Resampling output shape does not match dimensions.")
        if not all(isinstance(_, int) for _ in out_shape):
            raise TypeError("Resampling output shape must be int or list of int.")
        return out_shape
    elif isinstance(out_shape, int):
        return [out_shape] * dims
    else:
        raise TypeError("Invalid resampling output shape: %s", out_shape)


def sample_transform_to_uniform_grid(
    data, transform, out_shape, transform_uniform="AABB_OUTER", fill_max_steps=0
):
    dims = len(data.size()) - 2
    # out_shape is x,y,z
    out_shape = get_output_shape(out_shape, dims)
    vertex_coords = ortho_transform_to_coords(
        transform, dims
    )  # NCDHW with C=x,y,z coords
    cell_coords = coords_to_center_coords(vertex_coords)

    mat = get_uniform_transform(
        transform_uniform, vertex_coords, out_shape, dims, dtype=data.dtype
    )

    out_data, out_weights = PISOtorch.SampleTransformedGridLocalToGlobal(
        data,
        cell_coords,
        mat,
        torch.tensor(out_shape, dtype=torch.int32),
        fillMaxSteps=fill_max_steps,
    )

    return out_data


# compatibility
sample_to_uniform_grid = sample_transform_to_uniform_grid


def sample_coords_to_uniform_grid(
    data,
    coords,
    out_shape,
    is_cell_coords=False,
    transform_uniform="AABB_OUTER",
    fill_max_steps=0,
):
    dims = len(data.size()) - 2
    # out_shape is x,y,z
    # coords: NCDHW with C=x,y,z coords
    out_shape = get_output_shape(out_shape, dims)
    if is_cell_coords:
        if isinstance(transform_uniform, str) and transform_uniform.startswith("AABB"):
            raise NotImplementedError("vertex_coords are required for bounding box.")
        else:
            vertex_coords = None
        cell_coords = coords
    else:
        vertex_coords = coords
        cell_coords = coords_to_center_coords(vertex_coords)

    mat = get_uniform_transform(
        transform_uniform, vertex_coords, out_shape, dims, dtype=data.dtype
    )

    out_data, out_weights = PISOtorch.SampleTransformedGridLocalToGlobal(
        data,
        cell_coords,
        mat,
        torch.tensor(out_shape, dtype=torch.int32),
        fillMaxSteps=fill_max_steps,
    )

    return out_data


def sample_multi_coords_to_uniform_grid(
    data_list,
    coords_list,
    out_shape,
    is_cell_coords=False,
    transform_uniform="AABB_OUTER",
    fill_max_steps=0,
):
    assert len(data_list) == len(coords_list)
    assert len(data_list) > 0
    dims = len(data_list[0].size()) - 2
    # out_shape is x,y,z
    # coords: NCDHW with C=x,y,z coords
    out_shape = get_output_shape(out_shape, dims)

    vertex_coords_list = []
    cell_coords_list = []
    for coords in coords_list:
        if is_cell_coords:
            if isinstance(transform_uniform, str) and transform_uniform.startswith(
                "AABB"
            ):
                raise NotImplementedError(
                    "vertex_coords are required for bounding box."
                )
            cell_coords_list.append(coords)
        else:
            vertex_coords = coords
            vertex_coords_list.append(vertex_coords)
            cell_coords_list.append(coords_to_center_coords(vertex_coords))

    # get bounding box, assuming N=1
    if is_cell_coords:
        vertex_coords = None
    else:
        vertex_coords = torch.cat(
            [_.view(dims, -1) for _ in vertex_coords_list], dim=-1
        )

    mat = get_uniform_transform(
        transform_uniform, vertex_coords, out_shape, dims, dtype=data_list[0].dtype
    )

    out_shape = (
        out_shape.to(torch.int32)
        if isinstance(out_shape, torch.Tensor)
        else torch.tensor(out_shape, dtype=torch.int32)
    )
    out_data, out_weights = PISOtorch.SampleTransformedGridLocalToGlobalMulti(
        data_list, cell_coords_list, mat, out_shape, fillMaxSteps=fill_max_steps
    )

    return out_data


def sample_transform_from_uniform_grid(
    data, transform, transform_uniform="AABB_OUTER", boundary_mode="CLAMP"
):
    shape = data.size()
    dims = len(shape) - 2
    in_shape = [shape[-(d + 1)] for d in range(dims)]

    vertex_coords = ortho_transform_to_coords(
        transform, dims
    )  # NCDHW with C=x,y,z coords
    # print("vertex_coords:", vertex_coords.size())
    cell_coords = coords_to_center_coords(vertex_coords)  # block.getCellCoordinates()

    mat = get_uniform_transform(
        transform_uniform, vertex_coords, in_shape, dims, dtype=data.dtype
    )

    if boundary_mode == "CLAMP":
        boundary_mode = PISOtorch.BoundarySampling.CLAMP
    elif boundary_mode == "CONSTANT":
        boundary_mode = PISOtorch.BoundarySampling.CONSTANT

    out_data = PISOtorch.SampleTransformedGridGlobalToLocal(
        data, mat, cell_coords, boundary_mode, torch.zeros([1], dtype=data.dtype)
    )

    return out_data


sample_from_uniform_grid = sample_transform_from_uniform_grid


def sample_coords_from_uniform_grid(
    data,
    coords,
    is_cell_coords=False,
    transform_uniform="AABB_OUTER",
    boundary_mode="CLAMP",
):
    shape = data.size()
    dims = len(shape) - 2
    in_shape = [shape[-(d + 1)] for d in range(dims)]

    # NCDHW with C=x,y,z coords
    if is_cell_coords:
        if isinstance(transform_uniform, str) and transform_uniform.startswith("AABB"):
            raise NotImplementedError("vertex_coords are required for bounding box.")
        else:
            vertex_coords = None
        cell_coords = coords
    else:
        vertex_coords = coords
        cell_coords = coords_to_center_coords(vertex_coords)
    # print("cell_coords:", cell_coords)

    mat = get_uniform_transform(
        transform_uniform, vertex_coords, in_shape, dims, dtype=data.dtype
    )

    if boundary_mode == "CLAMP":
        boundary_mode = PISOtorch.BoundarySampling.CLAMP
    elif boundary_mode == "CONSTANT":
        boundary_mode = PISOtorch.BoundarySampling.CONSTANT

    out_data = PISOtorch.SampleTransformedGridGlobalToLocal(
        data, mat, cell_coords, boundary_mode, torch.zeros([1], dtype=data.dtype)
    )

    return out_data


def sample_multi_coords_from_uniform_grid(
    data,
    coords_list,
    out_shape=None,
    is_cell_coords=False,
    transform_uniform="AABB_OUTER",
    boundary_mode="CLAMP",
):
    assert len(coords_list) > 0
    dims = data.dim() - 2
    # out_shape is x,y,z
    # coords: NCDHW with C=x,y,z coords
    if out_shape is None:
        shape = data.size()
        out_shape = [shape[-(d + 1)] for d in range(dims)]
    else:
        out_shape = get_output_shape(out_shape, dims)

    vertex_coords_list = []
    cell_coords_list = []
    for coords in coords_list:
        if is_cell_coords:
            if isinstance(transform_uniform, str) and transform_uniform.startswith(
                "AABB"
            ):
                raise NotImplementedError(
                    "vertex_coords are required for bounding box."
                )
            cell_coords_list.append(coords)
        else:
            vertex_coords = coords
            vertex_coords_list.append(vertex_coords)
            cell_coords_list.append(coords_to_center_coords(vertex_coords))

    # get bounding box, assuming N=1
    if is_cell_coords:
        vertex_coords = None
    else:
        vertex_coords = torch.cat(
            [_.view(dims, -1) for _ in vertex_coords_list], dim=-1
        )

    mat = get_uniform_transform(
        transform_uniform, vertex_coords, out_shape, dims, dtype=data.dtype
    )
    if boundary_mode == "CLAMP":
        boundary_mode = PISOtorch.BoundarySampling.CLAMP
    elif boundary_mode == "CONSTANT":
        boundary_mode = PISOtorch.BoundarySampling.CONSTANT

    data_list = []
    for cell_coords in cell_coords_list:
        out_data = PISOtorch.SampleTransformedGridGlobalToLocal(
            data, mat, cell_coords, boundary_mode, torch.zeros([1], dtype=data.dtype)
        )
        data_list.append(out_data)

    return data_list


# ---------------------------------------------------------------------------
# Differentiable multi-block resampling.
#
# `PISOtorch.SampleTransformedGridLocalToGlobalMulti` is a raw pybind binding
# with no autograd::Function wrapper, so every resampled observation is detached
# from the simulation graph. The functions below reimplement it in pure torch so
# gradients reach the source cells, which full-BPTT / SHAC-style training needs
# for the terminal-value and closed-loop terms of the policy gradient.
#
# Nothing here is wired into the observation pipeline yet; see
# `tests/simulation/test_torch_resample.py` for the validation against the
# compiled kernel.
# ---------------------------------------------------------------------------


def sample_multi_coords_to_uniform_grid_diff(
    data_list,
    coords_list,
    out_shape,
    is_cell_coords=False,
    transform_uniform="AABB_OUTER",
    fill_max_steps=0,
):
    """Differentiable, pure-torch re-implementation of
    :func:`sample_multi_coords_to_uniform_grid`.

    The compiled ``PISOtorch.SampleTransformedGridLocalToGlobalMulti`` kernel
    performs a bilinear *splat* (scatter) of every source cell centre onto a
    uniform output grid, accumulating value-weighted contributions and a
    per-cell weight, and finally normalising by that weight. That operation is
    linear in the cell *values* (the geometry -- ``coords_list`` and the
    world->index transform -- is static), so it can be expressed with
    ``index_add`` and stays differentiable w.r.t. ``data_list``.

    This is the straightforward reference version: it rebuilds all of the
    (static) geometry on every call, which makes the autograd tape grow by
    ~8 * n_source_cells index and weight entries *per call*. For anything in a
    BPTT loop use :class:`DiffMultiblockResampler`, which caches that geometry
    and can restrict the output to the handful of cells an observation actually
    reads.

    Parameters
    ----------
    data_list: Sequence[torch.Tensor]
        Per-block cell data, each of shape ``[1, C, *spatial]`` (NCDHW / NCHW).
    coords_list: Sequence[torch.Tensor]
        Per-block coordinates, matching :func:`sample_multi_coords_to_uniform_grid`.
    out_shape: int | list | tuple | torch.Tensor
        Output grid shape in ``(x, y[, z])`` order.
    is_cell_coords: bool
        Whether ``coords_list`` holds cell-centre (True) or vertex (False)
        coordinates. Defaults to False.
    transform_uniform: str | torch.Tensor
        World->index transform, see :func:`get_uniform_transform`.
    fill_max_steps: int
        Number of hole-filling iterations. Each iteration assigns every empty
        cell that borders a filled cell the mean of its filled face-neighbours
        (4-connected in 2D, 6-connected in 3D), matching the compiled kernel's
        ``fillMaxSteps``. Defaults to 0.

    Returns
    -------
    torch.Tensor
        Resampled data of shape ``[1, C, *out_spatial]`` where ``out_spatial``
        is ``out_shape`` reversed (``(y, x)`` for 2D, ``(z, y, x)`` for 3D).
        Cells that receive no contribution are zero (matching the kernel).
    """
    assert len(data_list) == len(coords_list)
    assert len(data_list) > 0
    dims = len(data_list[0].size()) - 2
    device = data_list[0].device
    dtype = data_list[0].dtype
    out_shape = get_output_shape(out_shape, dims)

    # Cell-centre coordinates per block, and the joint vertex set for the AABB.
    cell_coords_list = []
    vertex_coords_list = []
    for coords in coords_list:
        if is_cell_coords:
            cell_coords_list.append(coords)
        else:
            vertex_coords_list.append(coords)
            cell_coords_list.append(coords_to_center_coords(coords))

    if is_cell_coords:
        vertex_coords = None
    else:
        vertex_coords = torch.cat(
            [_.view(dims, -1) for _ in vertex_coords_list], dim=-1
        )

    mat = get_uniform_transform(
        transform_uniform, vertex_coords, out_shape, dims, dtype
    )
    # mat maps output index -> world; invert to map world -> continuous index.
    inv = torch.inverse(mat[0].to(device=device, dtype=torch.float64))

    # Output spatial shape is out_shape reversed: (x, y[, z]) -> ([z,] y, x).
    out_spatial = [int(out_shape[dims - 1 - d]) for d in range(dims)]
    n_cells = 1
    for s in out_spatial:
        n_cells *= s
    # Strides for a linear index over (x, y[, z]) axes into out_spatial layout.
    axis_stride = [0] * dims  # stride per index axis (x=0, y=1, z=2)
    stride = 1
    for spatial_dim in range(dims):  # innermost (x) first
        axis_stride[spatial_dim] = stride
        stride *= out_spatial[dims - 1 - spatial_dim]

    channels = data_list[0].size(1)
    acc = torch.zeros((channels, n_cells), device=device, dtype=dtype)
    wacc = torch.zeros((n_cells,), device=device, dtype=dtype)

    for data, cell_coords in zip(data_list, cell_coords_list):
        pts = cell_coords.reshape(dims, -1).to(torch.float64)  # [dims, N] world x,y[,z]
        ones = torch.ones((1, pts.size(1)), device=device, dtype=torch.float64)
        gi = inv @ torch.cat([pts, ones], dim=0)  # [dims+1, N] continuous index
        gi = gi[:dims]  # per-axis continuous index (x, y[, z])

        base = torch.floor(gi).to(torch.int64)  # [dims, N]
        frac = gi - base  # [dims, N] in [0, 1)

        vals = data.reshape(channels, -1)  # [C, N]

        # Splat to every corner of the surrounding cell (2**dims corners).
        for corner in range(2**dims):
            idx = torch.zeros_like(base[0])
            weight = torch.ones_like(gi[0])
            valid = torch.ones_like(gi[0], dtype=torch.bool)
            for axis in range(dims):
                offset = (corner >> axis) & 1
                coord = base[axis] + offset
                weight = weight * (frac[axis] if offset else (1.0 - frac[axis]))
                valid = valid & (coord >= 0) & (coord < out_spatial[dims - 1 - axis])
                idx = idx + coord * axis_stride[axis]

            lin = idx[valid]
            w = weight[valid].to(dtype)
            wacc.index_add_(0, lin, w)
            acc.index_add_(1, lin, vals[:, valid] * w)

    written = wacc > 0
    out = torch.zeros_like(acc)
    out[:, written] = acc[:, written] / wacc[written]
    out = out.reshape([channels, *out_spatial])

    if fill_max_steps > 0:
        out = _fill_empty_cells(
            out, written.reshape(out_spatial), fill_max_steps, dims
        )

    return out.reshape([1, channels, *out_spatial])


def _fill_empty_cells(values, filled, max_steps, dims):
    """Iteratively fill empty cells with the mean of their filled face-neighbours.

    Reproduces the hole-filling of ``SampleTransformedGridLocalToGlobalMulti``:
    each step, every empty cell adjacent to a filled cell takes the mean of its
    filled face-neighbours; newly filled cells become sources for later steps.

    Parameters
    ----------
    values: torch.Tensor
        Cell values of shape ``[C, *spatial]`` (zero in empty cells).
    filled: torch.Tensor
        Boolean mask of shape ``spatial`` marking already-filled cells.
    max_steps: int
        Maximum number of fill iterations.
    dims: int
        Spatial dimensionality (2 or 3).

    Returns
    -------
    torch.Tensor
        Values with empty cells filled, same shape as ``values``.
    """
    conv = torch.nn.functional.conv2d if dims == 2 else torch.nn.functional.conv3d
    k = _face_neighbour_kernel(dims, values.dtype, values.device)

    channels = values.size(0)
    filled = filled.to(values.dtype)
    pad = 1
    for _ in range(max_steps):
        num = conv(
            (values * filled).view(channels, 1, *values.shape[1:]), k, padding=pad
        )[:, 0]
        den = conv(filled.view(1, 1, *filled.shape), k, padding=pad)[0, 0]
        newly = (den > 0) & (filled == 0)
        if not bool(newly.any()):
            break
        update = torch.where(newly, num / den.clamp_min(1.0), torch.zeros_like(num))
        values = values + update
        filled = filled + newly.to(values.dtype)

    return values


def _face_neighbour_kernel(dims, dtype, device):
    """Face-neighbour ("cross") convolution kernel: 1 at +/-1 along each axis."""
    k = torch.zeros([3] * dims, dtype=dtype, device=device)
    centre = tuple([1] * dims)
    for axis in range(dims):
        for offset in (0, 2):
            idx = list(centre)
            idx[axis] = offset
            k[tuple(idx)] = 1.0
    return k.view(1, 1, *([3] * dims))


class DiffMultiblockResampler:
    """Differentiable multi-block -> uniform-grid resampler with cached geometry.

    Same operation and same result as :func:`sample_multi_coords_to_uniform_grid`
    (and its reference torch port
    :func:`sample_multi_coords_to_uniform_grid_diff`), but built for use inside a
    BPTT graph:

    1. **Cached geometry.** The splat indices, weights and per-cell normalisation
       depend only on the grid, which never changes over the lifetime of an
       environment. They are computed once here. Every call then reuses the *same*
       tensors, so autograd stores references to one shared copy instead of a
       fresh ~8 * n_source_cells index/weight pair per call. The per-call tape
       growth drops to the accumulator itself.

    2. **Optional output restriction.** Point-sensor observations read a handful
       of cells out of the whole uniform grid. Splat + weight-normalisation +
       hole-filling together form a *fixed linear map* from source cells to
       output cells (all the masks are geometry, not data), so the rows for the
       requested output cells can be extracted once into a tiny operator. Pass
       ``out_indices`` and the call cost drops from the full grid to those rows:
       the result is identical to indexing the full result, but with orders of
       magnitude less work and tape.

    Memory note: building the unrestricted operator holds
    ``2**dims * n_source_cells`` int64 indices plus the same number of weights
    (~1 GB at 5M source cells in fp64, 3D). That is a one-time cost shared by
    every call and every field. In restricted mode those arrays are released
    after construction and only the small operator is kept.

    Parameters
    ----------
    coords_list: Sequence[torch.Tensor]
        Per-block coordinates (NCDHW with C = dims), as passed to
        :func:`sample_multi_coords_to_uniform_grid`.
    out_shape: int | list | tuple | torch.Tensor
        Output grid shape in ``(x, y[, z])`` order.
    dtype: torch.dtype
        Dtype of the data that will be resampled. Weights are stored in it.
    is_cell_coords: bool
        Whether ``coords_list`` holds cell-centre (True) or vertex (False)
        coordinates. Defaults to False.
    transform_uniform: str | torch.Tensor
        World->index transform, see :func:`get_uniform_transform`.
    fill_max_steps: int
        Hole-filling iterations, matching the kernel's ``fillMaxSteps``.
        Defaults to 0.
    device: torch.device | None
        Device to build on. Defaults to the device of ``coords_list[0]``.
    out_indices: torch.Tensor | None
        Output cells to restrict to. Either a 1D tensor of flat indices into the
        flattened output grid (``out_spatial`` layout, i.e. ``(z, y, x)`` in 3D),
        or a ``[dims, R]`` integer tensor of per-axis ``(x, y[, z])`` indices.
        If None (default), the full grid is produced.

    Attributes
    ----------
    out_spatial: list[int]
        Output spatial shape, ``out_shape`` reversed.
    """

    def __init__(
        self,
        coords_list,
        out_shape,
        dtype,
        is_cell_coords=False,
        transform_uniform="AABB_OUTER",
        fill_max_steps=0,
        device=None,
        out_indices=None,
    ):
        assert len(coords_list) > 0
        dims = len(coords_list[0].size()) - 2
        if dims not in (2, 3):
            raise ValueError("Only 2D and 3D resampling is supported.")
        device = coords_list[0].device if device is None else device

        self.dims = dims
        self.dtype = dtype
        self.device = device
        self.fill_max_steps = fill_max_steps

        out_shape = get_output_shape(out_shape, dims)
        self.out_spatial = [int(out_shape[dims - 1 - d]) for d in range(dims)]
        n_cells = 1
        for s in self.out_spatial:
            n_cells *= s
        self.n_cells = n_cells

        # --- geometry: world -> continuous output index -------------------
        cell_coords_list = []
        vertex_coords_list = []
        for coords in coords_list:
            if is_cell_coords:
                cell_coords_list.append(coords)
            else:
                vertex_coords_list.append(coords)
                cell_coords_list.append(coords_to_center_coords(coords))

        if is_cell_coords:
            vertex_coords = None
        else:
            vertex_coords = torch.cat(
                [_.view(dims, -1) for _ in vertex_coords_list], dim=-1
            )

        mat = get_uniform_transform(
            transform_uniform, vertex_coords, out_shape, dims, dtype
        )
        inv = torch.inverse(mat[0].to(device=device, dtype=torch.float64))

        axis_stride = [0] * dims
        stride = 1
        for spatial_dim in range(dims):  # innermost (x) first
            axis_stride[spatial_dim] = stride
            stride *= self.out_spatial[dims - 1 - spatial_dim]

        # Source cells of all blocks are addressed as one concatenated axis, in
        # block order; `block_sizes` is how a data list is flattened at call time.
        self.block_sizes = []
        rows = []  # output cell (linear index into out_spatial)
        cols = []  # source cell (linear index into the concatenated source)
        weights = []
        offset = 0
        for cell_coords in cell_coords_list:
            pts = cell_coords.reshape(dims, -1).to(torch.float64)
            n_src = pts.size(1)
            self.block_sizes.append(n_src)
            ones = torch.ones((1, n_src), device=device, dtype=torch.float64)
            gi = (inv @ torch.cat([pts, ones], dim=0))[:dims]

            base = torch.floor(gi).to(torch.int64)
            frac = gi - base
            src = torch.arange(n_src, device=device, dtype=torch.int64) + offset

            for corner in range(2**dims):
                idx = torch.zeros_like(base[0])
                weight = torch.ones_like(gi[0])
                valid = torch.ones_like(gi[0], dtype=torch.bool)
                for axis in range(dims):
                    corner_offset = (corner >> axis) & 1
                    coord = base[axis] + corner_offset
                    weight = weight * (
                        frac[axis] if corner_offset else (1.0 - frac[axis])
                    )
                    valid = valid & (coord >= 0)
                    valid = valid & (coord < self.out_spatial[dims - 1 - axis])
                    idx = idx + coord * axis_stride[axis]

                rows.append(idx[valid])
                cols.append(src[valid])
                weights.append(weight[valid].to(dtype))

            offset += n_src

        self.n_source = offset
        rows = torch.cat(rows)
        cols = torch.cat(cols)
        weights = torch.cat(weights)

        # --- weight normalisation ------------------------------------------
        wacc = torch.zeros((n_cells,), device=device, dtype=dtype)
        wacc.index_add_(0, rows, weights)
        written = wacc > 0
        # Normalised splat weights: the kernel divides each output cell by its
        # accumulated weight and leaves untouched cells at zero. A row can only
        # have zero accumulated weight if all its contributions are themselves
        # zero (a cell centre landing exactly on a grid line), so dividing by one
        # there reproduces the kernel's zero.
        denom = wacc[rows]
        weights = weights / torch.where(denom > 0, denom, torch.ones_like(denom))

        self._written = written

        # --- hole filling: the masks are data-independent, so cache them ----
        # Each step is  v <- v + conv(v * filled) * scale,  with scale zero
        # outside the newly filled cells. That is exactly the reference update
        # `where(newly, num / den.clamp_min(1), 0)`.
        self._fill_steps = []
        if fill_max_steps > 0:
            conv = (
                torch.nn.functional.conv2d if dims == 2 else torch.nn.functional.conv3d
            )
            kernel = _face_neighbour_kernel(dims, dtype, device)
            filled = written.to(dtype).reshape(self.out_spatial)
            for _ in range(fill_max_steps):
                den = conv(filled.view(1, 1, *filled.shape), kernel, padding=1)[0, 0]
                newly = (den > 0) & (filled == 0)
                if not bool(newly.any()):
                    break
                scale = torch.where(
                    newly, 1.0 / den.clamp_min(1.0), torch.zeros_like(den)
                )
                self._fill_steps.append((filled, newly, scale))
                filled = filled + newly.to(dtype)

        # --- restriction to the requested output cells ----------------------
        self.out_indices = None
        if out_indices is None:
            self._rows = rows
            self._cols = cols
            self._weights = weights
        else:
            flat = self._flatten_out_indices(out_indices)
            self.out_indices = flat
            self._rows, self._cols, self._weights = self._restrict(
                flat, rows, cols, weights
            )
            # The full-grid operator and the fill masks are no longer needed.
            self._fill_steps = []

    @property
    def n_out(self):
        """Number of output values a call produces (per channel)."""
        return self.n_cells if self.out_indices is None else self.out_indices.numel()

    def _flatten_out_indices(self, out_indices):
        """Accept flat indices or per-axis ``(x, y[, z])`` indices."""
        out_indices = torch.as_tensor(out_indices, device=self.device)
        if out_indices.dim() == 2:
            if out_indices.size(0) != self.dims:
                raise ValueError(
                    "Per-axis out_indices must have shape [dims, R], got "
                    f"{tuple(out_indices.size())}."
                )
            flat = torch.zeros_like(out_indices[0], dtype=torch.int64)
            stride = 1
            for spatial_dim in range(self.dims):  # innermost (x) first
                flat = flat + out_indices[spatial_dim].to(torch.int64) * stride
                stride *= self.out_spatial[self.dims - 1 - spatial_dim]
        elif out_indices.dim() == 1:
            flat = out_indices.to(torch.int64)
        else:
            raise ValueError("out_indices must be 1D (flat) or 2D ([dims, R]).")

        if flat.numel() and (
            int(flat.min()) < 0 or int(flat.max()) >= self.n_cells
        ):
            raise ValueError("out_indices out of range for the output grid.")
        return flat

    def _restrict(self, flat, rows, cols, weights):
        """Extract the rows of the full linear operator for the given output cells.

        The operator is ``F @ diag(1/wacc) @ S`` with ``S`` the splat and ``F``
        the hole-filling. ``F`` is a product of ``(I + M_k)``, so the row
        selection is propagated backwards through the fill steps first (which
        only expands support through cells that the fill actually writes, i.e.
        not at all when the grid has no holes), and the resulting output-cell
        combination is then composed with the splat rows.
        """
        n_sel = flat.numel()
        # G: sparse [n_sel, n_cells] as (row, col, value) triples. Starts as the
        # selection, then absorbs each fill step from the last one backwards.
        g_row = torch.arange(n_sel, device=self.device, dtype=torch.int64)
        g_col = flat
        g_val = torch.ones(n_sel, device=self.device, dtype=self.dtype)

        for filled, newly, scale in reversed(self._fill_steps):
            # G <- G (I + M_k). Only entries pointing at a cell this step fills
            # produce anything new; if none do, the step is a no-op.
            hit = newly.reshape(-1)[g_col]
            if not bool(hit.any()):
                continue
            h_row = g_row[hit]
            h_col = g_col[hit]
            h_val = g_val[hit] * scale.reshape(-1)[h_col]

            filled_flat = filled.reshape(-1)
            new_rows = []
            new_cols = []
            new_vals = []
            for axis in range(self.dims):
                # Axis `axis` of the (x, y[, z]) index order lives at
                # out_spatial[dims - 1 - axis]; step along it in the flat layout.
                stride = 1
                for spatial_dim in range(axis):
                    stride *= self.out_spatial[self.dims - 1 - spatial_dim]
                extent = self.out_spatial[self.dims - 1 - axis]
                coord = torch.div(h_col, stride, rounding_mode="floor") % extent
                for step in (-1, 1):
                    in_bounds = (coord + step >= 0) & (coord + step < extent)
                    neighbour = h_col + step * stride
                    ok = in_bounds.clone()
                    ok[in_bounds] &= filled_flat[neighbour[in_bounds]] > 0
                    new_rows.append(h_row[ok])
                    new_cols.append(neighbour[ok])
                    new_vals.append(h_val[ok])

            g_row = torch.cat([g_row] + new_rows)
            g_col = torch.cat([g_col] + new_cols)
            g_val = torch.cat([g_val] + new_vals)

        # Compose G with the splat: for each entry (r, o, g), take row `o` of the
        # normalised splat operator and scale it by `g`.
        order = torch.argsort(rows)
        s_rows = rows[order]
        s_cols = cols[order]
        s_vals = weights[order]
        # CSR-style row pointers over the output cells.
        counts = torch.bincount(s_rows, minlength=self.n_cells)
        row_start = torch.cat(
            [
                torch.zeros(1, device=self.device, dtype=torch.int64),
                torch.cumsum(counts, dim=0),
            ]
        )

        take = counts[g_col]
        total = int(take.sum())
        if total == 0:
            empty_i = torch.zeros(0, device=self.device, dtype=torch.int64)
            empty_v = torch.zeros(0, device=self.device, dtype=self.dtype)
            return empty_i, empty_i.clone(), empty_v

        # Expand each G entry into the `take` splat entries of its output row.
        out_row = torch.repeat_interleave(g_row, take)
        gather_base = torch.repeat_interleave(row_start[g_col], take)
        entry_end = torch.cumsum(take, dim=0)
        within = torch.arange(total, device=self.device, dtype=torch.int64)
        within = within - torch.repeat_interleave(entry_end - take, take)
        pos = gather_base + within

        r_rows = out_row
        r_cols = s_cols[pos]
        r_vals = s_vals[pos] * torch.repeat_interleave(g_val, take)
        return r_rows, r_cols, r_vals

    def __call__(self, data_list):
        """Resample per-block cell data onto the uniform grid, differentiably.

        Parameters
        ----------
        data_list: Sequence[torch.Tensor]
            Per-block cell data, each of shape ``[1, C, *spatial]``, in the same
            block order as the ``coords_list`` this was built from.

        Returns
        -------
        torch.Tensor
            ``[1, C, *out_spatial]`` for the full grid, or ``[1, C, R]`` when
            built with ``out_indices`` (in the order the indices were given).
        """
        if len(data_list) != len(self.block_sizes):
            raise ValueError(
                f"Expected {len(self.block_sizes)} blocks, got {len(data_list)}."
            )
        channels = data_list[0].size(1)
        flat_blocks = []
        for data, n_src in zip(data_list, self.block_sizes):
            block = data.reshape(channels, -1)
            if block.size(1) != n_src:
                raise ValueError(
                    f"Block has {block.size(1)} cells, operator expects {n_src}."
                )
            flat_blocks.append(block)
        vals = torch.cat(flat_blocks, dim=1) if len(flat_blocks) > 1 else flat_blocks[0]

        contrib = vals.index_select(1, self._cols) * self._weights
        out = torch.zeros(
            (channels, self.n_out), device=vals.device, dtype=vals.dtype
        )
        out = out.index_add(1, self._rows, contrib)

        if self.out_indices is not None:
            return out.reshape(1, channels, -1)

        out = out.reshape([channels, *self.out_spatial])
        if not self._fill_steps:
            return out.reshape([1, channels, *self.out_spatial])

        conv = (
            torch.nn.functional.conv2d
            if self.dims == 2
            else torch.nn.functional.conv3d
        )
        kernel = _face_neighbour_kernel(self.dims, out.dtype, out.device)
        for filled, _newly, scale in self._fill_steps:
            num = conv(
                (out * filled).view(channels, 1, *out.shape[1:]), kernel, padding=1
            )[:, 0]
            out = out + num * scale

        return out.reshape([1, channels, *self.out_spatial])
