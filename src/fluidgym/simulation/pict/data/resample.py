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
    differentiable=False,
):
    """Resample multiple local grids onto a shared uniform grid.

    Dispatches to a compiled or a differentiable implementation. Both produce
    numerically-matching output (see ``tests/simulation/test_torch_resample.py``);
    the differentiable one keeps gradients flowing w.r.t. ``data_list``.

    Parameters
    ----------
    data_list, coords_list, out_shape, is_cell_coords, transform_uniform, \
    fill_max_steps
        See :func:`sample_multi_coords_to_uniform_grid_nondiff`.
    differentiable: bool
        If True, use the pure-torch autograd-friendly implementation
        (:func:`sample_multi_coords_to_uniform_grid_diff`); otherwise use the
        compiled kernel (:func:`sample_multi_coords_to_uniform_grid_nondiff`).
        Defaults to False.

    Returns
    -------
    torch.Tensor
        Resampled data of shape ``[1, C, *out_spatial]``.
    """
    impl = (
        sample_multi_coords_to_uniform_grid_diff
        if differentiable
        else sample_multi_coords_to_uniform_grid_nondiff
    )
    return impl(
        data_list,
        coords_list,
        out_shape,
        is_cell_coords=is_cell_coords,
        transform_uniform=transform_uniform,
        fill_max_steps=fill_max_steps,
    )


def sample_multi_coords_to_uniform_grid_nondiff(
    data_list,
    coords_list,
    out_shape,
    is_cell_coords=False,
    transform_uniform="AABB_OUTER",
    fill_max_steps=0,
):
    """Resample multiple local grids onto a uniform grid via the compiled kernel.

    Wraps ``PISOtorch.SampleTransformedGridLocalToGlobalMulti``. This is fast but
    breaks the autograd graph; use ``differentiable=True`` on
    :func:`sample_multi_coords_to_uniform_grid` when gradients are needed.
    """
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


def sample_multi_coords_to_uniform_grid_diff(
    data_list,
    coords_list,
    out_shape,
    is_cell_coords=False,
    transform_uniform="AABB_OUTER",
    fill_max_steps=0,
):
    """Differentiable, pure-torch re-implementation of
    :func:`sample_multi_coords_to_uniform_grid_nondiff`.

    The compiled ``PISOtorch.SampleTransformedGridLocalToGlobalMulti`` kernel
    performs a bilinear *splat* (scatter) of every source cell centre onto a
    uniform output grid, accumulating value-weighted contributions and a
    per-cell weight, and finally normalising by that weight. That operation is
    linear in the cell *values* (the geometry -- ``coords_list`` and the
    world->index transform -- is static), so it can be expressed with
    ``index_add`` and stays differentiable w.r.t. ``data_list``. This lets
    gradients flow through resampled observations, which the compiled kernel
    breaks.

    The result matches the compiled kernel (for the ``fillMaxSteps`` values used
    by FluidGym) to floating-point precision; see
    ``tests/simulation/test_torch_resample.py``.

    Parameters
    ----------
    data_list: Sequence[torch.Tensor]
        Per-block cell data, each of shape ``[1, C, *spatial]`` (NCDHW / NCHW).
    coords_list: Sequence[torch.Tensor]
        Per-block coordinates, matching ``sample_multi_coords_to_uniform_grid``.
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
        ``fillMaxSteps``. Filled cells keep a weight of zero. Defaults to 0.

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

    for data, cell_coords in zip(data_list, cell_coords_list, strict=True):
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
    # Face-neighbour ("cross") kernel: 1 at +/-1 along each axis, 0 in the centre.
    k = torch.zeros([3] * dims, dtype=values.dtype, device=values.device)
    centre = tuple([1] * dims)
    for axis in range(dims):
        for offset in (0, 2):
            idx = list(centre)
            idx[axis] = offset
            k[tuple(idx)] = 1.0
    k = k.view(1, 1, *([3] * dims))

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
