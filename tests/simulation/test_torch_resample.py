"""Tests for the differentiable pure-torch multi-block resampling.

``sample_multi_coords_to_uniform_grid_diff`` reimplements the compiled
``PISOtorch.SampleTransformedGridLocalToGlobalMulti`` kernel so that gradients
flow through resampled observations. These tests pin it against the kernel on
the (genuinely multi-block) cylinder environment and check differentiability.
"""

import pytest
import torch

import fluidgym
from fluidgym.simulation.extensions import PISOtorch
from fluidgym.simulation.pict.data.resample import (
    get_uniform_transform,
    sample_multi_coords_to_uniform_grid,
    sample_multi_coords_to_uniform_grid_diff,
)
from fluidgym.simulation.pict.data.shapes import coords_to_center_coords

ENV_ID = "CylinderJet2D-easy-v0"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="The compiled PISOtorch resampling kernel requires CUDA.",
)


@pytest.fixture(scope="module")
def cylinder_env():
    env = fluidgym.make(ENV_ID, differentiable=True)
    env.reset(seed=42)
    yield env


def _kernel_resample(env, data_list, fill_max_steps):
    """Reference resampling via the compiled kernel, mirroring the obs pipeline."""
    sim = env._sim
    dims = env._ndims
    vertex_coords = sim.output_resampling_coords
    out_shape = torch.tensor(sim.output_resampling_shape, dtype=torch.int32)

    cell_coords = [coords_to_center_coords(v) for v in vertex_coords]
    all_vertices = torch.cat([v.view(dims, -1) for v in vertex_coords], dim=-1)
    mat = get_uniform_transform(
        "AABB_OUTER", all_vertices, out_shape, dims, dtype=vertex_coords[0].dtype
    )
    out, _ = PISOtorch.SampleTransformedGridLocalToGlobalMulti(
        data_list, cell_coords, mat, out_shape, fillMaxSteps=fill_max_steps
    )
    return out


def test_cylinder_is_multi_block(cylinder_env):
    # The whole point of the "Multi" kernel: several curvilinear blocks resampled
    # onto one uniform grid. Guard the assumption the other tests rely on.
    assert len(cylinder_env._domain.getBlocks()) > 1


@pytest.mark.parametrize("fill_max_steps", [0, 16])
def test_matches_compiled_kernel(cylinder_env, fill_max_steps):
    env = cylinder_env
    data_list = [b.velocity.detach().clone() for b in env._domain.getBlocks()]

    reference = _kernel_resample(env, data_list, fill_max_steps)
    result = sample_multi_coords_to_uniform_grid_diff(
        data_list,
        env._sim.output_resampling_coords,
        list(env._sim.output_resampling_shape),
        is_cell_coords=False,
        fill_max_steps=fill_max_steps,
    )

    assert result.shape == reference.shape
    # float32 splat + fill accumulation: agreement is at rounding level.
    torch.testing.assert_close(result, reference, atol=1e-3, rtol=1e-3)


def test_matches_observation_pipeline(cylinder_env):
    # The velocity observation is exactly this resample at the pipeline's fill.
    env = cylinder_env
    data_list = [b.velocity.detach().clone() for b in env._domain.getBlocks()]

    reference = _kernel_resample(
        env, data_list, env._sim.output_resampling_fill_max_steps
    )
    result = sample_multi_coords_to_uniform_grid_diff(
        data_list,
        env._sim.output_resampling_coords,
        list(env._sim.output_resampling_shape),
        fill_max_steps=env._sim.output_resampling_fill_max_steps,
    )
    torch.testing.assert_close(result, reference, atol=1e-3, rtol=1e-3)


def test_dispatch_matches_and_controls_grad(cylinder_env):
    # The public entry point routes to the compiled or torch implementation and
    # only the differentiable route keeps a graph.
    env = cylinder_env
    coords = env._sim.output_resampling_coords
    out_shape = list(env._sim.output_resampling_shape)
    fill = env._sim.output_resampling_fill_max_steps
    data_list = [b.velocity.detach().clone() for b in env._domain.getBlocks()]

    nondiff = sample_multi_coords_to_uniform_grid(
        data_list, coords, out_shape, fill_max_steps=fill, differentiable=False
    )
    diff = sample_multi_coords_to_uniform_grid(
        data_list, coords, out_shape, fill_max_steps=fill, differentiable=True
    )
    torch.testing.assert_close(diff, nondiff, atol=1e-3, rtol=1e-3)

    leaves = [d.clone().requires_grad_(True) for d in data_list]
    assert sample_multi_coords_to_uniform_grid(
        leaves, coords, out_shape, fill_max_steps=fill, differentiable=True
    ).grad_fn is not None


def test_is_differentiable(cylinder_env):
    # The compiled kernel detaches; the torch version must keep a graph so that
    # gradients of a resampled observation w.r.t. the velocity field exist.
    env = cylinder_env
    leaves = [b.velocity.detach().clone().requires_grad_(True)
              for b in env._domain.getBlocks()]

    out = sample_multi_coords_to_uniform_grid_diff(
        leaves,
        env._sim.output_resampling_coords,
        list(env._sim.output_resampling_shape),
        fill_max_steps=env._sim.output_resampling_fill_max_steps,
    )
    assert out.grad_fn is not None

    grads = torch.autograd.grad(out.sum(), leaves)
    # Every block feeds the uniform grid, so each must receive gradient.
    for g in grads:
        assert g is not None
        assert torch.count_nonzero(g) > 0


def test_gradient_matches_finite_difference(cylinder_env):
    # Bilinear splat + neighbour-mean fill are linear in the cell values, so the
    # analytic gradient of a scalar readout must match a finite-difference probe.
    env = cylinder_env
    coords = env._sim.output_resampling_coords
    out_shape = list(env._sim.output_resampling_shape)
    fill = env._sim.output_resampling_fill_max_steps

    base = [b.velocity.detach().clone() for b in env._domain.getBlocks()]
    leaves = [d.clone().requires_grad_(True) for d in base]

    # A fixed random linear readout of the resampled grid.
    out = sample_multi_coords_to_uniform_grid_diff(leaves, coords, out_shape,
                                                    fill_max_steps=fill)
    torch.manual_seed(0)
    w = torch.randn_like(out)
    loss = (out * w).sum()
    (analytic,) = torch.autograd.grad(loss, leaves[0])

    # Finite difference on a handful of entries of the first block.
    flat = base[0].reshape(-1)
    probe = torch.linspace(0, flat.numel() - 1, steps=5).long()
    eps = 1e-2

    def readout(block0):
        data = [block0] + base[1:]
        o = sample_multi_coords_to_uniform_grid_diff(data, coords, out_shape,
                                                      fill_max_steps=fill)
        return (o * w).sum()

    for p in probe:
        d = base[0].clone().reshape(-1)
        d[p] += eps
        plus = readout(d.reshape(base[0].shape))
        d[p] -= 2 * eps
        minus = readout(d.reshape(base[0].shape))
        fd = ((plus - minus) / (2 * eps)).item()
        torch.testing.assert_close(
            analytic.reshape(-1)[p].item(), fd, atol=1e-2, rtol=1e-2
        )
