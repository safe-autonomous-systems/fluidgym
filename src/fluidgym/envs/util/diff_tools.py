"""Utility functions for differentiating the full state of an environment."""

import torch

from fluidgym.envs import FluidEnv


def _get_state_tensors(env: FluidEnv) -> list[torch.Tensor]:
    """Collects the differentiable state tensors of every block.

    The state of a block is its velocity field, plus its passive scalar field if
    the domain transports a passive scalar.
    """
    has_passive_scalar = env._domain.hasPassiveScalar()

    tensors = []
    for block in env._domain.getBlocks():
        tensors.append(block.velocity)
        if has_passive_scalar:
            tensors.append(block.passiveScalar)

    return tensors


def get_flat_state(env: FluidEnv) -> torch.Tensor:
    """Returns the differentiable state of the environment as one flat vector.

    The state of a block is its velocity field, plus its passive scalar field if
    the domain transports a passive scalar. Every block's tensors are flattened
    and concatenated into a single vector, block by block.

    Parameters
    ----------
    env: FluidEnv
        The fluid environment.

    Returns
    -------
        A tensor of shape [sum(t.numel() for t in the state tensors)].
    """
    return torch.cat([tensor.reshape(-1) for tensor in _get_state_tensors(env)])


def mark_state_differentiable(env: FluidEnv) -> list[torch.Tensor]:
    """Marks the state tensors of the environment as gradient leaves.

    Call this before stepping the environment; the returned list is the `inputs`
    argument of `state_vjp`.

    Parameters
    ----------
    env: FluidEnv
        The fluid environment, created with `differentiable=True`.

    Returns
    -------
        The state tensors, now requiring gradients.
    """
    return [tensor.requires_grad_(True) for tensor in _get_state_tensors(env)]
