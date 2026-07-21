"""Utility modules for FluidGym environments."""

from fluidgym.envs.util.diff_tools import (
    get_flat_state,
    mark_state_differentiable,
    state_vjp,
)

__all__ = [
    "get_flat_state",
    "mark_state_differentiable",
    "state_vjp",
]
