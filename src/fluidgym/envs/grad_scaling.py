"""Scaling of the adjoint at env-step boundaries, for TD(lambda)-style BPTT.

Every tensor handed from one env step to the next is passed through
:class:`ScaleGrad`: unchanged in the forward, gradient multiplied by ``lambda`` in
backward. Reward ``k`` then reaches the state and action of step ``t`` weighted by
``lambda^(k - t)``. See ``FluidEnv.adjoint_lambda``.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

Accessor = tuple[Callable[[], Any], Callable[[Any], None]]


class ScaleGrad(torch.autograd.Function):
    """Identity in the forward, gradient multiplied by ``scale`` in backward."""

    @staticmethod
    def forward(ctx, tensor, scale):  # type: ignore[override]
        ctx.scale = scale
        # A copy rather than a view: the solver writes into the tensors it is
        # handed, and an in-place write on a view of this output is an error
        return tensor.clone()

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        return grad * ctx.scale, None


def scale_grad(value: Any, scale: float) -> Any:
    """Scale the gradient of every graph-connected tensor in `value`.

    `value` is a tensor or a (nested) mapping of them. Tensors that are not in the
    graph, and anything that is not a tensor, are returned as they are.
    """
    if isinstance(value, Mapping):
        return {key: scale_grad(item, scale) for key, item in value.items()}
    if torch.is_tensor(value) and value.requires_grad:
        return ScaleGrad.apply(value, scale)
    return value


def scale_carry_grad(accessors: Sequence[Accessor], scale: float) -> bool:
    """Rebind every graph-connected carry tensor with its gradient scaled.

    Returns whether anything was rebound, so the caller knows whether the solver
    has to refresh its views of the domain data.
    """
    rebound = False
    for getter, setter in accessors:
        tensor = getter()
        if torch.is_tensor(tensor) and tensor.requires_grad:
            setter(ScaleGrad.apply(tensor, scale))
            rebound = True
    return rebound
