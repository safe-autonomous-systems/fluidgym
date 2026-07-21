import torch

import fluidgym
from fluidgym.envs.util.diff_tools import (
    get_flat_state,
    mark_state_differentiable,
)

env = fluidgym.make(
    "CylinderJet2D-easy-v0",
    differentiable=True,  # This flag enables backpropagation through the environment
)
env.reset(seed=42)

action = env.sample_action()

# The differentiable state of the fluid is the per-block velocity field, plus the
# per-block passive scalar field for domains that transport one. Mark all of them
# as leaves we want gradients for.
inputs = mark_state_differentiable(env)

env.step(action)

outputs = get_flat_state(env)
cotangent = torch.ones_like(outputs)

grad = torch.autograd.grad(
    outputs,
    inputs,
    grad_outputs=cotangent,
    retain_graph=True,
    create_graph=False,
    allow_unused=True,
    materialize_grads=True,
)[0]

print("State dim :", outputs.numel())
print("VJP shape :", tuple(grad.shape))  # [input_dim], one backward pass

# Detach the env as soon as a new horizon is entered
env.detach()
