import torch

import fluidgym

# Create a FluidGym environment. With differentiable=True the observation
# returned by env.step() keeps a gradient graph back to the velocity field, so we
# can differentiate observations w.r.t. the flow state.
env = fluidgym.make(
    "CylinderJet2D-easy-v0",
    differentiable=True,  # This flag enables backpropagation through the environment
)
env.reset(seed=42)

action = env.sample_action()

# The differentiable "state" of the fluid is the per-block velocity field. Mark
# the velocity components of every block as leaves we want gradients for.
velocity = [block.velocity.requires_grad_(True) for block in env._domain.getBlocks()]

# Step the simulation. new_obs["velocity"] has shape [n_sensors, 2] and is now a
# differentiable function of the velocity components marked above.
new_obs, reward, terminated, truncated, info = env.step(action)
obs = new_obs["velocity"]

# Compute the Jacobian d(obs) / d(velocity) for the first block, one output row
# at a time. Row i holds the gradient of obs.flatten()[i] w.r.t. the (flattened)
# velocity field of block 0.
outputs = obs.reshape(-1)
jacobian = torch.stack(
    [
        torch.autograd.grad(output, velocity[0], retain_graph=True)[0].reshape(-1)
        for output in outputs
    ]
)  # shape: [n_sensors * 2, 2 * H * W]

print("Observation shape    :", tuple(obs.shape))
print("Velocity field shape :", tuple(velocity[0].shape))
print("Jacobian shape       :", tuple(jacobian.shape))

# For many sensors, materializing the full Jacobian is expensive. To get the
# gradient of a single scalar readout instead (a vector-Jacobian product), simply
# backpropagate from that scalar:
#   torch.autograd.grad(obs[0, 0], velocity)
# or use torch.autograd.functional.jacobian for a batched computation.

# Detach the environment from the computation graph before the next step to avoid
# accumulating the graph across steps:
env.detach()
