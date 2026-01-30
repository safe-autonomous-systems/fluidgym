import fluidgym

# Create a FluidGym environment
env = fluidgym.make(
    "CylinderJet2D-easy-v0",
    differentiable=True,  # This flag enables backpropagation through the environment
)
obs, info = env.reset(seed=42)

for _ in range(50):
    action = env.sample_action()

    # Enable gradient tracking for the action
    action.requires_grad_(True)

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset(seed=42)

    reward.backward()
    print("Action gradients:", action.grad)

    # To detach the environment from the computation graph, use:
    env.detach()

    env.render()
env.save_gif("cylinder.gif")
