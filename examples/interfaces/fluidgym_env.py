import fluidgym

# Create a FluidGym environment
env = fluidgym.make(
    "CylinderJet2D-easy-v0",
)

# We need to pass a reset seed to ensure reproducibility
obs, info = env.reset(seed=42)

# Now, we can interact with the environment as usual
for _ in range(50):
    action = env.sample_action()
    obs, reward, terminated, truncated, info = env.step(action)

    # For the gif, we need to render at each step
    env.render()

    # Important: All FluidGym environments only set the
    # truncation flag to True since they do not naturally
    # terminate
    if terminated or truncated:
        break

# This will save a gif of the rendered environment
env.save_gif("cylinder.gif")
