import fluidgym
from fluidgym.wrappers import VideoRecorder

# Create a FluidGym environment
env = fluidgym.make(
    "RBC2D-easy-v0",
)

# Record every episode as gif, this renders a frame after every step
env = VideoRecorder(env, filename="rbc")

# We need to pass a reset seed to ensure reproducibility
obs, info = env.reset(seed=42)

# Now, we can interact with the environment as usual
for _ in range(50):
    action = env.sample_action()
    obs, reward, terminated, truncated, info = env.step(action)

    # Important: All FluidGym environments only set the
    # truncation flag to True since they do not naturally
    # terminate
    if terminated or truncated:
        break

# Closing the environment saves the gif, e.g. as rbc_ep1.gif
env.close()
