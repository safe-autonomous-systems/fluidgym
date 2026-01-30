import fluidgym
from fluidgym.wrappers import FlattenObservation
from fluidgym.integration.gymnasium import GymFluidEnv

# Create a FluidGym environment
fluid_env = fluidgym.make(
    "CylinderJet2D-easy-v0",
)

# First, we flatten the observation space to receive a 1D array of observations
fluid_env = FlattenObservation(fluid_env)

# First, we flatten the observation space to receive a 1D array of observations.
# This is not required, you can also use gymnasium.wrappers after wrapping the
# the fluidgym environment
env = GymFluidEnv(fluid_env)

obs, info = env.reset(seed=42)

for i in range(10):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    print(f"Step {i}: Reward = {reward:.4f}")

    env.render()

    # Important: All FluidGym environments only set the
    # truncation flag to True since they do not naturally
    # terminate
    if term or trunc:
        break

env.save_gif("cylinder.gif")
