import fluidgym
from fluidgym.wrappers import SensorNoise

env = fluidgym.make(
    "CylinderJet2D-easy-v0",
)

# Now, we add action noise to the environment's actions
env = SensorNoise(env, sigma=0.1, seed=42)

obs, info = env.reset(seed=42)

action = env.sample_action()

# Now, if we take a step, the observation will have noise added to it
obs, reward, terminated, truncated, info = env.step(action)
