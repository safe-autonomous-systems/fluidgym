import fluidgym
from fluidgym.wrappers import PowerPenalty

env = fluidgym.make(
    "CylinderJet2D-easy-v0",
)

# Now, we charge the reward for the power the actuation draws
env = PowerPenalty(env, penalty=0.1)

obs, info = env.reset(seed=42)

action = env.sample_action()

# Now, if we take a step, the reward is reduced by 0.1 * ||action||^2
obs, reward, terminated, truncated, info = env.step(action)
