import fluidgym
from fluidgym.wrappers import FlattenObservation

env = fluidgym.make(
    "CylinderJet2D-easy-v0",
)

# This will give you a dict observation space with keys ["velocity", "pressure"]
print("Original observation space:", env.observation_space)

# Now, flatten the observation space to receive a 1D array of observations
env = FlattenObservation(env)
print("Flattened observation space:", env.observation_space)

obs, info = env.reset(seed=42)

print("Flattened observation shape:", obs.shape)
