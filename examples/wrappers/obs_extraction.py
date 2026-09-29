import fluidgym
from fluidgym.wrappers import ObsExtraction

env = fluidgym.make(
    "CylinderJet2D-easy-v0",
)

# This will give you a dict observation space with keys ["velocity", "pressure"]
print("Original observation space:", env.observation_space)

# Now, we extract only the "velocity" component from the observation dict
env = ObsExtraction(env, keys=["velocity"])
print("New observation space:", env.observation_space)

obs, info = env.reset(seed=42)

print("Extracted observation shapes:")
for key in obs:
    print(f"  {key}: {obs[key].shape}")
