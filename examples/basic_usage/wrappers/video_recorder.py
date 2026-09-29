from pathlib import Path

import fluidgym
from fluidgym.wrappers import VideoRecorder

env = fluidgym.make(
    "CylinderJet2D-easy-v0",
)

# Now, we record every episode of the environment as gif
env = VideoRecorder(env, filename="cylinder", output_path=Path("videos"))

obs, info = env.reset(seed=42)

# Every step renders a frame
for _ in range(20):
    action = env.sample_action()
    obs, reward, terminated, truncated, info = env.step(action)

# The next reset saves the finished episode, e.g. as videos/cylinder_ep1.gif
obs, info = env.reset()

# Closing the environment saves the current episode as well
env.close()
