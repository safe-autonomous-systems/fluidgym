import numpy as np

import fluidgym
from fluidgym.wrappers import FlattenObservation
from fluidgym.integration.sb3 import VecEnv

fluid_env = fluidgym.make(
    "CylinderJet3D-easy-v0",
    use_marl=True,
)
fluid_env = FlattenObservation(fluid_env)

env = VecEnv(fluid_env)

obs = env.reset(seed=42)

for i in range(50):
    actions = np.array([env.action_space.sample() for _ in range(env.num_envs)])
    obs, reward, done, info = env.step(actions)
    print(f"Step: {i}; Rewards:", reward.tolist())

    env.render()

    if np.any(done):
        break

env.save_gif("cylinder.gif")
