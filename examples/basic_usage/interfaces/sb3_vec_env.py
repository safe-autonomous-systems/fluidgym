import numpy as np

import fluidgym
from fluidgym.integration.sb3 import VecFluidEnv
from fluidgym.wrappers import FlattenObservation, VideoRecorder

fluid_env = fluidgym.make(
    "Airfoil3D-easy-v0",
    use_marl=True,
)

# We flatten the observation space to receive a 1D array of observations
fluid_env = FlattenObservation(fluid_env)

# Record every episode as gif, this renders a frame after every step
fluid_env = VideoRecorder(fluid_env, filename="airfoil")

# For the SB3 VecEnv interface, wrap the FluidGym environment. This will give us a
# vectorized environment with a pseudo-enviroment for each agent
env = VecFluidEnv(fluid_env)

obs = env.reset(seed=42)

for i in range(50):
    actions = np.array([env.action_space.sample() for _ in range(env.num_envs)])
    obs, reward, done, info = env.step(actions)
    print(f"Step: {i}; Rewards:", reward.tolist())

    if np.any(done):
        break

# Closing the environment saves the gif, e.g. as airfoil_ep1.gif
env.close()
