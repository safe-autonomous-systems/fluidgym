Stable-Baselines3
=================

In order to use Stable-Baselines3 in the MARL setting, we can leverage the VecEnv
interface. FluidGym environments are compatible with the SB3 VecEnv interface, allowing
seamless integration with various reinforcement learning libraries that support SB3.
Here is a simple example from ``examples/interfaces/sb3_vec_env.py``:

.. code-block:: python

    import numpy as np

    import fluidgym
    from fluidgym.wrappers import FlattenObservation
    from fluidgym.integration.sb3 import VecFluidEnv

    fluid_env = fluidgym.make(
        "Airfoil3D-easy-v0",
        use_marl=True,
    )

    # We flatten the observation space to receive a 1D array of observations
    fluid_env = FlattenObservation(fluid_env)

    # For the SB3 VecEnv interface, wrap the FluidGym environment. This will give us a
    # vectorized environment with a pseudo-enviroment for each agent
    env = VecFluidEnv(fluid_env)

    obs = env.reset(seed=42)

    for i in range(50):
        actions = np.array([env.action_space.sample() for _ in range(env.num_envs)])
        obs, reward, done, info = env.step(actions)
        print(f"Step: {i}; Rewards:", reward.tolist())

        env.render()

        if np.any(done):
            break

    env.save_gif("cylinder.gif")

