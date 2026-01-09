Stable-Baselines3
=================

Here is a simple example from ``examples/interfaces/sb3_marl_env.py``:

.. code-block:: python

    import numpy as np

    import fluidgym
    from fluidgym.envs.multi_agent_fluid_env import MultiAgentFluidEnv
    from fluidgym.integration.sb3 import MultiAgentVecEnv

    fluid_env = fluidgym.make(
        "CylinderJet3D-easy-v0",
    )
    assert isinstance(fluid_env, MultiAgentFluidEnv)
    env = MultiAgentVecEnv(fluid_env)

    obs, info = env.reset(seed=42)

    for i in range(50):
        actions = np.array([env.action_space.sample() for _ in range(env.num_envs)])
        obs, reward, done, info = env.step(actions)
        print(f"Step {i}: Reward = {reward:.4f}")

        env.render()

        if np.any(done):
            break

    env.save_gif("cylinder.gif")
