Pettingzoo
==========

For mulit-Agent RL, FluidGym environments are compatible with the Pettingzoo interface,
allowing seamless integration with various RL libraries that support Pettingzoo.

To use a FluidGym environment with Pettingzoo, you can create an instance of the desired
environment using the ``make`` function from FluidGym and the ``PettingZooFluidEnv`` wrapper.

Here is a simple example from ``examples/interfaces/pettingzoo_env.py``:

.. code-block:: python
    
    import fluidgym
    from fluidgym.envs.multi_agent_fluid_env import MultiAgentFluidEnv
    from fluidgym.integration.pettingzoo import PettingZooFluidEnv

    # Create a FluidGym environment
    env = fluidgym.make("CylinderJet3D-easy-v0")
    assert isinstance(env, MultiAgentFluidEnv)
    env.seed(42)

    # Wrap the FluidGym environment with the Pettingzoo wrapper
    pz_env = PettingZooFluidEnv(env)

    obs = pz_env.reset()

    for i in range(10):
        action = {i: space.sample() for i, space in pz_env.action_spaces.items()}
        obs, reward, term, trunc, info = pz_env.step(action)
        for agent, reward in reward.items():
            print(f"Agent {agent} Step {i}: Reward = {reward:.4f}")

        pz_env.render()

        # Important: All FluidGym environments only set the
        # truncation flag to True since they do not naturally
        # terminate
        if term or trunc:
            break