Pettingzoo
==========

For mulit-Agent RL, FluidGym environments are compatible with the Pettingzoo interface,
allowing seamless integration with various RL libraries that support Pettingzoo.

To use a FluidGym environment with Pettingzoo, you can create an instance of the desired
environment using the ``make`` function from FluidGym and the ``PettingZooFluidEnv`` wrapper.

Here is a simple example from ``examples/interfaces/pettingzoo.py``:

.. code-block:: python
    
    import fluidgym
    from fluidgym.wrappers import FlattenObservation
    from fluidgym.integration.pettingzoo import PettingZooFluidEnv

    # Create a FluidGym environment, it has to be a multi-agent environment
    env = fluidgym.make("CylinderJet3D-easy-v0", use_marl=True)

    # We flatten the observation space to receive a 1D array of observations
    env = FlattenObservation(env)

    env.seed(42)

    # For the PettingZoo interface, wrap the FluidGym environment
    pz_env = PettingZooFluidEnv(env)

    obs = pz_env.reset()

    for i in range(10):
        action = {i: space.sample() for i, space in pz_env.action_spaces.items()}
        obs, reward, term, trunc, info = pz_env.step(action)
        print(term, trunc)
        for agent, reward in reward.items():
            print(f"Agent {agent} Step {i}: Reward = {reward:.4f}")

        pz_env.render()

        # Important: All FluidGym environments only set the
        # truncation flag to True since they do not naturally
        # terminate. For MARL, all we can do is check the first agent
        # since all agents share the same term and trunc flags.
        if term[0] or trunc[0]:
            break
