Gymnasium
=========

FluidGym environments are compatible with the Gymnasium interface, allowing seamless 
integration with various reinforcement learning libraries that support Gymnasium.

To use a FluidGym environment with Gymnasium, you can create an instance of the desired
environment using the ``make`` function from FluidGym and the ``GymFluidEnv`` wrapper.

Here is a simple example from ``examples/interfaces/gymnasium_env.py``:

.. code-block:: python

    import fluidgym
    from fluidgym.integration.gymnasium import GymFluidEnv

    # Create a FluidGym environment
    env = fluidgym.make("CylinderJet2D-easy-v0")

    # All FluidGym environments require seeding for reproducibility
    env.seed(42)

    # Wrap the FluidGym environment with the Gymnasium wrapper
    gym_env = GymFluidEnv(env)

    obs, info = env.reset()

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = gym_env.step(action)
        print(f"Step {i}: Reward = {reward:.4f}")

        gym_env.render()

        # Important: All FluidGym environments only set the
        # truncation flag to True since they do not naturally
        # terminate
        if term or trunc:
            break

    gym_env.save_gif("cylinder.gif")