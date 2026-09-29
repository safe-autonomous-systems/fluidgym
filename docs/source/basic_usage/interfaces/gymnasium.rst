Gymnasium
=========

FluidGym environments are compatible with the Gymnasium interface, allowing seamless 
integration with various reinforcement learning libraries that support Gymnasium.

To use a FluidGym environment with Gymnasium, you can create an instance of the desired
environment using the ``make`` function from FluidGym and the ``GymFluidEnv`` wrapper.

Here is a simple example from ``examples/basic_usage/interfaces/gymnasium_env.py``:

.. code-block:: python

    import fluidgym
    from fluidgym.integration.gymnasium import GymFluidEnv
    from fluidgym.wrappers import FlattenObservation, VideoRecorder

    # Create a FluidGym environment
    fluid_env = fluidgym.make(
        "CylinderJet2D-easy-v0",
    )

    # First, we flatten the observation space to receive a 1D array of observations
    fluid_env = FlattenObservation(fluid_env)

    # Record every episode as gif, this renders a frame after every step
    fluid_env = VideoRecorder(fluid_env, filename="cylinder")

    # First, we flatten the observation space to receive a 1D array of observations.
    # This is not required, you can also use gymnasium.wrappers after wrapping the
    # the fluidgym environment
    env = GymFluidEnv(fluid_env)

    obs, info = env.reset(seed=42)

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"Step {i}: Reward = {reward:.4f}")

        # Important: All FluidGym environments only set the
        # truncation flag to True since they do not naturally
        # terminate
        if term or trunc:
            break

    # Closing the environment saves the gif, e.g. as cylinder_ep1.gif
    env.close()
