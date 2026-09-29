FluidGym
========

FluidGym environments natively support SARL and MARL using the same interface.

Here is a simple example for SARL from ``examples/basic_usage/interfaces/fluidgym_env.py``:

.. code-block:: python

    import fluidgym
    from fluidgym.wrappers import VideoRecorder

    # Create a FluidGym environment
    env = fluidgym.make(
        "RBC2D-easy-v0",
    )

    # Record every episode as gif, this renders a frame after every step
    env = VideoRecorder(env, filename="rbc")

    # We need to pass a reset seed to ensure reproducibility
    obs, info = env.reset(seed=42)

    # Now, we can interact with the environment as usual
    for _ in range(50):
        action = env.sample_action()
        obs, reward, terminated, truncated, info = env.step(action)

        # Important: All FluidGym environments only set the
        # truncation flag to True since they do not naturally
        # terminate
        if terminated or truncated:
            break

    # Closing the environment saves the gif, e.g. as rbc_ep1.gif
    env.close()


For MARL, you need to set the ``use_marl=True`` flag when creating the environment.
The ``step()`` and ``reset()`` functions will then return observations and rewards
for all agents in the environment. By default, only the 3D RBC and TCF environments have
MARL activated. Here is an example for MARL:

.. code-block:: python

    import fluidgym
    from fluidgym.wrappers import VideoRecorder

    # Create a FluidGym environment, now it is a multi-agent environment
    env = fluidgym.make("CylinderJet3D-easy-v0", use_marl=True)
    env = VideoRecorder(env, filename="cylinder")

    # All FluidGym environments require seeding for reproducibility
    env.seed(42)

    obs, info = env.reset()

    for i in range(10):
        action = env.sample_action()
        
        # The function now returns a tensor of observations and 
        # rewards for all agents. The remaining return values are the same,
        # since all agents share the same termination and truncation flags as
        # well as the info dictionary.
        obs, reward, term, trunc, info = env.step(action)
        print(f"Step {i}: Reward = {reward.detach().cpu().numpy()}")

        # Important: All FluidGym environments only set the
        # truncation flag to True since they do not naturally
        # terminate
        if term or trunc:
            break

    env.close()