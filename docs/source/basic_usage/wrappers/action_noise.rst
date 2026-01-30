ActionNoise Wrapper
===================

To learn and evaluate control policies to noisy actuators, FluidGym provides the
``ActionNoise`` wrapper, which adds noise to the actions taken by the agent. This is
particularly useful for simulating real-world scenarios where actuators may not perform
perfectly.

FluidGym wrappers can be used by importing them from the ``fluidgym.wrappers`` module
and wrapping the environment instance.

Here is a simple example from ``examples/wrappers/action_noise.py``:

.. code-block:: python

    import fluidgym
    from fluidgym.wrappers import ActionNoise

    env = fluidgym.make(
        "CylinderJet2D-easy-v0",
    )

    # Now, we add action noise to the environment's actions
    env = ActionNoise(env, sigma=0.1, seed=42)

    obs, info = env.reset(seed=42)

    action = env.sample_action()

    # Now, if we take a step, the action will have noise added to it
    obs, reward, terminated, truncated, info = env.step(action)

