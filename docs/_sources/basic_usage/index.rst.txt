Basic Usage
===========

This section covers everything you need to get started with FluidGym — from choosing
an environment to running your first training loop.

FluidGym provide GPU-accelerated fluid simulations in a Gymnasium-like
interface. Each environment simulates a distinct flow control problem and exposes
structured observations (velocity fields, pressure, sensor readings) and continuous
action spaces for actuator control.

**Quick start** — create an environment, interact with it, and save a visualisation:

.. code-block:: python

    import fluidgym

    env = fluidgym.make("CylinderJet2D-easy-v0")
    obs, info = env.reset(seed=42)

    for _ in range(50):
        action = env.sample_action()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            break

    env.save_gif("cylinder.gif")

Browse the sections below to learn about the available environments, how to plug
FluidGym into common RL frameworks, the observation wrappers, and rendering options.

.. toctree::
   :hidden:
   :maxdepth: 2

   environments
   interfaces/index
   wrappers/index
   rendering
