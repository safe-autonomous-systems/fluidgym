FluidGym
========

FluidGym environments natively support SARL and MARL using the same interface.

Here is a simple example for SARL from ``examples/interfaces/fluidgym.py``:

.. code-block:: python
    
    import fluidgym

    # Create a FluidGym environment
    env = fluidgym.make(
        "CylinderJet2D-easy-v0",
    )

    # We need to pass a reset seed to ensure reproducibility
    obs, info = env.reset(seed=42)

    # Now, we can interact with the environment as usual
    for _ in range(50):
        action = env.sample_action()
        obs, reward, terminated, truncated, info = env.step(action)

        # For the gif, we need to render at each step
        env.render()

        # Important: All FluidGym environments only set the
        # truncation flag to True since they do not naturally
        # terminate
        if terminated or truncated:
            break

    # This will save a gif of the rendered environment
    env.save_gif("cylinder.gif")


For MARL, you need to set the ``use_marl=True`` flag when creating the environment.
The ``step()`` and ``reset()`` functions will then return observations and rewards
for all agents in the environment. By default, only the 3D RBC and TCF environments have
MARL activated. Here is an example for MARL:

.. code-block:: python

    import fluidgym
    from fluidgym.integration.gymnasium import GymFluidEnv

    # Create a FluidGym environment, now it is a multi-agent environment
    env = fluidgym.make("CylinderJet3D-easy-v0", use_marl=True)

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

        env.render()

        # Important: All FluidGym environments only set the
        # truncation flag to True since they do not naturally
        # terminate
        if term or trunc:
            break

    env.save_gif("cylinder.gif")