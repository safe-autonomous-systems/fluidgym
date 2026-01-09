FluidGym
========

FluidGym environments natively support SARL and MARL using the same interface.

Here is a simple example for SARL:

.. code-block:: python
    
    import fluidgym
    from fluidgym.integration.gymnasium import GymFluidEnv

    # Create a FluidGym environment
    env = fluidgym.make("CylinderJet2D-easy-v0")

    # All FluidGym environments require seeding for reproducibility
    env.seed(42)

    obs, info = env.reset()

    for i in range(10):
        action = env.sample_action()
        obs, reward, term, trunc, info = env.step(action)
        print(f"Step {i}: Reward = {reward:.4f}")

        env.render()

        # Important: All FluidGym environments only set the
        # truncation flag to True since they do not naturally
        # terminate
        if term or trunc:
            break

    env.save_gif("cylinder.gif")

MARL environments are instances of the ``MultiAgentFluidEnv`` class and provide 
``reset_marl()`` and ``step_marl()`` methods for multi-agent interaction.
Here is a simple example for MARL:

.. code-block:: python

    import fluidgym
    from fluidgym.integration.gymnasium import GymFluidEnv

    # Create a FluidGym environment, now it is a multi-agent environment
    env = fluidgym.make("CylinderJet3D-easy-v0")

    # All FluidGym environments require seeding for reproducibility
    env.seed(42)

    obs, info = env.reset_marl()

    for i in range(10):
        action = env.sample_action()
        
        # The function now returns a tensor of observations and 
        # rewards for all agents. The remaining return values are the same,
        # since all agents share the same termination and truncation flags as
        # well as the info dictionary.
        obs, reward, term, trunc, info = env.step_marl(action)
        print(f"Step {i}: Reward = {reward.detach().cpu().numpy()}")

        env.render()

        # Important: All FluidGym environments only set the
        # truncation flag to True since they do not naturally
        # terminate
        if term or trunc:
            break

    env.save_gif("cylinder.gif")