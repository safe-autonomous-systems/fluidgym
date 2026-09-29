Gradient-Based Methods
======================

.. warning::

    If you encounter issues, please report them on our GitHub 
    issues page.

In addition to classic RL interfaces, FluidGym environments also support
gradient-based methods for policy optimization. This allows users to leverage
techniques DPC to be applied directly within FluidGym environments.

Here is a simple example from ``examples/basic_usage/interfaces/gradient_based_methods.py``:

.. code-block:: python

    import fluidgym
    from fluidgym.wrappers import VideoRecorder

    # Create a FluidGym environment
    env = fluidgym.make(
        "CylinderJet2D-easy-v0",
        differentiable=True,  # This flag enables backpropagation through the environment
    )
    # Record every episode as gif, this renders a frame after every step
    env = VideoRecorder(env, filename="cylinder")
    obs, info = env.reset(seed=42)

    for _ in range(50):
        action = env.sample_action()

        # Enable gradient tracking for the action
        action.requires_grad_(True)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset(seed=42)

        reward.backward()
        print("Action gradients:", action.grad)

        # To detach the environment from the computation graph, use:
        env.detach()

    # Closing the environment saves the gif of the last episode
    env.close()
