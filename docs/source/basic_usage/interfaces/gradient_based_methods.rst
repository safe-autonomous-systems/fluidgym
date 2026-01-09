Gradient-Based Methods
======================

.. warning::

    If you encounter issues, please report them on our GitHub 
    issues page.

In addition to classic RL interfaces, FluidGym environments also support
gradient-based methods for policy optimization. This allows users to leverage
techniques DPC to be applied directly within FluidGym environments.

Here is a simple example from ``examples/interfaces/gradient_based_methods.py``:

.. code-block:: python
    
    import fluidgym

    env = fluidgym.make(
        "JetCylinder2D-easy-v0",
        differentiable=True,
    )

    env.init()

    obs, info = env.reset(seed=42)

    for _ in range(50):
        action = env.sample_action()
        action.requires_grad_(True)

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset(seed=42)

        reward.backward()
        print("Action gradients:", action.grad)

        env.render()
    env.save_gif("cylinder.gif")

