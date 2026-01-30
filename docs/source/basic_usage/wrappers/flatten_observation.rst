FlattenObservation Wrapper
==========================

Standard observations in FluidGym >= 0.0.2 are dictionarys containing multiple arrays,
e.g., velocity field, pressure field, etc. To enable compatibility with interfaces that
expect flat observation spaces (e.g., Gymnasium, Stable-Baselines3, etc.), we provide the
``FlattenObservation`` wrapper, which flattens the observation dictionary into a single
1D array.

FluidGym wrappers can be used by importing them from the ``fluidgym.wrappers`` module
and wrapping the environment instance.

Here is a simple example from ``examples/wrappers/flatten_observation.py``:

.. code-block:: python

    import fluidgym
    from fluidgym.wrappers import FlattenObservation

    env = fluidgym.make(
        "CylinderJet2D-easy-v0",
    )

    # This will give you a dict observation space with keys ["velocity", "pressure"]
    print("Original observation space:", env.observation_space)

    # Now, flatten the observation space to receive a 1D array of observations
    env = FlattenObservation(env)
    print("Flattened observation space:", env.observation_space)

    obs, info = env.reset(seed=42)

    print("Flattened observation shape:", obs.shape)
