ObsExtraction Wrapper
=====================

Standard observations in FluidGym >= 0.0.2 are dictionarys containing multiple arrays,
e.g., velocity field, pressure field, etc. To extract only specific components from the
observation dictionary, we provide the ``ObsExtraction`` wrapper, which allows you to
specify which keys to extract from the observation dictionary.

FluidGym wrappers can be used by importing them from the ``fluidgym.wrappers`` module
and wrapping the environment instance.

Here is a simple example from ``examples/wrappers/obs_extraction.py``:

.. code-block:: python

    import fluidgym
    from fluidgym.wrappers import ObsExtraction

    env = fluidgym.make(
        "CylinderJet2D-easy-v0",
    )

    # This will give you a dict observation space with keys ["velocity", "pressure"]
    print("Original observation space:", env.observation_space)

    # Now, we extract only the "velocity" component from the observation dict
    env = ObsExtraction(env, keys=["velocity"])
    print("New observation space:", env.observation_space)

    obs, info = env.reset(seed=42)

    print("Extracted observation shapes:")
    for key in obs:
        print(f"  {key}: {obs[key].shape}")
