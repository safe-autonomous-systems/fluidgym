VideoRecorder Wrapper
=====================

To inspect what a policy does to the flow, FluidGym provides the
``VideoRecorder`` wrapper, which records every episode of the environment as a
GIF. It renders a frame after every ``reset()`` and ``step()``, keeps the frames
itself and saves the episode when the next ``reset()`` begins or the environment
is closed with ``close()``.

The GIFs are named ``<filename>_ep<k>.gif`` and are written to ``output_path``
(the current directory by default, created if it does not exist). They show the
frame returned by :meth:`render`, i.e. the default render slice of the
environment. ``render_kwargs`` are passed to every :meth:`render` call, e.g.
``{"render_3d": True}`` for detailed 3D renderings. For a vectorized
environment, including a ``ParallelFluidEnv``, ``env_ids`` selects which
environments to record, each into its own GIF (``<filename>_env<i>_ep<k>.gif``).
By default, all of them are recorded.

Since ``VideoRecorder`` is a FluidGym wrapper, apply it before wrapping the
environment for Gymnasium, Stable-Baselines3 or PettingZoo, e.g.
``GymFluidEnv(VideoRecorder(env, filename="cylinder"))``. Their ``reset()`` and
``close()`` then save the GIFs.

FluidGym wrappers can be used by importing them from the ``fluidgym.wrappers`` module
and wrapping the environment instance.

Here is a simple example from ``examples/basic_usage/wrappers/video_recorder.py``:

.. code-block:: python

    from pathlib import Path

    import fluidgym
    from fluidgym.wrappers import VideoRecorder

    env = fluidgym.make(
        "CylinderJet2D-easy-v0",
    )

    # Now, we record every episode of the environment as gif
    env = VideoRecorder(env, filename="cylinder", output_path=Path("videos"))

    obs, info = env.reset(seed=42)

    # Every step renders a frame
    for _ in range(20):
        action = env.sample_action()
        obs, reward, terminated, truncated, info = env.step(action)

    # The next reset saves the finished episode, e.g. as videos/cylinder_ep1.gif
    obs, info = env.reset()

    # Closing the environment saves the current episode as well
    env.close()
