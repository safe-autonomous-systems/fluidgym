Running FluidGym on multiple GPUs in parallel
=============================================

.. warning::

    Experimental feature. If you encounter issues, please report them on our GitHub 
    issues page.

FluidGym supports running experiments on multiple GPUs in parallel using the
`ParallelFluidEnv` class to speed up training of RL agents.

.. code-block:: python

    from fluidgym.envs.parallel_env import ParallelFluidEnv

    env = ParallelFluidEnv(
        env_id="CylinderJet3D-easy-v0",
        cuda_ids=[0, 1],  # List of GPU IDs to use
    )

However, as of now, the ``ParallelFluidEnv`` class is not directly integrated with the
general ``FluidEnv`` and ``MultiAgentFluidEnv`` classes. Therefore, not all herited functions
and properties are available when using ``ParallelFluidEnv``. We plan to address this
in future releases.