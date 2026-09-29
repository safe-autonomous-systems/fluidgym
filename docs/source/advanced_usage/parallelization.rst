Running FluidGym on multiple GPUs in parallel
=============================================

.. warning::

    Experimental feature. If you encounter issues, please report them on our GitHub 
    issues page.

FluidGym supports running experiments on multiple GPUs in parallel using the
`ParallelFluidEnv` class to speed up training of RL agents.

.. code-block:: python

    from fluidgym.envs.parallel_env import ParallelFluidEnv

    # Since the process is forked, we need to protect the entry point
    if __name__ == "__main__":
        # Instead of using fluidgym.make, we directly create a ParallelFluidEnv
        env = ParallelFluidEnv(
            env_id="Airfoil3D-easy-v0",
            cuda_ids=[0, 0],  # List of GPU IDs to use
            use_marl=False,
        )
        try:
            # From now on, everything works as usual with the first dimension
            # being the number of parallel environments for SARL or
            # number of parallel envs * number of agents for MARL.
            env.seed(42)

            obs, info = env.reset()
            action = env.sample_action()

            obs, reward, terminated, truncated, info = env.step(action)

        finally:
            env.close()
