from fluidgym.envs.parallel_env import ParallelFluidEnv

if __name__ == "__main__":
    env = ParallelFluidEnv(
        env_id="Airfoil3D-easy-v0",
        cuda_ids=[0, 0],  # List of GPU IDs to use
        use_marl=False,
    )
    try:
        env.seed(42)

        obs, info = env.reset()
        action = env.sample_action()

        obs, reward, terminated, truncated, info = env.step(action)

    finally:
        env.close()
