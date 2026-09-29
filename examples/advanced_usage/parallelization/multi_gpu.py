import fluidgym

# Since the worker processes are spawned, we need to protect the entry point
if __name__ == "__main__":
    # 128 environments: one worker process per GPU, each simulating 64
    # environments batched together. n_envs must be divisible by len(devices)
    env = fluidgym.make_vec("CylinderJet2D-easy-v0", n_envs=128, devices=[0, 1])
    try:
        # Everything has a leading env dim: [n_envs, ...], or
        # [n_envs, n_agents, ...] for MARL. Worker r is seeded with seed + r
        env.seed(42)

        obs, info = env.reset()
        action = env.sample_action()

        obs, reward, terminated, truncated, info = env.step(action)

        # Results of a ParallelFluidEnv are returned on the CPU
    finally:
        env.close()
