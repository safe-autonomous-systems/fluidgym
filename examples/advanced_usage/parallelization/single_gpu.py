import fluidgym

# Four environments batched in one simulation on one GPU. They share the grid,
# the solver setup and every kernel launch, so a small batch costs little more
# than a single environment. fluidgym.make_vec(id, n_envs=4) is equivalent
env = fluidgym.make("CylinderJet2D-easy-v0", n_envs=4)
env.seed(42)

# Everything has a leading env dim: [n_envs, ...], or [n_envs, n_agents, ...]
# for MARL
obs, info = env.reset()
action = env.sample_action()

obs, reward, terminated, truncated, info = env.step(action)
env.render(save=True)
print(reward)
# reward: [4], terminated/truncated: bool tensors [4]

# The batch shares the episode clock, but every env can start from its own
# initial domain
obs, info = env.reset(domain_idx=[0, 1, 2, 3])
