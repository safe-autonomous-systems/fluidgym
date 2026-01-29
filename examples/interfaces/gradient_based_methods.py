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
