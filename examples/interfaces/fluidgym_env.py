import fluidgym

env = fluidgym.make(
    "JetCylinder2D-easy-v0",
)
obs, info = env.reset(seed=42)

for _ in range(50):
    action = env.sample_action()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        break

env.save_gif("cylinder.gif")
