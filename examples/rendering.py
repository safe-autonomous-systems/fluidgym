from pathlib import Path

import fluidgym
from fluidgym.integration.gymnasium import GymFluidEnv

env = fluidgym.make(
    "CylinderJet3D-easy-v0",
)
env = GymFluidEnv(env)

obs, info = env.reset(seed=42)

done = False
i = 0
while not done:
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    done = term or trunc

    # Now, we also want fancy 3d plots of the environment
    env.render(
        # Defines, whether to save individual rendered frames to plots/images
        save=True,
        # Enables 3D rendering for fancy, i.e., more detailed cube and isosurface plots
        render_3d=True,
        # You can also specify a specific tag/filename for each frame
        filename=f"cylinder_3d_step_{i:02d}.png",
        # Specify output path for rendered frames
        output_path=Path("./renders"),
    )

    print(f"Step {i}: Reward = {reward:.4f}")

    i += 1

env.save_gif("cylinder_3d.gif")
