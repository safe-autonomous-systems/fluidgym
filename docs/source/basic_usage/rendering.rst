Rendering Environments
======================

FluidGym environments support rendering of the fluid flow states during simulation.
This is useful for visualizing the dynamics and understanding the agent's interactions
with the fluid environment. Rendering differs for 2D and 3D environments, with 3D
environments offering more advanced visualization options.

Per default, for 2D environments, the `render` method generates 2D plots of the fluid
state, while for 3D environments, it creates 3D visualizations along slices and 
optionally more detailed 3D plots. The latter can be enabled by setting the `render_3d`
parameter to `True`. We note that rendering 3D environments with detailed plots can be
computationally intensive and may slow down the simulation. Therefore, 3D rendering is
disabled by default.

Here is a simple example from `examples/rendering.py`:

.. code-block:: python

    from pathlib import Path

    import fluidgym

    env = fluidgym.make_gym(
        "CylinderJet3D-easy-v0",
    )

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