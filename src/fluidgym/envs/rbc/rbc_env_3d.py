"""3D Rayleigh-Bénard Convection (RBC) environment."""

from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from fluidgym.envs.rbc.rbc_env_base import RBCEnvBase
from fluidgym.envs.util.obs_extraction import extract_moving_window_3d
from fluidgym.envs.util.visualization import (
    render_3d_voxels,
)
from fluidgym.simulation.helpers import get_cell_size

RBC_3D_DEFAULT_CONFIG = {
    "rayleigh_number": 2e3,
    "prandtl_number": 0.7,
    "n_heaters": 8,
    "resolution": 8,
    "dt": 0.05,
    "adaptive_cfl": 0.8,
    "step_length": 1.0,
    "episode_length": 200,
    "local_obs_window": 3,  # in number of agents
    "local_reward_weight": 0.0015,  # Based on beta in doi.org/10.1063/5.0153181
    "uniform_grid": False,
    "aspect_ratio": 1.0,  # equivalent to pi
    "use_marl": True,
    "dtype": torch.float32,
    "load_initial_domain": True,
    "load_domain_statistics": True,
    "randomize_initial_state": True,
    "enable_actions": True,
    "differentiable": False,
}


class RBCEnv3D(RBCEnvBase):
    """Environment for 3D Rayleigh-Bénard Convection (RBC).

    Parameters
    ----------
    rayleigh_number: float
        The Rayleigh number for the simulation.

    prandtl_number: float
        The Prandtl number for the simulation.

    n_heaters: int
        The number of heaters in the domain.

    resolution: int
        The width (resolution) of each heater in grid cells.

    adaptive_cfl: float
        Target CFL number for adaptive time stepping.

    dt: float
        The time step size for the simulation.

    step_length: float
        The physical time length of each environment step.

    episode_length: int,
        The number of steps per episode.

    local_obs_window: int
        The size of the local observation window for each agent.

    local_reward_weight: float | None
        Weighting factor for local rewards in multi-agent settings.
        Has to be set for multi-agent RL. Defaults to None.

    uniform_grid: bool
        Whether to use a uniform grid. If False, a non-uniform grid is used.

    aspect_ratio: float
        The aspect ratio (L/H) of the domain in multiples of π.

    use_marl: bool
        Whether to enable multi-agent reinforcement learning mode.

    dtype: torch.dtype
        The data type for the simulation tensors. Defaults to torch.float32.

    cuda_device: torch.device | None
        The CUDA device to use for the simulation. If None, the default cuda device is
        used. Defaults to None.

    load_initial_domain: bool
        Whether to load the initial domain from file. Defaults to True.

    load_domain_statistics: bool
        Whether to load precomputed domain statistics. Defaults to True.

    randomize_initial_state: bool
        Whether to randomize the initial state of the simulation. Defaults to False.

    enable_actions: bool
        Whether to enable action application in the environment. Defaults to True.

    differentiable: bool
        Whether to enable differentiable simulation. Defaults to False.

    References
    ----------
    [1] J. Vasanth, J. Rabault, F. Alcántara-Ávila, M. Mortensen, and R. Vinuesa,
    “Multi-agent Reinforcement Learning for the Control of Three-Dimensional
    Rayleigh-Bénard Convection,” Flow, Turbulence and Combustion, Dec. 2024,
    doi: 10.1007/s10494-024-00619-2.
    """

    _default_render_key: str = "3d_temperature"
    _ndims = 3

    # Based on reference [1] with half domain size (division by sqrt(2))
    _initial_domain_steps = 1500

    def _get_action_space(self) -> spaces.Box:
        """Per-agent action space."""
        shape: tuple[int, ...]

        if self.use_marl:
            shape = (1,)
        else:
            shape = (self._n_heaters, self._n_heaters, 1)

        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=shape,
            dtype=np.float32,
        )

    def _get_observation_space(self) -> spaces.Dict:
        """Per-agent observation space."""
        if self._use_marl:
            shape = (
                self._n_sensors_per_heater * self._local_obs_window,
                self._n_sensors_y,
                self._n_sensors_per_heater * self._local_obs_window,
            )
        else:
            shape = (
                (self._n_sensors_per_heater * self._n_heaters),
                self._n_sensors_y,
                (self._n_sensors_per_heater * self._n_heaters),
            )

        return spaces.Dict(
            {
                "temperature": spaces.Box(
                    low=self._T_cold,
                    high=self._T_hot + self._heater_limit,
                    shape=shape,
                    dtype=np.float32,
                ),
                "velocity": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._ndims,) + shape,
                    dtype=np.float32,
                ),
                "pressure": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=shape,
                    dtype=np.float32,
                ),
            }
        )

    @property
    def render_shape(self) -> tuple[int, int, int]:
        """The shape of the rendered domain."""
        nx = self._n_heaters * 20
        height = round(nx / self._aspect_ratio)

        return (nx, height, nx)

    def _get_sensor_locations(self) -> torch.Tensor:
        sensor_locations_2d = self._get_sensor_locations_2d()
        nz = self.render_shape[-1]
        n_sensors_z = self._n_sensors_per_heater * self._n_heaters
        sensor_z = torch.linspace(
            start=0,
            end=self.render_shape[-1],
            steps=n_sensors_z + 1,
        )[:-1] + nz / (2 * n_sensors_z)
        sensor_z = sensor_z.round().to(torch.int).to(self._cuda_device)

        # Repeat each (x, y) for each z
        x = sensor_locations_2d[0].repeat_interleave(n_sensors_z)
        y = sensor_locations_2d[1].repeat_interleave(n_sensors_z)
        z = sensor_z.repeat(sensor_locations_2d.shape[1])

        sensor_locations_3d = torch.stack([x, y, z], dim=0)
        return sensor_locations_3d

    def __smooth_action_profile_1d(self, T_action: torch.Tensor) -> torch.Tensor:
        heater_width = self._heater_width
        alpha = 0.1  # % of heater_width to smooth over
        blended_heater_width = round(heater_width * alpha)

        def cubic_blend(
            t: torch.Tensor, A: torch.Tensor, B: torch.Tensor
        ) -> torch.Tensor:
            s = t * t * (3 - 2 * t)
            return (1 - s) * A + s * B

        # shift left/right along the last dimension
        T_left = torch.roll(T_action, shifts=1, dims=-1)
        T_right = torch.roll(T_action, shifts=-1, dims=-1)

        # heater segment indexing
        x_idx = torch.arange(self._x, device=T_action.device, dtype=torch.long)
        seg_id = x_idx // heater_width
        x_pos = x_idx % heater_width

        # gather neighboring values for each batch
        T0 = T_left.gather(-1, seg_id.unsqueeze(0).expand(T_action.size(0), -1))
        T1 = T_action.gather(-1, seg_id.unsqueeze(0).expand(T_action.size(0), -1))
        T2 = T_right.gather(-1, seg_id.unsqueeze(0).expand(T_action.size(0), -1))

        # zones
        left_zone = x_pos < blended_heater_width
        right_zone = x_pos >= heater_width - blended_heater_width

        # blending parameters
        tL = (x_pos.to(torch.float32) / blended_heater_width + 0.5).clamp(0.0, 1.0)
        tR = 1 - torch.roll(tL, shifts=heater_width - blended_heater_width + 1, dims=-1)

        # cubic blends
        TL = cubic_blend(tL, T0, T1)
        TR = cubic_blend(tR, T1, T2)

        # piecewise assembly
        T_smooth = torch.where(left_zone, TL, torch.where(right_zone, TR, T1))
        return T_smooth

    def __smooth_action_profile_2d(self, T_action: torch.Tensor) -> torch.Tensor:
        smooth_x = self.__smooth_action_profile_1d(T_action.T)
        smooth_xz = self.__smooth_action_profile_1d(smooth_x.T)
        return smooth_xz

    def _action_to_control(self, action: torch.Tensor) -> torch.Tensor:
        # First, we bring the action to the shape (n_heaters, n_heaters)
        transformed_action = action.view(self._n_heaters, self._n_heaters)

        # Cf. eq. (8) in https://doi.org/10.1063/5.0153181
        T_shifted = transformed_action - transformed_action.mean()

        # Cf. eq. (9) in https://doi.org/10.1063/5.0153181
        T_action = T_shifted / (
            torch.clamp(T_shifted.abs(), min=1.0) / self._heater_limit
        )

        # So far, we have computed the derivation from the bottom temperature.
        # We need to shift it to the actual temperature range.
        T_action += self._T_hot

        T_smooth = self.__smooth_action_profile_2d(T_action=T_action)

        return T_smooth

    def _apply_action(self, action: torch.Tensor) -> None:
        """Apply the given action to the simulation."""
        control = self._action_to_control(action)
        control = control[None, None, :, None, :]

        self._bottom_plate.setPassiveScalar(control)

    def __get_agent_range(
        self, agent_idx: int, userender_shape: bool
    ) -> tuple[int, int, int, int]:
        # TODO what to include in 3D local obs and rewards?
        x_idx = agent_idx % self._n_heaters
        z_idx = agent_idx // self._n_heaters

        if userender_shape:
            heater_width_x_z = self.render_shape[0] // self._n_heaters
        else:
            heater_width_x_z = self._heater_width

        x_min, x_max = x_idx * heater_width_x_z, (x_idx + 1) * heater_width_x_z
        z_min, z_max = z_idx * heater_width_x_z, (z_idx + 1) * heater_width_x_z

        return x_min, x_max, z_min, z_max

    def _get_global_obs(self) -> dict[str, torch.Tensor]:
        T = self.get_temperature()
        u = self.get_velocity()
        p = self.get_pressure()

        T = T[
            self._sensor_locations[2],
            self._sensor_locations[1],
            self._sensor_locations[0],
        ]
        T = T.reshape(self._n_sensors_x, self._n_sensors_y, self._n_sensors_x).permute(
            2, 1, 0
        )

        u = u.permute(1, 2, 3, 0)
        u = u[
            self._sensor_locations[2],
            self._sensor_locations[1],
            self._sensor_locations[0],
            :,
        ]
        u = u.reshape(
            self._n_sensors_x, self._n_sensors_y, self._n_sensors_x, 3
        ).permute(3, 2, 1, 0)

        p = p[
            self._sensor_locations[2],
            self._sensor_locations[1],
            self._sensor_locations[0],
        ]
        p = p.reshape(self._n_sensors_x, self._n_sensors_y, self._n_sensors_x).permute(
            2, 1, 0
        )

        return {
            "temperature": T,
            "velocity": u,
            "pressure": p,
        }

    def _get_local_obs(self) -> dict[str, torch.Tensor]:
        global_obs = self._get_global_obs()
        T = global_obs["temperature"]
        u = global_obs["velocity"]
        p = global_obs["pressure"]

        u_x = u[0, ...]  # [Z, Y, X]
        u_y = u[1, ...]  # [Z, Y, X]
        u_z = u[2, ...]  # [Z, Y, X]

        local_obs_T = extract_moving_window_3d(
            field=T,
            n_agents=self._n_heaters,  # n_agents per dim
            agent_width=self._n_sensors_per_heater,
            n_agents_per_window=self._local_obs_window,
        )

        local_obs_u_x = extract_moving_window_3d(
            field=u_x,
            n_agents=self._n_heaters,  # n_agents per dim
            agent_width=self._n_sensors_per_heater,
            n_agents_per_window=self._local_obs_window,
        )

        local_obs_u_y = extract_moving_window_3d(
            field=u_y,
            n_agents=self._n_heaters,  # n_agents per dim
            agent_width=self._n_sensors_per_heater,
            n_agents_per_window=self._local_obs_window,
        )

        local_obs_u_z = extract_moving_window_3d(
            field=u_z,
            n_agents=self._n_heaters,  # n_agents per dim
            agent_width=self._n_sensors_per_heater,
            n_agents_per_window=self._local_obs_window,
        )
        local_obs_u = torch.stack((local_obs_u_x, local_obs_u_y, local_obs_u_z), dim=1)

        local_obs_p = extract_moving_window_3d(
            field=p,
            n_agents=self._n_heaters,  # n_agents per dim
            agent_width=self._n_sensors_per_heater,
            n_agents_per_window=self._local_obs_window,
        )

        return {
            "temperature": local_obs_T,
            "velocity": local_obs_u,
            "pressure": local_obs_p,
        }

    def _get_local_rewards(self) -> torch.Tensor:
        T: torch.Tensor = self._block.passiveScalar[0, 0]  # [Z, Y, X]

        u: torch.Tensor = self._block.getVelocity(False)
        u_y = u[0, 1]  # [Z, Y, X]

        cell_size = get_cell_size(self._block).squeeze()
        local_cell_size = cell_size[
            : self._local_obs_window * self._heater_width,
            :,
            : self._local_obs_window * self._heater_width,
        ]

        local_T = extract_moving_window_3d(
            field=T,
            n_agents=self._n_heaters,  # n_agents per dim
            agent_width=self._heater_width,
            n_agents_per_window=self._local_obs_window,
        )

        local_u_y = extract_moving_window_3d(
            field=u_y,
            n_agents=self._n_heaters,  # n_agents per dim
            agent_width=self._heater_width,
            n_agents_per_window=self._local_obs_window,
        )

        local_nu = self._compute_nusselt(
            T=local_T,
            u_y=local_u_y,
            cell_size=local_cell_size,
        )  # [n_agents,]

        return self.nu_ref - local_nu

    def plot(self) -> None:
        """Plot the environments configuration."""
        # Plot sensor locations in 3D
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        ax = plt.axes(projection="3d")
        all_sensor_locs = self._sensor_locations.cpu().numpy()

        for n in range(self.n_agents):
            x_min, x_max, z_min, z_max = self.__get_agent_range(n, userender_shape=True)

            # Select sensor locations based on x and z
            x_mask = (all_sensor_locs[0] > x_min) & (all_sensor_locs[0] < x_max)
            z_mask = (all_sensor_locs[2] > z_min) & (all_sensor_locs[2] < z_max)
            sensor_locs = all_sensor_locs[:, x_mask & z_mask]

            # select color based no agent index
            x_idx = n % self._n_heaters
            z_idx = n // self._n_heaters

            color = "blue" if (x_idx + z_idx) % 2 == 0 else "red"

            ax.scatter(
                sensor_locs[0],
                sensor_locs[2],
                sensor_locs[1],
                marker="o",
                color=color,
                s=10,  # type: ignore
                label="Sensors",
            )
        ax.set_xlabel("X axis")
        ax.set_ylabel("Z axis")
        ax.set_zlabel("Y axis")  # type: ignore

        ax.set_xlim(0, self.render_shape[0])
        ax.set_ylim(0, self.render_shape[2])
        ax.set_zlim(0, self.render_shape[1])  # type: ignore

        plt.title("3D Sensor Locations")
        plt.savefig("3d_sensor_locations.png", dpi=300)
        plt.close()

    def plot_actuation(self, action: torch.Tensor, action_smooth: torch.Tensor) -> None:
        """Plot the heater actuation profiles before and after smoothing.

        Parameters
        ----------
        action: torch.Tensor
            The original heater action profile of shape (n_heaters, n_heaters).

        action_smooth: torch.Tensor
            The smoothed heater action profile of shape (n_heaters, n_heaters).
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 5))
        plt.imshow(
            action.cpu(),
            origin="lower",
            cmap="rainbow",
        )
        plt.colorbar(label="Heater Temperature")
        plt.title("Smoothed Heater Actuation Profile")
        plt.xlabel("X axis")
        plt.ylabel("Z axis")
        plt.savefig("action.png", dpi=300)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.imshow(
            action_smooth.cpu(),
            origin="lower",
            cmap="rainbow",
        )
        plt.colorbar(label="Heater Temperature")
        plt.title("Smoothed Heater Actuation Profile")
        plt.xlabel("X axis")
        plt.ylabel("Z axis")
        plt.savefig("heater_actuation_profile.png", dpi=300)
        plt.close()

    def _get_render_data(
        self,
        render_3d: bool,
        output_path: Path | None = None,
    ) -> dict[str, np.ndarray]:
        render_data = super()._get_render_data(
            render_3d=render_3d, output_path=output_path
        )
        T = self.get_temperature().squeeze(0).detach().cpu().numpy()

        if render_3d:
            if output_path is not None:
                output_path_3d = (
                    output_path
                    / f"3d_temperature_fig_{self._n_episodes}_{self._n_steps}.png"
                )
            else:
                output_path_3d = None

            render_data["3d_temperature"] = render_3d_voxels(
                field=T,
                ds=4,
                field_range=(self._T_cold, self._T_hot + self._heater_limit),
                output_path=output_path_3d,
                colormap="rainbow",
                view_kwargs={"elev": 15, "azim": 45},
            )

        return render_data
