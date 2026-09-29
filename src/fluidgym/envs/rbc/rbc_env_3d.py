"""3D Rayleigh-Bénard Convection (RBC) environment."""

from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from phipict.grid.helpers import get_cell_size
from phipict.solvers.tolerance import SolverTolerance

from fluidgym.envs.rbc.rbc_env_base import RBCEnvBase
from fluidgym.envs.util.obs_extraction import extract_moving_window_3d
from fluidgym.envs.util.smoothing import smooth_segment_profile
from fluidgym.envs.util.visualization import (
    render_3d_voxels,
)

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
    "pressure_tol_intermediate": SolverTolerance(atol=1e-4),
    "pressure_warm_start": True,
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

    # A quarter of the 20 PISO steps of a registered env step: the 3D replay tape
    # is what caps a differentiable rollout here (see
    # ``FluidEnv.bptt_segment_size``)
    _default_bptt_segment_size: int | None = 5

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
    def obs_resampling_shape(self) -> tuple[int, int, int]:
        """The shape of the observation resampling grid."""
        nx = self._n_heaters * 20
        height = round(nx / self._aspect_ratio)

        return (nx, height, nx)

    def _get_sensor_locations(self) -> torch.Tensor:
        sensor_locations_2d = self._get_sensor_locations_2d()
        nz = self.obs_resampling_shape[-1]
        n_sensors_z = self._n_sensors_per_heater * self._n_heaters
        sensor_z = torch.linspace(
            start=0,
            end=self.obs_resampling_shape[-1],
            steps=n_sensors_z + 1,
        )[:-1] + nz / (2 * n_sensors_z)
        sensor_z = sensor_z.round().to(torch.int).to(self._cuda_device)

        # Repeat each (x, y) for each z
        x = sensor_locations_2d[0].repeat_interleave(n_sensors_z)
        y = sensor_locations_2d[1].repeat_interleave(n_sensors_z)
        z = sensor_z.repeat(sensor_locations_2d.shape[1])

        sensor_locations_3d = torch.stack([x, y, z], dim=0)
        return sensor_locations_3d

    def _smooth_action_profile_2d(self, T_action: torch.Tensor) -> torch.Tensor:
        # [E, n_heaters_z, n_heaters_x], smoothed along each of the last two dims
        width, alpha = self._heater_width, self._heater_smoothing_alpha
        smooth_x = smooth_segment_profile(T_action.transpose(-1, -2), width, alpha)
        smooth_xz = smooth_segment_profile(smooth_x.transpose(-1, -2), width, alpha)
        return smooth_xz

    def _shift_and_limit(self, action: torch.Tensor, limit: float) -> torch.Tensor:
        """Map an action to a zero-mean deviation bounded by ``limit``.

        Returns the per-agent deviation of shape ``(E, n_heaters, n_heaters)``.
        """
        # First, we bring the action to the shape (E, n_heaters, n_heaters)
        transformed_action = action.reshape(
            action.size(0), self._n_heaters, self._n_heaters
        )

        # Cf. eq. (8) in https://doi.org/10.1063/5.0153181 (per environment)
        shifted = transformed_action - transformed_action.mean(
            dim=(-2, -1), keepdim=True
        )

        # Cf. eq. (9) in https://doi.org/10.1063/5.0153181
        return shifted / (torch.clamp(shifted.abs(), min=1.0) / limit)

    def _action_to_control(self, action: torch.Tensor) -> torch.Tensor:
        T_action = self._shift_and_limit(action, self._heater_limit)

        # So far, we have computed the derivation from the bottom temperature
        # We need to shift it to the actual temperature range
        T_action = T_action + self._T_hot

        T_smooth = self._smooth_action_profile_2d(T_action=T_action)

        return T_smooth

    def _apply_action(self, action: torch.Tensor) -> None:
        """Apply the given action to the simulation."""
        # one heater field per environment: [E, 1, nz, 1, nx]
        control = self._action_to_control(action)
        control = control[:, None, :, None, :]

        self._bottom_plate.setPassiveScalar(control)

    def __get_agent_range(
        self, agent_idx: int, userender_shape: bool
    ) -> tuple[int, int, int, int]:
        # TODO what to include in 3D local obs and rewards?
        x_idx = agent_idx % self._n_heaters
        z_idx = agent_idx // self._n_heaters

        if userender_shape:
            heater_width_x_z = self.obs_resampling_shape[0] // self._n_heaters
        else:
            heater_width_x_z = self._heater_width

        x_min, x_max = x_idx * heater_width_x_z, (x_idx + 1) * heater_width_x_z
        z_min, z_max = z_idx * heater_width_x_z, (z_idx + 1) * heater_width_x_z

        return x_min, x_max, z_min, z_max

    def _get_global_obs(self) -> dict[str, torch.Tensor]:
        # [E, C, Z, Y, X] -> sensor readings [E, C, R]
        at = (
            slice(None),
            slice(None),
            self._sensor_locations[2],
            self._sensor_locations[1],
            self._sensor_locations[0],
        )
        T = self._temperature_fields()[at]
        u = self._velocity_fields()[at]
        p = self._pressure_fields()[at]
        n_envs = T.size(0)
        grid = (self._n_sensors_x, self._n_sensors_y, self._n_sensors_x)

        T = T.reshape(n_envs, *grid).permute(0, 3, 2, 1)
        u = u.reshape(n_envs, 3, *grid).permute(0, 1, 4, 3, 2)
        p = p.reshape(n_envs, *grid).permute(0, 3, 2, 1)

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

        u_x = u[:, 0]  # [E, Z, Y, X]
        u_y = u[:, 1]  # [E, Z, Y, X]
        u_z = u[:, 2]  # [E, Z, Y, X]

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
        local_obs_u = torch.stack((local_obs_u_x, local_obs_u_y, local_obs_u_z), dim=2)

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
        assert isinstance(self._block.passiveScalar, torch.Tensor)

        T: torch.Tensor = self._block.passiveScalar[:, 0]  # [E, Z, Y, X]

        u: torch.Tensor = self._block.getVelocity(False)
        u_y = u[:, 1]  # [E, Z, Y, X]

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
        )  # [E, n_agents]

        return self.nu_ref - local_nu

    def plot(self, output_path: Path | None = None) -> None:
        """Plot the environments configuration."""
        # Plot sensor locations in 3D
        import matplotlib.pyplot as plt

        if output_path is None:
            output_path = Path(".")

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

        ax.set_xlim(0, self.obs_resampling_shape[0])
        ax.set_ylim(0, self.obs_resampling_shape[2])
        ax.set_zlim(0, self.obs_resampling_shape[1])  # type: ignore

        plt.title("3D Sensor Locations")
        plt.savefig(output_path / "3d_sensor_locations.png", dpi=300)
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
        T = self._render_env_field(self._temperature_fields())[0].detach().cpu().numpy()

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
