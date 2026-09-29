"""2D Rayleigh-Bénard Convection (RBC) environment."""

import numpy as np
import torch
from gymnasium import spaces
from phipict.grid.helpers import get_cell_size

from fluidgym.envs.rbc.rbc_env_base import RBCEnvBase
from fluidgym.envs.util.obs_extraction import extract_moving_window_2d
from fluidgym.envs.util.smoothing import smooth_segment_profile

RBC_2D_DEFAULT_CONFIG = {
    "rayleigh_number": 8e4,
    "prandtl_number": 0.7,
    "n_heaters": 12,
    "resolution": 8,
    "dt": 0.05,
    "adaptive_cfl": 0.8,
    "step_length": 1.0,
    "episode_length": 200,
    "local_obs_window": 11,  # in number of agents
    "local_reward_weight": 0.2,
    "uniform_grid": False,
    "aspect_ratio": 1.0,  # equivalent to pi
    "use_marl": False,
    "dtype": torch.float32,
    "load_initial_domain": True,
    "load_domain_statistics": True,
    "randomize_initial_state": True,
    "enable_actions": True,
    "differentiable": False,
}


class RBCEnv2D(RBCEnvBase):
    """Environment for 2D Rayleigh-Bénard Convection (RBC).

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
    [1] C. Vignon, J. Rabault, J. Vasanth, F. Alcántara-Ávila, M. Mortensen, and
    R. Vinuesa, “Effective control of two-dimensional Rayleigh-Bénard convection:
    Invariant multi-agent reinforcement learning is all you need,” Physics of Fluids,
    vol. 35, no. 6, p. 065146, June 2023, doi: 10.1063/5.0153181.
    """

    _ndims = 2

    # Based on https://doi.org/10.1007/s10494-024-00619-2
    # with half domain size (division by sqrt(2))
    _initial_domain_steps = 283

    def _get_action_space(self) -> spaces.Box:
        """Per-agent action space."""
        shape: tuple[int, ...]

        if self.use_marl:
            shape = (1,)
        else:
            shape = (
                self._n_heaters,
                1,
            )

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
                self._n_sensors_y,
                self._n_sensors_per_heater * self._local_obs_window,
            )
        else:
            shape = (
                self._n_sensors_y,
                self._n_heaters * self._n_sensors_per_heater,
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
    def obs_resampling_shape(self) -> tuple[int, ...]:
        """The shape of the observation resampling grid."""
        nx = self._n_heaters * 20
        height = round(nx / self._aspect_ratio)

        return (nx, height, nx)

    def _get_global_obs(self) -> dict[str, torch.Tensor]:
        # [E, C, Y, X] -> sensor readings [E, C, R]
        at = (
            slice(None),
            slice(None),
            self._sensor_locations[1],
            self._sensor_locations[0],
        )
        T = self._temperature_fields()[at]
        u = self._velocity_fields()[at]
        p = self._pressure_fields()[at]
        n_envs = T.size(0)
        grid = (self._n_sensors_x, self._n_sensors_y)

        T = T.reshape(n_envs, *grid).transpose(1, 2)
        u = u.reshape(n_envs, 2, *grid).transpose(2, 3)
        p = p.reshape(n_envs, *grid).transpose(1, 2)

        return {
            "temperature": T,
            "velocity": u,
            "pressure": p,
        }

    def _get_sensor_locations(self) -> torch.Tensor:
        """
        Get the locations of the sensors in the flow domain.

        Args:
            domain_shape (tuple): (nx, ny) size of the domain
            n_sensors (tuple): (n_sensors_x, n_sensors_y)

        Returns
        -------
            torch.Tensor of shape (2, n_sensors_x * n_sensors_y), dtype=torch.int32
        """
        return self._get_sensor_locations_2d()

    def __action_to_control(self, action: torch.Tensor) -> torch.Tensor:
        # scaled_action = action * self.heater_limit

        # Cf. eq. (8) in https://doi.org/10.1063/5.0153181 (per environment)
        T_shifted = action - action.mean(dim=-1, keepdim=True)

        # Cf. eq. (9) in https://doi.org/10.1063/5.0153181
        T_action = T_shifted / (
            torch.clamp(T_shifted.abs(), min=1.0) / self._heater_limit
        )

        # So far, we have computed the derivation from the bottom temperature
        # We need to shift it to the actual temperature range
        T_action += self._T_hot

        # Smoothing according to https://doi.org/10.1063/5.0153181
        T_smooth = smooth_segment_profile(
            T_action, self._heater_width, self._heater_smoothing_alpha
        )

        # [E, nx]
        return T_smooth

    def _apply_action(self, action: torch.Tensor) -> None:
        """Apply the given action to the simulation."""
        # one heater profile per environment: [E, n_heaters] -> [E, 1, 1, nx]
        flat_action = action.reshape(action.size(0), -1)
        control = self.__action_to_control(flat_action)
        control = control[:, None, None, :]

        self._bottom_plate.setPassiveScalar(control)

    def _get_local_obs(self) -> dict[str, torch.Tensor]:
        global_obs = self._get_global_obs()

        T = global_obs["temperature"]  # [E, Y, X]
        u = global_obs["velocity"]  # [E, 2, Y, X]
        p = global_obs["pressure"]  # [E, Y, X]

        u_x = u[:, 0]  # [E, Y, X]
        u_y = u[:, 1]  # [E, Y, X]

        local_obs_T = extract_moving_window_2d(
            field=T,
            n_agents=self.n_agents,
            agent_width=self._n_sensors_per_heater,
            n_agents_per_window=self._local_obs_window,
        )

        local_obs_u_x = extract_moving_window_2d(
            field=u_x,
            n_agents=self.n_agents,
            agent_width=self._n_sensors_per_heater,
            n_agents_per_window=self._local_obs_window,
        )
        local_obs_u_y = extract_moving_window_2d(
            field=u_y,
            n_agents=self.n_agents,
            agent_width=self._n_sensors_per_heater,
            n_agents_per_window=self._local_obs_window,
        )
        locla_obs_u = torch.stack([local_obs_u_x, local_obs_u_y], dim=2)

        local_obs_p = extract_moving_window_2d(
            field=p,
            n_agents=self.n_agents,
            agent_width=self._n_sensors_per_heater,
            n_agents_per_window=self._local_obs_window,
        )

        return {
            "temperature": local_obs_T,
            "velocity": locla_obs_u,
            "pressure": local_obs_p,
        }

    def _get_local_rewards(self) -> torch.Tensor:
        assert isinstance(self._block.passiveScalar, torch.Tensor)

        T: torch.Tensor = self._block.passiveScalar[:, 0]  # [E, Y, X]

        u: torch.Tensor = self._block.getVelocity(False)
        u_y = u[:, 1]  # [E, Y, X]

        cell_size = get_cell_size(self._block).squeeze()
        local_cell_size = cell_size[:, : self._local_obs_window * self._heater_width]

        local_T = extract_moving_window_2d(
            field=T,
            n_agents=self.n_agents,
            agent_width=self._heater_width,
            n_agents_per_window=self._local_obs_window,
        )  # [E, n_agents, Y, agent_window * n_obs_per_agent]

        local_u_y = extract_moving_window_2d(
            field=u_y,
            n_agents=self.n_agents,
            agent_width=self._heater_width,
            n_agents_per_window=self._local_obs_window,
        )  # [n_agents, Y, agent_window * n_obs_per_agent]

        local_nu = self._compute_nusselt(
            T=local_T,
            u_y=local_u_y,
            cell_size=local_cell_size,
        )  # [E, n_agents]

        return self.nu_ref - local_nu

    def plot_actuation(
        self,
        action: torch.Tensor,
        action_smooth: torch.Tensor,
    ) -> None:
        """Plot the actuation profile."""
        if self._ndims != 2:
            self._logger.warning(
                "Plotting actuation is only implemented for 2D RBC environments.",
            )
            return

        import matplotlib.pyplot as plt

        _T_action = torch.repeat_interleave(
            action,
            repeats=self._heater_width,
            dim=0,
        )
        plt.figure(figsize=(10, 5))
        plt.plot(_T_action, marker=None, linestyle="-", color="b")
        plt.plot(action_smooth, marker=None, linestyle="--", color="r")
        plt.title("Actuation Profile")
        plt.xticks(np.arange(0, self._x, self._heater_width))
        plt.xlabel("Heater Index")
        plt.ylabel("Actuation Strength")
        plt.grid()
        plt.tight_layout()
        plt.savefig("actuation.png", dpi=500)
