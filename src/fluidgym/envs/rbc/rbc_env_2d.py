"""2D Rayleigh-Bénard Convection (RBC) environment."""

import numpy as np
import torch
from gymnasium import spaces

from fluidgym.envs.rbc.rbc_env_base import RBCEnvBase
from fluidgym.envs.util.obs_extraction import extract_moving_window_2d
from fluidgym.simulation.helpers import get_cell_size

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
    def render_shape(self) -> tuple[int, ...]:
        """The shape of the rendered domain."""
        nx = self._n_heaters * 20
        height = round(nx / self._aspect_ratio)

        return (nx, height, nx)

    def _get_global_obs(self) -> dict[str, torch.Tensor]:
        T = self.get_temperature()
        u = self.get_velocity()
        p = self.get_pressure()

        T = T[self._sensor_locations[1], self._sensor_locations[0]]
        T = T.reshape(self._n_sensors_x, self._n_sensors_y).T

        u = u.permute(1, 2, 0)
        u = u[self._sensor_locations[1], self._sensor_locations[0], :]
        u = u.reshape(self._n_sensors_x, self._n_sensors_y, 2).permute(2, 1, 0)

        p = p[self._sensor_locations[1], self._sensor_locations[0]]
        p = p.reshape(self._n_sensors_x, self._n_sensors_y).T

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

    def __smooth_action_profile_1d(self, T_action: torch.Tensor) -> torch.Tensor:
        heater_width = self._heater_width
        alpha = 0.1  # % of heater_width to smooth over
        blended_heater_width = round(heater_width * alpha)

        def cubic_blend(
            t: torch.Tensor, A: torch.Tensor, B: torch.Tensor
        ) -> torch.Tensor:
            s = t * t * (3 - 2 * t)
            return (1 - s) * A + s * B

        # shift left/right
        T_left = torch.roll(T_action, shifts=1, dims=0)
        T_right = torch.roll(T_action, shifts=-1, dims=0)

        # heater segment indexing
        x_idx = torch.arange(self._x, device=T_action.device, dtype=torch.long)
        seg_id = x_idx // heater_width
        x_pos = x_idx % heater_width

        # gather neighboring values
        T0 = T_left[seg_id]
        T1 = T_action[seg_id]
        T2 = T_right[seg_id]

        # zones
        left_zone = x_pos < blended_heater_width
        right_zone = x_pos >= heater_width - blended_heater_width

        # blending parameters
        tL = (x_pos.to(torch.float32) / blended_heater_width + 0.5).clamp(0.0, 1.0)
        tR = 1 - torch.roll(tL, shifts=heater_width - blended_heater_width + 1, dims=0)

        # cubic blends
        TL = cubic_blend(tL, T0, T1)
        TR = cubic_blend(tR, T1, T2)

        # piecewise assembly
        T_smooth = torch.where(left_zone, TL, torch.where(right_zone, TR, T1))

        return T_smooth

    def __action_to_control(self, action: torch.Tensor) -> torch.Tensor:
        # scaled_action = action * self.heater_limit

        # Cf. eq. (8) in https://doi.org/10.1063/5.0153181
        T_shifted = action - action.mean()

        # Cf. eq. (9) in https://doi.org/10.1063/5.0153181
        T_action = T_shifted / (
            torch.clamp(T_shifted.abs(), min=1.0) / self._heater_limit
        )

        # So far, we have computed the derivation from the bottom temperature.
        # We need to shift it to the actual temperature range.
        T_action += self._T_hot

        # Smoothing according to https://doi.org/10.1063/5.0153181
        T_smooth = self.__smooth_action_profile_1d(T_action=T_action)

        # broadcast along y and add leading channel dim: (1, nx, ny)
        # (expand returns a view; use .clone() if you need a writable contiguous tensor)
        control = T_smooth.expand(self._x)

        return control

    def _apply_action(self, action: torch.Tensor) -> None:
        """Apply the given action to the simulation."""
        flat_action = action.squeeze()
        control = self.__action_to_control(flat_action)
        control = control[None, None, None, ...]

        self._bottom_plate.setPassiveScalar(control)

    def _get_local_obs(self) -> dict[str, torch.Tensor]:
        global_obs = self._get_global_obs()

        T = global_obs["temperature"]  # [Y, X]
        u = global_obs["velocity"]  # [Y, X, 2]
        p = global_obs["pressure"]  # [Y, X]

        u_x = u[0, ...]  # [Y, X]
        u_y = u[1, ...]  # [Y, X]

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
        locla_obs_u = torch.stack([local_obs_u_x, local_obs_u_y], dim=1)

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
        T: torch.Tensor = self._block.passiveScalar[0, 0]  # [Y, X]

        u: torch.Tensor = self._block.getVelocity(False)
        u_y = u[0, 1]  # [Y, X]

        cell_size = get_cell_size(self._block).squeeze()
        local_cell_size = cell_size[:, : self._local_obs_window * self._heater_width]

        local_T = extract_moving_window_2d(
            field=T,
            n_agents=self.n_agents,
            agent_width=self._heater_width,
            n_agents_per_window=self._local_obs_window,
        )  # [n_agents, Y, agent_window * n_obs_per_agent]

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
        )  # [n_agents,]

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
