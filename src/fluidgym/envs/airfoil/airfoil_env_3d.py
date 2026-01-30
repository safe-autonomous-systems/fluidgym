"""Environment for 3D airfoil drag and lift reduction."""

from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from fluidgym.envs.airfoil.airfoil_env_base import AirfoilEnvBase
from fluidgym.envs.util.obs_extraction import (
    extract_global_3d_obs,
    transform_global_to_local_obs_3d,
)
from fluidgym.envs.util.visualization import render_3d_iso
from fluidgym.simulation.extensions import (
    PISOtorch,  # type: ignore[import-untyped,import-not-found]
)
from fluidgym.simulation.pict.util.domain_io import load_domain
from fluidgym.types import EnvMode

VORTICITY_RENDER_LEVELS = {
    1000: 2.0,
    3000: 3.5,
    5000: 4.5,
}


AIRFOIL_3D_DEFAULT_CONFIG = {
    "n_agents": 4,
    "reynolds_number": 3e3,
    "dt": 0.05,
    "adaptive_cfl": 0.8,
    "step_length": 0.25,
    "episode_length": 200,
    "attack_angle_deg": 10.0,
    "local_obs_window": 1,
    "use_marl": False,
    "local_reward_weight": 0.5,  # Based on doi.org/10.48550/arXiv.2509.10185
    "local_2d_obs": False,
    "init_from_2d": True,
    "dtype": torch.float32,
    "load_initial_domain": True,
    "load_domain_statistics": True,
    "randomize_initial_state": True,
    "enable_actions": True,
    "differentiable": False,
}


class AirfoilEnv3D(AirfoilEnvBase):
    """Environment for 3D airfoil aerodynamic efficiency improvement.

    Parameters
    ----------
    n_agents: int
        The number of agents controlling the synthetic jets.

    reynolds_number: float
        The Reynolds number of the flow.

    adaptive_cfl: float
        Target CFL number for adaptive time stepping.

    step_length: float
        The non-dimensional time length of each environment step.

    episode_length: int
        The number of steps per episode.

    dt: float
        The time step size to use in the simulation.

    attack_angle_deg: float
        The angle of attack of the airfoil in degrees.

    local_obs_window: int
        The size of the local observation window for each agent.

    use_marl: bool
        Whether to enable multi-agent reinforcement learning mode.

    local_reward_weight: float | None
        The weight of the local reward in the combined reward for each agent. Has to be
        set if multi-agent step function is used. Defaults to None.

    local_2d_obs: bool
        Whether to use 2D local observations (velocity in x and y directions only, a
        single sensor layer and window = 1). Defaults to False.

    init_from_2d: bool
        Whether to initialize the 3D flow field from a 2D initial domain. This
        significantly reduces the initial transient phase. If the initial domain is
        loaded from disk, this parameter is not active. Defaults to True.

    dtype: torch.dtype
        The data type to use for the simulation. Defaults to torch.float32.

    cuda_device: torch.device | None
        The CUDA device to use for the simulation. If None, the default cuda device is
        used. Defaults to None.

    debug: bool
        Whether to enable debug mode. Defaults to False.

    load_initial_domain: bool
        Whether to load initial domain states from disk. Defaults to True.

    load_domain_statistics: bool
        Whether to load domain statistics from disk. Defaults to True.

    randomize_initial_state: bool
        Whether to randomize the initial state on reset. Defaults to True.

    enable_actions: bool
        Whether to enable actions. If False, the environment will be run in
        uncontrolled mode. Defaults to True.

    differentiable: bool
        Whether to enable differentiable simulation mode. Defaults to False.

    References
    ----------
    [1] R. Montalà et al., “Discovering Flow Separation Control Strategies in 3D Wings
    via Deep Reinforcement Learning,” Sept. 12, 2025, arXiv: arXiv:2509.10185.
    doi: 10.48550/arXiv.2509.10185.
    """

    _default_render_key: str = "3d_vorticity"

    _n_sensors_per_agent: int = 1
    _supports_marl: bool = True

    def __init__(
        self,
        n_agents: int,
        reynolds_number: float,
        adaptive_cfl: float,
        step_length: float,
        episode_length: int,
        dt: float,
        attack_angle_deg: float,
        local_obs_window: int,
        use_marl: bool,
        local_reward_weight: float | None,
        local_2d_obs: bool = False,
        init_from_2d: bool = True,
        dtype: torch.dtype = torch.float32,
        cuda_device: torch.device | None = None,
        debug: bool = False,
        load_initial_domain: bool = True,
        load_domain_statistics: bool = True,
        randomize_initial_state: bool = True,
        enable_actions: bool = True,
        differentiable: bool = False,
    ):
        if n_agents < 1 or self._res_z % n_agents != 0:
            raise ValueError(
                "n_agents must be a positive integer that evenly divides"
                "circle_resolution_angular."
            )

        if local_2d_obs and not use_marl:
            raise ValueError(
                "Local 2D observations are only supported in multi-agent mode."
            )

        self._local_2d_obs = local_2d_obs
        self._n_agents = n_agents
        self._local_obs_window = local_obs_window
        self._local_reward_weight = local_reward_weight
        self._init_from_2d = init_from_2d

        # In case we already have vortex shedding in the 2D initial domain,
        # we need less initial steps to reach a realistic 3D flow field
        if init_from_2d:
            self._initial_domain_steps //= 2

        if local_2d_obs:
            self._logger.info(
                "Using 2D local observations (velocity in x and y directions only, a"
                "single sensor layer and window = 1)."
            )
            self._n_sensors_per_agent = 1
            self._local_obs_window = 1

        super().__init__(
            ndims=3,
            reynolds_number=reynolds_number,
            adaptive_cfl=adaptive_cfl,
            step_length=step_length,
            episode_length=episode_length,
            dt=dt,
            attack_angle_deg=attack_angle_deg,
            use_marl=use_marl,
            dtype=dtype,
            cuda_device=cuda_device,
            debug=debug,
            load_initial_domain=load_initial_domain,
            load_domain_statistics=load_domain_statistics,
            randomize_initial_state=randomize_initial_state,
            enable_actions=enable_actions,
            differentiable=differentiable,
        )

    def _get_action_space(self) -> spaces.Box:
        shape: tuple[int, ...]
        if self._use_marl:
            shape = (self._n_jets,)
        else:
            shape = (
                self._n_agents,
                self._n_jets,
            )

        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=shape,
            dtype=np.float32,
        )

    def _get_observation_space(self) -> spaces.Dict:
        n_sensors_x_y = self._sensor_locations.shape[-1]
        velocity_shape: tuple[int, ...]
        pressure_shape: tuple[int, ...]

        if self._use_marl:
            if self._local_2d_obs:
                velocity_shape = (
                    n_sensors_x_y,
                    self._ndims - 1,
                )
                pressure_shape = (n_sensors_x_y,)
            else:
                velocity_shape = (
                    self._local_obs_window,
                    self._n_sensors_per_agent,
                    self._ndims,
                    n_sensors_x_y,
                )
                pressure_shape = (
                    self._local_obs_window,
                    self._n_sensors_per_agent,
                    n_sensors_x_y,
                )
        else:
            velocity_shape = (
                self._n_agents,
                self._n_sensors_per_agent,
                self._ndims,
                n_sensors_x_y,
            )
            pressure_shape = (
                self._n_agents,
                self._n_sensors_per_agent,
                n_sensors_x_y,
            )

        return spaces.Dict(
            {
                "velocity": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=velocity_shape,
                    dtype=np.float32,
                ),
                "pressure": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=pressure_shape,
                    dtype=np.float32,
                ),
            }
        )

    @property
    def n_agents(self) -> int:
        """The number of agents in the environment."""
        return self._n_agents

    @property
    def _n_sensors_z(self) -> int:
        return self._n_agents * self._n_sensors_per_agent

    @property
    def _nz_per_agent(self) -> int:
        return self._res_z // self._n_agents

    @property
    def _cd_ref(self) -> float:
        if "drag" in self._metrics_stats:
            return self._metrics_stats["drag"].mean
        else:
            return 0.0

    @property
    def _cl_ref(self) -> float:
        if "abs_lift" in self._metrics_stats:
            return self._metrics_stats["abs_lift"].mean
        else:
            return 0.0

    def _get_sensor_locations(self) -> torch.Tensor:
        sensor_locations = self._get_sensor_locations_3d()
        grid_coords = self._physical_locations_to_grid_coords(sensor_locations)

        grid_coords = torch.stack(
            [
                grid_coords[0].reshape(-1, self._n_sensors_z).T,
                grid_coords[1].reshape(-1, self._n_sensors_z).T,
                grid_coords[2].reshape(-1, self._n_sensors_z).T,
            ]
        )

        all_sensors = []

        # Filter out airfoil mask
        for i in range(grid_coords.shape[2]):
            x_idx = grid_coords[0, :, i]
            y_idx = grid_coords[1, :, i]

            if self._airfoil_mask[0, y_idx, x_idx].any():
                continue

            all_sensors.append(grid_coords[:, :, i : i + 1])

        grid_coords = torch.cat(all_sensors, dim=2)

        return grid_coords

    def _get_sensor_locations_3d(self) -> torch.Tensor:
        sensor_locations_2d = self._get_sensor_locations_2d()
        sensor_z = torch.linspace(
            start=-self.H / 2,
            end=self.H / 2,
            steps=self._n_sensors_z + 1,
        )[:-1] + self.H / (2 * self._n_sensors_z)

        x = sensor_locations_2d[0].unsqueeze(0).expand(self._n_sensors_z, -1).T
        y = sensor_locations_2d[1].unsqueeze(0).expand(self._n_sensors_z, -1).T
        z = sensor_z.unsqueeze(1).expand(-1, sensor_locations_2d.shape[1]).T
        sensor_locations_3d = torch.stack([x, y, z], dim=0)

        return sensor_locations_3d

    def _get_global_obs(self) -> dict[str, torch.Tensor]:
        """Return the current observation."""
        return extract_global_3d_obs(
            env=self,
            sensor_locations=self._sensor_locations,
            n_agents=self._n_agents,
            n_sensors_per_agent=self._n_sensors_per_agent,
            n_sensors_z=self._n_sensors_z,
            local_2d_obs=self._local_2d_obs,
        )

    def _get_local_obs(self) -> dict[str, torch.Tensor]:
        global_obs = self._get_global_obs()
        return transform_global_to_local_obs_3d(
            global_obs=global_obs,
            local_obs_window=self._local_obs_window,
            n_agents=self._n_agents,
            local_2d_obs=self._local_2d_obs,
        )

    def _get_base_jet_profiles(self) -> torch.Tensor:
        base_profile_2d = super()._get_base_jet_profiles()

        # Insert zero z-velocity component
        zeros = torch.zeros_like(
            base_profile_2d[:, 0, :, :],
        )
        base_profile_3d = torch.stack(
            [base_profile_2d[:, 0, :, :], base_profile_2d[:, 1, :, :], zeros],
            dim=1,
        )

        # Repeat along z-axis for 3D
        base_profile_3d = base_profile_3d.unsqueeze(-2).repeat(1, 1, self._res_z, 1, 1)

        return base_profile_3d

    def _action_to_control(self, action: torch.Tensor) -> torch.Tensor:
        assert self._jet_locations_top is not None

        v_action = action - action.mean(dim=1, keepdim=True)

        # Ensure max abs. value of 1.0
        max_v = torch.max(torch.abs(v_action), dim=1, keepdim=True).values
        v_action = torch.where(max_v > 1.0, v_action / max_v, v_action)

        # Expand action to full z-resolution
        v_action = v_action.repeat_interleave(self._nz_per_agent, dim=0)
        top_profile = self._top_base_profile.clone()

        for i in range(self._n_jets):
            start_idx_top, end_idx_top = self._jet_locations_top[i]

            top_profile[
                0,
                :2,
                :,
                0,
                start_idx_top : end_idx_top + 1,
            ] *= v_action[:, i, None]

        return top_profile

    def _step_impl(
        self, action: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, bool, dict[str, torch.Tensor]]:
        obs, reward, term, info = super()._step_impl(action)

        all_cds = info.pop("drag")
        all_cls = info.pop("lift")

        # Sum over all cylinder cells to get total drag and lift
        cd = torch.sum(all_cds) / self.D
        cl = torch.sum(all_cls) / self.D

        reward = (cl / cd) - self._cl_cd_ref

        info["drag"] = cd
        info["lift"] = cl

        info["all_cds"] = all_cds
        info["all_cls"] = all_cls

        return obs, reward, term, info

    def _step_marl_impl(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, bool, dict[str, torch.Tensor]]:
        if self._local_reward_weight is None:
            raise ValueError("local_reward_weight must be set for multi-agent step.")

        _, global_reward, terminated, info = self._step_impl(actions)

        local_obs = self._get_local_obs()

        all_cds = info.pop("all_cds")
        lift = info.pop("all_cls")

        # First mean over agents cylinder cells
        local_cd = all_cds.view(self._n_agents, -1).sum(dim=1)
        local_cl = lift.view(self._n_agents, -1).sum(dim=1)

        local_cd = local_cd / (self.D / self._n_agents)
        local_cl = local_cl / (self.D / self._n_agents)

        local_rewards = (local_cl / local_cd) - self._cl_cd_ref
        agent_rewards = (
            self._local_reward_weight * local_rewards
            + (1 - self._local_reward_weight) * global_reward
        )
        info["global_reward"] = global_reward

        return local_obs, agent_rewards, terminated, info

    def _get_render_data(
        self,
        render_3d: bool,
        output_path: Path | None = None,
    ) -> dict[str, np.ndarray]:
        render_data = super()._get_render_data(
            render_3d=render_3d, output_path=output_path
        )

        curl = self.get_vorticity().squeeze(0)
        u = self.get_velocity().squeeze(0)

        u_magn: torch.Tensor = torch.linalg.norm(u, dim=0)
        u_arr = u_magn.detach().cpu().numpy()
        u_arr = u_arr.transpose(2, 1, 0)

        curl_magn: torch.Tensor = torch.linalg.norm(curl, dim=0)
        curl_arr = curl_magn.detach().cpu().numpy()
        curl_arr = curl_arr.transpose(2, 1, 0)

        if self._velocity_stats:
            # From our experiments, we have seen that these ranges work well
            # for vorticity rendering
            u_min = self._velocity_stats.min
            u_max = self._velocity_stats.mean * 0.6
        else:
            u_min = float(torch.min(u_magn).item())
            u_max = float(torch.max(u_magn).item())

        if render_3d:
            # We set vorticity at bottom and top to zero to make rendering cleaner
            curl_arr[:, :15, :] = 0.0
            curl_arr[:, -15:, :] = 0.0

            if output_path is not None:
                output_path = (
                    output_path / f"vorticity_{self._n_episodes}_{self._n_steps}.png"
                )

            iso_val = VORTICITY_RENDER_LEVELS[int(self._reynolds_number)]
            render_data["3d_vorticity"] = render_3d_iso(
                iso_field=curl_arr,
                iso=[iso_val],
                output_path=output_path,
                color_field=u_arr,
                color_range=(u_min, u_max),
                colormap="rainbow",
                extent=(
                    (-1.5, 4.5),
                    (
                        -self.H / 2,
                        self.H / 2,
                    ),
                    (
                        -self.D / 2,
                        self.D / 2,
                    ),
                ),
                view_kwargs={"elev": 10, "azim": 60},
                airfoil_coords=self._airfoil_coords.detach().cpu().numpy(),
            )

        return render_data

    def _get_domain(self) -> PISOtorch.Domain:
        domain_3d = super()._get_domain()
        if not self._init_from_2d:
            return domain_3d

        try:
            # Get random 2D initial domain
            idx = int(self._np_rng.integers(0, 10))
            domain_2d = self._load_2d_domain(EnvMode.TRAIN, idx=idx)
        except FileNotFoundError as e:
            self._logger.error(
                "2D initial domain not found on disk but attempting to init from 2D."
            )
            raise e

        assert domain_2d is not None

        for block_3d, block_2d in zip(
            domain_3d.getBlocks(), domain_2d.getBlocks(), strict=False
        ):
            vel_2d = block_2d.velocity

            if block_3d.velocity[:, :2, 0, :, :].shape != vel_2d.shape:
                self._logger.error(
                    "Shape mismatch when initializing 3D domain from 2D:"
                    f"vel_3d: {block_3d.velocity[:, :2, 0, :, :].shape}, vel_2d:"
                    f" {vel_2d.shape}. Using 3D initial domain as fallback."
                )
                return domain_3d

            # Expand 2D velocity to 3D by adding zero z-component
            vel_3d_new = torch.zeros_like(block_3d.velocity)
            vel_3d_new[:, :2, :, :, :] = vel_2d.unsqueeze(2).expand(
                -1, -1, vel_3d_new.shape[2], -1, -1
            )
            vel_3d_new = vel_3d_new.contiguous()

            block_3d.setVelocity(vel_3d_new)

        return domain_3d

    def _load_2d_domain(self, mode: EnvMode, idx: int) -> PISOtorch.Domain:
        """Load the 2D initial domain from disk.

        Parameters
        ----------
        mode: EnvMode
            Environment mode ('train', 'val', 'test').

        idx: int
            Index of the initial domain to load.

        Returns
        -------
        PISOtorch.Domain
            The loaded domain.
        """
        out_dir = self._get_domain_dir(idx)
        original_dir = str(out_dir)
        original_dir = original_dir.replace("airfoil_3D", "airfoil_2D")
        original_dir = original_dir.replace("Re10000", "Re3000")

        out_dir = Path(original_dir)
        domain = load_domain(
            path=str(out_dir / mode.value),
            dtype=self._dtype,
            device=self._cuda_device,
            with_scalar=True,
        )
        return domain
