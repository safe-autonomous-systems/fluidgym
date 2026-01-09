"""Environment for 3D airfoil drag and lift reduction."""

from pathlib import Path

import numpy as np
import torch

from fluidgym.envs.airfoil.airfoil_env_base import AirfoilEnvBase
from fluidgym.envs.fluid_env import EnvMode
from fluidgym.envs.multi_agent_fluid_env import MultiAgentFluidEnv
from fluidgym.envs.util.visualization import render_3d_iso
from fluidgym.simulation.extensions import (
    PISOtorch,  # type: ignore[import-untyped,import-not-found]
)
from fluidgym.simulation.pict.util.domain_io import load_domain
from fluidgym.simulation.pict.util.output import _resample_block_data

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


class AirfoilEnv3D(AirfoilEnvBase, MultiAgentFluidEnv):
    """Environment for 3D airfoil drag and lift reduction.

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

    _n_sensors_per_agent: int = 1

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
            dtype=dtype,
            cuda_device=cuda_device,
            debug=debug,
            load_initial_domain=load_initial_domain,
            load_domain_statistics=load_domain_statistics,
            randomize_initial_state=randomize_initial_state,
            enable_actions=enable_actions,
            differentiable=differentiable,
        )

    @property
    def action_space_shape(self) -> tuple[int, ...]:
        """The shape of the action space."""
        return (
            self._n_agents,
            self._n_jets,
        )

    @property
    def observation_space_shape(self) -> tuple[int, ...]:
        """The shape of the observation space."""
        n_sensors_x_y = self._sensors_locations.shape[-1]
        n_sensors_total = n_sensors_x_y * self._n_agents * self._n_sensors_per_agent
        return (n_sensors_total * self._ndims,)

    @property
    def local_observation_space_shape(self) -> tuple[int, ...]:
        """The shape of the local observation space for each agent."""
        n_sensors_x_y = self._sensors_locations.shape[-1]
        if self._local_2d_obs:
            return (n_sensors_x_y * 2,)
        else:
            return (
                self._n_sensors_per_agent
                * self._local_obs_window
                * n_sensors_x_y
                * self._ndims,
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

    def _get_global_obs(self) -> torch.Tensor:
        """Return the current observation."""
        u_list = [block.velocity for block in self._domain.getBlocks()]

        u = _resample_block_data(
            u_list,
            self._sim.output_resampling_coords,
            self._sim.output_resampling_shape,
            self._ndims,
            fill_max_steps=self._sim.output_resampling_fill_max_steps,
        )

        if self._local_2d_obs:
            u = u[:, :2, :, :, :]
        u = u.squeeze()
        u = u.permute(1, 2, 3, 0)

        sensor_locations = self._sensors_locations.flatten(start_dim=1)
        u = u[
            sensor_locations[2],
            sensor_locations[1],
            sensor_locations[0],
            :,
        ]
        u = u.view(self._n_sensors_z, -1)
        u = u.view(self._n_agents, self._n_sensors_per_agent, -1)

        return u

    def _get_local_obs(self, global_obs: torch.Tensor) -> torch.Tensor:
        offset = self._local_obs_window // 2

        # First, shift the global obs to start with the first agents sensor
        # window at zero
        shifted_obs = torch.roll(global_obs, shifts=offset, dims=0)

        local_obs = []
        for _ in range(self._n_agents):
            window = shifted_obs[: self._local_obs_window].reshape(-1)
            local_obs += [window]

            shifted_obs = torch.roll(shifted_obs, shifts=-1, dims=0)

        return torch.stack(local_obs, dim=0)

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
    ) -> tuple[torch.Tensor, torch.Tensor, bool, dict[str, torch.Tensor]]:
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

    def reset_marl(
        self,
        seed: int | None = None,
        randomize: bool | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Reset the environment to the initial state for multiple agents.

        Parameters
        ----------
        seed: int | None
            The seed to use for random number generation. If None, the current seed is
            used.

        randomize: bool | None
            Whether to randomize the initial state. If None, the default behavior is
            used.

        Returns
        -------
        tuple[torch.Tensor, dict]
            A tuple containing the initial observations for all agents and an info
            dictionary.
        """
        _, info = self.reset(seed=seed, randomize=randomize)
        local_obs = self._get_local_obs(info.pop("full_obs"))
        return local_obs, info

    def step_marl(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool, bool, dict[str, torch.Tensor]]:
        """Take a step in the environment using the given actions for multiple agents.

        Parameters
        ----------
        actions: torch.Tensor
            The actions to take for each agent.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, bool, bool, dict[str, torch.Tensor]]
            A tuple containing the observations, rewards, terminated flag, truncated
            flag, and info dictionary.
        """
        if self._local_reward_weight is None:
            raise ValueError("local_reward_weight must be set for multi-agent step.")

        _, global_reward, terminated, truncated, info = self.step(actions)

        local_obs = self._get_local_obs(info.pop("full_obs"))

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

        return local_obs, agent_rewards, terminated, truncated, info

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
            domain_2d = self._load_2d_domain(EnvMode.TRAIN, idx=0)
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

            # Expand 2D velocity to 3D by adding zero z-component
            vel_3d_new = torch.randn_like(block_3d.velocity) * 0.01
            vel_3d_new[:, :2, :, :, :] += vel_2d.unsqueeze(2).expand(
                -1, -1, vel_3d_new.shape[2], -1, -1
            )
            vel_3d_new[:, 2, :, :, :] += (
                torch.randn_like(vel_3d_new[:, 2, :, :, :]) * 0.05
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
