"""3D Environment for flow around a cylinder with jet actuation."""

from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from fluidgym.envs.cylinder.cylinder_env_base import CylinderEnvBase
from fluidgym.envs.util.obs_extraction import extract_global_3d_obs
from fluidgym.envs.util.profiles import get_jet_profile
from fluidgym.envs.util.visualization import render_3d_iso
from fluidgym.simulation.pict.PISOtorch_simulation import balance_boundary_fluxes

VORTICITY_RENDER_LEVELS = {
    100: 1.5,
    250: 2.5,
    500: 3.5,
}


CYLINDER_JET_3D_DEFAULT_CONFIG = {
    "n_jets": 8,
    "reynolds_number": 1e2,
    "resolution": 24,
    "dt": 1e-2,
    "adaptive_cfl": 0.8,
    "step_length": 0.25,
    "lift_penalty": 1.0,
    "episode_length": 80,
    "local_obs_window": 3,
    "local_reward_weight": 0.8,  # Based on doi.org/10.1038/s44172-025-00446-x
    "local_2d_obs": False,
    "use_marl": False,
    "dtype": torch.float32,
    "load_initial_domain": True,
    "load_domain_statistics": True,
    "randomize_initial_state": True,
    "enable_actions": True,
    "differentiable": False,
}


class CylinderJetEnv3D(CylinderEnvBase):
    """3D Environment for flow around a cylinder with jet actuation.

    This environment extends the 2D jet cylinder environment to 3D, allowing for
    multiple jet actuators distributed around the cylinder. Each jet can be controlled
    independently, enabling multi-agent reinforcement learning scenarios.

    Parameters
    ----------
    n_jets: int
        The number of jet actuators (or agents in case of MARL) distributed along
        the cylinder.

    reynolds_number: float
        The Reynolds number for the simulation.

    circle_resolution_angular: int
        The angular resolution of the cylinder boundary.

    dt: float
        The time step size for the simulation.

    adaptive_cfl: float
        The adaptive CFL number for time step adjustment.

    step_length: float
        The physical time duration of each environment step.

    episode_length: int
        The number of steps per episode.

    lift_penalty: float
        The penalty factor for lift in the reward calculation.

    local_reward_weight: float | None, optional
        Weighting factor for local rewards in multi-agent settings.
        Has to be set for multi-agent RL. Defaults to None.

    local_2d_obs: bool, optional
        Whether to use 2D local observations (velocity in x and y directions only).
        Defaults to False.

    use_marl: bool
        Whether to enable multi-agent reinforcement learning mode.

    dtype: torch.dtype, optional
        The data type for the simulation tensors. Defaults to torch.float32.

    load_initial_domain: bool, optional
        Whether to load the initial domain from file. Defaults to True.

    load_domain_statistic: bool, optional
        Whether to load precomputed domain statistics. Defaults to True.

    randomize_initial_state : bool, optional
        Whether to randomize the initial state of the simulation. Defaults to False.

    enable_actions: bool, optional
        Whether to enable action application in the environment. Defaults to True.

    differentiable: bool, optional
        Whether to enable differentiable simulation. Defaults to False.

    References
    ----------
    [1] P. Suárez et al., “Active Flow Control for Drag Reduction Through Multi-agent
    Reinforcement Learning on a Turbulent Cylinder at $$Re_D=3900$$,” Flow Turbulence
    Combust, vol. 115, no. 1, pp. 3-27, June 2025, doi: 10.1007/s10494-025-00642-x.

    [2] P. Suárez et al., “Flow control of three-dimensional cylinders transitioning to
    turbulence via multi-agent reinforcement learning,” Commun Eng, vol. 4, no. 1,
    p. 113, June 2025, doi: 10.1038/s44172-025-00446-x.
    """

    _default_render_key: str = "3d_vorticity"

    _jet_angle: float = 10.0  # degrees
    _n_sensors_per_agent: int = 2
    _supports_marl: bool = True

    def __init__(
        self,
        n_jets: int,
        reynolds_number: float,
        resolution: int,
        dt: float,
        adaptive_cfl: float,
        step_length: float,
        episode_length: int,
        lift_penalty: float,
        local_obs_window: int,
        use_marl: bool,
        local_reward_weight: float | None,
        local_2d_obs: bool = False,
        dtype: torch.dtype = torch.float32,
        cuda_device: torch.device | None = None,
        load_initial_domain: bool = True,
        load_domain_statistics: bool = True,
        randomize_initial_state: bool = True,
        enable_actions: bool = True,
        differentiable: bool = False,
    ):
        if n_jets < 1 or resolution % n_jets != 0:
            raise ValueError(
                "n_agents must be a positive integer that evenly divides"
                "circle_resolution_angular."
            )

        if local_2d_obs and not use_marl:
            raise ValueError(
                "Local 2D observations are only supported in multi-agent mode."
            )

        self._local_2d_obs = local_2d_obs
        self._n_jets = n_jets
        self._local_obs_window = local_obs_window
        self._local_reward_weight = local_reward_weight

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
            resolution=resolution,
            dt=dt,
            adaptive_cfl=adaptive_cfl,
            step_length=step_length,
            episode_length=episode_length,
            lift_penalty=lift_penalty,
            use_marl=use_marl,
            dtype=dtype,
            cuda_device=cuda_device,
            load_initial_domain=load_initial_domain,
            load_domain_statistics=load_domain_statistics,
            randomize_initial_state=randomize_initial_state,
            enable_actions=enable_actions,
            differentiable=differentiable,
        )

    def _get_action_space(self) -> spaces.Box:
        shape: tuple[int, ...]

        if self._use_marl:
            shape = (1,)
        else:
            shape = (
                self._n_jets,
                1,
            )

        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=shape,
            dtype=np.float32,
        )

    def _get_observation_space(self) -> spaces.Dict:
        velocity_shape: tuple[int, ...]
        pressure_shape: tuple[int, ...]

        if self._use_marl:
            if self._local_2d_obs:
                velocity_shape = (
                    self._n_sensors_x_y,
                    self._ndims - 1,
                )
                pressure_shape = (self._n_sensors_x_y,)
            else:
                velocity_shape = (
                    self._local_obs_window,
                    self._n_sensors_per_agent,
                    self._ndims,
                    self._n_sensors_x_y,
                )
                pressure_shape = (
                    self._local_obs_window,
                    self._n_sensors_per_agent,
                    self._n_sensors_x_y,
                )
        else:
            velocity_shape = (
                self._n_jets,
                self._n_sensors_per_agent,
                self._ndims,
                self._n_sensors_x_y,
            )
            pressure_shape = (
                self._n_jets,
                self._n_sensors_per_agent,
                self._n_sensors_x_y,
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
        if self.use_marl:
            return self._n_jets
        else:
            return 1

    @property
    def _n_sensors_z(self) -> int:
        return self._n_jets * self._n_sensors_per_agent

    def _additional_initialization(self) -> None:
        super()._additional_initialization()
        (self._top_velocity, self._bottom_velocity, self._nz_per_agent) = (
            self.__get_boundary_velocities()
        )

    def _get_sensor_locations(self) -> torch.Tensor:
        sensor_locations = self._get_sensor_locations_3d()
        grid_coords = self._sensor_locations_to_grid_coords(sensor_locations)

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

    def _get_global_obs(self) -> dict[str, torch.Tensor]:
        return extract_global_3d_obs(
            env=self,
            sensor_locations=self._sensor_locations,
            n_agents=self._n_jets,
            n_sensors_per_agent=self._n_sensors_per_agent,
            n_sensors_z=self._n_sensors_z,
            local_2d_obs=self._local_2d_obs,
        )

    def _get_local_obs(self) -> dict[str, torch.Tensor]:
        global_obs = self._get_global_obs()
        offset = self._local_obs_window // 2

        local_obs = {}
        for k, v in global_obs.items():
            # First, shift the global obs to start with the first agents sensor
            # window at zero
            shifted_obs = torch.roll(v, shifts=offset, dims=0)

            local_obs_list = []
            for _ in range(self._n_jets):
                window = shifted_obs[: self._local_obs_window]

                if self._local_2d_obs:
                    window = window.squeeze()

                local_obs_list += [window]

                shifted_obs = torch.roll(shifted_obs, shifts=-1, dims=0)

            local_obs[k] = torch.stack(local_obs_list, dim=0)

        return local_obs

    def __get_boundary_velocities(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        def coords_to_velocities(
            coords_boundary: torch.Tensor, direction: str
        ) -> torch.Tensor:
            centers = 0.5 * (coords_boundary[:, :-1] + coords_boundary[:, 1:])

            if direction == "top":
                angles = torch.pi / 2 - torch.atan2(centers[1, :], centers[0, :])
            else:
                angles = -torch.pi / 2 - torch.atan2(centers[1, :], centers[0, :])

            angles_deg = torch.rad2deg(angles)
            angles_deg_abs = torch.abs(angles_deg)

            angles_deg_abs[angles_deg_abs > self._jet_angle] = 0.0
            min_idx, max_idx = torch.where(angles_deg_abs > 0.0)[0][[0, -1]]
            min_idx = min_idx - 1
            max_idx = max_idx + 1

            profile_width = max_idx - min_idx + 1
            u_profile = get_jet_profile(
                h=int(profile_width), dtype=self._dtype, device=self._cuda_device
            )

            velocities = torch.zeros_like(centers)
            for i, u_magn in zip(range(min_idx, max_idx + 1), u_profile, strict=False):
                angle = angles_deg[i]
                velocities[0, i] = u_magn * torch.sin(torch.deg2rad(angle))
                velocities[1, i] = u_magn * torch.cos(torch.deg2rad(angle))

            return velocities

        vertex_coords = self._domain.getVertexCoordinates()
        top_coords = vertex_coords[self._top_block_idx]
        bottom_coords = vertex_coords[self._bottom_block_idx]

        # We set z = 0
        top_coords_boundary = top_coords[0, :, 0, 0, :]
        bottom_coords_boundary = bottom_coords[0, :, 0, -1, :]

        top_velocities = coords_to_velocities(top_coords_boundary, direction="top")
        bottom_velocities = coords_to_velocities(
            bottom_coords_boundary, direction="bottom"
        )

        # We insert z dimension
        n_z = top_coords.shape[2] - 1
        top_velocities = top_velocities[None, :, None, None, :].repeat(1, 1, n_z, 1, 1)
        bottom_velocities = bottom_velocities[None, :, None, None, :].repeat(
            1, 1, n_z, 1, 1
        )

        return top_velocities, bottom_velocities, n_z // self._n_jets

    def _apply_action(self, action: torch.Tensor) -> None:
        """Apply the given action to the simulation.

        Parameters
        ----------
        action: torch.Tensor
            The actions to apply for each jet actuator.
        """
        action = action.flatten()

        # We need to repeat the action in z-direction, having blocks of nz_per_agent
        # values per agent
        action = action.repeat_interleave(self._nz_per_agent, dim=0)
        assert action.shape[0] == self._top_velocity.shape[2]

        # Then we expand dimensions to match the velocity shape
        action = action[None, None, :, None, None]

        self._top_boundary.setVelocity(self._top_velocity.clone() * action)
        self._bottom_boundary.setVelocity(self._bottom_velocity.clone() * action)
        out_bounds = [
            self._domain.getBlock(self._top_block_idx).getBoundary("-y"),
            self._domain.getBlock(self._bottom_block_idx).getBoundary("+y"),
            self._domain.getBlock(self._vortex_street_block_idx).getBoundary("+x"),
        ]
        balance_boundary_fluxes(self._domain, out_bounds, tol=1e-7)

    @property
    def id(self) -> str:
        """Unique identifier for the environment."""
        return f"JetCylinder3D_Re{self._reynolds_number}"

    def _step_impl(
        self, action: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, bool, dict[str, torch.Tensor]]:
        obs, reward, term, info = super()._step_impl(action)

        all_cds = info.pop("drag")
        all_cls = info.pop("lift")

        # Sum over all cylinder cells to get total drag and lift
        cd = torch.sum(all_cds) / self.D
        cl = torch.sum(all_cls) / self.D

        reward = self._cd_ref - cd - self._lift_penalty * torch.abs(cl)

        # To obtain the 3D drag and lift coefficients, we have to divide by the depth of
        # the domain
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
        local_cd = all_cds.view(self._n_jets, -1).sum(dim=1)
        local_cl = lift.view(self._n_jets, -1).sum(dim=1)

        local_cd = local_cd / (self.D / self._n_jets)
        local_cl = local_cl / (self.D / self._n_jets)

        local_rewards = (
            self._cd_ref - local_cd - self._lift_penalty * torch.abs(local_cl)
        )
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
            iso_val = VORTICITY_RENDER_LEVELS[int(self._reynolds_number)]

            # We set vorticity at bottom, top and in front of the cylinder
            #  to zero to make rendering cleaner
            curl_arr[:, :15, :] = 0.0
            curl_arr[:, -15:, :] = 0.0
            curl_arr[:20, :, :] = 0.0

            if output_path is not None:
                output_path = (
                    output_path / f"vorticity_{self._n_episodes}_{self._n_steps}.png"
                )

            render_data["3d_vorticity"] = render_3d_iso(
                iso_field=curl_arr,
                iso=[iso_val],
                output_path=output_path,
                color_field=u_arr,
                color_range=(u_min, u_max),
                colormap="rainbow",
                extent=(
                    (-2 * self.cylinder_diameter, self.L - 2 * self.cylinder_diameter),
                    (
                        -self.H / 2 + self.cylinder_offset_y,
                        self.H / 2 + self.cylinder_offset_y,
                    ),
                    (0, self.H),
                ),
                view_kwargs={"elev": 10, "azim": 60},
                cylinder_kwargs={
                    "radius": 0.5,
                    "center_x": 0.0,
                    "center_y": 0.0,
                },
            )

        return render_data
