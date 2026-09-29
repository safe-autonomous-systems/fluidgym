"""Environment for turbulent channel flow control."""

from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

from fluidgym.config import config as global_config
from fluidgym.envs.fluid_env import FluidEnv, Stats
from fluidgym.envs.tcf.grid import (
    get_van_driest_sqr,
    make_channel_flow_domain,
    set_dynamic_forcing,
)
from fluidgym.envs.util.obs_extraction import extract_moving_window_2d_x_z
from fluidgym.envs.util.visualization import (
    render_3d_iso,
)
from fluidgym.simulation.extensions import (
    PISOtorch,  # type: ignore[import-untyped,import-not-found]
)
from fluidgym.simulation.helpers import get_cell_centers, get_cell_size
from fluidgym.simulation.pict.data import TCF_tools
from fluidgym.simulation.pict.PISOtorch_simulation import (
    append_prep_fn,
)
from fluidgym.simulation.pict.util.output import _resample_block_data
from fluidgym.simulation.simulation import Simulation
from fluidgym.types import EnvMode

Q_CRITERION_ISOS = {
    np.pi / 2: {
        180: 0.05,
        330: 0.05,
        550: 0.05,
    },
    np.pi: {
        180: 0.05,
        330: 0.05,
        550: 0.05,
    },
}

VELOCITY_MAX = {
    np.pi / 2: {
        180: 0.9,
        330: 0.9,
        550: 0.9,
    },
    np.pi: {
        180: 0.9,
        330: 0.9,
        550: 0.9,
    },
}

SMALL_TCF_3D_DEFAULT_CONFIG = {
    "resolution_y": 65,
    "resolution_x_z": 64,
    "actor_size": 2,
    "L": np.pi,
    "D": np.pi / 2,
    "reynolds_number_wall": 180,
    "adaptive_cfl": 0.1,
    "step_length": 0.6,
    "episode_length": 1000,
    "local_obs_window": 1,
    "local_reward_weight": 0.0,
    "use_marl": True,
    "C_smag": 0.0,
    "use_van_driest": False,
    "init_with_noise": True,
    "dtype": torch.float32,
    "load_initial_domain": True,
    "load_domain_statistics": True,
    "randomize_initial_state": True,
    "enable_actions": True,
    "differentiable": False,
}

LARGE_TCF_3D_DEFAULT_CONFIG = {
    **SMALL_TCF_3D_DEFAULT_CONFIG,
    "resolution_x_z": 128,
    "L": 2 * np.pi,
    "D": np.pi,
}


class TCF3DBottomEnv(FluidEnv):
    """Environment for turbulent channel flow control.

    Parameters
    ----------
    resolution_y: int
        The resolution of the simulation grid in the wall-normal direction.

    resolution_x_z: int
        The resolution of the simulation grid in the streamwise and spanwise
        directions.

    L: float
        The length of the domain in the streamwise direction.

    D: float
        The length of the domain in the spanwise direction.

    actor_size: int
        The size of each actor region in grid cells.

    reynolds_number_wall: float
        The Reynolds number based on the wall shear velocity and half channel height.

    adaptive_cfl: float
        Target CFL number for adaptive time stepping.

    step_length: float
        The non-dimensional time length of each environment step.

    episode_length: int
        The number of steps per episode.

    local_obs_window: int
        The size of the local observation window for each agent.

    local_reward_weight: float
        The weight of the local reward in the total reward.

    use_marl: bool
        Whether to enable multi-agent reinforcement learning mode.

    C_smag: float
        The Smagorinsky constant for the LES model. If 0, no LES model is used.
        Defaults to 0.0.

    use_van_driest: bool
        Whether to use Van Driest damping for the LES model. Defaults to False.

    init_with_noise: bool
        Whether to initialize the velocity field with added noise. Defaults to True.

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
    [1] L. Guastoni, J. Rabault, P. Schlatter, H. Azizpour, and R. Vinuesa, “Deep
    reinforcement learning for turbulent drag reduction in channel flows,”
    Eur. Phys. J. E, vol. 46, no. 4, p. 27, Apr. 2023,
    doi: 10.1140/epje/s10189-023-00285-8.

    [2] Z. Zhao et al., “Physics-informed Neural-operator Predictive Control for Drag
    Reduction in Turbulent Flows,” Oct. 03, 2025, arXiv: arXiv:2510.03360.
    doi: 10.48550/arXiv.2510.03360.
    """

    _default_render_key: str = "3d_q_criterion"

    _actuation: str = "bottom"
    _supports_marl: bool = True

    # We need to be able to disable action scaling for opposition control
    _scale_actions: bool = True
    _action_smoothing_alpha: float = 0.1

    _delta: float = 1.0  # half channel height
    _H: float = 2.0 * _delta  # channel height
    _L: float
    _D: float

    _action_range = (-1.0, 1.0)
    _observation_range = (-2.5, 2.5)

    _y_obs_wall: float = 15.0  # See reference [1]

    _actor_size: int
    _local_obs_window: int

    _metrics: list[str] = ["wall_stress", "wall_stress_bottom", "wall_stress_top"]
    _vorticity_stats: Stats | None = None

    # Domain generation
    _initial_domain_ett: float = 50.0
    _initial_domain_restart: bool = False

    _wall_distance: float | None = None

    def __init__(
        self,
        resolution_y: int,
        resolution_x_z: int,
        L: float,
        D: float,
        actor_size: int,
        reynolds_number_wall: float,
        adaptive_cfl: float,
        step_length: float,
        episode_length: int,
        local_obs_window: int,
        local_reward_weight: float,
        use_marl: bool,
        C_smag: float = 0.0,
        use_van_driest: bool = False,
        init_with_noise: bool = True,
        dtype: torch.dtype = torch.float32,
        cuda_device: torch.device | None = None,
        debug: bool = False,
        load_initial_domain: bool = True,
        load_domain_statistics: bool = True,
        randomize_initial_state: bool = True,
        enable_actions: bool = True,
        differentiable: bool = False,
    ):
        self._L = L
        self._D = D

        self._debug = debug
        self._re_wall = reynolds_number_wall
        self._re_center: float = TCF_tools.Re_wall_to_cl(self._re_wall)
        self._viscosity = torch.tensor([self._delta / self._re_center], dtype=dtype)
        self._u_wall = self._re_wall / self._re_center
        self._x = resolution_x_z
        self._y = resolution_y
        self._z = resolution_x_z
        self._grid_refinement_strength = 2 if resolution_x_z < 64 else 1
        self._C_smag = C_smag
        self._use_van_driest = use_van_driest
        self._init_with_noise = init_with_noise
        self._actor_size = actor_size
        self._local_obs_window = local_obs_window
        self._local_reward_weight = local_reward_weight

        # We dt from wall units to physical units
        step_length = self._t_wall_to_t(step_length)

        # See https://doi.org/10.1017/jfm.2023.147
        dt = step_length / 10

        super().__init__(
            dt=dt,
            adaptive_cfl=adaptive_cfl,
            step_length=step_length,
            episode_length=episode_length,
            ndims=3,
            use_marl=use_marl,
            dtype=dtype,
            cuda_device=cuda_device,
            load_initial_domain=load_initial_domain,
            load_domain_statistics=load_domain_statistics,
            randomize_initial_state=randomize_initial_state,
            enable_actions=enable_actions,
            differentiable=differentiable,
        )
        self._viscosity = self._viscosity.to(self._cpu_device)

        target_t = TCF_tools.ETT_to_t(
            self._initial_domain_ett,
            self._u_wall,
            self._delta,  # type: ignore
        )
        self._initial_domain_steps = round(target_t / self._step_length)

        # For the small domain with Re = 180, we need more initial steps to reach a
        # turbulent state
        if self._L < 3.0 and self._re_wall < 330:
            self._initial_domain_steps *= 2

    @property
    def render_shape(self) -> tuple[int, ...]:
        """The shape of the rendered domain."""
        x_render_size = 2 * self._x
        y_render_size = int(x_render_size / self._L * self._H)
        z_render_size = int(x_render_size / self._L * self._D)
        return (x_render_size, y_render_size, z_render_size)

    def _get_domain(self) -> PISOtorch.Domain:
        domain = make_channel_flow_domain(
            H=self._H,
            L=self._L,
            D=self._D,
            x=self._x,
            y=self._y,
            z=self._z,
            refinement_strength=self._grid_refinement_strength,
            n_dims=self._ndims,
            u_wall=self._u_wall,
            viscosity=self._viscosity,
            cuda_device=self._cuda_device,
            init_with_noise=self._init_with_noise,
        )

        # finalize domain
        domain.PrepareSolve()
        return domain

    def _t_to_t_wall(self, t: float) -> float:
        return t / TCF_tools.t_star(self._viscosity.item(), self._u_wall)

    def _t_wall_to_t(self, t_wall: float) -> float:
        return t_wall * TCF_tools.t_star(self._viscosity.item(), self._u_wall)

    def _x_to_x_wall(self, pos: float) -> float:
        return pos * ((1 / self._viscosity.item()) * self._u_wall)

    def _x_wall_to_x(self, pos_wall: float) -> float:
        return pos_wall / ((1 / self._viscosity.item()) * self._u_wall)

    def _y_to_y_wall(self, pos: float) -> float:
        pos = pos + self._delta
        return pos * ((1 / self._viscosity.item()) * self._u_wall)

    def _y_wall_to_y(self, pos_wall: float) -> float:
        pos = pos_wall / ((1 / self._viscosity.item()) * self._u_wall)
        return -self._delta + pos

    def __get_y_obs_idx(self, y_wall: float) -> int:
        vertex_coords: torch.Tensor = self._domain.getVertexCoordinates()[0]
        cell_centers: torch.Tensor = get_cell_centers(vertex_coords)
        y_centers = cell_centers[1, 0, :, 0]

        y_obs = self._y_wall_to_y(y_wall)

        # find index of y center closedst to self._y_obs_slice
        center_distances = torch.abs(y_centers - y_obs)
        return int(torch.argmin(center_distances).item())

    @property
    def _n_actors_x(self) -> int:
        return self._x // self._actor_size

    @property
    def _n_actors_z(self) -> int:
        return self._z // self._actor_size

    @property
    def n_agents(self) -> int:
        """The number of agents in the environment."""
        return self._n_actors_x * self._n_actors_z

    def _get_action_space(self) -> spaces.Box:
        """Per-agent action space."""
        shape: tuple[int, ...]

        if self.use_marl:
            shape = (1,)
        else:
            shape = (
                self.n_agents,
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
        velocity_shape: tuple[int, ...]
        pressure_shape: tuple[int, ...]

        if self._use_marl:
            velocity_shape = (
                self._local_obs_window,
                self._local_obs_window,
                2,
            )
            pressure_shape = (
                self._local_obs_window,
                self._local_obs_window,
            )
        else:
            velocity_shape = (
                2,
                self._z,
                self._x,
            )
            pressure_shape = (
                self._z,
                self._x,
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
    def scale_actions(self) -> bool:
        r"""Whether actions are scaled by :math:`u_{\\mathrm{wall}}`."""
        return self._scale_actions

    @scale_actions.setter
    def scale_actions(self, value: bool) -> None:
        self._scale_actions = value

    def _get_prep_fn(self, domain: PISOtorch.Domain) -> dict[str, Any]:
        prep_fn: dict[str, Any] = {}
        set_dynamic_forcing(self._ndims, domain, prep_fn)

        if self._C_smag != 0:
            block = domain.getBlock(0)
            SGS_coefficient = torch.tensor(
                [self._C_smag], dtype=self._dtype, device=self._cpu_device
            )

            if self._use_van_driest:
                van_driest_scale_sqr = [
                    get_van_driest_sqr(block, domain, self._u_wall, self._cuda_device)
                ]

            def get_SGS_viscosity(domain):
                return PISOtorch.SGSviscosityIncompressibleSmagorinsky(
                    domain, SGS_coefficient
                )

            def add_block_SGS_viscosity(domain, **kwargs):
                domain.UpdateDomainData()

                SGS_viscosities = get_SGS_viscosity(domain)
                base_viscosity = domain.viscosity.to(self._cuda_device)

                for idx, (block, visc) in enumerate(
                    zip(domain.getBlocks(), SGS_viscosities, strict=True)
                ):
                    if self._use_van_driest:
                        visc = visc * van_driest_scale_sqr[idx]
                    visc = visc + base_viscosity
                    block.setViscosity(visc)

                domain.UpdateDomainData()

            append_prep_fn(prep_fn, "PRE", add_block_SGS_viscosity)

        return prep_fn

    def _get_simulation(
        self,
        domain: PISOtorch.Domain,
        prep_fn: dict[str, Any],
    ) -> Simulation:
        sim = Simulation(
            domain=domain,
            prep_fn=prep_fn,
            substeps="ADAPTIVE",
            dt=self._dt,
            corrector_steps=2,
            advection_use_BiCG=True,
            advection_tol=1e-6,
            pressure_tol=1e-6,
            adaptive_CFL=self._adaptive_cfl,
            advect_non_ortho_steps=1,
            pressure_non_ortho_steps=1,
            pressure_return_best_result=True,
            velocity_corrector="FD",
            non_orthogonal=True,
            output_resampling_shape=self.render_shape[: self._ndims],
            output_resampling_fill_max_steps=16,
            differentiable=self._differentiable,
        )

        sim.solver_double_fallback = False
        sim.preconditionBiCG = False
        sim.BiCG_precondition_fallback = True

        PISOtorch.CopyVelocityResultFromBlocks(self._domain)

        sim.make_divergence_free()

        return sim

    def _additional_initialization(self) -> None:
        self._block = self._domain.getBlock(0)
        self._bottom_plate = self._block.getBoundary("-y")
        self._inflow = self._block.getBoundary("-x")
        self._top_plate = self._block.getBoundary("+y")
        self._outflow = self._block.getBoundary("+x")
        self._y_obs_bottom_idx = self.__get_y_obs_idx(y_wall=self._y_obs_wall)

    def _action_to_control(self, action: torch.Tensor) -> torch.Tensor:
        # For opposition control, we do not scale the actions. For the RL case,
        # we ensure a zero mean and max abs. value of u_wall
        if self._scale_actions:
            # Ensure zero-net mass flux
            scaled_action = action - torch.mean(action)

            # Ensure max abs. value of u_wall
            scaled_action = (
                self._u_wall
                * scaled_action
                / (torch.clamp(scaled_action.abs(), min=1.0))
            )
            scaled_action -= torch.mean(scaled_action)
        else:
            scaled_action = action

        velocity_profile = torch.zeros(
            (1, 3, self._z, 1, self._x), device=self._cuda_device
        )
        v_expanded = scaled_action.repeat_interleave(
            self._actor_size, dim=0
        ).repeat_interleave(self._actor_size, dim=1)

        velocity_profile[0, 1, :, 0, :] = v_expanded.transpose(1, 0)

        return velocity_profile

    def _apply_action(self, action: torch.Tensor) -> None:
        """Apply the given action to the simulation."""
        reshaped_action = action.view(self._n_actors_x, self._n_actors_z)

        control = self._action_to_control(reshaped_action)
        self._bottom_plate.setVelocity(control)

    @property
    def tau_ref(self) -> float:
        """Reference bottom wall shear stress for normalization."""
        if "wall_stress_bottom" in self._metrics_stats:
            return self._metrics_stats["wall_stress_bottom"].mean
        else:
            return 1.0

    def _get_wall_stress(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the wall shear stress at both walls.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The wall shear stress at the bottom and top walls.
        """
        block = self._domain.getBlock(0)
        viscosity = self._domain.viscosity.to(self._domain.getDevice())

        pos_y = torch.mean(
            self._domain.getBlock(0).getCellCoordinates()[0, 1], dim=(0, 2)
        )
        d_y = (1 + pos_y[0].cpu().numpy(), 1 - pos_y[-1].cpu().numpy())

        mean_vel_u = torch.mean(block.velocity[0, 0], dim=(0, 2))
        tau_wall_bottom = viscosity * mean_vel_u[0] / d_y[0]
        tau_wall_top = viscosity * mean_vel_u[-1] / d_y[-1]

        return tau_wall_bottom, tau_wall_top

    def _get_q_criterion(self) -> torch.Tensor:
        """
        Compute the Q-criterion of the domain.

        Returns
        -------
        torch.Tensor
            The Q-criterion tensor.

        References
        ----------
        [1] J. Jeong and F. Hussain, “On the identification of a vortex,” Journal of
        Fluid Mechanics, vol. 285, pp. 69-94, 1995. doi:10.1017/S0022112095000462
        """
        self._domain.UpdateDomainData()
        gradients = PISOtorch.ComputeSpatialVelocityGradients(self._domain)
        d_dx, d_dy, d_dz = gradients[0]

        du_dx = d_dx[0, 0, ...]
        du_dy = d_dy[0, 0, ...]
        du_dz = d_dz[0, 0, ...]

        dv_dy = d_dy[0, 1, ...]
        dv_dx = d_dx[0, 1, ...]
        dv_dz = d_dz[0, 1, ...]

        dw_dx = d_dx[0, 2, ...]
        dw_dy = d_dy[0, 2, ...]
        dw_dz = d_dz[0, 2, ...]

        grad_u = torch.stack(
            [
                torch.stack([du_dx, du_dy, du_dz], dim=0),
                torch.stack([dv_dx, dv_dy, dv_dz], dim=0),
                torch.stack([dw_dx, dw_dy, dw_dz], dim=0),
            ],
            dim=0,
        )

        # Compute the symmetric and antisymmetric parts
        S = 0.5 * (grad_u + grad_u.transpose(1, 0))
        Omega = 0.5 * (grad_u - grad_u.transpose(1, 0))

        # Compute the Frobenius norms
        S_norm_sq = torch.sum(S**2, dim=(0, 1))
        Omega_norm_sq = torch.sum(Omega**2, dim=(0, 1))

        # Compute Q
        Q = 0.5 * (Omega_norm_sq - S_norm_sq)

        Q = _resample_block_data(
            data_list=[Q[None, None, ...]],
            vertex_coord_list=self._sim.output_resampling_coords,
            resampling_out_shape=self._sim.output_resampling_shape,
            ndims=self._ndims,
            fill_max_steps=self._sim.output_resampling_fill_max_steps,
        )

        return Q

    def _get_global_obs(self, y_idx: int | None = None) -> dict[str, torch.Tensor]:
        """Return the current observation."""
        if y_idx is None:
            y_idx = self._y_obs_bottom_idx

        u: torch.Tensor = self._domain.getBlock(0).getVelocity(False)
        u = u.squeeze()

        p: torch.Tensor = self._domain.getBlock(0).pressure
        p = p.squeeze()

        cell_size = get_cell_size(self._domain.getBlock(0))
        cell_size = cell_size.squeeze()

        # Compute spatial mean velocity
        mean_u = (u * cell_size[None, ...]).sum(
            dim=(1, 2, 3), keepdim=True
        ) / cell_size.sum()
        u_prime = u - mean_u

        u_slice = u_prime[
            :2,  # We only take u_x and u_y components
            :,
            y_idx,
            :,
        ]
        p_slice = p[:, y_idx, :]

        return {
            "velocity": u_slice,
            "pressure": p_slice,
        }

    def _get_render_data(
        self,
        render_3d: bool,
        output_path: Path | None = None,
    ) -> dict[str, np.ndarray]:
        y_wall = 150
        y = self._y_wall_to_y(y_wall)
        y_shape_idx = round((y + self._delta) / self._H * self.render_shape[1])

        q = self._get_q_criterion()
        q_arr = q.squeeze().detach().cpu().numpy()

        u = self.get_velocity()
        u = u.squeeze()
        u = torch.linalg.vector_norm(u, dim=0)

        # Flip x- and y-axis
        u = torch.flip(u, dims=[-2, -1])

        vorticity = self.get_vorticity()
        vorticity = vorticity.squeeze()

        # Flip x- and y-axis
        vorticity = torch.flip(vorticity, dims=[-2, -1])
        vorticity_arr = vorticity.detach().cpu().numpy()

        u_min = 0.0
        u_max = VELOCITY_MAX[self._D][int(self._re_wall)]

        format_velocity = partial(
            self._format_render_data,
            v_min=u_min,
            v_max=u_max,
            cmap="viridis",
        )

        if self._vorticity_stats:
            vort_min = self._vorticity_stats.min
            vort_max = self._vorticity_stats.p95
        else:
            vort_min = vorticity_arr.min().item()
            vort_max = vorticity_arr.max().item()

        # We take the max abs value for symmetric colormap
        abs_max = max(abs(vort_min), abs(vort_max))
        vort_min = -abs_max
        vort_max = abs_max

        format_vorticity = partial(
            self._format_render_data,
            v_min=vort_min,
            v_max=vort_max,
            cmap="icefire",
        )

        render_data = {}
        u_xy = u[u.shape[0] // 2, :, :]
        u_xz = u[:, y_shape_idx // 2, :]
        u_yz = u[:, :, u.shape[2] // 2]

        render_data["x-y-velocity"] = format_velocity(data=u_xy.detach().cpu().numpy())
        render_data["x-z-velocity"] = format_velocity(data=u_xz.detach().cpu().numpy())
        render_data["y-z-velocity"] = format_velocity(
            data=u_yz.T.detach().cpu().numpy()
        )

        vort_xy = vorticity_arr[2, vorticity_arr.shape[0] // 2, :, :]
        vort_xz = vorticity_arr[1, :, y_shape_idx // 2, :]
        vort_yz = vorticity_arr[0, :, :, vorticity_arr.shape[2] // 2]

        render_data["x-y-vorticity"] = format_vorticity(data=vort_xy)
        render_data["x-z-vorticity"] = format_vorticity(data=vort_xz)
        render_data["y-z-vorticity"] = format_vorticity(data=vort_yz.T)

        u_arr = u.detach().cpu().numpy()
        q_arr = q_arr.transpose(2, 1, 0)
        u_arr = u_arr.transpose(2, 1, 0)

        if render_3d:
            q_wall = q_arr[:, :y_shape_idx, :]
            u_wall = u_arr[:, :y_shape_idx, :]

            if output_path is not None:
                output_path = output_path / f"q_{self._n_episodes}_{self._n_steps}.png"

            q_iso_value = Q_CRITERION_ISOS[self._D][int(self._re_wall)]
            render_data["3d_q_criterion"] = render_3d_iso(
                iso_field=q_wall,
                iso=[q_iso_value],
                color_range=(u_min, u_max),
                output_path=output_path,
                color_field=u_wall,
                colormap="rainbow",
                extent=(
                    (0, self._x_to_x_wall(self._L)),
                    (self._y_to_y_wall(-self._delta), y_wall),
                    (0, self._x_to_x_wall(self._D)),
                ),
                view_kwargs={"elev": 15, "azim": 60},
            )

        return render_data

    def _get_reward(
        self, tau_total: torch.Tensor, tau_bottom: torch.Tensor
    ) -> torch.Tensor:
        # For the bottom case, we only consider the bottom wall stress
        return 1 - tau_bottom / self.tau_ref

    def _step_impl(
        self, action: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, bool, dict[str, torch.Tensor]]:
        flat_action = action.squeeze()

        if self._enable_actions:
            self._apply_action(flat_action)

        tau_top_list = []
        tau_bottom_list = []
        for _ in range(self._n_sim_steps):
            self._sim.single_step()
            _tau_bottom, _tau_top = self._get_wall_stress()

            tau_bottom_list += [_tau_bottom]
            tau_top_list += [_tau_top]

        # Average over simulation steps
        tau_bottom = torch.stack(tau_bottom_list).mean()
        tau_top = torch.stack(tau_top_list).mean()

        tau_total = 0.5 * (tau_bottom + tau_top)

        reward = self._get_reward(
            tau_total=tau_total,
            tau_bottom=tau_bottom,
        )

        obs = self._get_global_obs()

        info = {
            "wall_stress": tau_total,
            "wall_stress_bottom": tau_bottom,
            "wall_stress_top": tau_top,
        }

        return obs, reward, False, info

    def plot(self, output_path: Path | None = None) -> None:
        """Plot the environments configuration.

        Parameters
        ----------
        output_path: Path | None
            Path to save the plot. If None, the current directory is used. Defaults to
            None.
        """
        if output_path is None:
            output_path = Path(".")

        y_sensor = self._y_wall_to_y(self._y_obs_wall)

        colors = global_config.palette

        plt.figure(figsize=(10, 5))
        ax = plt.gca()

        # add vertical line for y_sensor as dotted line
        plt.axhline(y=y_sensor, color=colors[2], linestyle="dotted", linewidth=2)
        plt.axhline(y=-y_sensor, color=colors[2], linestyle="dotted", linewidth=2)

        plt.xlim(0, self._L)
        plt.ylim(-self._H / 2, self._H / 2)

        ax.set_yticks([-self._H / 2, 0, self._H / 2])
        ax.set_yticklabels([f"{-self._H / 2:.1f}", "0.0", f"{self._H / 2:.1f}"])

        ax.set_xticks([0, self._L])

        aspect = round(self._L / torch.pi)
        aspect_str = "" if aspect == 1 else str(aspect)
        ax.set_xticklabels(["0", aspect_str + r"$\pi$"])

        ax.set_xlabel("L")
        ax.set_ylabel("H")

        plt.savefig(output_path / f"{self.id}.pdf")

    @property
    def initial_domain_id(self) -> str:
        """Unique identifier for the initial domain."""
        return (
            f"channel_flow3D_L{self._L:.2f}_Re{int(self._re_wall)}_Res{self._x}"
            f"_Ref{self._grid_refinement_strength}"
        )

    @property
    def id(self) -> str:
        """Unique identifier for the environment."""
        return f"ChannelFlow3D_Re{int(self._re_wall)}_L{self._L:.2f}"

    def _randomize_domain(self) -> None:
        velocity_noise = 0.01
        pressure_noise = 0.01

        max_n_steps = int(0.01 * self._episode_length)
        n_steps = self._np_rng.integers(int(0.5 * max_n_steps), max_n_steps) + 1

        blocks = self._domain.getBlocks()
        for block in blocks:
            u = block.velocity
            p = block.pressure

            u += (
                torch.normal(
                    mean=0.0,
                    std=1.0,
                    size=u.shape,
                    device=self._cuda_device,
                    generator=self._torch_rng_cuda,
                )
                * velocity_noise
            )
            p += (
                torch.normal(
                    mean=0.0,
                    std=1.0,
                    size=p.shape,
                    device=self._cuda_device,
                    generator=self._torch_rng_cuda,
                )
                * pressure_noise
            )

            block.setVelocity(u)
            block.setPressure(p)

        for _ in range(n_steps):
            self._sim.single_step()

    def _get_local_obs(
        self, y_idx: int | None = None, flip_obs: bool = False
    ) -> dict[str, torch.Tensor]:
        if y_idx is None:
            y_idx = self._y_obs_bottom_idx

        u: torch.Tensor = self._domain.getBlock(0).getVelocity(False)
        u = u.squeeze()

        p: torch.Tensor = self._domain.getBlock(0).pressure
        p = p.squeeze()

        u_slice = u[
            :2,  # We only take u_x and u_y components
            :,
            y_idx,
            :,
        ]
        p_slice = p[:, y_idx, :]

        # Compute spatial mean velocity for the slice
        mean_u = u_slice.mean(dim=(1, 2), keepdim=True)
        u_prime = u_slice - mean_u

        u_x = u_prime[0, :, :]
        u_y = u_prime[1, :, :]

        local_obs_u_x = extract_moving_window_2d_x_z(
            field=u_x,
            n_agents_x=self._n_actors_x,
            n_agents_z=self._n_actors_z,
            agent_width=self._actor_size,
            n_agents_per_window_x=self._local_obs_window,
            n_agents_per_window_z=self._local_obs_window,
            pad_x=self._local_obs_window - 1,
            pad_z=self._local_obs_window // 2,
        )
        if flip_obs:
            local_obs_u_x = torch.flip(local_obs_u_x, dims=[2])

        local_obs_u_y = extract_moving_window_2d_x_z(
            field=u_y,
            n_agents_x=self._n_actors_x,
            n_agents_z=self._n_actors_z,
            agent_width=self._actor_size,
            n_agents_per_window_x=self._local_obs_window,
            n_agents_per_window_z=self._local_obs_window,
            pad_x=self._local_obs_window,
            pad_z=self._local_obs_window // 2,
        )

        # This is for the top wall observation to have a consistent orientation
        if flip_obs:
            local_obs_u_y = torch.flip(local_obs_u_y, dims=[2])
            local_obs_u_y *= -1

        local_obs_p = extract_moving_window_2d_x_z(
            field=p_slice,
            n_agents_x=self._n_actors_x,
            n_agents_z=self._n_actors_z,
            agent_width=self._actor_size,
            n_agents_per_window_x=self._local_obs_window,
            n_agents_per_window_z=self._local_obs_window,
            pad_x=self._local_obs_window,
            pad_z=self._local_obs_window // 2,
        )
        if flip_obs:
            local_obs_p = torch.flip(local_obs_p, dims=[1])

        local_obs_u = torch.stack((local_obs_u_x, local_obs_u_y), dim=-1)

        return {
            "velocity": local_obs_u,
            "pressure": local_obs_p,
        }

    def _step_marl_impl(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, bool, dict[str, torch.Tensor]]:
        if self._local_reward_weight is None:
            raise ValueError("local_reward_weight must be set for multi-agent step.")

        _, global_reward, terminated, info = self._step_impl(actions)

        local_obs = self._get_local_obs()
        agent_rewards = global_reward * torch.ones(
            self.n_agents,
            device=global_reward.device,
            dtype=global_reward.dtype,
        )
        info["global_reward"] = global_reward

        return local_obs, agent_rewards, terminated, info

    def _load_domain_statistics(self) -> dict[str, dict[str, float]]:
        stats = super()._load_domain_statistics()
        self._vorticity_stats = Stats(**stats["vorticity_magnitude"])
        return stats

    def save_opposition_control_episode(
        self, idx: int, mode: EnvMode, df: pd.DataFrame
    ) -> None:
        """Save the opposition control episode data to a CSV file.

        Parameters
        ----------
        idx: int
            The index of the episode.

        mode: EnvMode
            The mode of the environment (e.g., training, evaluation).

        df: pd.DataFrame
            The DataFrame containing the episode data to save.
        """
        path = self._get_domain_dir(idx=idx)

        df.to_csv(
            path / f"{mode.value}_opposition_control_{self._actuation}_episode.csv",
            index=False,
        )

    def load_opposition_control_episode(self, idx: int, mode: EnvMode) -> pd.DataFrame:
        """Load the opposition control episode data from a CSV file.

        Parameters
        ----------
        idx: int
            The index of the episode.

        mode: EnvMode
            The mode of the environment (e.g., training, evaluation).

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the episode data.
        """
        path = self._get_domain_dir(idx=idx)

        df = pd.read_csv(
            path / f"{mode.value}_opposition_control_{self._actuation}_episode.csv"
        )

        return df


class TCF3DBothEnv(TCF3DBottomEnv):
    """Environment for turbulent channel flow control with both walls actuated.

    The first half of the agents control the bottom wall, while the second half control
    the top wall.

    References
    ----------
    [1] T. R. Bewley, P. Moin, and R. Temam, “DNS-based predictive
    control of turbulence: an optimal benchmark for feedback
    algorithms,” J. Fluid Mech., vol. 447, pp. 179-225, Nov. 2001,
    doi: 10.1017/S0022112001005821.
    """

    _actuation: str = "both"

    def _get_observation_space(self) -> spaces.Dict:
        """Per-agent observation space."""
        velocity_shape: tuple[int, ...]
        pressure_shape: tuple[int, ...]

        if self._use_marl:
            velocity_shape = (
                self._local_obs_window,
                self._local_obs_window,
                2,
            )
            pressure_shape = (
                self._local_obs_window,
                self._local_obs_window,
            )
        else:
            velocity_shape = (
                2,
                2,
                self._z,
                self._x,
            )
            pressure_shape = (
                2,
                self._z,
                self._x,
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
        return 2 * super().n_agents

    @property
    def tau_ref(self) -> float:
        """Reference overall wall shear stress for normalization."""
        if "wall_stress" in self._metrics_stats:
            return self._metrics_stats["wall_stress"].mean
        else:
            return 1.0

    def _additional_initialization(self) -> None:
        super()._additional_initialization()
        self._y_obs_top_idx = self._y - self._y_obs_bottom_idx

    def _apply_action(self, action: torch.Tensor) -> None:
        action_bottom = action[: self.n_agents // 2]
        action_top = action[self.n_agents // 2 :]

        action_bottom = action_bottom.view(self._n_actors_x, self._n_actors_z)
        action_top = action_top.view(self._n_actors_x, self._n_actors_z)

        control_bottom = self._action_to_control(action_bottom)
        control_top = -1 * self._action_to_control(action_top)

        self._bottom_plate.setVelocity(control_bottom)
        self._top_plate.setVelocity(control_top)

    def _get_reward(
        self, tau_total: torch.Tensor, tau_bottom: torch.Tensor
    ) -> torch.Tensor:
        # We take both the stress at both plates for the reward
        return 1 - tau_total / self.tau_ref

    def _get_local_rewards(self, full_wall_stress: torch.Tensor) -> torch.Tensor:
        # We take both the stress at both plates for local rewards
        return 1 - full_wall_stress.flatten() / self.tau_ref

    def _get_global_obs(self, y_idx: int | None = None) -> dict[str, torch.Tensor]:
        """Return the current observation."""
        bottom_obs = super()._get_global_obs(y_idx=self._y_obs_bottom_idx)
        top_obs = super()._get_global_obs(y_idx=self._y_obs_top_idx)

        full_obs = {
            "velocity": torch.stack(
                (bottom_obs["velocity"], top_obs["velocity"]), dim=0
            ),
            "pressure": torch.stack(
                (bottom_obs["pressure"], top_obs["pressure"]), dim=0
            ),
        }

        return full_obs

    def _get_local_obs(
        self, y_idx: int | None = None, flip_obs: bool = False
    ) -> dict[str, torch.Tensor]:
        bottom_obs = super()._get_local_obs(
            y_idx=self._y_obs_bottom_idx, flip_obs=False
        )
        top_obs = super()._get_local_obs(y_idx=self._y_obs_top_idx, flip_obs=True)

        full_obs = {
            "velocity": torch.cat((bottom_obs["velocity"], top_obs["velocity"]), dim=0),
            "pressure": torch.cat((bottom_obs["pressure"], top_obs["pressure"]), dim=0),
        }
        return full_obs
