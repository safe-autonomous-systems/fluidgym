"""Rayleigh-Bénard Convection (RBC) environment base class."""

from abc import abstractmethod
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

import fluidgym.simulation.pict.data.shapes as shapes
from fluidgym import config as global_config
from fluidgym.envs import FluidEnv
from fluidgym.simulation import Simulation
from fluidgym.simulation.extensions import (
    PISOtorch,  # type: ignore[import-untyped,import-not-found]
)
from fluidgym.simulation.helpers import get_cell_size
from fluidgym.simulation.pict.util.output import _resample_block_data


class RBCEnvBase(FluidEnv):
    """Abstract base class for Rayleigh-Bénard Convection (RBC)
    environments.

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
    [1] C. Vignon, J. Rabault, J. Vasanth, F. Alcántara-Ávila,
    M. Mortensen, and R. Vinuesa, “Effective control of two-dimensional
    Rayleigh-Bénard convection: Invariant multi-agent reinforcement
    learning is all you need,” Physics of Fluids, vol. 35, no. 6,
    p. 065146, June 2023, doi: 10.1063/5.0153181.

    [2] J. Vasanth, J. Rabault, F. Alcántara-Ávila, M. Mortensen, and
    R. Vinuesa, “Multi-agent Reinforcement Learning for the Control of
    Three-Dimensional Rayleigh-Bénard Convection,” Flow, Turbulence and
    Combustion, Dec. 2024, doi: 10.1007/s10494-024-00619-2.
    """

    _default_render_key: str = "temperature"
    _supports_marl = True

    _T_cold: float = 0.0
    _T_hot: float = 1.0
    _heater_width: int
    _heater_limit: float = 0.75
    _n_sensors_y: int = 8
    _n_sensors_per_heater: int = 4
    _step_length: float = 1.0
    _resolution_scale_y: float = 2.0  # Twice the resolution in y-direction
    _non_uniform_grid_base = 1.02

    _H: float = 1.0
    _buoyancy_factor: float = 1.0

    _action_range: tuple[float, float] = (-1.0, 1.0)
    _observation_range: tuple[float, float] = (-2.5, 2.5)

    _metrics: list[str] = ["nusselt"]

    # Domain generation
    _initial_domain_restart = True

    def __init__(
        self,
        rayleigh_number: float,
        prandtl_number: float,
        n_heaters: int,
        resolution: int,
        adaptive_cfl: float,
        dt: float,
        step_length: float,
        episode_length: int,
        local_obs_window: int,
        local_reward_weight: float | None,
        uniform_grid: bool,
        aspect_ratio: float,
        use_marl: bool,
        dtype: torch.dtype = torch.float32,
        cuda_device: torch.device | None = None,
        load_initial_domain: bool = True,
        load_domain_statistics: bool = True,
        randomize_initial_state: bool = True,
        enable_actions: bool = True,
        differentiable: bool = False,
    ):
        self._rayleigh_number = rayleigh_number
        self._prandtl_number = prandtl_number
        self._heater_width = resolution
        self._n_heaters = n_heaters
        self._local_reward_weight = local_reward_weight
        self._local_obs_window = local_obs_window
        self._uniform_grid = uniform_grid

        super().__init__(
            dt=dt,
            adaptive_cfl=adaptive_cfl,
            step_length=step_length,
            episode_length=episode_length,
            ndims=self._ndims,
            dtype=dtype,
            use_marl=use_marl,
            cuda_device=cuda_device,
            load_initial_domain=load_initial_domain,
            load_domain_statistics=load_domain_statistics,
            randomize_initial_state=randomize_initial_state,
            enable_actions=enable_actions,
            differentiable=differentiable,
        )

        self._aspect_ratio = aspect_ratio * torch.pi

        self._x = int(resolution * self._n_heaters)
        self._y = round(self._resolution_scale_y * self._x / self._aspect_ratio)
        self._L = self._H * self._aspect_ratio

        self._kinematic_viscosity = torch.tensor(
            [(prandtl_number / rayleigh_number) ** 0.5], dtype=dtype
        )
        self._thermal_diffusivity = torch.tensor(
            [(rayleigh_number * prandtl_number) ** -0.5], dtype=dtype
        )

        self._sensor_locations = self._get_sensor_locations()

    def _get_domain(self) -> PISOtorch.Domain:
        grid = shapes.make_wall_refined_ortho_grid(
            self._x,
            self._y,
            corner_lower=(0, -self._H / 2),
            corner_upper=(self._L, self._H / 2),
            wall_refinement=["-y", "+y"],
            base=1.0 if self._uniform_grid else self._non_uniform_grid_base,
        )

        if self._ndims == 3:
            grid = shapes.extrude_grid_z(
                grid=grid,
                res_z=self._x,
                start_z=0.0,  # type: ignore
                end_z=self._L,  # type: ignore
                weights_z=None,
                exp_base=1,
            )

        grid = grid.cuda().contiguous()

        # create domain
        _domain = PISOtorch.Domain(
            self._ndims,
            self._kinematic_viscosity,
            passiveScalarChannels=1,
            name="RBCDomain",
            device=self._cuda_device,
            dtype=self._dtype,
        )
        _domain.setScalarViscosity(self._thermal_diffusivity.to(self._cuda_device))

        # create block from the mesh on the domain (missing settings, fields and
        # transformation metrics are created automatically)
        block = _domain.CreateBlock(vertexCoordinates=grid, name="RBCBlock")
        block.CloseBoundary("-y")
        block.CloseBoundary("+y")

        grad = torch.linspace(
            self._T_hot,
            self._T_cold,
            steps=self._y,
            dtype=self._dtype,
            device=self._cuda_device,
        )  # shape (y,)

        if self._ndims == 2:
            grad = grad[:, None].expand(self._y, self._x)
        else:
            grad = grad[None, :, None].expand(self._x, self._y, self._x)
        initial_temperature = grad.unsqueeze(0).unsqueeze(0)

        # add random perturbations
        initial_temperature = initial_temperature.clone() + torch.normal(
            mean=0.0,
            std=1.0,
            size=initial_temperature.shape,
            device=self._cuda_device,
            generator=self._torch_rng_cuda,
        ) * 0.1 * (self._T_hot - self._T_cold)
        initial_temperature = torch.clamp(
            initial_temperature, min=self._T_cold, max=self._T_hot
        )

        block.setPassiveScalar(initial_temperature.contiguous())

        initial_velocity = (
            torch.normal(
                mean=0.0,
                std=1.0,
                size=block.velocity.shape,
                device=self._cuda_device,
                generator=self._torch_rng_cuda,
            )
            * 0.05
        )
        block.setVelocity(initial_velocity.contiguous())

        block.getBoundary("-y").setPassiveScalar(
            torch.tensor([[self._T_hot]], dtype=self._dtype, device=self._cuda_device)
        )
        block.getBoundary("+y").setPassiveScalar(
            torch.tensor([[self._T_cold]], dtype=self._dtype, device=self._cuda_device)
        )

        # finalize domain
        _domain.PrepareSolve()
        return _domain

    def _get_prep_fn(self, domain: PISOtorch.Domain) -> dict[str, Any]:
        block = domain.getBlock(0)

        vel_src_velX_pad = torch.zeros_like(block.passiveScalar)

        def buoyancy_fn_2d(domain, time_step, **kwargs):
            T = block.passiveScalar
            source = torch.cat([vel_src_velX_pad, T * self._buoyancy_factor], dim=1)
            block.setVelocitySource(source)
            domain.UpdateDomainData()

        def buoyancy_fn_3d(domain, time_step, **kwargs):
            T = block.passiveScalar
            source = torch.cat(
                [vel_src_velX_pad, T * self._buoyancy_factor, vel_src_velX_pad], dim=1
            )
            block.setVelocitySource(source)
            domain.UpdateDomainData()

        prep_fn = {
            "PRE_VELOCITY_SETUP": [
                buoyancy_fn_2d if self._ndims == 2 else buoyancy_fn_3d,
            ],
        }
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
            adaptive_CFL=self._adaptive_cfl,
            dt=self._dt,
            corrector_steps=2,
            pressure_tol=1e-5,
            advect_non_ortho_steps=1,
            pressure_non_ortho_steps=1,
            pressure_return_best_result=True,
            velocity_corrector="FD",
            non_orthogonal=False,
            output_resampling_shape=self.render_shape[: self._ndims],
            output_resampling_fill_max_steps=16,
            differentiable=self._differentiable,
        )

        return sim

    def _additional_initialization(self) -> None:
        self._block = self._domain.getBlock(0)
        self._bottom_plate = self._block.getBoundary("-y")
        self._top_plate = self._block.getBoundary("+y")

    def _randomize_domain(self) -> None:
        velocity_noise = 0.05
        temperature_noise = 0.05

        T = self._block.passiveScalar
        u = self._block.getVelocity(False)

        # 1) Randomly flip along x-axis (and z-axis for 3D)
        if self._np_rng.uniform(0.0, 1.0) > 0.5:
            T = torch.flip(T, dims=[-1])
            u = torch.flip(u, dims=[-1])

            u[:, 0, ...] *= -1.0  # flip y-velocity

        if self._ndims == 3 and self._np_rng.uniform(0.0, 1.0) > 0.5:
            T = torch.flip(T, dims=[-3])
            u = torch.flip(u, dims=[-3])

            u[:, 2, ...] *= -1.0  # flip z-velocity

        # 2) Translate in x-direction (and z-direction for 3D)
        x_shift = int(self._np_rng.integers(0, self._x))
        T = torch.roll(T, shifts=x_shift, dims=-1)
        u = torch.roll(u, shifts=x_shift, dims=-1)
        if self._ndims == 3:
            z_shift = int(self._np_rng.integers(0, self._x))
            T = torch.roll(T, shifts=z_shift, dims=-3)
            u = torch.roll(u, shifts=z_shift, dims=-3)

        # 3) Add small random noise
        T += (
            torch.normal(
                mean=0.0,
                std=1.0,
                size=T.shape,
                device=self._cuda_device,
                generator=self._torch_rng_cuda,
            )
            * temperature_noise
        ).to(self._dtype)
        T = torch.clamp(T, min=self._T_cold, max=self._T_hot)
        u += (
            torch.normal(
                mean=0.0,
                std=1.0,
                size=u.shape,
                device=self._cuda_device,
                generator=self._torch_rng_cuda,
            )
            * velocity_noise
        ).to(self._dtype)

        self._block.setPassiveScalar(T)
        self._block.setVelocity(u)

        # 4) Run simulation for a random number of steps
        min_sim_time = 1.0
        max_sim_time = 2.0
        sim_time = self._np_rng.uniform(min_sim_time, max_sim_time)
        n_steps = int(sim_time / self._dt)
        for _ in range(n_steps):
            self._sim.single_step()

    @property
    def render_shape(self) -> tuple[int, ...]:
        """The shape of the rendered domain."""
        nx = self._n_heaters * 20
        height = round(nx / self._aspect_ratio)

        return (nx, height, nx)

    @property
    def nu_ref(self) -> float:
        """Reference Nusselt number for reward normalization."""
        if "nusselt" in self._metrics_stats:
            if self._ndims == 2:
                return self._metrics_stats["nusselt"].p50
            else:
                return self._metrics_stats["nusselt"].mean
        else:
            return 0.0

    @property
    def n_agents(self) -> int:
        """The number of agents in the environment."""
        if self._use_marl:
            if self._ndims == 2:
                return self._n_heaters
            else:
                return self._n_heaters**2
        else:
            return 1

    @property
    def _n_sensors_x(self) -> int:
        return self._n_heaters * self._n_sensors_per_heater

    @abstractmethod
    def _get_sensor_locations(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _get_local_obs(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _get_local_rewards(self) -> torch.Tensor:
        raise NotImplementedError

    def _get_sensor_locations_2d(self) -> torch.Tensor:
        """
        Get the locations of the sensors in the flow domain.

        Args:
            domain_shape (tuple): (nx, ny) size of the domain
            n_sensors (tuple): (n_sensors_x, n_sensors_y)

        Returns
        -------
            torch.Tensor of shape (2, n_sensors_x * n_sensors_y), dtype=torch.int32
        """
        nx, ny = self.render_shape[:-1]

        # sensor grid positions (exclude domain boundaries)
        sensor_x = torch.linspace(0, nx, self._n_sensors_x + 1)[:-1] + nx / (
            2 * self._n_sensors_x
        )
        sensor_y = torch.linspace(0, ny, self._n_sensors_y + 1)[:-1] + ny / (
            2 * self._n_sensors_y
        )

        grid_x, grid_y = torch.meshgrid(sensor_x, sensor_y, indexing="ij")
        sensor_locations = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2).T

        return sensor_locations.round().to(torch.int).to(self._cuda_device)

    def get_temperature(
        self,
    ) -> torch.Tensor:
        """Return the current temperature field resampled to the render shape.

        Returns
        -------
        torch.Tensor
            Temperature field resampled to the render shape.
        """
        T = _resample_block_data(
            [self._block.passiveScalar],
            self._sim.output_resampling_coords,
            self._sim.output_resampling_shape,
            self._ndims,
            fill_max_steps=self._sim.output_resampling_fill_max_steps,
        )
        return T.squeeze()

    def _compute_nusselt(
        self, T: torch.Tensor, u_y: torch.Tensor, cell_size: torch.Tensor
    ) -> torch.Tensor:
        is_batched = T.ndim == self._ndims + 1
        if is_batched:
            cell_size = cell_size.unsqueeze(0)
            sum_dim = tuple(range(1, self._ndims + 1))
        else:
            sum_dim = tuple(range(0, self._ndims))

        vol_mean_uy_T = (u_y * T * cell_size).sum(dim=sum_dim) / cell_size.sum(
            dim=sum_dim
        )
        Nu = (
            1.0
            + torch.sqrt(
                torch.as_tensor(self._rayleigh_number)
                * torch.as_tensor(self._prandtl_number)
            )
            * vol_mean_uy_T
        )

        return Nu

    def compute_global_nusselt(self) -> torch.Tensor:
        """Compute the global Nusselt number of the current state.

        Returns
        -------
        torch.Tensor
            The global Nusselt number.
        """
        T = self._block.passiveScalar[0]
        T = T.squeeze()

        u = self._block.getVelocity(False)
        u = u.squeeze()

        if self._ndims == 2:
            u = u.permute(1, 2, 0)
        elif self._ndims == 3:
            u = u.permute(1, 2, 3, 0)
        u_y = u[..., 1]

        cell_size = get_cell_size(self._block)
        cell_size = cell_size.squeeze()
        return self._compute_nusselt(
            T=T.unsqueeze(0), u_y=u_y.unsqueeze(0), cell_size=cell_size
        )

    def _get_render_data(
        self,
        render_3d: bool,
        output_path: Path | None = None,
    ) -> dict[str, np.ndarray]:
        T = self.get_temperature()

        min_val = self._T_cold
        max_val = self._T_hot + self._heater_limit
        T = (T - min_val) / (max_val - min_val)

        render_data = {}
        if self._ndims == 2:
            render_data["temperature"] = self._format_render_data(
                data=T.detach().cpu().numpy(), v_min=0, v_max=1.0, cmap="rainbow"
            )
            render_data["temperature"] = np.flipud(render_data["temperature"])
        else:
            T_xy = T[T.shape[0] // 2, :, :]
            T_xz = T[:, T.shape[1] // 2, :]
            T_yz = T[:, :, T.shape[2] // 2]

            render_data["x-y-temperature"] = self._format_render_data(
                data=T_xy.detach().cpu().numpy(), v_min=0, v_max=1.0, cmap="rainbow"
            )
            render_data["x-y-temperature"] = np.flipud(render_data["x-y-temperature"])
            render_data["x-z-temperature"] = self._format_render_data(
                data=T_xz.detach().cpu().numpy(), v_min=0, v_max=1.0, cmap="rainbow"
            )
            render_data["y-z-temperature"] = self._format_render_data(
                data=T_yz.detach().cpu().numpy(), v_min=0, v_max=1.0, cmap="rainbow"
            )
            render_data["y-z-temperature"] = render_data["y-z-temperature"].transpose(
                1, 0, 2
            )

        return render_data

    def _step_impl(
        self, action: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, bool, dict[str, torch.Tensor]]:
        if self._enable_actions:
            self._apply_action(action)
        for _ in range(self._n_sim_steps):
            self._sim.single_step()

        nu = self.compute_global_nusselt()
        obs = self._get_global_obs()

        reward = self.nu_ref - nu
        info = {
            "nusselt": nu[0],
        }

        return obs, reward, False, info

    @property
    def id(self) -> str:
        """Unique identifier for the environment."""
        return (
            f"RBC{self._ndims}d_Ra{self._rayleigh_number}_Pr{self._prandtl_number}"
            f"_NH{self._n_heaters}_HW{self._heater_width}"
        )

    @property
    def initial_domain_id(self) -> str:
        """Unique identifier for the initial domain."""
        return (
            f"rbc_{self._ndims}d_Ra{self._rayleigh_number}_Pr{self._prandtl_number}"
            f"_NH{self._n_heaters}_HW{self._heater_width}"
        )

    def _step_marl_impl(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool, dict[str, torch.Tensor]]:
        if self._local_reward_weight is None:
            raise ValueError("local_reward_weight must be set for multi-agent step.")

        _, global_reward, terminated, info = self._step_impl(actions)

        local_obs = self._get_local_obs()

        if self._local_reward_weight > 0:
            local_rewards = self._get_local_rewards()
        else:
            local_rewards = torch.zeros(
                (self.n_agents,), dtype=self._dtype, device=self._cuda_device
            )

        agent_rewards = (
            self._local_reward_weight * local_rewards
            + (1 - self._local_reward_weight) * global_reward
        )

        info["global_reward"] = global_reward

        return local_obs, agent_rewards, terminated, info

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

        sensor_x = torch.linspace(0, self._L, self._n_sensors_x + 1)[:-1] + self._L / (
            2 * self._n_sensors_x
        )
        sensor_y = (
            torch.linspace(0, self._H, self._n_sensors_y + 1)[:-1]
            + self._H / (2 * self._n_sensors_y)
            - self._H / 2
        )
        grid_x, grid_y = torch.meshgrid(sensor_x, sensor_y, indexing="ij")
        sensor_locations = (
            torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2).T.numpy()
        )

        colors = global_config.palette

        plt.figure(figsize=(10, 5))
        ax = plt.gca()

        # Plot sensors
        plt.scatter(
            sensor_locations[0],
            sensor_locations[1],
            marker="o",
            color=colors[2],
            s=5,
            label="Sensors",
        )

        # Plot heaters using vertical lines
        for i in range(1, self._n_heaters):
            heater_x = i * self._L / self._n_heaters
            plt.axvline(
                x=heater_x,
                color=colors[0],
                linestyle="--",
                label="Heater" if i == 0 else None,
            )

        plt.xlim(0, self._L)
        plt.ylim(-self._H / 2, self._H / 2)

        ax.set_yticks([-self._H / 2, 0, self._H / 2])
        ax.set_yticklabels([f"{-self._H / 2:.1f}", "0.0", f"{self._H / 2:.1f}"])

        ax.set_xticks([0, self._L])

        aspect = round(self._aspect_ratio / torch.pi)
        aspect_str = "" if aspect == 1 else str(aspect)
        ax.set_xticklabels(["0", aspect_str + r"$\pi$"])

        ax.set_xlabel("L")
        ax.set_ylabel("H")

        plt.savefig(output_path / f"{self.id}.pdf")
