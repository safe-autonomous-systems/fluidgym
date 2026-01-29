"""Abstract base class for cylinder flow environments (van Karman vortex street)."""

from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium import spaces

from fluidgym import config as fluidgym_config
from fluidgym.envs.cylinder.grid import make_vortex_street_domain
from fluidgym.envs.fluid_env import EnvState, FluidEnv, Stats
from fluidgym.envs.util.forces import (
    collect_boundary_coords,
    collect_boundary_fields,
    compute_forces_2d,
    compute_forces_3d,
    wall_distance_from_vertices,
)
from fluidgym.envs.util.obs_extraction import extract_global_2d_obs
from fluidgym.simulation.extensions import (
    PISOtorch,  # type: ignore[import-untyped,import-not-found]
)
from fluidgym.simulation.pict.PISOtorch_simulation import (
    update_advective_boundaries,
)
from fluidgym.simulation.simulation import Simulation

VORTICITY_RENDER_RANGE = (-10, 10)


class CylinderEnvBase(FluidEnv, ABC):
    """Abstract base class for cylinder flow environments (van Karman vortex street).

    Parameters
    ----------
    reynolds_number: float
        Reynolds number of the flow.

    resolution: int
        Number of grid cells along the circumference of the cylinder.

    dt: float
        Time step size for the simulation.

    adaptive_cfl: float
        Target CFL number for adaptive time stepping.

    step_length: float
        Physical time length of one environment step.

    episode_length: int
        Number of steps per episode.

    ndims: int
        Number of spatial dimensions (2 or 3).

    lift_penalty: float
        Penalty factor for the lift coefficient in the reward computation.

    use_marl: bool
        Whether to enable multi-agent reinforcement learning mode.

    dtype: torch.dtype, optional
        Data type for the simulation. Defaults to torch.float32.

    debug: bool, optional
        Whether to enable debug mode. Defaults to False.

    load_initial_domain: bool, optional
        Whether to load a precomputed initial domain. Defaults to True.

    load_domain_statistics: bool, optional
        Whether to load precomputed domain statistics. Defaults to True.

    randomize_initial_state: bool, optional
        Whether to randomize the initial state of the environment. Defaults to
        False.

    enable_actions: bool, optional
        Whether to enable action application. If False, actions are ignored.
        Defaults to True.

    differentiable: bool, optional
        Whether to enable differentiable simulation. Defaults to False.

    Notes
    -----
    The environment simulates the flow around a cylinder in a 2D or 3D domain.
    The cylinder is placed in a channel with a parabolic inflow profile. The flow
    is governed by the incompressible Navier-Stokes equations. The Reynolds number
    can be specified to control the flow regime. The environment provides a set of
    sensors in the wake of the cylinder to observe the flow field. The action space
    consists of a single continuous action that is further defined in the subclasses.
    The reward is based on the drag and lift coefficients acting on the cylinder.

    References
    ----------
    [1] J. Rabault, M. Kuchta, A. Jensen, U. Réglade, and N. Cerardi,
    “Artificial neural networks trained through deep reinforcement learning discover
    control strategies for active flow control,” Journal of Fluid Mechanics,
    vol. 865, pp. 281-302, Apr. 2019, doi: 10.1017/jfm.2019.62.

    [2] F. Ren, J. Rabault, and H. Tang, “Applying deep reinforcement earning to active
    flow control in weakly turbulent conditions,” Physics of Fluids, vol. 33, no. 3,
    p. 037121, Mar. 2021, doi: 10.1063/5.0037371.

    [3] P. Suárez et al., “Active Flow Control for Drag Reduction Through Multi-agent
    Reinforcement Learning on a Turbulent Cylinder at $$Re_D=3900$$,” Flow Turbulence
    Combust, vol. 115, no. 1, pp. 3-27, June 2025, doi: 10.1007/s10494-025-00642-x.

    [4] P. Suárez et al., “Flow control of three-dimensional cylinders transitioning to
    turbulence via multi-agent reinforcement learning,” Commun Eng, vol. 4, no. 1,
    p. 113, June 2025, doi: 10.1038/s44172-025-00446-x.
    """

    _default_render_key: str = "vorticity"
    _action_smoothing_alpha: float = 0.1

    H: float = 4.1
    L: float = 22.0
    D: float = 4.0  # in case of 3D
    cylinder_diameter: float = 1.0
    _U_mean: float = 1.0
    cylinder_offset_y: float = 0.05
    _n_sensors_x_y: int = 151

    _vortex_street_refinement_base: float = 0.95

    _metrics: list[str] = ["drag", "lift"]
    _vorticity_stats: Stats | None = None

    # Domain generation
    _initial_domain_steps = 400
    _initial_domain_restart = False

    __wall_normals: torch.Tensor | None = None
    __wall_distances: torch.Tensor | None = None
    __wall_face_lengths: torch.Tensor | None = None

    def __init__(
        self,
        reynolds_number: float,
        resolution: int,
        dt: float,
        adaptive_cfl: float,
        step_length: float,
        episode_length: int,
        ndims: int,
        lift_penalty: float,
        use_marl: bool,
        dtype: torch.dtype = torch.float32,
        cuda_device: torch.device | None = None,
        debug: bool = False,
        load_initial_domain: bool = True,
        load_domain_statistics: bool = True,
        randomize_initial_state: bool = True,
        enable_actions: bool = True,
        differentiable: bool = False,
    ):
        self._reynolds_number = reynolds_number
        self._circle_resolution_angular = resolution
        self._lift_penalty = lift_penalty

        super().__init__(
            dt=dt,
            adaptive_cfl=adaptive_cfl,
            step_length=step_length,
            episode_length=episode_length,
            ndims=ndims,
            use_marl=use_marl,
            dtype=dtype,
            cuda_device=cuda_device,
            load_initial_domain=load_initial_domain,
            load_domain_statistics=load_domain_statistics,
            randomize_initial_state=randomize_initial_state,
            enable_actions=enable_actions,
            differentiable=differentiable,
        )

        self._debug = debug
        self._viscosity = torch.tensor(
            [self._U_mean / self._reynolds_number], device=self._cuda_device
        )

        (
            self._left_block_idx,
            self._top_block_idx,
            self._right_block_idx,
            self._bottom_block_idx,
            self._vortex_street_block_idx,
        ) = range(5)
        self.__last_control = torch.zeros(
            (1,), device=self._cuda_device, requires_grad=False
        )
        self._sensor_locations = self._get_sensor_locations()
        self._cylinder_mask = self._get_cylinder_mask()

    def _get_action_space(self) -> spaces.Box:
        """Per-agent action space."""
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

    def _get_observation_space(self) -> spaces.Dict:
        """Per-agent observation space."""
        return spaces.Dict(
            {
                "velocity": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        self._n_sensors_x_y,
                        self._ndims,
                    ),
                    dtype=np.float32,
                ),
                "pressure": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._n_sensors_x_y,),
                    dtype=np.float32,
                ),
            }
        )

    @property
    def render_shape(self) -> tuple[int, int, int]:
        """The shape of the rendered domain."""
        z_res = self._circle_resolution_angular * 4
        y_res = z_res
        x_res = int(z_res / self.H * self.L)
        return (x_res, y_res, z_res)

    @property
    def n_agents(self) -> int:
        """The number of agents in the environment."""
        return 1

    def _get_domain(self) -> PISOtorch.Domain:
        domain = make_vortex_street_domain(
            ndims=self._ndims,
            viscosity=self._viscosity.to(self._cpu_device),
            domain_height=self.H,
            domain_length=self.L,
            cylinder_radius=self.cylinder_diameter / 2,
            cylinder_offset_y=self.cylinder_offset_y,
            circle_thickness=self.cylinder_diameter / 2,
            quad_thickness_x=self.cylinder_diameter,
            circle_resolution_angular=self._circle_resolution_angular,
            vortex_street_refinement_base=self._vortex_street_refinement_base,
            vortex_street_refinement_axes=["+y", "-y"],
            cuda_device=self._cuda_device,
            dtype=self._dtype,
            debug=self._debug,
        )

        # finalize domain
        domain.PrepareSolve()

        return domain

    @property
    def _cd_ref(self) -> float:
        if "drag" in self._metrics_stats:
            return self._metrics_stats["drag"].mean
        else:
            return 0.0

    def _get_prep_fn(self, domain: PISOtorch.Domain) -> dict[str, Any]:
        if self._ndims == 2:
            char_vel = torch.tensor(
                [[self._U_mean, 0.0]], device=self._cuda_device, dtype=self._dtype
            )
        else:
            char_vel = torch.tensor(
                [[self._U_mean, 0.0, 0.0]], device=self._cuda_device, dtype=self._dtype
            )

        bound = domain.getBlock(self._vortex_street_block_idx).getBoundary("+x")

        def pre_fn(domain, time_step, **kwargs):
            boundary = domain.getBlock(self._vortex_street_block_idx).getBoundary("+x")
            boundary.setTransform(bound.transform.to(self._cuda_device))
            out_bounds = [bound]
            update_advective_boundaries(
                domain, out_bounds, char_vel, time_step.to(self._cuda_device), tol=5e-6
            )

        prep_fn = {
            "PRE": pre_fn,
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
            pressure_tol=1e-5 if self._ndims == 2 else 5e-7,
            advect_non_ortho_steps=1,
            pressure_non_ortho_steps=1 if self._ndims == 2 else 4,
            pressure_return_best_result=True,
            velocity_corrector="FD",
            non_orthogonal=True,
            output_resampling_shape=self.render_shape[: self._ndims],
            output_resampling_fill_max_steps=16,
            differentiable=self._differentiable,
        )

        sim.preconditionBiCG = False
        sim.BiCG_precondition_fallback = True

        sim.make_divergence_free()

        return sim

    def _additional_initialization(self) -> None:
        """Perform any additional initialization after the domain and simulation are
        created.
        """
        self._left_boundary = self._domain.getBlock(self._left_block_idx).getBoundary(
            "+x"
        )
        self._top_boundary = self._domain.getBlock(self._top_block_idx).getBoundary(
            "-y"
        )
        self._right_boundary = self._domain.getBlock(self._right_block_idx).getBoundary(
            "-x"
        )
        self._bottom_boundary = self._domain.getBlock(
            self._bottom_block_idx
        ).getBoundary("+y")

        self.__prepare_drag_and_lift_computation()

        assert self.__wall_normals is not None
        assert self.__wall_distances is not None
        assert self.__wall_face_lengths is not None
        assert self.__wall_normals.grad_fn is None
        assert self.__wall_distances.grad_fn is None
        assert self.__wall_face_lengths.grad_fn is None

        self.__last_control = torch.zeros(
            (1,), device=self._cuda_device, requires_grad=False
        )

    def _randomize_domain(self) -> None:
        strouhal_number = 0.3
        vortex_shedding_period = 1 / (
            strouhal_number * self._U_mean / self.cylinder_diameter
        )
        velocity_noise = 0.025
        pressure_noise = 0.025

        max_n_steps = 2 * int(vortex_shedding_period / self._step_length) - 1
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

    def get_vorticity(self) -> torch.Tensor:
        """Get the vorticity field of the fluid with the cylinder region masked.

        Returns
        -------
        torch.Tensor
            The vorticity field as a tensor.
        """
        vorticity = super().get_vorticity()

        if self._ndims == 2:
            vorticity[self._cylinder_mask] = 0.0
        else:
            vorticity[:, self._cylinder_mask] = 0.0

        return vorticity

    def get_velocity(self) -> torch.Tensor:
        """Get the velocity field of the fluid with the cylinder region masked.

        Returns
        -------
        torch.Tensor
            The velocity field as a tensor.
        """
        u = super().get_velocity()

        u[:, self._cylinder_mask] = 0.0
        return u

    def _sensor_locations_to_grid_coords(
        self, physical_coords: torch.Tensor
    ) -> torch.Tensor:
        # Now we need to convert the physical locations to grid indices
        physical_coords[0, :] += 2.0
        physical_coords[0, :] *= (self.render_shape[0] - 1) / (self.L - 2.0)
        physical_coords[1, :] += self.H / 2
        physical_coords[1, :] *= (self.render_shape[1] - 1) / self.H

        if self._ndims == 3:
            physical_coords[2, :] += self.H / 2
            physical_coords[2, :] *= (self.render_shape[1] - 1) / self.H

        return torch.round(physical_coords).to(torch.int32)

    def _get_sensor_locations(self) -> torch.Tensor:
        sensor_locations = self._get_sensor_locations_2d()
        grid_coords = self._sensor_locations_to_grid_coords(sensor_locations)

        return grid_coords

    def _get_sensor_locations_2d(self) -> torch.Tensor:
        # First, we get the main grid in the cylinder wake
        x_indices = torch.arange(
            1.0,
            5.0,
            step=0.5,
        )
        y_indices = torch.arange(
            -1.5,
            1.75,
            step=0.5,
        )

        xy_grid = torch.meshgrid(x_indices, y_indices, indexing="ij")
        x_flat = xy_grid[0].ravel()
        y_flat = xy_grid[1].ravel()
        sensor_locations = torch.stack([x_flat, y_flat], dim=0)

        # Now, we add the remaining sensors close to the cylinder
        x_1 = torch.arange(-0.25, 1, 0.25)
        y_1a = torch.full_like(x_1, -1.5)
        y_1b = torch.full_like(x_1, 1.5)

        x_2 = torch.arange(0.25, 1.25, 0.25)
        x_2 = torch.concatenate([torch.tensor([-0.25]), x_2])
        y_2a = torch.full_like(x_2, self.cylinder_diameter)
        y_2b = torch.full_like(x_2, -self.cylinder_diameter)

        x_3 = torch.tensor([0.75] * 3)
        y_3 = torch.tensor([-0.5, 0, 0.5])

        # Combine all sensor locations
        additional_sensors = torch.stack(
            [
                torch.concatenate([x_1, x_1, x_2, x_2, x_3]),
                torch.concatenate([y_1a, y_1b, y_2a, y_2b, y_3]),
            ],
            dim=0,
        )

        # Finally, we add two circles around the cylinder
        cylinder_radius = 0.5
        angles = torch.linspace(0, 2 * torch.pi, steps=36)

        radius_one = 2 * cylinder_radius
        radius_two = 1.25 * cylinder_radius

        circle_one_x = radius_one * torch.cos(angles)
        circle_one_y = radius_one * torch.sin(angles)
        circle_one = torch.stack([circle_one_x, circle_one_y], dim=0)

        circle_two_x = radius_two * torch.cos(angles)
        circle_two_y = radius_two * torch.sin(angles)
        circle_two = torch.stack([circle_two_x, circle_two_y], dim=0)

        all_locations = torch.concatenate(
            [sensor_locations, circle_one, circle_two, additional_sensors], dim=1
        )

        return all_locations

    def _get_cylinder_mask(self) -> np.ndarray:
        cylinder_radius = (
            self.cylinder_diameter / 2 * (self.render_shape[1] - 1) / self.H
        )

        center_x = round((self.render_shape[0] - 1) / self.L * 2.0)
        center_y = round((self.render_shape[1] - 1) / self.H * (2.0))

        Y, X = np.ogrid[: self.render_shape[1], : self.render_shape[0]]
        dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        cylinder_mask = dist_from_center <= cylinder_radius

        if self._ndims == 3:
            cylinder_mask = np.repeat(
                cylinder_mask[None, :, :], self.render_shape[2], axis=0
            )

        return cylinder_mask

    @abstractmethod
    def _apply_action(self, action: torch.Tensor) -> None:
        """Apply the given action to the simulation."""
        raise NotImplementedError

    def _get_global_obs(self) -> dict[str, torch.Tensor]:
        return extract_global_2d_obs(
            env=self,
            sensor_locations=self._sensor_locations,
        )

    def __collect_boundary_coords(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        block_idxs = [
            self._left_block_idx,
            self._top_block_idx,
            self._right_block_idx,
            self._bottom_block_idx,
        ]
        boundary_cell_slices = [
            (..., slice(None), -1),
            (..., 0, slice(None)),
            (..., slice(None), 0),
            (..., -1, slice(None)),
        ]
        flip_dims = [
            [],
            [],
            [-1],
            [-1],
        ]

        return collect_boundary_coords(
            domain=self._domain,
            block_idxs=block_idxs,
            boundary_cell_slices=boundary_cell_slices,
            flip_dims=flip_dims,
        )

    def __collect_boundary_fields(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_idxs = [
            self._left_block_idx,
            self._top_block_idx,
            self._right_block_idx,
            self._bottom_block_idx,
        ]
        boundary_faces = [
            "+x",
            "-y",
            "-x",
            "+y",
        ]
        boundary_cell_slices = [
            (..., slice(None), -1),
            (..., 0, slice(None)),
            (..., slice(None), 0),
            (..., -1, slice(None)),
        ]
        flip_dims = [
            [],
            [],
            [-1],
            [-1],
        ]

        return collect_boundary_fields(
            domain=self._domain,
            block_idxs=block_idxs,
            boundary_faces=boundary_faces,
            boundary_cell_slices=boundary_cell_slices,
            flip_dims=flip_dims,
        )

    def __prepare_drag_and_lift_computation(
        self,
    ) -> None:
        # Check if this was already computed
        if self.__wall_distances is not None:
            return

        cell_coords, cell_centers = self.__collect_boundary_coords()
        cell_centers = cell_centers.detach()
        cell_coords = cell_coords.detach()

        cell_centers_left = torch.roll(cell_centers, shifts=-1, dims=-1)
        cell_centers_right = torch.roll(cell_centers, shifts=1, dims=-1)
        self.__tangent_lengths = torch.sqrt(
            torch.sum((cell_centers_left - cell_centers_right) ** 2, dim=0)
        )

        # For 3D we only consider a single slice in z-direction,
        # and expand the final tensors to 3D afterwards. Thus,
        # we can share the logic with the 2D case.
        if self._ndims == 3:
            cell_coords = cell_coords[:2, 0, :]
            cell_centers = cell_centers[:2, 0, :]

        self.__wall_distances, self.__wall_normals = wall_distance_from_vertices(
            cell_coords, cell_centers
        )

        # Cell segment lengths
        xw, yw = cell_coords[0, ...], cell_coords[1, ...]
        self.__wall_face_lengths = torch.sqrt(
            (xw[1:] - xw[:-1]) ** 2 + (yw[1:] - yw[:-1]) ** 2
        )

        if self._ndims == 3:
            # We need to add a dummy dimension for z
            self.__wall_distances = self.__wall_distances[..., None]
            self.__wall_normals = self.__wall_normals[..., None]
            # For the tangents, we only need the first z-slice
            self.__tangent_lengths = self.__tangent_lengths[:1, :].transpose(1, 0)

    def _get_drag_and_lift(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.__wall_normals is not None
        assert self.__wall_face_lengths is not None
        assert self.__wall_distances is not None

        u_cell, u_cylinder, p_cell = self.__collect_boundary_fields()

        if self._ndims == 2:
            forces = compute_forces_2d(
                u_cell=u_cell,
                u_boundary=u_cylinder,
                p_cell=p_cell,
                wall_normals=self.__wall_normals,
                wall_distances=self.__wall_distances,
                tangent_lengths=self.__tangent_lengths,
                wall_face_lengths=self.__wall_face_lengths,
                viscosity=self._viscosity,
            )
        else:
            face_areas = self.__wall_face_lengths * (
                self.D / self._circle_resolution_angular
            )
            forces = compute_forces_3d(
                u_cell=u_cell,
                u_boundary=u_cylinder,
                p_cell=p_cell,
                wall_normals=self.__wall_normals,
                wall_distances=self.__wall_distances,
                tangent_lengths=self.__tangent_lengths,
                wall_face_areas=face_areas,
                viscosity=self._viscosity,
            )

        drag = forces[0]
        lift = forces[1]

        cd = drag / (0.5 * self._U_mean**2 * self.cylinder_diameter)
        cl = lift / (0.5 * self._U_mean**2 * self.cylinder_diameter)

        return cd, cl

    def _get_render_data(
        self,
        render_3d: bool,
        output_path: Path | None = None,
    ) -> dict[str, np.ndarray]:
        vorticity = self.get_vorticity().squeeze()
        vorticity = torch.flip(vorticity, dims=[-1])

        vort_min, vort_max = VORTICITY_RENDER_RANGE

        format_vorticity = partial(
            self._format_render_data,
            v_min=vort_min,
            v_max=vort_max,
            cmap="icefire",
        )

        render_data = {}
        if self._ndims == 2:
            x_y_vorticity = format_vorticity(
                data=vorticity.detach().cpu().numpy(),
            )
            x_y_vorticity[self._cylinder_mask] = 0
            render_data["vorticity"] = x_y_vorticity
        else:  # ndims == 3
            vort_xy = vorticity[2, vorticity.shape[0] // 2, :, :]
            vort_xz = vorticity[1, :, vorticity.shape[1] // 2, :]
            vort_yz = vorticity[0, :, :, int(vorticity.shape[2] * 0.8)]

            x_y_vorticity = format_vorticity(data=vort_xy.detach().cpu().numpy())
            x_y_vorticity[self._cylinder_mask[0, :, :]] = 0
            render_data["x-y-vorticity"] = x_y_vorticity
            x_z_vorticity = format_vorticity(data=vort_xz.detach().cpu().numpy())
            x_z_vorticity[self._cylinder_mask[:, 0, :]] = 0
            render_data["x-z-vorticity"] = x_z_vorticity
            y_z_vorticity = format_vorticity(data=vort_yz.detach().cpu().numpy().T)
            y_z_vorticity[self._cylinder_mask[:, :, 0]] = 0
            render_data["y-z-vorticity"] = y_z_vorticity

        return render_data

    def _step_impl(
        self, action: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, bool, dict[str, torch.Tensor]]:
        all_cds = []
        all_cls = []
        for _ in range(self._n_sim_steps):
            # We apply the action smoothing as proposed by Rabault et al. (2020)
            control = self.__last_control + self._action_smoothing_alpha * (
                action - self.__last_control
            )
            self.__last_control = control
            if self._enable_actions:
                self._apply_action(control)

            self._sim.single_step()
            _cd, _cl = self._get_drag_and_lift()
            all_cds += [_cd]
            all_cls += [_cl]

        obs = self._get_global_obs()

        all_cds_tensor = torch.stack(all_cds).mean(dim=0)
        all_cls_tensor = torch.stack(all_cls).mean(dim=0)

        # For 3D, we sum over the z-direction, for 2D it's just a scalar
        cd = torch.sum(all_cds_tensor)
        cl = torch.sum(all_cls_tensor)

        reward = self._cd_ref - cd - self._lift_penalty * torch.abs(cl)

        info = {
            "drag": all_cds_tensor,
            "lift": all_cls_tensor,
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

        colors = fluidgym_config.palette

        plt.figure(figsize=(10, 2.5))

        plt.xlim(-2, self.L - 2)
        plt.ylim(-self.H / 2, self.H / 2 + self.cylinder_offset_y)

        plt.grid()

        # Then, we add the cylinder position
        cylinder_radius = self.cylinder_diameter / 2
        circle = patches.Circle((0, 0), cylinder_radius, color=colors[0], fill=True)
        ax = plt.gca()
        ax.add_artist(circle)
        ax.set_aspect("equal")

        ax.set_xlabel("L")
        ax.set_ylabel("H")

        ax.set_xticks(np.linspace(-2, self.L, 13))
        ax.set_xticklabels([f"{int(x)}" for x in np.linspace(-2, self.L, 13)])

        # We want domain coordinates
        sensor_locations = self._get_sensor_locations_2d()

        # Then, we add the sensor locations
        for sensor in sensor_locations.T:
            plt.scatter(sensor[0], sensor[1], color=colors[2], label="Sensors", s=5)

        plt.tight_layout()
        plt.savefig(output_path / f"{self.id}.pdf")

    @property
    def initial_domain_id(self) -> str:
        """Unique identifier for the initial domain."""
        return (
            f"cylinder_{self._ndims}D_Re{int(self._reynolds_number)}"
            f"_Res{self._circle_resolution_angular}"
        )

    def _load_domain_statistics(self) -> dict[str, dict[str, float]]:
        """Load statistics from a JSON file.

        Returns
        -------
        dict[str, dict[str, float]]
            The loaded statistics.
        """
        stats = super()._load_domain_statistics()
        self._vorticity_stats = Stats(**stats["vorticity_magnitude"])
        return stats

    def detach(self) -> None:
        """Detach all tensors from the current computation graph."""
        super().detach()
        self.__last_control = self.__last_control.detach()

    def get_state(self) -> EnvState:
        """Get the current state of the environment.

        Returns
        -------
        EnvState
            The current state of the environment.
        """
        state = super().get_state()
        state.additional_info["last_control"] = self.__last_control.clone()
        return state

    def set_state(self, state: EnvState) -> None:
        """Set the current state of the environment.

        Parameters
        ----------
        state: EnvState
            The state to set the environment to.
        """
        super().set_state(state)
        last_control: torch.Tensor = state.additional_info["last_control"]
        self.__last_control = last_control
