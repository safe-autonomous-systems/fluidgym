"""Abstract base class for airfoil flow environments."""

from abc import abstractmethod
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap, to_rgb

from fluidgym.config import config as global_config
from fluidgym.envs.airfoil.grid import (
    get_jet_locations,
    make_airfoil_domain,
    read_airfoil,
)
from fluidgym.envs.fluid_env import EnvState, FluidEnv, Stats
from fluidgym.envs.util.forces import (
    collect_boundary_coords,
    collect_boundary_fields,
    compute_forces_2d,
    compute_forces_3d,
    wall_distance_from_vertices,
)
from fluidgym.envs.util.obs_extraction import extract_global_2d_obs
from fluidgym.envs.util.profiles import get_jet_profile
from fluidgym.simulation.extensions import (
    PISOtorch,  # type: ignore[import-untyped,import-not-found]
)
from fluidgym.simulation.pict.PISOtorch_simulation import (
    balance_boundary_fluxes,
    update_advective_boundaries,
)
from fluidgym.simulation.simulation import Simulation

VORTICITY_RENDER_RANGE = {
    1000: (-10, 10),
    3000: (-12.5, 12.5),
    5000: (-15, 15),
}


class AirfoilEnvBase(FluidEnv):
    """Environment for turbulent channel flow control.

    References
    ----------
    [1] X. Garcia et al., “Deep-reinforcement-learning-based separation control in a
    two-dimensional airfoil,” Feb. 24, 2025, arXiv: arXiv:2502.16993.
    doi: 10.48550/arXiv.2502.16993.

    [2] Y.-Z. Wang, Y.-F. Mei, N. Aubry, Z. Chen, P. Wu, and W.-T. Wu,
    “Deep reinforcement learning based synthetic jet control on disturbed flow over
    airfoil,” Physics of Fluids, 2022, [Online].
    Available: https://doi.org/10.1063/5.0080922

    [3] R. Montalà et al., “Discovering Flow Separation Control Strategies in 3D Wings
    via Deep Reinforcement Learning,” Sept. 12, 2025, arXiv: arXiv:2509.10185.
    doi: 10.48550/arXiv.2509.10185.
    """

    _default_render_key: str = "vorticity"

    _action_smoothing_alpha: float = 0.1

    _n_jets: int = 3
    _res_z: int = 96
    U_mean: float = 0.3
    airfoil_length: float = 1.0
    H: float = 1.4
    L: float = 4.5
    D: float = 1.4

    _action_range = (-1.0, 1.0)
    _observation_range = (-2.5, 2.5)

    _jet_locations_top: list[list[int]] | None = None

    _metrics: list[str] = ["drag", "lift"]
    _vorticity_stats: Stats | None = None

    # Domain generation
    _initial_domain_steps = 400
    _initial_domain_restart = False

    _airfoil_resolution: int | None = None

    _wall_normals: torch.Tensor | None = None
    __wall_distances: torch.Tensor | None = None
    __wall_face_lengths: torch.Tensor | None = None

    def __init__(
        self,
        ndims: int,
        reynolds_number: float,
        adaptive_cfl: float,
        step_length: float,
        episode_length: int,
        dt: float,
        attack_angle_deg: float,
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
        if attack_angle_deg < 0.0 or attack_angle_deg > 20.0:
            raise ValueError("Attack angle must be between 0 and 20 degrees.")

        self._debug = debug
        self._reynolds_number = reynolds_number
        self._viscosity = torch.tensor(
            [(self.U_mean * self.airfoil_length) / reynolds_number], dtype=dtype
        )
        self._step_length = step_length
        self._attack_angle_deg = attack_angle_deg

        self._ndims = ndims
        _, self._airfoil_coords = read_airfoil(
            attack_angle_deg=self._attack_angle_deg,
            cpu_device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self._airfoil_coords = self._airfoil_coords.squeeze()
        self._airfoil_mask = self._get_airfoil_mask()
        self._sensor_locations = self._get_sensor_locations()

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
        (
            self._left_block_idx,
            self._airfoil_front_block_idx,
            self._airfoil_top_block_idx,
            self._airfoil_bot_block_idx,
            self._tail_upper_block_idx,
            self._tail_lower_block_idx,
        ) = range(6)

        self.__last_control = torch.zeros((1,), device=self._cuda_device)
        self._viscosity = self._viscosity.to(self._cuda_device)

    @property
    def render_shape(self) -> tuple[int, ...]:
        """The shape of the rendered domain."""
        return (600, 150, 150)

    @property
    def _cl_cd_ref(self) -> float:
        """Reference lift-to-drag ratio for the airfoil."""
        if "lift" in self._metrics_stats and "drag" in self._metrics_stats:
            return self._metrics_stats["lift"].mean / self._metrics_stats["drag"].mean
        else:
            return 0.0

    def _get_airfoil_mask(self) -> np.ndarray:
        """Compute a mask for the airfoil location in the grid.

        Returns
        -------
        np.ndarray
            Boolean mask of shape (nx, ny, (nz)) where True indicates the presence of
            the airfoil.
        """
        from matplotlib.path import Path

        coords = self._physical_locations_to_grid_coords(self._airfoil_coords)
        coords_np = coords.cpu().numpy()

        xs, ys = coords_np

        polygon = np.vstack((xs, ys)).T  # shape (N, 2)

        nx, ny = self.render_shape[0], self.render_shape[1]
        x_min, x_max = 0, self.render_shape[0] - 1
        y_min, y_max = 0, self.render_shape[1] - 1

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, nx),
            np.linspace(y_min, y_max, ny),
        )

        path = Path(polygon)
        grid_points = np.vstack((xx.ravel(), yy.ravel())).T
        mask = path.contains_points(grid_points).reshape(xx.shape)
        mask = mask.copy()

        if self._ndims == 3:
            mask = np.repeat(mask[None, :, :], self.render_shape[2], axis=0)
        return mask

    def _get_domain(self) -> PISOtorch.Domain:
        # For the hard case in 3D we need a finer grid at the outflow
        if self._ndims == 3 and self._reynolds_number >= 5000:
            tail_grow_mul = 1.001
        else:
            tail_grow_mul = 1.01

        domain = make_airfoil_domain(
            n_dims=self._ndims,
            res_z=self._res_z,
            H=self.H,
            L=self.L,
            vel_in=self.U_mean,
            attack_angle_deg=self._attack_angle_deg,
            viscosity=self._viscosity.to(self._cpu_device),
            resolution_div=1,
            tail_grow_mul=tail_grow_mul,
            cpu_device=self._cpu_device,
            cuda_device=self._cuda_device,
        )

        # finalize domain
        domain.PrepareSolve()
        return domain

    def _get_prep_fn(self, domain: PISOtorch.Domain) -> dict[str, Any]:
        if self._ndims == 2:
            char_vel = torch.tensor(
                [[self.U_mean, 0.0]], device=self._cuda_device, dtype=self._dtype
            )
        else:
            char_vel = torch.tensor(
                [[self.U_mean, 0.0, 0.0]], device=self._cuda_device, dtype=self._dtype
            )

        # callback function to update the outflow during the simulation
        def update_outflow(domain, time_step, **kwargs):
            out_bounds = [
                domain.getBlock(self._tail_lower_block_idx).getBoundary("+x"),
                domain.getBlock(self._tail_upper_block_idx).getBoundary("+x"),
            ]
            update_advective_boundaries(
                domain=domain,
                bounds=out_bounds,
                velms=char_vel,
                dt=time_step.cuda(),
            )

        return {"PRE": update_outflow}

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
            advection_tol=1e-6,
            pressure_tol=1e-7 if self._ndims == 2 else 1e-8,
            advect_non_ortho_steps=2,
            pressure_non_ortho_steps=4,
            pressure_return_best_result=True,
            velocity_corrector="FD",
            non_orthogonal=True,
            output_resampling_shape=self.render_shape[: self._ndims],
            output_resampling_fill_max_steps=128,
            differentiable=self._differentiable,
        )

        sim.preconditionBiCG = False
        sim.BiCG_precondition_fallback = True

        sim.make_divergence_free()

        return sim

    def _additional_initialization(self) -> None:
        self._airfoil_top_boundary = self._domain.getBlock(
            self._airfoil_top_block_idx
        ).getBoundary("-y")
        self._airfoil_bot_boundary = self._domain.getBlock(
            self._airfoil_bot_block_idx
        ).getBoundary("+y")
        self.__prepare_drag_and_lift_computation()
        self._jet_locations_top = get_jet_locations(self._domain)
        self._top_base_profile = self._get_base_jet_profiles()

    def _randomize_domain(self) -> None:
        velocity_noise = 0.01
        pressure_noise = 0.01

        max_n_steps = int(0.05 * self._episode_length)
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

    def __collect_boundary_coords(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        block_idxs = [
            self._airfoil_front_block_idx,
            self._airfoil_top_block_idx,
            self._airfoil_bot_block_idx,
        ]
        boundary_cell_slices = [
            (..., slice(None), -1),
            (..., 0, slice(None)),
            (..., -1, slice(None)),
        ]
        flip_dims = [
            [],
            [],
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
            self._airfoil_front_block_idx,
            self._airfoil_top_block_idx,
            self._airfoil_bot_block_idx,
        ]
        boundary_faces = [
            "+x",
            "-y",
            "+y",
        ]
        boundary_cell_slices = [
            (..., slice(None), -1),
            (..., 0, slice(None)),
            (..., -1, slice(None)),
        ]
        flip_dims = [
            [],
            [],
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

        self.__wall_distances, self._wall_normals = wall_distance_from_vertices(
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
            self._wall_normals = self._wall_normals[..., None]
            # For the tangents, we only need the first z-slice
            self.__tangent_lengths = self.__tangent_lengths[:1, :].transpose(1, 0)

    def _get_drag_and_lift(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._wall_normals is not None
        assert self.__wall_face_lengths is not None
        assert self.__wall_distances is not None

        u_cell, u_airfoil, p_cell = self.__collect_boundary_fields()

        if self._ndims == 2:
            forces = compute_forces_2d(
                u_cell=u_cell,
                u_boundary=u_airfoil,
                p_cell=p_cell,
                wall_normals=self._wall_normals,
                wall_distances=self.__wall_distances,
                tangent_lengths=self.__tangent_lengths,
                wall_face_lengths=self.__wall_face_lengths,
                viscosity=self._viscosity,
            )
        else:
            face_areas = self.__wall_face_lengths * (self.D / self._res_z)
            forces = compute_forces_3d(
                u_cell=u_cell,
                u_boundary=u_airfoil,
                p_cell=p_cell,
                wall_normals=self._wall_normals,
                wall_distances=self.__wall_distances,
                tangent_lengths=self.__tangent_lengths,
                wall_face_areas=face_areas,
                viscosity=self._viscosity,
            )

        drag = forces[0]
        lift = forces[1]

        cd = drag / (0.5 * self.U_mean**2 * self.airfoil_length)
        cl = lift / (0.5 * self.U_mean**2 * self.airfoil_length)

        return cd, cl

    def _get_base_jet_profiles(self) -> torch.Tensor:
        assert self._jet_locations_top is not None
        assert self._wall_normals is not None

        grids: list[torch.Tensor] = self._domain.getVertexCoordinates()
        n_boundary_cells_top = grids[2].shape[-1] - 1  # AirfoilTop block

        velocity_profile_top = torch.zeros(
            (1, 2, 1, n_boundary_cells_top), device=self._cuda_device
        )

        if self._ndims == 2:
            wall_normals = self._wall_normals
        else:
            # Remove dummy z-dimension
            wall_normals = self._wall_normals[..., 0]

        def get_profile(idx_start: int, idx_end: int) -> torch.Tensor:
            assert self._wall_normals is not None
            # Since wall_normals are concatenated, including all values from
            # front->top->bottom, we need to offset the indices for the top boundary
            front_block_coords = self._domain.getVertexCoordinates()[
                self._airfoil_front_block_idx
            ]
            n_boundary_cells = front_block_coords.shape[-2]  # +y boundary
            normal_idx_start = n_boundary_cells + idx_start
            normal_idx_end = n_boundary_cells + idx_end

            profile = get_jet_profile(
                h=idx_end - idx_start + 3, dtype=self._dtype, device=self._cuda_device
            )
            # Remove first and last element (= 0) since we have
            # the between cells there
            profile = profile[1:-1]

            profile /= profile.sum()

            # "Rotate" profile with wall normals
            n = wall_normals[:, normal_idx_start : normal_idx_end + 1]

            profile = profile.unsqueeze(0) * n

            return profile

        for i in range(self._n_jets):
            start_idx_top, end_idx_top = self._jet_locations_top[i]

            velocity_profile_top[
                0,
                :,
                0,
                start_idx_top : end_idx_top + 1,
            ] = get_profile(start_idx_top, end_idx_top)

        return velocity_profile_top

    def get_vorticity(self) -> torch.Tensor:
        """Get the vorticity field of the fluid with the airfoil region masked.

        Returns
        -------
        torch.Tensor
            The vorticity field as a tensor.
        """
        vorticity = super().get_vorticity()

        if self._ndims == 2:
            vorticity[self._airfoil_mask] = 0.0
        else:
            vorticity[:, self._airfoil_mask] = 0.0

        return vorticity

    def get_velocity(self) -> torch.Tensor:
        """Get the velocity field of the fluid with the airfoil region masked.

        Returns
        -------
        torch.Tensor
            The velocity field as a tensor.
        """
        u = super().get_velocity()

        u[:, self._airfoil_mask] = 0.0
        return u

    def _physical_locations_to_grid_coords(
        self, physical_coords: torch.Tensor
    ) -> torch.Tensor:
        physical_coords = physical_coords.clone()

        # Now we need to convert the physical locations to grid indices
        physical_coords[0, :] += 1.5
        physical_coords[0, :] *= (self.render_shape[0]) / (self.L + 1.5)
        physical_coords[1, :] += self.H / 2
        physical_coords[1, :] *= (self.render_shape[1]) / self.H

        if physical_coords.shape[0] == 3:
            physical_coords[2, :] += self.D / 2
            physical_coords[2, :] *= (self.render_shape[1]) / self.D

        return torch.round(physical_coords).to(torch.int32)

    def _get_sensor_locations(self) -> torch.Tensor:
        sensor_locations = self._get_sensor_locations_2d()
        grid_coords = self._physical_locations_to_grid_coords(sensor_locations)

        all_sensors = []

        # Filter out airfoil mask
        for i in range(grid_coords.shape[1]):
            x_idx = grid_coords[0, i]
            y_idx = grid_coords[1, i]

            if self._airfoil_mask[y_idx, x_idx]:
                continue

            all_sensors.append(grid_coords[:, i : i + 1])

        grid_coords = torch.cat(all_sensors, dim=1)

        return grid_coords

    def _get_sensor_locations_2d(self) -> torch.Tensor:
        def idxs_to_locs(x_idxs: torch.Tensor, y_idxs: torch.Tensor) -> torch.Tensor:
            xy_grid = torch.meshgrid(x_indices, y_indices, indexing="ij")
            x_flat = xy_grid[0].ravel()
            y_flat = xy_grid[1].ravel()
            return torch.stack([x_flat, y_flat], dim=0)

        # First, we get the main grid in the airfoil wake
        x_indices = torch.arange(
            1.5,
            2.6,
            step=0.125,
        )
        y_indices = torch.linspace(
            start=-self.H / 2,
            end=self.H / 2,
            steps=10,
        )[1:-1]
        sensors_wake_coarse = idxs_to_locs(x_indices, y_indices)

        # Then, we add the fine grid right behind the airfoil
        x_indices = torch.arange(
            1.05,
            1.5 - 0.05,
            step=0.05,
        )
        y_indices = torch.linspace(
            start=-self.H / 2,
            end=self.H / 2,
            steps=10,
        )[1:-1]
        sensors_wake_fine = idxs_to_locs(x_indices, y_indices)

        # Finally, we add some sensors close to the airfoil surface
        x_indices = torch.linspace(
            start=-0.125,
            end=self.airfoil_length,
            steps=10,
        )
        y_indices = torch.linspace(
            start=-0.5,
            end=0.125,
            steps=8,
        )
        sensors_airfoil = idxs_to_locs(x_indices, y_indices)

        all_locations = torch.cat(
            [sensors_wake_coarse, sensors_wake_fine, sensors_airfoil], dim=1
        )
        return all_locations

    def _get_global_obs(self) -> dict[str, torch.Tensor]:
        return extract_global_2d_obs(
            env=self,
            sensor_locations=self._sensor_locations,
        )

    def _get_render_data(
        self,
        render_3d: bool,
        output_path: Path | None = None,
    ) -> dict[str, np.ndarray]:
        vorticity = self.get_vorticity().squeeze()
        vorticity = torch.flip(vorticity, dims=[-2, -1])

        vort_min, vort_max = VORTICITY_RENDER_RANGE[int(self._reynolds_number)]

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
            render_data["vorticity"] = x_y_vorticity
        else:  # ndims == 3
            vort_xy = vorticity[2, vorticity.shape[0] // 2, :, :]
            vort_xz = vorticity[1, :, vorticity.shape[1] // 2, :]
            vort_yz = vorticity[0, :, :, int(vorticity.shape[2] * 0.8)]

            render_data["x-y-vorticity"] = format_vorticity(
                data=vort_xy.detach().cpu().numpy()
            )
            render_data["x-z-vorticity"] = format_vorticity(
                data=vort_xz.detach().cpu().numpy()
            )
            render_data["y-z-vorticity"] = format_vorticity(
                data=vort_yz.detach().cpu().numpy().T
            )

        return render_data

    @abstractmethod
    def _action_to_control(self, action: torch.Tensor) -> torch.Tensor:
        """Convert the action to boundary control values."""
        raise NotImplementedError

    def _apply_action(self, action: torch.Tensor) -> None:
        """Apply the given action to the simulation."""
        control_top = self._action_to_control(action)
        self._airfoil_top_boundary.setVelocity(control_top)
        out_bounds = [
            self._domain.getBlock(self._tail_lower_block_idx).getBoundary("+x"),
            self._domain.getBlock(self._tail_upper_block_idx).getBoundary("+x"),
            self._airfoil_top_boundary,
        ]
        balance_boundary_fluxes(self._domain, out_bounds)

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

        # For 3D, we sum over the z-direction
        cd = torch.sum(all_cds_tensor)
        cl = torch.sum(all_cls_tensor)

        reward: torch.Tensor = (cl / cd) - self._cl_cd_ref

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

        # First, we plot the domain shape
        plt.figure(figsize=(10, 2.5))
        ax = plt.gca()

        plt.xlim(0, self.render_shape[0] - 1)
        plt.ylim(0, self.render_shape[1] - 1)

        plt.grid()

        colors = global_config.palette

        rgb = to_rgb(colors[0])
        cmap = ListedColormap(
            [
                (1.0, 1.0, 1.0),
                rgb,
            ]
        )

        plt.imshow(
            self._airfoil_mask,
            extent=(0, self.render_shape[0], 0, self.render_shape[1]),
            origin="lower",
            cmap=cmap,
        )

        # Then, we add the sensor locations
        for sensor in self._sensor_locations.T:
            plt.scatter(
                sensor[0], sensor[1], color=colors[2], label="Sensor Location", s=5
            )

        total_length = 1.5 + self.L

        ax.set_yticks(
            [
                0,
                int(self.render_shape[1] / 2),
                self.render_shape[1] - 1,
            ]
        )
        ax.set_yticklabels([f"-{self.H / 2:.1f}", "0.0", f"{self.H / 2:.1f}"])
        ax.set_ylabel("H")

        ax.set_xticks(
            [
                0,
                int(self.render_shape[0] / total_length) * 1.5,
                int(self.render_shape[0] / total_length) * 2.5,
                self.render_shape[0],
            ]
        )
        ax.set_xticklabels(["-1.5", "0.0", "1.0", f"{self.L}"])
        ax.set_xlabel("L")

        plt.tight_layout()

        plt.savefig(output_path / f"{self.id}.pdf")

    @property
    def initial_domain_id(self) -> str:
        """Unique identifier for the initial domain."""
        return f"airfoil_{self._ndims}D_Re{int(self._reynolds_number)}"

    @property
    def id(self) -> str:
        """Unique identifier for the environment."""
        return f"Airfoil{self._ndims}D_Re{int(self._reynolds_number)}"

    def _load_domain_statistics(self) -> dict[str, dict[str, float]]:
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
