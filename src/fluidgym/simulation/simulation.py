"""FluidGym simulation class based on PISOtorch implemented in PICT."""

from typing import Literal

import torch

from fluidgym.simulation.extensions import (
    PISOtorch,  # type: ignore[import-untyped,import-not-found]
)
from fluidgym.simulation.pict import PISOtorch_diff
from fluidgym.simulation.pict.PISOtorch_simulation import (
    Simulation as PISOtorchSimulation,
)
from fluidgym.simulation.pict.PISOtorch_simulation import (
    get_max_time_step,
)
from fluidgym.simulation.pict.util.logging import get_logger
from fluidgym.simulation.pict.util.output import numerical_to_numpy as ntonp


class Simulation(PISOtorchSimulation):
    """FluidGym simulation class based on PISOtorch implemented in PICT.

    Parameters
    ----------
    domain: PISOtorch.Domain
        The simulation domain.

    dt: float
        The simulation time step.

    verbose: bool
        If True, enable verbose logging. Defaults to False.

    substeps: int | Literal["ADAPTIVE"]
        Number of substeps per simulation step or "ADAPTIVE" for adaptive substepping.
        Defaults to 1.

    corrector_steps: int
        Number of corrector steps in the PISO algorithm. Defaults to 2.

    density_viscosity: float | None
        Fluid density viscosity. If None, uses domain default. Defaults to None.

    adaptive_CFL: float
        CFL number for adaptive time stepping. Defaults to 0.8.

    prep_fn: dict | None
        Preparation function to be called before each step. Defaults to None.

    advection_use_BiCG: bool
        Whether to use BiCG solver for advection. Defaults to True.

    pressure_use_BiCG: bool
        Whether to use BiCG solver for pressure. Defaults to False.

    scipy_solve_advection: bool
        Whether to use SciPy solver for advection. Defaults to False.

    scipy_solve_pressure: bool
        Whether to use SciPy solver for pressure. Defaults to False.

    preconditionBiCG: bool
        Whether to precondition BiCG solver. Defaults to False.

    BiCG_precondition_fallback: bool
        Whether to fallback if preconditioning fails. Defaults to True.

    advection_tol: float | None
        Tolerance for advection solver. If None, uses default. Defaults to None.

    pressure_tol: float | None
        Tolerance for pressure solver. If None, uses default. Defaults to None.

    flux_balance_tol: float
        Tolerance for flux balance check before each step. Defaults to 1e-5.

    convergence_tol: float | None
        Convergence tolerance for the solver. If None, uses default. Defaults to None.

    solver_double_fallback: bool
        Whether to fallback to double precision if single precision fails.
        Defaults to False.

    advect_non_ortho_steps: int
        Number of non-orthogonal corrector steps for advection. Defaults to 1.

    pressure_non_ortho_steps: int
        Number of non-orthogonal corrector steps for pressure. Defaults to 1.

    normalize_pressure_result: bool
        Whether to normalize pressure results. Defaults to True.

    pressure_return_best_result: bool
        Whether to return the best pressure result. Defaults to False.

    advect_passive_scalar: bool
        Whether to advect passive scalars. Defaults to True.

    pressure_time_step_normalized: bool
        Whether pressure is normalized by time step. Defaults to False.

    velocity_corrector: str
        Type of velocity corrector to use. Defaults to "FD".

    non_orthogonal: bool
        Whether to use non-orthogonal corrections. Defaults to True.

    differentiable: bool
        Whether to use the differentiable solver backend. Defaults to False.

    exclude_advection_solve_gradients: bool
        Whether to exclude gradients from advection solve. Defaults to False.

    exclude_pressure_solve_gradients: bool
        Whether to exclude gradients from pressure solve. Defaults to False.

    output_resampling_shape: tuple[int, ...] | None
        Shape for output resampling. If None, dynamic shape is used. Defaults to None.

    output_resampling_fill_max_steps: int
        Number of steps to fill for max step resampling. Defaults to 0.
    """

    def __init__(
        self,
        domain: PISOtorch.Domain,
        dt: float,
        verbose: bool = False,
        substeps: int | Literal["ADAPTIVE"] = 1,
        corrector_steps: int = 2,
        density_viscosity: float | None = None,
        adaptive_CFL: float = 0.8,
        prep_fn: dict | None = None,
        advection_use_BiCG: bool = True,
        pressure_use_BiCG: bool = False,
        scipy_solve_advection: bool = False,
        scipy_solve_pressure: bool = False,
        preconditionBiCG: bool = False,
        BiCG_precondition_fallback: bool = True,
        advection_tol: float | None = None,
        pressure_tol: float | None = None,
        flux_balance_tol: float = 1e-5,
        convergence_tol: float | None = None,
        solver_double_fallback: bool = False,
        advect_non_ortho_steps: int = 1,
        pressure_non_ortho_steps: int = 1,
        normalize_pressure_result: bool = True,
        pressure_return_best_result: bool = False,
        advect_passive_scalar: bool = True,
        pressure_time_step_normalized: bool = False,
        velocity_corrector: str = "FD",
        non_orthogonal: bool = True,
        differentiable: bool = False,
        exclude_advection_solve_gradients: bool = False,
        exclude_pressure_solve_gradients: bool = False,
        output_resampling_shape: tuple[int, ...] | None = None,
        output_resampling_fill_max_steps: int = 0,
    ):
        super().__init__(
            domain=domain,
            time_step=dt,
            substeps=substeps,  # type: ignore
            corrector_steps=corrector_steps,
            density_viscosity=density_viscosity,  # type: ignore
            adaptive_CFL=adaptive_CFL,
            prep_fn=prep_fn,
            advection_use_BiCG=advection_use_BiCG,
            pressure_use_BiCG=pressure_use_BiCG,
            scipy_solve_advection=scipy_solve_advection,
            scipy_solve_pressure=scipy_solve_pressure,
            preconditionBiCG=preconditionBiCG,
            BiCG_precondition_fallback=BiCG_precondition_fallback,
            advection_tol=advection_tol,  # type: ignore
            pressure_tol=pressure_tol,  # type: ignore
            convergence_tol=convergence_tol,  # type: ignore
            solver_double_fallback=solver_double_fallback,
            advect_non_ortho_steps=advect_non_ortho_steps,
            pressure_non_ortho_steps=pressure_non_ortho_steps,
            normalize_pressure_result=normalize_pressure_result,
            pressure_return_best_result=pressure_return_best_result,
            advect_passive_scalar=advect_passive_scalar,
            pressure_time_step_normalized=pressure_time_step_normalized,
            velocity_corrector=velocity_corrector,
            non_orthogonal=non_orthogonal,
            differentiable=differentiable,
            exclude_advection_solve_gradients=exclude_advection_solve_gradients,
            exclude_pressure_solve_gradients=exclude_pressure_solve_gradients,
            log_dir=None,  # type: ignore
            log_interval=0,
            log_images=False,
            log_vtk=False,
            norm_vel=False,
            block_layout=None,
            log_fn=None,
            output_resampling_coords=None,
            output_resampling_shape=output_resampling_shape,
            output_resampling_fill_max_steps=output_resampling_fill_max_steps,
            save_domain_name=None,  # type: ignore
            stop_fn=lambda: False,
        )

        if not verbose:
            # We don't want extensive logging
            get_logger("PISOsim").setLevel("ERROR")

        self.flux_balance_tol = flux_balance_tol
        self._cpu_device = torch.device("cpu")

    def single_step(self, static: bool = False) -> bool:
        """Perform a single simulation step.

        Parameters
        ----------
        static: bool
            If True, perform a static step without adaptive substepping.
            Default is False.

        Returns
        -------
        bool
            True if the simulation step was successful, False otherwise.
        """
        domain_flux_balance = self.domain.GetBoundaryFluxBalance()
        if ntonp(torch.abs(domain_flux_balance)) > ntonp(self.flux_balance_tol):
            raise RuntimeError(
                f"Domain boundary fluxes not balanced, cannot proceed with simulation "
                f"step. Flux balance: {domain_flux_balance}, "
                f"flux_balance_tol: {self.flux_balance_tol}"
            )

        time_step_target = self.time_step
        substeps = self.substeps
        CFL_cond = 0.8
        adaptive_step = False
        time_step = None

        if isinstance(substeps, int) and substeps > 0:
            # just fixed substeps.
            # 1 iteration with have physical time = time_step*substeps.
            pass
        elif substeps == -1:
            # compute max time step for each iteration/substep based on current
            # velocity. 1 iteration with have physical time = time_step.
            adaptive_step = True
        elif substeps == -2:
            # compute max time step based on initial conditions, then keep it constant.
            # 1 iteration with have physical time = time_step.
            time_step, substeps = get_max_time_step(
                self.domain, time_step_target, CFL_cond, with_transformations=True
            )
            self.__LOG.info(
                "Setting time step to %.02e,"
                "substeps to %d based on initial conditions.",
                time_step,
                substeps,
            )
            time_step = torch.tensor(
                [time_step],
                dtype=self.domain.getBlock(0).velocity.dtype,
                device=self._cpu_device,
            )
        else:
            raise ValueError("Invalid substeps")

        try:
            if static:
                sim_ok = self.advect_static(iterations=substeps, time_step=time_step)
            elif adaptive_step:
                sim_ok = self._PISO_adaptive_step()
            else:
                sim_ok = self._PISO_split_step(iterations=substeps, time_step=time_step)
        except PISOtorch_diff.LinsolveError:
            self.__LOG.exception(
                "Simulation failed in step (total step %d):",
                self.total_step,
            )
            return False

        return sim_ok
