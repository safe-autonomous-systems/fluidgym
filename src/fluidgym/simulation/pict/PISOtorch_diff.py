# Copyright 2025 Aleksandra Franz, Nils Thürey
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os

import torch
from fluidgym.simulation.extensions import PISOtorch  # type: ignore[import-untyped]

from fluidgym.simulation import solver_stats
from fluidgym.simulation.amg import amg_pcg_solve
from fluidgym.simulation.pict.util.profiling import SAMPLE
from fluidgym.simulation.pict.util.output import StringWriter

import logging

_LOG = logging.getLogger("fluidgym.Diff")
_LOG_DEBUG = False


_LINSOLVE_BACKEND = None


def set_linsolve_backend(backend):
    """Select the backend used for every linear solve.

    Parameters
    ----------
    backend : str or None
        ``"piso"`` for the bundled CUDA CG/BiCGStab kernels (default),
        ``"sla"`` for the torch-sla backed solver. ``None`` restores the
        default, re-reading ``FLUIDGYM_LINSOLVE_BACKEND``.

    Notes
    -----
    Every solve in the codebase funnels through ``_linear_solve_wrapper``, so
    this switches both the differentiable and non-differentiable paths, forward
    and backward, at once.
    """
    global _LINSOLVE_BACKEND
    if backend is None:
        _LINSOLVE_BACKEND = None
        return
    if backend not in ("piso", "sla"):
        raise ValueError(
            "Unknown linear solve backend %r, expected 'piso' or 'sla'." % (backend,)
        )
    if backend == "sla":
        from fluidgym.simulation.pict.PISOtorch_sla import PISOtorchSLA

        _LINSOLVE_BACKEND = PISOtorchSLA()
    else:
        _LINSOLVE_BACKEND = PISOtorch


def _get_linsolve_backend():
    """Return the module-like object providing ``SolveLinear``."""
    global _LINSOLVE_BACKEND
    if _LINSOLVE_BACKEND is None:
        set_linsolve_backend(os.environ.get("FLUIDGYM_LINSOLVE_BACKEND", "piso"))
    return _LINSOLVE_BACKEND


_EXCLUDED_GRADIENTS = {}


def exclude_gradient(function_name: str, grad_name: str, is_gradient_input: bool):
    global _EXCLUDED_GRADIENTS
    if function_name not in _EXCLUDED_GRADIENTS:
        raise ValueError("Function '%s' does not exist" % (function_name,))
    if not grad_name.endswith("_GRAD"):
        raise ValueError(
            "Only gradients can be excluded (name should end with '_GRAD')."
        )
    _EXCLUDED_GRADIENTS[function_name][0 if is_gradient_input else 1].add(grad_name)


def clear_excluded_gradients():
    global _EXCLUDED_GRADIENTS
    for function_name, (input_set, output_set) in _EXCLUDED_GRADIENTS.items():
        input_set.clear()
        output_set.clear()


def _set_gradient_input(var_grad, grad_setter, function_name, grad_name):
    if grad_name in _EXCLUDED_GRADIENTS[function_name][0]:
        grad_setter(torch.zeros_like(var_grad))
    else:
        grad_setter(var_grad)


def check_non_ortho_rhs(non_ortho_flags):
    # these must match the definition in 'PISO_multiblock_cuda.h'
    NON_ORTHO_DIRECT_MATRIX = 1
    NON_ORTHO_DIRECT_RHS = 2  # less stable than NON_ORTHO_DIRECT_MATRIX
    NON_ORTHO_DIAGONAL_MATRIX = 4  # not implemented
    NON_ORTHO_DIAGONAL_RHS = 8
    NON_ORTHO_CENTER_MATRIX = 16

    return (non_ortho_flags & NON_ORTHO_DIRECT_RHS) > 0 or (
        non_ortho_flags & NON_ORTHO_DIAGONAL_RHS
    ) > 0


def flatten_domain(
    domain,
    tensor_filter: list,
    clone: bool = False,
    empty: bool = False,
    exclusion_list: list = [],
):
    tensor_list = []
    tensor_idx = 0

    def add_tensor(dict, name, tensor, on_cpu=False):
        if tensor_filter is None or name in tensor_filter:
            nonlocal tensor_idx
            dict[name] = tensor_idx
            if empty or name in exclusion_list:
                tensor = None
            elif (
                tensor is not None
            ):  # some fields are optional and return None if not set
                # if tensor is None: raise ValueError("Tensor '"+name+"' is None.")
                if clone:
                    tensor = tensor.clone()
                if on_cpu:
                    tensor = tensor.cpu()
            tensor_list.append(tensor)
            tensor_idx += 1

    domain_dict = {}
    add_tensor(domain_dict, "C", domain.C.value)
    add_tensor(domain_dict, "C_GRAD", domain.CGrad.value)
    add_tensor(domain_dict, "A", domain.A)
    add_tensor(domain_dict, "A_GRAD", domain.AGrad)
    add_tensor(domain_dict, "P", domain.P.value)
    add_tensor(domain_dict, "P_GRAD", domain.PGrad.value)
    add_tensor(domain_dict, "VISCOSITY", domain.viscosity)
    add_tensor(domain_dict, "VISCOSITY_GRAD", domain.viscosityGrad, on_cpu=True)
    add_tensor(domain_dict, "PASSIVE_SCALAR_VISCOSITY", domain.passiveScalarViscosity)
    add_tensor(
        domain_dict, "PASSIVE_SCALAR_VISCOSITY_GRAD", domain.passiveScalarViscosityGrad
    )

    add_tensor(domain_dict, "PASSIVE_SCALAR_RESULT", domain.scalarResult)
    add_tensor(domain_dict, "PASSIVE_SCALAR_RESULT_GRAD", domain.scalarResultGrad)
    add_tensor(domain_dict, "VELOCITY_RESULT", domain.velocityResult)
    add_tensor(domain_dict, "VELOCITY_RESULT_GRAD", domain.velocityResultGrad)
    add_tensor(domain_dict, "PRESSURE_RESULT", domain.pressureResult)
    add_tensor(domain_dict, "PRESSURE_RESULT_GRAD", domain.pressureResultGrad)
    add_tensor(domain_dict, "EPOT_RESULT", domain.epotResult)
    add_tensor(domain_dict, "EPOT_RESULT_GRAD", domain.epotResultGrad)

    add_tensor(domain_dict, "PASSIVE_SCALAR_RHS", domain.scalarRHS)
    add_tensor(domain_dict, "PASSIVE_SCALAR_RHS_GRAD", domain.scalarRHSGrad)
    add_tensor(domain_dict, "VELOCITY_RHS", domain.velocityRHS)
    add_tensor(domain_dict, "VELOCITY_RHS_GRAD", domain.velocityRHSGrad)
    add_tensor(domain_dict, "PRESSURE_RHS", domain.pressureRHS)
    add_tensor(domain_dict, "PRESSURE_RHS_GRAD", domain.pressureRHSGrad)
    add_tensor(domain_dict, "PRESSURE_RHS_DIV", domain.pressureRHSdiv)
    add_tensor(domain_dict, "PRESSURE_RHS_DIV_GRAD", domain.pressureRHSdivGrad)

    domain_dict["BLOCKS"] = []
    for block in domain.getBlocks():
        block_dict = {}

        add_tensor(block_dict, "VISCOSITY_BLOCK", block.viscosity)
        add_tensor(block_dict, "VISCOSITY_BLOCK_GRAD", block.viscosityGrad)
        add_tensor(block_dict, "VELOCITY", block.velocity)
        add_tensor(block_dict, "VELOCITY_GRAD", block.velocityGrad)
        add_tensor(block_dict, "VELOCITY_SOURCE", block.velocitySource)
        add_tensor(block_dict, "VELOCITY_SOURCE_GRAD", block.velocitySourceGrad)
        add_tensor(block_dict, "PASSIVE_SCALAR", block.passiveScalar)
        add_tensor(block_dict, "PASSIVE_SCALAR_GRAD", block.passiveScalarGrad)
        add_tensor(block_dict, "PRESSURE", block.pressure)
        add_tensor(block_dict, "PRESSURE_GRAD", block.pressureGrad)
        add_tensor(block_dict, "EPOT", block.epot)
        add_tensor(block_dict, "EPOT_GRAD", block.epotGrad)

        block_dict["BOUNDARIES"] = [None] * (domain.getSpatialDims() * 2)
        for boundary_idx in range(domain.getSpatialDims() * 2):
            bound = block.getBoundary(boundary_idx)
            if isinstance(
                bound, PISOtorch.FixedBoundary
            ):  # bound.type==PISOtorch.FIXED:
                bound_dict = {}

                add_tensor(bound_dict, "BOUNDARY_VELOCITY", bound.velocity)
                add_tensor(bound_dict, "BOUNDARY_VELOCITY_GRAD", bound.velocityGrad)
                add_tensor(bound_dict, "BOUNDARY_PASSIVE_SCALAR", bound.passiveScalar)
                add_tensor(
                    bound_dict, "BOUNDARY_PASSIVE_SCALAR_GRAD", bound.passiveScalarGrad
                )

                block_dict["BOUNDARIES"][boundary_idx] = bound_dict

        domain_dict["BLOCKS"].append(block_dict)

    return domain_dict, tensor_list


def set_domain_tensors_from_flat(
    domain, domain_dict, tensor_list, tensor_filter: list = None, clone=False
):
    def set_tensor(dict, name, setter, clear_fn=None):
        if (tensor_filter is None or name in tensor_filter) and (name in dict):
            # _LOG.debug("set '"+name+"' to domain")
            tensor = tensor_list[dict[name]]
            if tensor is not None:
                if clone:
                    tensor = tensor.clone()
                setter(tensor)
            elif clear_fn is not None:
                clear_fn()

    set_tensor(domain_dict, "C", domain.C.setValue)
    set_tensor(domain_dict, "C_GRAD", domain.CGrad.setValue)
    set_tensor(domain_dict, "A", domain.setA)
    set_tensor(domain_dict, "A_GRAD", domain.setAGrad)
    set_tensor(domain_dict, "P", domain.P.setValue)
    set_tensor(domain_dict, "P_GRAD", domain.PGrad.setValue)
    set_tensor(domain_dict, "VISCOSITY", domain.setViscosity)
    # set_tensor(domain_dict, "VISCOSITY_GRAD", domain.setViscosityGrad)
    set_tensor(
        domain_dict,
        "PASSIVE_SCALAR_VISCOSITY",
        domain.setScalarViscosity,
        domain.clearScalarViscosity,
    )

    set_tensor(domain_dict, "PASSIVE_SCALAR_RESULT", domain.setScalarResult)
    set_tensor(domain_dict, "PASSIVE_SCALAR_RESULT_GRAD", domain.setScalarResultGrad)
    set_tensor(domain_dict, "VELOCITY_RESULT", domain.setVelocityResult)
    set_tensor(domain_dict, "VELOCITY_RESULT_GRAD", domain.setVelocityResultGrad)
    set_tensor(domain_dict, "PRESSURE_RESULT", domain.setPressureResult)
    set_tensor(domain_dict, "PRESSURE_RESULT_GRAD", domain.setPressureResultGrad)
    set_tensor(domain_dict, "EPOT_RESULT", domain.setEpotResult)
    set_tensor(domain_dict, "EPOT_RESULT_GRAD", domain.setEpotResultGrad)

    set_tensor(domain_dict, "PASSIVE_SCALAR_RHS", domain.setScalarRHS)
    set_tensor(domain_dict, "PASSIVE_SCALAR_RHS_GRAD", domain.setScalarRHSGrad)
    set_tensor(domain_dict, "VELOCITY_RHS", domain.setVelocityRHS)
    set_tensor(domain_dict, "VELOCITY_RHS_GRAD", domain.setVelocityRHSGrad)
    set_tensor(domain_dict, "PRESSURE_RHS", domain.setPressureRHS)
    set_tensor(domain_dict, "PRESSURE_RHS_GRAD", domain.setPressureRHSGrad)
    set_tensor(domain_dict, "PRESSURE_RHS_DIV", domain.setPressureRHSdiv)
    set_tensor(domain_dict, "PRESSURE_RHS_DIV_GRAD", domain.setPressureRHSdivGrad)

    for block_idx, block in enumerate(domain.getBlocks()):
        block_dict = domain_dict["BLOCKS"][block_idx]

        set_tensor(
            block_dict, "VISCOSITY_BLOCK", block.setViscosity, block.clearViscosity
        )
        set_tensor(block_dict, "VISCOSITY_BLOCK_GRAD", block.setViscosityGrad)
        set_tensor(block_dict, "VELOCITY", block.setVelocity)
        set_tensor(block_dict, "VELOCITY_GRAD", block.setVelocityGrad)
        set_tensor(
            block_dict,
            "VELOCITY_SOURCE",
            block.setVelocitySource,
            block.clearVelocitySource,
        )
        set_tensor(block_dict, "VELOCITY_SOURCE_GRAD", block.setVelocitySourceGrad)
        set_tensor(block_dict, "PASSIVE_SCALAR", block.setPassiveScalar)
        set_tensor(block_dict, "PASSIVE_SCALAR_GRAD", block.setPassiveScalarGrad)
        set_tensor(block_dict, "PRESSURE", block.setPressure)
        set_tensor(block_dict, "PRESSURE_GRAD", block.setPressureGrad)
        set_tensor(block_dict, "EPOT", block.setEpot)
        set_tensor(block_dict, "EPOT_GRAD", block.setEpotGrad)

        for boundary_idx in range(domain.getSpatialDims() * 2):
            bound = block.getBoundary(boundary_idx)
            if isinstance(
                bound, PISOtorch.FixedBoundary
            ):  # bound.type==PISOtorch.FIXED:
                bound_dict = block_dict["BOUNDARIES"][boundary_idx]

                set_tensor(bound_dict, "BOUNDARY_VELOCITY", bound.setVelocity)
                set_tensor(bound_dict, "BOUNDARY_VELOCITY_GRAD", bound.setVelocityGrad)
                set_tensor(
                    bound_dict, "BOUNDARY_PASSIVE_SCALAR", bound.setPassiveScalar
                )
                set_tensor(
                    bound_dict,
                    "BOUNDARY_PASSIVE_SCALAR_GRAD",
                    bound.setPassiveScalarGrad,
                )


def _get_solver_tolerance(tol=None, dtype=torch.float32):
    if tol is None:
        if dtype == torch.float64:
            tol = 1e-8
        elif dtype == torch.float32:
            tol = 1e-5
    return tol


def _get_solver_tolerance_torch(tol=None, dtype=torch.float32):
    tol = _get_solver_tolerance(tol, dtype)
    tol_torch = torch.tensor([tol], dtype=dtype)
    return tol_torch


def _tol_value(tol) -> float:
    """A tolerance as a Python float, whether it arrives as a tensor or a number.

    ``amg_pcg_solve`` takes a float where the CUDA kernels take a 1-element
    tensor; this is the same conversion ``Simulation.linear_solve_AMG`` does.
    """
    if isinstance(tol, torch.Tensor):
        return float(tol.detach().cpu().reshape(-1)[0])
    return float(tol)


# The forward stops the moment it crosses its threshold, so the relative accuracy
# it *achieved* is essentially that threshold. Asking the adjoint for exactly the
# same on a different right-hand side therefore has no margin over the same
# stagnation floor, and misses it about as often as not: on the 2D cylinder every
# non-converged adjoint solve of a checkpointed backward lands 1.16-1.28x above
# what it was asked for, having found that iterate by iteration ~180 and then
# spending the remaining ~4800 up to `max_iter` on nothing. This factor buys the
# margin back
_BWD_TOL_SAFETY = 2.0


def _rescaled_bwd_tolerance(
    tol_torch,
    fwd_rhs_norm,
    grad_x,
    n_rows=None,
    min_relative=1e-12,
    safety=_BWD_TOL_SAFETY,
):
    """Hold the adjoint solve to the forward's *relative* accuracy, both ways.

    The kernels take an absolute tolerance (ConvergenceCriterion.NORM2_NORMALIZED
    is ``||r|| / sqrt(n)``), and the number handed to a solve is resolved once,
    against the forward RHS. The adjoint system has the same operator but a
    completely different right-hand side, so reusing that number silently changes
    the requested tolerance.
    """
    fwd = float(fwd_rhs_norm)
    if not math.isfinite(fwd) or fwd <= 0:
        return tol_torch
    bwd = float(grad_x.detach().norm())
    if not math.isfinite(bwd) or bwd <= 0:
        return tol_torch

    scaled = _tol_value(tol_torch) * (bwd / fwd) * safety

    # The criterion divides by sqrt(n), so the attainable floor does too
    n = float(n_rows) if n_rows else float(grad_x.numel())
    rms = bwd / math.sqrt(max(n, 1.0))
    floor = min_relative * rms
    value = max(scaled, floor)

    # Never tighter than the forward itself, unless that target is
    # already met at the zero iterate, which would return a zero gradient
    fwd_tol = _tol_value(tol_torch)
    if rms > fwd_tol:
        value = max(value, fwd_tol)

    if isinstance(tol_torch, torch.Tensor):
        return torch.full_like(tol_torch, value)
    return value


class LinsolveError(RuntimeError):
    pass


def _check_solver_return_infos(
    solver_infos,
    transpose,
    use_BiCG,
    tol,
    max_iter,
    return_best_result,
    is_FWD=None,
    debug_out=False,
):
    # solver_infos: list(PISOtorch.LinearSolverResultInfo), as returned by PISOtorch.SolveLinear()

    fwd_str = "" if is_FWD is None else ("FWD, " if is_FWD else "BWD, ")

    if any(not solver_info.isFiniteResidual for solver_info in solver_infos):
        non_finite_batches = [
            i
            for i, solver_info in enumerate(solver_infos)
            if not solver_info.isFiniteResidual
        ]
        s = StringWriter()
        for i, solver_info in enumerate(solver_infos):
            s.write_line("\tRHS %d: %s", i, solver_info)
        raise LinsolveError(
            "Linear solve (%sT:%s, BiCG:%s, tol:%.03e) reported non-finite residual for RHS %s of %d. Solver infos:\n%s."
            % (
                fwd_str,
                transpose,
                use_BiCG,
                tol,
                non_finite_batches,
                len(solver_infos),
                s,
            )
        )
        s.reset()

    elif any(not solver_info.converged for solver_info in solver_infos):
        not_converged_batches = [
            i for i, solver_info in enumerate(solver_infos) if not solver_info.converged
        ]
        msg = (
            "Linear solve (%sT:%s, BiCG:%s, tol:%.03e) RHS %s of %d did not converge after %d iterations."
            % (
                fwd_str,
                transpose,
                use_BiCG,
                tol,
                not_converged_batches,
                len(solver_infos),
                max_iter,
            )
        )
        if return_best_result:
            _LOG.warning(msg)
            s = StringWriter()
            for i in not_converged_batches:
                s.write_line(
                    "\tRHS %d: best residual %.03e from iteration %d.",
                    i,
                    solver_infos[i].finalResidual,
                    solver_infos[i].usedIterations,
                )
            _LOG.warning("Using best results:\n%s", s)
            s.reset()
            if debug_out:
                converged_batches = [
                    i
                    for i, solver_info in enumerate(solver_infos)
                    if solver_info.converged
                ]
                for i in converged_batches:
                    s.write_line(
                        "\tRHS %d: residual %.03e from iteration %d.",
                        i,
                        solver_infos[i].finalResidual,
                        solver_infos[i].usedIterations,
                    )
                _LOG.debug("Converged results:\n%s", s)
                s.reset()
        else:
            s = StringWriter()
            for i, solver_info in enumerate(solver_infos):
                s.write_line("\tRHS %d: %s", i, solver_info)
            msg += " Solver infos:\n%s" % (s,)
            raise LinsolveError(msg)

    elif debug_out:
        s = StringWriter()
        for i, solver_info in enumerate(solver_infos):
            s.write_line(
                "\tRHS %d: residual %.03e from iteration %d.",
                i,
                solver_info.finalResidual,
                solver_info.usedIterations,
            )
        _LOG.debug(
            "Linear solve (%sT:%s, BiCG:%s, tol:%.03e) converged:\n%s",
            fwd_str,
            transpose,
            use_BiCG,
            tol,
            s,
        )
        s.reset()


def _linear_solve_wrapper(
    csrMat: PISOtorch.CSRmatrix,
    rhs: torch.Tensor,
    result: torch.Tensor,
    maxit_torch: torch.Tensor,
    tol_torch: torch.Tensor,
    convergence_criterion: PISOtorch.ConvergenceCriterion,
    use_BiCG: bool,
    matrix_rank_deficient: bool,
    residual_reset_step: int,
    transpose: bool,
    print_residual: bool,
    return_best_result: bool,
    is_FWD: bool = None,
    debug_out: bool = False,
    double_fallback: bool = False,
    BiCG_with_preconditioner: bool = True,
    BiCG_precondition_fallback: bool = False,
    tag: str = "unknown",
):
    if not rhs.eq(0).all():
        backend = _get_linsolve_backend()

        # Opt-in telemetry (fluidgym.simulation.solver_stats). Nothing is
        # measured, allocated or synchronized unless a recorder is active.
        probe = solver_stats.Probe(rhs, tag) if solver_stats.is_active() else None
        if probe is not None:
            probe.start()

        # with SAMPLE("PISOtorch.SolveLinear"):
        solver_infos = backend.SolveLinear(
            csrMat,
            rhs,
            result,
            maxit_torch,
            tol_torch,
            convergence_criterion,
            use_BiCG,
            matrix_rank_deficient,
            residual_reset_step,
            transpose,
            print_residual,
            return_best_result,
            BiCGwithPreconditioner=BiCG_with_preconditioner,
        )

        def not_solved(sinfos):
            return (
                not return_best_result and any(not sinfo.converged for sinfo in sinfos)
            ) or (
                return_best_result
                and any(not sinfo.isFiniteResidual for sinfo in sinfos)
            )

        if double_fallback and rhs.dtype == torch.float32 and not_solved(solver_infos):
            _LOG.warning("Single precision solve failed, trying double precision.")
            debug_out = True
            if debug_out:
                _LOG.debug(
                    "Single precision solver infos:\n%s", [str(_) for _ in solver_infos]
                )

            dp = torch.float64
            result_dp = torch.zeros_like(
                result, dtype=dp
            )  # do not start with a possibly corrupted result tensor
            solver_infos = backend.SolveLinear(
                csrMat.toType(dp),
                rhs.to(dp),
                result_dp,
                maxit_torch,
                tol_torch.to(dp),
                convergence_criterion,
                use_BiCG,
                matrix_rank_deficient,
                residual_reset_step,
                transpose,
                print_residual,
                return_best_result,
                BiCGwithPreconditioner=BiCG_with_preconditioner,
            )
            result = result_dp.to(result.dtype)
        # elif not double_fallback:
        #    raise RuntimeWarning("double_fallback is off!")

        if (
            BiCG_precondition_fallback
            and use_BiCG
            and not BiCG_with_preconditioner
            and not_solved(solver_infos)
        ):
            _LOG.warning("Not preconditioned BiCG solve failed, trying preconditioned.")
            debug_out = True
            if debug_out:
                _LOG.debug("Solver infos:\n%s", [str(_) for _ in solver_infos])

            result.zero_()  # may contain nan after failed solve

            solver_infos = backend.SolveLinear(
                csrMat,
                rhs,
                result,
                maxit_torch,
                tol_torch,
                convergence_criterion,
                use_BiCG,
                matrix_rank_deficient,
                residual_reset_step,
                transpose,
                print_residual,
                return_best_result,
                BiCGwithPreconditioner=True,
            )

        tol_value = tol_torch.detach().cpu().numpy()[0]
        maxit_value = maxit_torch.detach().cpu().numpy()[0]

        if probe is not None:
            probe.stop(
                solver_infos,
                tolerance=float(tol_value),
                max_iterations=int(maxit_value),
                use_BiCG=use_BiCG,
                transpose=transpose,
                is_fwd=is_FWD,
            )

        _check_solver_return_infos(
            solver_infos,
            transpose,
            use_BiCG,
            tol_value,
            maxit_value,
            return_best_result,
            is_FWD=is_FWD,
            debug_out=debug_out,
        )

    else:
        result.zero_()


def _project_out_constant(v: torch.Tensor, n_rows: int) -> torch.Tensor:
    """Remove the constant mode from each right-hand side in ``v``.

    For a singular operator whose nullspace is the constant vector (a pure
    Neumann Laplacian, i.e. every boundary carrying a flux condition and none
    prescribing a value) ``A y = g`` is solvable only for ``g`` orthogonal to
    ``null(A^T)``, and its solution is unique only up to ``null(A)``. Both spaces
    are ``span(1)`` for the symmetric case, so projecting the RHS makes the solve
    well posed and projecting the result selects the minimum-norm solution.
    """
    if n_rows <= 0 or v.numel() % n_rows != 0:
        return v
    shaped = v.reshape(-1, n_rows)
    return (shaped - shaped.mean(dim=1, keepdim=True)).reshape(v.shape)


def linear_solve_GPU(
    csrMat: PISOtorch.CSRmatrix,
    rhs: torch.Tensor,
    transpose=False,
    use_BiCG=False,
    tol=None,
    max_iter=5000,
    return_best_result=False,
    double_fallback: bool = False,
    BiCG_with_preconditioner: bool = True,
    BiCG_precondition_fallback: bool = False,
    adjoint_rank_deficient: bool = False,
    tag: str = "unknown",
    x0: torch.Tensor = None,
    amg_hierarchy=None,
    matrix_rank_deficient: bool = False,
    residual_reset_step: int = 0,
):
    # The matrix is *not* cloned into this closure: a deep copy (value + index
    # + row) per solve is retained for the whole graph and, living in a Python
    # closure, is invisible to torch.autograd.graph hooks. Only the integer
    # sparsity pattern is captured -- it is never differentiated and is the same
    # object on every step, so it costs one pattern rather than one per solve.
    # The values travel through save_for_backward instead, which keeps a single
    # copy.
    A_index = csrMat.index
    A_row = csrMat.row

    def _make_matrix(values):
        """Rebuild the operator from a value vector and the shared pattern."""
        return PISOtorch.CSRmatrix(values.detach(), A_index, A_row)

    convergence_criterion = PISOtorch.ConvergenceCriterion.NORM2_NORMALIZED
    dtype = rhs.dtype

    # Captured at function scope and detached, so it can never become an autograd
    # input. Exact: the map being differentiated is `b -> A^-1 b`, whose Jacobian
    # does not depend on the initial iterate.
    x0_const = (
        None
        if x0 is None
        else x0.detach().reshape(rhs.shape).contiguous()
    )

    # Safe here because forward and backward both run with grad disabled, so no
    # V-cycle is traced; `not use_BiCG` asserts the operator is symmetric, which is
    # what makes the same hierarchy valid for the adjoint solve.
    use_amg = amg_hierarchy is not None and not use_BiCG

    # `adjoint_rank_deficient` says the operator has the constant nullspace. It
    # deliberately does *not* touch the forward solve -- neither the kernel flag
    # above nor the RHS -- so forward trajectories stay bit-identical and this
    # change is confined to the gradient. The forward is well posed as it stands
    # because its RHS is compatible by construction (for the pressure Poisson,
    # `balance_boundary_fluxes` enforces zero net boundary flux). The adjoint RHS
    # is an incoming gradient with no such guarantee, which is what makes the
    # backward solve ill-posed on a duct where every boundary prescribes velocity.
    A_rows = csrMat.getRows()

    tol_torch = _get_solver_tolerance_torch(
        tol, dtype
    )  # torch.tensor([tol], dtype=dtype)
    maxit_torch = torch.IntTensor([max_iter])

    class LinearSolveFunction(torch.autograd.Function):
        # solves Ax=b
        @staticmethod
        def forward(ctx, A_val: torch.Tensor, b: torch.Tensor):
            with SAMPLE("%sCG-FWD" % ("Bi" if use_BiCG else "",)):
                if _LOG_DEBUG:
                    _LOG.debug("linsolve %sCG forward", "Bi" if use_BiCG else "")

                # Cloned because the solvers write the iterate into `x` in place.
                if x0_const is None:
                    x = torch.zeros_like(b)
                else:
                    x = x0_const.to(dtype=b.dtype, device=b.device).clone()

                if use_amg:
                    # Probed like every other solve, or an AMG solve would be
                    # invisible to solver_stats and its cost unattributable.
                    probe = (
                        solver_stats.Probe(b, tag) if solver_stats.is_active() else None
                    )
                    if probe is not None:
                        probe.start()
                    with torch.no_grad():
                        solver_infos = amg_pcg_solve(
                            b,
                            x,
                            amg_hierarchy,
                            tol=_tol_value(tol_torch),
                            max_iter=max_iter,
                            return_best_result=return_best_result,
                        )
                    if probe is not None:
                        probe.stop(
                            solver_infos,
                            tolerance=_tol_value(tol_torch),
                            max_iterations=max_iter,
                            use_BiCG=use_BiCG,
                            transpose=transpose,
                            is_fwd=True,
                        )
                    _check_solver_return_infos(
                        solver_infos,
                        transpose,
                        use_BiCG,
                        _tol_value(tol_torch),
                        max_iter,
                        return_best_result,
                        is_FWD=True,
                        debug_out=False,
                    )
                else:
                    # Local, so it is released when forward returns; only A_val is
                    # retained, via save_for_backward below.
                    A_fwd = _make_matrix(A_val)

                    _linear_solve_wrapper(
                        A_fwd,
                        b,
                        x,
                        maxit_torch,
                        tol_torch,
                        convergence_criterion,
                        use_BiCG,
                        matrix_rank_deficient,
                        residual_reset_step,
                        transpose,
                        False,
                        return_best_result,
                        is_FWD=True,
                        debug_out=False,
                        double_fallback=double_fallback,
                        BiCG_with_preconditioner=BiCG_with_preconditioner,
                        BiCG_precondition_fallback=BiCG_precondition_fallback,
                        tag=tag,
                    )

                    del A_fwd

                # ctx.save_for_backward(A_val, b, x)
                ctx.save_for_backward(A_val, x)
                # Scale `tol` was resolved against. `tol_torch` is computed once,
                # from *this* RHS, but backward solves a system whose RHS is
                # grad_x -- see the tolerance rescale in backward().
                ctx.fwd_rhs_norm = b.detach().norm()

            return x  # , (0<=it and it<maxit)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_x):
            with SAMPLE("%sCG-BWD" % ("Bi" if use_BiCG else "",)):
                if _LOG_DEBUG:
                    _LOG.debug("linsolve %sCG backward", "Bi" if use_BiCG else "")

                grad_b = None
                grad_A_val = None
                if ctx.needs_input_grad[1] or ctx.needs_input_grad[0]:
                    with SAMPLE("RHSgrad"):
                        # A_val, b, x = ctx.saved_tensors
                        A_val, x = ctx.saved_tensors
                        A = _make_matrix(A_val)
                        grad_b = torch.zeros_like(grad_x)

                        # Make the adjoint solve well posed on a singular
                        # operator. Without this the incoming gradient carries a
                        # component along null(A^T) that no iterate can reduce,
                        # so CG cannot converge and returns a vector whose scale
                        # is set by the residual floor rather than by the
                        # gradient -- the mechanism behind the momentum adjoint
                        # gaining ~1e10 per step on the MHD duct.
                        if adjoint_rank_deficient:
                            grad_x = _project_out_constant(grad_x, A_rows)

                        # NORM2_NORMALIZED is an *absolute* RMS, and tol_torch
                        # was resolved against the forward RHS, so on its own it
                        # asks the adjoint for something unrelated to the adjoint:
                        # unreachable when ||grad_x|| >> ||b||, and already met at
                        # the zero iterate when ||grad_x|| << ||b||, which returns
                        # a zero adjoint and severs the chain. Rescaling holds it
                        # to the forward's relative accuracy either way.
                        #
                        # After the projection above, so the norm is the one the
                        # solve actually starts from
                        bwd_tol = _rescaled_bwd_tolerance(
                            tol_torch, ctx.fwd_rhs_norm, grad_x, n_rows=A_rows
                        )

                        # if not grad_x.eq(0).all(): # will not converge if all 0, but grad should be 0 anyways in that case
                        # solver_info = PISOtorch.SolveLinear(A, grad_x, grad_b, maxit_torch, tol_torch, conv, use_BiCG, False, 0, not transpose, False, return_best_result)

                        # _check_solver_return_infos(solver_info, not transpose, use_BiCG, tol, max_iter, return_best_result, is_FWD=False, debug_out=False)
                        if use_amg:
                            # A^T == A, so the same hierarchy preconditions the
                            # adjoint solve; grad_b is already a zero iterate.
                            probe = (
                                solver_stats.Probe(grad_x, tag)
                                if solver_stats.is_active()
                                else None
                            )
                            if probe is not None:
                                probe.start()
                            with torch.no_grad():
                                solver_infos = amg_pcg_solve(
                                    grad_x,
                                    grad_b,
                                    amg_hierarchy,
                                    tol=_tol_value(bwd_tol),
                                    max_iter=max_iter,
                                    return_best_result=return_best_result,
                                )
                            if probe is not None:
                                probe.stop(
                                    solver_infos,
                                    tolerance=_tol_value(bwd_tol),
                                    max_iterations=max_iter,
                                    use_BiCG=use_BiCG,
                                    transpose=not transpose,
                                    is_fwd=False,
                                )
                            _check_solver_return_infos(
                                solver_infos,
                                not transpose,
                                use_BiCG,
                                _tol_value(bwd_tol),
                                max_iter,
                                return_best_result,
                                is_FWD=False,
                                debug_out=False,
                            )
                        else:
                            _linear_solve_wrapper(
                                A,
                                grad_x,
                                grad_b,
                                maxit_torch,
                                bwd_tol,
                                convergence_criterion,
                                use_BiCG,
                                matrix_rank_deficient,
                                0,
                                not transpose,
                                False,
                                return_best_result,
                                is_FWD=False,
                                debug_out=False,
                                double_fallback=double_fallback,
                                BiCG_with_preconditioner=BiCG_with_preconditioner,
                                BiCG_precondition_fallback=BiCG_precondition_fallback,
                                tag=tag,
                            )

                    # Pick the minimum-norm adjoint: with a constant nullspace
                    # grad_b is determined only up to a constant, and letting
                    # that component ride along would feed an arbitrary offset
                    # into SparseOuterProduct below and into the next step.
                    if adjoint_rank_deficient:
                        grad_b = _project_out_constant(grad_b, A_rows)

                    # print("grad_b", grad_b)
                    if ctx.needs_input_grad[0]:  # gradient w.r.t. matrix
                        with SAMPLE("MatrixGrad"):
                            grad_A = (
                                A.WithZeroValue()
                            )  # creates new tensor for a.value initialized to 0
                            grad_A_val = grad_A.value
                            # if _LOG_DEBUG: _LOG.debug("A %s, db %s, x %s", grad_A, grad_b.size(), x.size())
                            if grad_A.getRows() == grad_b.size(0):  # no batched RHS
                                PISOtorch.SparseOuterProduct(grad_b, x, grad_A)
                            else:
                                grad_b_splits = torch.split(grad_b, grad_A.getRows())
                                x_splits = torch.split(x, grad_A.getRows())
                                for grad_b_split, x_split in zip(
                                    grad_b_splits, x_splits
                                ):
                                    grad_A_split = grad_A.WithZeroValue()
                                    PISOtorch.SparseOuterProduct(
                                        grad_b_split, x_split, grad_A_split
                                    )
                                    # A_val
                                    grad_A_val += grad_A_split.value
                            grad_A_val *= -1

            return grad_A_val, grad_b

    return LinearSolveFunction.apply(csrMat.value, rhs)


_EXCLUDED_GRADIENTS["SetupAdvectionMatrix"] = (set(), set())


def SetupAdvectionMatrix(
    domain, time_step, non_ortho_flags, forPassiveScalar=False, passiveScalarChannel=0
):
    # input tensors that need to be tracked by pytorch
    tracked_tensor_filter = ["VELOCITY", "BOUNDARY_VELOCITY", "VISCOSITY"]
    if forPassiveScalar:
        tracked_tensor_filter.append("PASSIVE_SCALAR_VISCOSITY")
    else:
        tracked_tensor_filter.append("VISCOSITY_BLOCK")

    # the filter to get the gradient tensors corresponding to the tracked input tensors
    grad_tensor_filter = [s + "_GRAD" for s in tracked_tensor_filter]

    domain_dict, tracked_tensors = flatten_domain(domain, tracked_tensor_filter)

    # filter for any forward tensors that need to be saved for backwards
    backwards_tensor_filter = []

    class SetupAdvectionMatrixFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *tracked_tensors):
            with SAMPLE("SetupAdvectionMatrix-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("SetupAdvectionMatrix forward")

                domain.CreateA()
                domain.C.CreateValue()
                domain.UpdateDomainData()
                PISOtorch.SetupAdvectionMatrix(
                    domain,
                    time_step,
                    non_ortho_flags,
                    forPassiveScalar=forPassiveScalar,
                    passiveScalarChannel=passiveScalarChannel,
                )

                saved_tensors = []
                saved_tensors.append(time_step)
                ctx.save_for_backward(*saved_tensors)

            return domain.A, domain.C.value  # .clone()

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, A_grad, C_val_grad):
            # if not non_ortho_flags==0: raise NotImplementedError("Only Orthogonal gradients are supported.")

            if any(ctx.needs_input_grad):
                with SAMPLE("SetupAdvectionMatrix-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("SetupAdvectionMatrix backward")
                    time_step = ctx.saved_tensors[-1]
                    # domain.setAGrad(A_grad)
                    _set_gradient_input(
                        A_grad, domain.setAGrad, "SetupAdvectionMatrix", "A_GRAD"
                    )
                    # domain.CGrad.setValue(C_val_grad)
                    _set_gradient_input(
                        C_val_grad,
                        domain.CGrad.setValue,
                        "SetupAdvectionMatrix",
                        "C_GRAD",
                    )
                    domain.CreateViscosityGrad()  # also creates passive scalar and per-cell viscosity gradient tensors if neccessary
                    domain.CreateVelocityGradOnBlocks()
                    domain.CreateVelocityGradOnBoundaries()
                    domain.UpdateDomainData()
                    PISOtorch.SetupAdvectionMatrixGrad(
                        domain,
                        time_step,
                        non_ortho_flags,
                        forPassiveScalar=forPassiveScalar,
                        passiveScalarChannel=passiveScalarChannel,
                    )
                    # block_velocity_grad = [block.velocityGrad for block in domain.getBlocks()]
                    _, grad_tensors = flatten_domain(
                        domain,
                        grad_tensor_filter,
                        exclusion_list=_EXCLUDED_GRADIENTS["SetupAdvectionMatrix"][1],
                    )
            else:
                if _LOG_DEBUG:
                    _LOG.debug("SetupAdvectionMatrix backward empty")
                # block_velocity_grad = [None] * domain.getNumBlocks()
                _, grad_tensors = flatten_domain(domain, grad_tensor_filter, empty=True)

            # if _LOG_DEBUG: _LOG.debug("->velocity grad: %s", block_velocity_grad)

            return (*grad_tensors,)

    return SetupAdvectionMatrixFunction.apply(*tracked_tensors)


_EXCLUDED_GRADIENTS["SetupAdvectionScalar"] = (set(), set())


def SetupAdvectionScalar(domain, time_step, non_ortho_flags):
    """
    inputs:
    - block.passiveScalar (not needed for backwards)
    - domain.scalarResult (if non-orthogonal)
    - block.transform (must be static, not differentiable)
    - domain.viscosity
    - boundary.passiveScalar
    - boundary.velocity
    - boundary.transform (must be static, not differentiable)
    outputs:
    - domain.scalarRHS
    """
    is_non_ortho = check_non_ortho_rhs(non_ortho_flags)
    # _LOG.debug("SetupAdvectionScalar is_non_ortho: %s",is_non_ortho)
    # additionally uses domain.scalarResult for non-orthogonal handling on the RHS

    tracked_tensor_filter = [
        "PASSIVE_SCALAR",
        "BOUNDARY_PASSIVE_SCALAR",
        "BOUNDARY_VELOCITY",
        "VISCOSITY",
        "PASSIVE_SCALAR_VISCOSITY",
    ]
    if is_non_ortho:
        tracked_tensor_filter.append("PASSIVE_SCALAR_RESULT")

    grad_tensor_filter = [s + "_GRAD" for s in tracked_tensor_filter]

    domain_dict, tracked_tensors = flatten_domain(domain, tracked_tensor_filter)

    backwards_tensor_filter = [
        "BOUNDARY_PASSIVE_SCALAR",
        "BOUNDARY_VELOCITY",
        "VISCOSITY",
        "PASSIVE_SCALAR_VISCOSITY",
    ]
    if is_non_ortho:
        backwards_tensor_filter.append("PASSIVE_SCALAR_RESULT")

    class SetupAdvectionScalarFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *tracked_tensors):
            with SAMPLE("SetupAdvectionScalar-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("scalarRHS forward")
                with torch.no_grad():
                    domain.CreateScalarRHS()
                    domain.UpdateDomainData()
                    PISOtorch.SetupAdvectionScalar(domain, time_step, non_ortho_flags)

                if backwards_tensor_filter:
                    domain_dict, saved_tensors = flatten_domain(
                        domain, backwards_tensor_filter
                    )
                    ctx.saved_tensors_domain_dict = domain_dict
                else:
                    ctx.saved_tensors_domain_dict = None
                    saved_tensors = []

                saved_tensors.append(time_step)
                ctx.save_for_backward(*saved_tensors)

            return domain.scalarRHS  # .clone()

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_rhs):
            # if not non_ortho_flags==0: raise NotImplementedError("Only Orthogonal gradients are supported.")
            with torch.no_grad():
                if any(ctx.needs_input_grad):
                    with SAMPLE("SetupAdvectionScalar-BWD"):
                        if _LOG_DEBUG:
                            _LOG.debug("scalarRHS backward")
                        time_step = ctx.saved_tensors[-1]
                        if ctx.saved_tensors_domain_dict is not None:
                            set_domain_tensors_from_flat(
                                domain, ctx.saved_tensors_domain_dict, ctx.saved_tensors
                            )

                        # domain.setScalarRHSGrad(grad_rhs)
                        _set_gradient_input(
                            grad_rhs,
                            domain.setScalarRHSGrad,
                            "SetupAdvectionScalar",
                            "PASSIVE_SCALAR_RHS_GRAD",
                        )
                        domain.CreateViscosityGrad()  # also creates passive scalar and per-cell viscosity gradient tensors if neccessary
                        # domain.CreatePassiveScalarViscosityGrad() # will clear the gradient field if dedicated passive scalar viscosity is not used
                        domain.CreatePassiveScalarGradOnBlocks()
                        domain.CreatePassiveScalarGradOnBoundaries()
                        domain.CreateVelocityGradOnBoundaries()
                        if is_non_ortho:
                            domain.CreateScalarResultGrad()
                        domain.UpdateDomainData()
                        PISOtorch.SetupAdvectionScalarGrad(
                            domain, time_step, non_ortho_flags
                        )
                        _, grad_tensors = flatten_domain(
                            domain,
                            grad_tensor_filter,
                            exclusion_list=_EXCLUDED_GRADIENTS["SetupAdvectionScalar"][
                                1
                            ],
                        )
                else:
                    if _LOG_DEBUG:
                        _LOG.debug("scalarRHS backward empty")
                    _, grad_tensors = flatten_domain(
                        domain, grad_tensor_filter, empty=True
                    )

            return (*grad_tensors,)

    rhs = SetupAdvectionScalarFunction.apply(*tracked_tensors)
    return rhs


_EXCLUDED_GRADIENTS["SetupAdvectionVelocity"] = (set(), set())


def SetupAdvectionVelocity(
    domain, time_step, non_ortho_flags, apply_pressure_gradient=False
):
    """
    inputs:
    - block.velocity (not needed for backwards)
    - block.velocitySource (not needed for backwards)
    - domain.velocityResult (if non-orthogonal)
    - block.transform (must be static, not differentiable)
    - domain.viscosity
    - block.viscosity
    - boundary.velocity
    - boundary.transform (must be static, not differentiable)
    outputs:
    - domain.velocityRHS
    """

    is_non_ortho = check_non_ortho_rhs(non_ortho_flags)

    tracked_tensor_filter = [
        "VELOCITY",
        "VELOCITY_SOURCE",
        "BOUNDARY_VELOCITY",
        "VISCOSITY",
        "VISCOSITY_BLOCK",
    ]
    if is_non_ortho:
        tracked_tensor_filter.append("VELOCITY_RESULT")

    grad_tensor_filter = [s + "_GRAD" for s in tracked_tensor_filter]

    domain_dict, tracked_tensors = flatten_domain(domain, tracked_tensor_filter)

    backwards_tensor_filter = ["BOUNDARY_VELOCITY", "VISCOSITY", "VISCOSITY_BLOCK"]
    backwards_tensor_filter.append(
        "VELOCITY_SOURCE"
    )  # DEBUG: value not needed but needs to be set to receive gradients
    if is_non_ortho:
        backwards_tensor_filter.append("VELOCITY_RESULT")

    class SetupAdvectionVelocityFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *tracked_tensors):
            with SAMPLE("SetupAdvectionVelocity-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("velocityRHS forward")

                domain.CreateVelocityRHS()
                domain.UpdateDomainData()
                PISOtorch.SetupAdvectionVelocity(
                    domain, time_step, non_ortho_flags, apply_pressure_gradient
                )

                if backwards_tensor_filter:
                    domain_dict, saved_tensors = flatten_domain(
                        domain, backwards_tensor_filter
                    )
                    ctx.saved_tensors_domain_dict = domain_dict
                else:
                    ctx.saved_tensors_domain_dict = None
                    saved_tensors = []

                saved_tensors.append(time_step)
                ctx.save_for_backward(*saved_tensors)

            return domain.velocityRHS  # .clone()

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_rhs):
            # if not non_ortho_flags==0: raise NotImplementedError("Only Orthogonal gradients are supported.")

            if any(ctx.needs_input_grad):
                with SAMPLE("SetupAdvectionVelocity-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("velocityRHS backward")
                    time_step = ctx.saved_tensors[-1]
                    if ctx.saved_tensors_domain_dict is not None:
                        set_domain_tensors_from_flat(
                            domain, ctx.saved_tensors_domain_dict, ctx.saved_tensors
                        )

                    # domain.setVelocityRHSGrad(grad_rhs)
                    _set_gradient_input(
                        grad_rhs,
                        domain.setVelocityRHSGrad,
                        "SetupAdvectionVelocity",
                        "VELOCITY_RHS_GRAD",
                    )
                    domain.CreateViscosityGrad()
                    domain.CreateVelocityGradOnBlocks()
                    domain.CreateVelocityGradOnBoundaries()
                    domain.CreateVelocitySourceGradOnBlocks()  # only create grad if velocity source exists, clears it otherwise
                    if is_non_ortho:
                        domain.CreateVelocityResultGrad()
                    domain.UpdateDomainData()

                    PISOtorch.SetupAdvectionVelocityGrad(
                        domain, time_step, non_ortho_flags, apply_pressure_gradient
                    )
                    # _LOG.debug("boundary[0].velocityGrad:\n%s", domain.getBlock(0).getBoundary(0).velocityGrad)
                    _, grad_tensors = flatten_domain(
                        domain,
                        grad_tensor_filter,
                        exclusion_list=_EXCLUDED_GRADIENTS["SetupAdvectionVelocity"][1],
                    )
            else:
                if _LOG_DEBUG:
                    _LOG.debug("velocityRHS backward empty")
                _, grad_tensors = flatten_domain(domain, grad_tensor_filter, empty=True)

            return (*grad_tensors,)

    return SetupAdvectionVelocityFunction.apply(*tracked_tensors)


def CopyScalarResultToBlocks(domain):
    """
    inputs:
    - domain.scalarResult
    outputs:
    - block.passiveScalar
    """

    class CopyScalarResultToBlocksFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, scalar_result):
            with SAMPLE("CopyScalarResultToBlocks-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("copy scalar forward")

                domain.CreatePassiveScalarOnBlocks()
                domain.UpdateDomainData()
                PISOtorch.CopyScalarResultToBlocks(domain)

                return (
                    *[block.passiveScalar for block in domain.getBlocks()],
                )  # .clone()

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, *blocks_passive_scalar_grad):
            if ctx.needs_input_grad[0]:
                with SAMPLE("CopyScalarResultToBlocks-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("copy scalar backward")
                    for block, s_grad in zip(
                        domain.getBlocks(), blocks_passive_scalar_grad
                    ):
                        block.setPassiveScalarGrad(s_grad)
                    domain.CreateScalarResultGrad()
                    domain.UpdateDomainData()
                    PISOtorch.CopyScalarResultGradFromBlocks(domain)
                    scalarResultGrad = domain.scalarResultGrad
            else:
                if _LOG_DEBUG:
                    _LOG.debug("copy scalar backward empty")
                scalarResultGrad = None

            return scalarResultGrad

    CopyScalarResultToBlocksFunction.apply(domain.scalarResult)


def CopyScalarResultFromBlocks(domain):
    """
    inputs:
    - block.passiveScalar
    outputs:
    - domain.scalarResult
    """

    class CopyScalarResultFromBlocksFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *tracked_tensors):
            with SAMPLE("CopyScalarResultFromBlocks-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("copy scalar forward")

                domain.CreateScalarResult()
                domain.UpdateDomainData()
                PISOtorch.CopyScalarResultFromBlocks(domain)

                return domain.scalarResult

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, scalar_result_grad):
            raise NotImplementedError(
                "TODO: requires PISOtorch.CopyScalarResultGradToBlocks"
            )

            if any(ctx.needs_input_grad):
                with SAMPLE("CopyScalarResultFromBlocks-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("copy scalar backward")
                    domain.setScalarResultGrad(scalar_result_grad)
                    domain.CreatePassiveScalarGradOnBlocks()
                    domain.UpdateDomainData()
                    PISOtorch.CopyScalarResultGradToBlocks(domain)
                    block_scalar_grad = [
                        block.passiveScalarGrad for block in domain.getBlocks()
                    ]
            else:
                if _LOG_DEBUG:
                    _LOG.debug("copy scalar backward empty")
                block_scalar_grad = [None] * domain.getNumBlocks()

            return (*block_scalar_grad,)

    block_scalar_data = [block.passiveScalar for block in domain.getBlocks()]
    CopyScalarResultFromBlocksFunction.apply(*block_scalar_data)


def CopyVelocityResultToBlocks(domain):
    class CopyVelocityResultToBlocksFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, velocity_result):
            with SAMPLE("CopyVelocityResultToBlocks-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("copy velocity to forward")

                domain.CreateVelocityOnBlocks()
                domain.UpdateDomainData()
                PISOtorch.CopyVelocityResultToBlocks(domain)

                return (*[block.velocity for block in domain.getBlocks()],)  # .clone()

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, *blocks_velocity_grad):
            if ctx.needs_input_grad[0]:
                with SAMPLE("CopyVelocityResultToBlocks-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("copy velocity to backward")
                    for block, v_grad in zip(domain.getBlocks(), blocks_velocity_grad):
                        v_grad = v_grad.contiguous()
                        block.setVelocityGrad(v_grad)
                    domain.CreateVelocityResultGrad()
                    domain.UpdateDomainData()
                    PISOtorch.CopyVelocityResultGradFromBlocks(domain)
                    velocityResultGrad = domain.velocityResultGrad
            else:
                if _LOG_DEBUG:
                    _LOG.debug("copy velocity to backward empty")
                velocityResultGrad = None

            return velocityResultGrad

    CopyVelocityResultToBlocksFunction.apply(domain.velocityResult)


def CopyVelocityResultFromBlocks(domain):
    class CopyVelocityResultFromBlocksFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *tracked_tensors):
            with SAMPLE("CopyVelocityResultFromBlocks-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("copy velocity from forward")

                domain.CreateVelocityResult()
                domain.UpdateDomainData()
                PISOtorch.CopyVelocityResultFromBlocks(domain)

                return domain.velocityResult

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, velocity_result_grad):
            if any(ctx.needs_input_grad):
                with SAMPLE("CopyVelocityResultFromBlocks-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("copy velocity from backward")
                    domain.setVelocityResultGrad(velocity_result_grad)
                    domain.CreateVelocityGradOnBlocks()
                    domain.UpdateDomainData()
                    PISOtorch.CopyVelocityResultGradToBlocks(domain)
                    block_velocity_grad = [
                        block.velocityGrad for block in domain.getBlocks()
                    ]
            else:
                if _LOG_DEBUG:
                    _LOG.debug("copy velocity from backward empty")
                block_velocity_grad = [None] * domain.getNumBlocks()

            return (*block_velocity_grad,)

    block_velocity_data = [block.velocity for block in domain.getBlocks()]
    CopyVelocityResultFromBlocksFunction.apply(*block_velocity_data)


def CopyPressureResultToBlocks(domain):
    class CopyPressureResultToBlocksFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, pressure_result):
            with SAMPLE("CopyPressureResultToBlocks-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("copy pressure to forward")

                domain.CreatePressureOnBlocks()
                domain.UpdateDomainData()
                PISOtorch.CopyPressureResultToBlocks(domain)

                return (*[block.pressure for block in domain.getBlocks()],)  # .clone()

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, *blocks_pressure_grad):
            if ctx.needs_input_grad[0]:
                with SAMPLE("CopyPressureResultToBlocks-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("copy pressure to backward")
                    for block, p_grad in zip(domain.getBlocks(), blocks_pressure_grad):
                        block.setPressureGrad(p_grad)
                    domain.CreatePressureResultGrad()
                    domain.UpdateDomainData()
                    PISOtorch.CopyPressureResultGradFromBlocks(domain)
                    pressureResultGrad = domain.pressureResultGrad
            else:
                if _LOG_DEBUG:
                    _LOG.debug("copy pressure to backward empty")
                pressureResultGrad = None

            return pressureResultGrad

    CopyPressureResultToBlocksFunction.apply(domain.pressureResult)


def CopyEpotResultToBlocks(domain):
    class CopyEpotResultToBlocksFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, epot_result):
            with SAMPLE("CopyEpotResultToBlocks-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("copy epot to blocks forward")

                domain.CreateEpotOnBlocks()
                domain.UpdateDomainData()
                PISOtorch.CopyEpotResultToBlocks(domain)

                return (*[block.epot for block in domain.getBlocks()],)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, *blocks_epot_grad):
            if ctx.needs_input_grad[0]:
                with SAMPLE("CopyEpotResultToBlocks-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("copy epot to blocks backward")
                    for block, e_grad in zip(domain.getBlocks(), blocks_epot_grad):
                        block.setEpotGrad(e_grad.contiguous())
                    domain.CreateEpotResultGrad()
                    domain.UpdateDomainData()
                    PISOtorch.CopyEpotResultGradFromBlocks(domain)
                    return domain.epotResultGrad
            else:
                if _LOG_DEBUG:
                    _LOG.debug("copy epot to blocks backward empty")
                return None

    CopyEpotResultToBlocksFunction.apply(domain.epotResult)


_EXCLUDED_GRADIENTS["ComputeEpotRHS"] = (set(), set())


def ComputeEpotRHS(domain, u_cross_eb_flat):
    class ComputeEpotRHSFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, vec_field):
            with SAMPLE("ComputeEpotRHS-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("ComputeEpotRHS forward")
                return PISOtorch.ComputeEpotRHS(domain, vec_field.contiguous())

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_divergence):
            if ctx.needs_input_grad[0]:
                with SAMPLE("ComputeEpotRHS-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("ComputeEpotRHS backward")
                    return PISOtorch.ComputeEpotRHSGrad(domain, grad_divergence)
            return None

    return ComputeEpotRHSFunction.apply(u_cross_eb_flat)


_EXCLUDED_GRADIENTS["ComputeCurrentDensityFaceBased"] = (set(), set())


def ComputeCurrentDensityFaceBased(domain, epot_result, u_cross_eb_flat):
    class ComputeCurrentDensityFaceBasedFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, epot, u_cross_eb):
            with SAMPLE("ComputeCurrentDensityFaceBased-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("ComputeCurrentDensityFaceBased forward")
                epot = epot.contiguous()
                u_cross_eb = u_cross_eb.contiguous()
                ctx.save_for_backward(epot, u_cross_eb)
                return PISOtorch.ComputeCurrentDensityFaceBased(domain, epot, u_cross_eb)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_J):
            epot, u_cross_eb = ctx.saved_tensors
            grad_epot, grad_ucb = PISOtorch.ComputeCurrentDensityFaceBasedGrad(
                domain, epot, u_cross_eb, grad_J
            )
            return grad_epot, grad_ucb

    return ComputeCurrentDensityFaceBasedFunction.apply(epot_result, u_cross_eb_flat)


_EXCLUDED_GRADIENTS["SetupPressureCorrection"] = (set(), set())


def SetupPressureCorrection(
    domain, time_step, non_ortho_flags, use_face_transform=False, timeStepNorm=False
):
    """
    combination of SetupPressureMatrix, SetupPressureRHS (which includes SetupPressureRHSdiv)
    """

    is_non_ortho = check_non_ortho_rhs(non_ortho_flags)

    tracked_tensor_filter = [
        "VELOCITY",
        "VELOCITY_SOURCE",
        "VELOCITY_RESULT",
        "BOUNDARY_VELOCITY",
        "VISCOSITY",
        "VISCOSITY_BLOCK",
        "A",
        "C",
    ]  # "C"
    if is_non_ortho:
        tracked_tensor_filter.append("PRESSURE_RESULT")

    grad_tensor_filter = [s + "_GRAD" for s in tracked_tensor_filter]

    domain_dict, tracked_tensors = flatten_domain(domain, tracked_tensor_filter)

    backwards_tensor_filter = [
        "BOUNDARY_VELOCITY",
        "VISCOSITY",
        "VISCOSITY_BLOCK",
        "A",
        "C",
        "VELOCITY_RESULT",
        "PRESSURE_RHS",
    ]
    backwards_tensor_filter.append(
        "VELOCITY_SOURCE"
    )  # DEBUG: value not needed but needs to be set to receive gradients
    if is_non_ortho:
        tracked_tensor_filter.append("PRESSURE_RESULT")

    class SetupPressureCorrectionFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *tracked_tensors):
            with SAMPLE("SetupPressureCorrection-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("setup pressure forward")

                domain.CreatePressureRHS()
                domain.CreatePressureRHSdiv()
                domain.P.CreateValue()
                domain.UpdateDomainData()
                PISOtorch.SetupPressureCorrection(
                    domain,
                    time_step,
                    non_ortho_flags,
                    use_face_transform,
                    timeStepNorm=timeStepNorm,
                )

                if backwards_tensor_filter:
                    domain_dict, saved_tensors = flatten_domain(
                        domain, backwards_tensor_filter
                    )
                    ctx.saved_tensors_domain_dict = domain_dict
                else:
                    ctx.saved_tensors_domain_dict = None
                    saved_tensors = []

                saved_tensors.append(time_step)
                ctx.save_for_backward(*saved_tensors)

            return domain.P.value, domain.pressureRHS, domain.pressureRHSdiv

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, P_value_grad, pressureRHS_grad, pressureRHSdiv_grad):
            # if not non_ortho_flags==0: raise NotImplementedError("SetupPressureCorrection: Only Orthogonal gradients are supported.")
            if use_face_transform:
                raise NotImplementedError(
                    "SetupPressureCorrection: Gradient does not support face transformations."
                )
            # if not timeStepNorm: raise NotImplementedError("SetupPressureCorrection: Gradient expect timeStepNorm.")

            # velocity_result_grad = None
            # block_velocity_grad = [None] * domain.getNumBlocks()

            if any(ctx.needs_input_grad):
                with SAMPLE("SetupPressureCorrection-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("setup pressure backward")

                    time_step = ctx.saved_tensors[-1]
                    if ctx.saved_tensors_domain_dict is not None:
                        set_domain_tensors_from_flat(
                            domain, ctx.saved_tensors_domain_dict, ctx.saved_tensors
                        )

                    # domain.PGrad.setValue(P_value_grad)
                    _set_gradient_input(
                        P_value_grad,
                        domain.PGrad.setValue,
                        "SetupPressureCorrection",
                        "P_GRAD",
                    )
                    # domain.setPressureRHSGrad(pressureRHS_grad)
                    _set_gradient_input(
                        pressureRHS_grad,
                        domain.setPressureRHSGrad,
                        "SetupPressureCorrection",
                        "PRESSURE_RHS_GRAD",
                    )
                    # domain.setPressureRHSdivGrad(pressureRHSdiv_grad)
                    _set_gradient_input(
                        pressureRHSdiv_grad,
                        domain.setPressureRHSdivGrad,
                        "SetupPressureCorrection",
                        "PRESSURE_RHS_DIV_GRAD",
                    )
                    domain.CreateAGrad()
                    domain.CGrad.CreateValue()
                    domain.CreateViscosityGrad()
                    domain.CreateVelocityResultGrad()
                    domain.CreateVelocityGradOnBlocks()
                    domain.CreateVelocitySourceGradOnBlocks()  # only create grad if velocity source exists, clears it otherwise
                    domain.CreateVelocityGradOnBoundaries()
                    if is_non_ortho:
                        domain.CreatePressureResultGrad()
                    domain.UpdateDomainData()

                    PISOtorch.SetupPressureCorrectionGrad(
                        domain,
                        time_step,
                        non_ortho_flags,
                        use_face_transform,
                        timeStepNorm,
                    )
                    _, grad_tensors = flatten_domain(
                        domain,
                        grad_tensor_filter,
                        exclusion_list=_EXCLUDED_GRADIENTS["SetupPressureCorrection"][
                            1
                        ],
                    )
            else:
                if _LOG_DEBUG:
                    _LOG.debug("setup pressure backward empty")
                _, grad_tensors = flatten_domain(domain, grad_tensor_filter, empty=True)

            return (*grad_tensors,)

    # block_velocity_data = [block.velocity for block in domain.getBlocks()]
    # if is_non_ortho:
    # block_velocity_data.append(domain.pressureResult)
    SetupPressureCorrectionFunction.apply(*tracked_tensors)


_EXCLUDED_GRADIENTS["SetupPressureMatrix"] = (set(), set())


def SetupPressureMatrix(domain, time_step, non_ortho_flags, use_face_transform=False):
    """
    inputs:
    - domain.A
    outputs:
    - domain.P
    """

    # is_non_ortho = check_non_ortho_rhs(non_ortho_flags)

    tracked_tensor_filter = ["A"]

    grad_tensor_filter = [s + "_GRAD" for s in tracked_tensor_filter]

    domain_dict, tracked_tensors = flatten_domain(domain, tracked_tensor_filter)

    backwards_tensor_filter = ["A"]

    class SetupPressureMatrixFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A_diag):
            with SAMPLE("SetupPressureMatrix-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("setup pressure matrix forward")

                domain.P.CreateValue()
                domain.UpdateDomainData()
                PISOtorch.SetupPressureMatrix(
                    domain, time_step, non_ortho_flags, use_face_transform
                )

                if backwards_tensor_filter:
                    domain_dict, saved_tensors = flatten_domain(
                        domain, backwards_tensor_filter
                    )
                    ctx.saved_tensors_domain_dict = domain_dict
                else:
                    ctx.saved_tensors_domain_dict = None
                    saved_tensors = []

                saved_tensors.append(time_step)
                ctx.save_for_backward(*saved_tensors)

            return domain.P.value

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, P_value_grad):
            if _LOG_DEBUG:
                _LOG.debug("setup pressure matrix backward (empty)")
            # if not non_ortho_flags==0: raise NotImplementedError("SetupPressureMatrix: Only Orthogonal gradients are supported.")
            if use_face_transform:
                raise NotImplementedError(
                    "SetupPressureMatrix: Gradient does not support face transformations."
                )

            A_diag_grad = None
            if any(ctx.needs_input_grad):
                with SAMPLE("SetupPressureMatrix-BWD"):
                    time_step = ctx.saved_tensors[-1]
                    if ctx.saved_tensors_domain_dict is not None:
                        set_domain_tensors_from_flat(
                            domain, ctx.saved_tensors_domain_dict, ctx.saved_tensors
                        )

                    # domain.PGrad.setValue(P_value_grad)
                    _set_gradient_input(
                        P_value_grad,
                        domain.PGrad.setValue,
                        "SetupPressureMatrix",
                        "P_GRAD",
                    )
                    domain.CreateAGrad()
                    domain.UpdateDomainData()

                    PISOtorch.SetupPressureMatrixGrad(
                        domain, time_step, non_ortho_flags, use_face_transform
                    )

                    # A_diag_grad = domain.AGrad
                    _, grad_tensors = flatten_domain(
                        domain,
                        grad_tensor_filter,
                        exclusion_list=_EXCLUDED_GRADIENTS["SetupPressureMatrix"][1],
                    )

            return (*grad_tensors,)

    SetupPressureMatrixFunction.apply(domain.A)


_EXCLUDED_GRADIENTS["SetupPressureRHS"] = (set(), set())


def SetupPressureRHS(
    domain, time_step, non_ortho_flags, use_face_transform=False, timeStepNorm=False
):
    """
    includes SetupPressureRHSdiv
    inputs:
    - block.velocity (not needed for backwards) [u from before advection]
    - block.velocitySource (not needed for backwards)
    - block.velocityResult [u* output from advection]
    - domain.C
    - domain.A
    - block.transform (must be static, not differentiable)
    - domain.viscosity
    - block.viscosity
    - boundary.velocity
    - boundary.transform (must be static, not differentiable)
    outputs:
    - domain.pressureRHS
    """

    is_non_ortho = check_non_ortho_rhs(non_ortho_flags)

    tracked_tensor_filter = [
        "VELOCITY",
        "VELOCITY_SOURCE",
        "VELOCITY_RESULT",
        "BOUNDARY_VELOCITY",
        "VISCOSITY",
        "VISCOSITY_BLOCK",
        "C",
        "A",
    ]  #
    if is_non_ortho:
        tracked_tensor_filter.append("PRESSURE_RESULT")

    grad_tensor_filter = [s + "_GRAD" for s in tracked_tensor_filter]

    domain_dict, tracked_tensors = flatten_domain(domain, tracked_tensor_filter)

    backwards_tensor_filter = [
        "BOUNDARY_VELOCITY",
        "VISCOSITY",
        "VISCOSITY_BLOCK",
        "A",
        "C",
        "VELOCITY_RESULT",
        "PRESSURE_RHS",
    ]
    backwards_tensor_filter.append(
        "VELOCITY_SOURCE"
    )  # DEBUG: value not needed but needs to be set to receive gradients
    # if is_non_ortho:
    #    tracked_tensor_filter.append("PRESSURE_RESULT")

    class SetupPressureRHSFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *tracked_tensors):
            with SAMPLE("SetupPressureRHS-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("setup pressure RHS forward")

                domain.CreatePressureRHS()
                domain.CreatePressureRHSdiv()
                domain.UpdateDomainData()
                PISOtorch.SetupPressureRHS(
                    domain,
                    time_step,
                    non_ortho_flags,
                    use_face_transform,
                    timeStepNorm=timeStepNorm,
                )

                if backwards_tensor_filter:
                    domain_dict, saved_tensors = flatten_domain(
                        domain, backwards_tensor_filter
                    )
                    ctx.saved_tensors_domain_dict = domain_dict
                else:
                    ctx.saved_tensors_domain_dict = None
                    saved_tensors = []

                saved_tensors.append(time_step)
                ctx.save_for_backward(*saved_tensors)

            return domain.pressureRHS, domain.pressureRHSdiv

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, pressureRHS_grad, pressureRHSdiv_grad):
            # if not non_ortho_flags==0: raise NotImplementedError("SetupPressureRHS: Only Orthogonal gradients are supported.")
            if use_face_transform:
                raise NotImplementedError(
                    "SetupPressureRHS: Gradient does not support face transformations."
                )
            # if not timeStepNorm: raise NotImplementedError("SetupPressureRHS: Gradient expect timeStepNorm.")

            # velocity_result_grad = None
            # block_velocity_grad = [None] * domain.getNumBlocks()

            if any(ctx.needs_input_grad):
                with SAMPLE("SetupPressureRHS-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("setup pressure RHS backward")

                    time_step = ctx.saved_tensors[-1]
                    if ctx.saved_tensors_domain_dict is not None:
                        set_domain_tensors_from_flat(
                            domain, ctx.saved_tensors_domain_dict, ctx.saved_tensors
                        )

                    # domain.setPressureRHSGrad(pressureRHS_grad)
                    _set_gradient_input(
                        pressureRHS_grad,
                        domain.setPressureRHSGrad,
                        "SetupPressureRHS",
                        "PRESSURE_RHS_GRAD",
                    )
                    # domain.setPressureRHSdivGrad(pressureRHSdiv_grad)
                    _set_gradient_input(
                        pressureRHSdiv_grad,
                        domain.setPressureRHSdivGrad,
                        "SetupPressureRHS",
                        "PRESSURE_RHS_DIV_GRAD",
                    )
                    domain.CreateAGrad()
                    domain.CGrad.CreateValue()
                    domain.CreateViscosityGrad()
                    domain.CreateVelocityResultGrad()
                    domain.CreateVelocityGradOnBlocks()
                    domain.CreateVelocitySourceGradOnBlocks()  # only create grad if velocity source exists, clears it otherwise
                    domain.CreateVelocityGradOnBoundaries()
                    if is_non_ortho:
                        domain.CreatePressureResultGrad()
                    domain.UpdateDomainData()

                    PISOtorch.SetupPressureRHSGrad(
                        domain,
                        time_step,
                        non_ortho_flags,
                        use_face_transform,
                        timeStepNorm,
                    )
                    _, grad_tensors = flatten_domain(
                        domain,
                        grad_tensor_filter,
                        exclusion_list=_EXCLUDED_GRADIENTS["SetupPressureRHS"][1],
                    )
            else:
                if _LOG_DEBUG:
                    _LOG.debug("setup pressure backward empty")
                _, grad_tensors = flatten_domain(domain, grad_tensor_filter, empty=True)

            return (*grad_tensors,)

    SetupPressureRHSFunction.apply(*tracked_tensors)


_EXCLUDED_GRADIENTS["SetupPressureRHSdiv"] = (set(), set())


def SetupPressureRHSdiv(
    domain, time_step, non_ortho_flags, use_face_transform=False, timeStepNorm=False
):
    """
    computes divergence of domain.pressureRHS
    adds non-orthogonal RHS compoents if non-orthogonal

    inputs:
    - domain.A (if non-orthogonal)
    - domain.pressureResult (if non-orthogonal)
    - domain.pressureRHS (not needed for backwards)
    - block.transform (must be static, not differentiable)
    - boundary.velocity
    - boundary.transform (must be static, not differentiable)
    outputs:
    - domain.pressureRHSdiv
    """

    is_non_ortho = check_non_ortho_rhs(non_ortho_flags)

    tracked_tensor_filter = ["PRESSURE_RHS", "BOUNDARY_VELOCITY"]  # "A"
    if is_non_ortho:
        tracked_tensor_filter.append("PRESSURE_RESULT")
        tracked_tensor_filter.append("A")

    grad_tensor_filter = [s + "_GRAD" for s in tracked_tensor_filter]

    domain_dict, tracked_tensors = flatten_domain(domain, tracked_tensor_filter)

    backwards_tensor_filter = []
    if is_non_ortho:
        backwards_tensor_filter.append("PRESSURE_RESULT")
        backwards_tensor_filter.append("A")

    class SetupPressureRHSdivFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *tracked_tensors):
            with SAMPLE("SetupPressureRHSdiv-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("setup pressure RHS div forward")

                domain.CreatePressureRHSdiv()
                domain.UpdateDomainData()
                PISOtorch.SetupPressureRHSdiv(
                    domain,
                    time_step,
                    non_ortho_flags,
                    use_face_transform,
                    timeStepNorm=timeStepNorm,
                )

                if backwards_tensor_filter:
                    domain_dict, saved_tensors = flatten_domain(
                        domain, backwards_tensor_filter
                    )
                    ctx.saved_tensors_domain_dict = domain_dict
                else:
                    ctx.saved_tensors_domain_dict = None
                    saved_tensors = []

                saved_tensors.append(time_step)
                ctx.save_for_backward(*saved_tensors)

            return domain.pressureRHSdiv

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, pressureRHSdiv_grad):
            # if not non_ortho_flags==0: raise NotImplementedError("SetupPressureRHSdiv: Only Orthogonal gradients are supported.")
            if use_face_transform:
                raise NotImplementedError(
                    "SetupPressureRHSdiv: Gradient does not support face transformations."
                )
            # if not timeStepNorm: raise NotImplementedError("SetupPressureRHSdiv: Gradient expect timeStepNorm.")

            # pressureRHS_grad = None

            if any(ctx.needs_input_grad):
                with SAMPLE("SetupPressureRHSdiv-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("setup pressure RHS div backward")

                    time_step = ctx.saved_tensors[-1]
                    if ctx.saved_tensors_domain_dict is not None:
                        set_domain_tensors_from_flat(
                            domain, ctx.saved_tensors_domain_dict, ctx.saved_tensors
                        )

                    # domain.setPressureRHSdivGrad(pressureRHSdiv_grad)
                    _set_gradient_input(
                        pressureRHSdiv_grad,
                        domain.setPressureRHSdivGrad,
                        "SetupPressureRHSdiv",
                        "PRESSURE_RHS_DIV_GRAD",
                    )
                    if is_non_ortho:
                        domain.CreateAGrad()
                    domain.CreatePressureRHSGrad()
                    domain.CreateVelocityGradOnBoundaries()
                    if is_non_ortho:
                        domain.CreatePressureResultGrad()
                    domain.UpdateDomainData()

                    PISOtorch.SetupPressureRHSdivGrad(
                        domain,
                        time_step,
                        non_ortho_flags,
                        use_face_transform,
                        timeStepNorm,
                    )
                    _, grad_tensors = flatten_domain(
                        domain,
                        grad_tensor_filter,
                        exclusion_list=_EXCLUDED_GRADIENTS["SetupPressureRHSdiv"][1],
                    )
            else:
                if _LOG_DEBUG:
                    _LOG.debug("setup pressure backward empty")
                _, grad_tensors = flatten_domain(domain, grad_tensor_filter, empty=True)

            return tuple(grad_tensors)

    SetupPressureRHSdivFunction.apply(*tracked_tensors)


_EXCLUDED_GRADIENTS["CorrectVelocity"] = (set(), set())


def CorrectVelocity(
    domain, time_step, version, timeStepNorm=False, exclude_pressure_grad=False
):
    """
    `exclude_pressure_grad` drops PRESSURE_GRAD from this call's backward only,
    without touching the global `_EXCLUDED_GRADIENTS` entry.

    inputs:
    - block.pressure
    - domain.A
    - domain.pressureRHS (not needed for backwards)
    - block.transform (must be static, not differentiable)
    - boundary.transform (must be static, not differentiable)
    outputs:
    - domain.velocityResult
    """

    # is_non_ortho = check_non_ortho_rhs(non_ortho_flags)

    tracked_tensor_filter = ["A", "PRESSURE", "PRESSURE_RHS"]

    grad_tensor_filter = [s + "_GRAD" for s in tracked_tensor_filter]

    domain_dict, tracked_tensors = flatten_domain(domain, tracked_tensor_filter)

    backwards_tensor_filter = ["A", "PRESSURE"]

    class CorrectVelocityFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *tracked_tensors):
            with SAMPLE("CorrectVelocity-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("correct velocity forward")

                domain.CreateVelocityResult()
                domain.UpdateDomainData()
                PISOtorch.CorrectVelocity(
                    domain, time_step, version, timeStepNorm=timeStepNorm
                )

                if backwards_tensor_filter:
                    domain_dict, saved_tensors = flatten_domain(
                        domain, backwards_tensor_filter
                    )
                    ctx.saved_tensors_domain_dict = domain_dict
                else:
                    ctx.saved_tensors_domain_dict = None
                    saved_tensors = []

                saved_tensors.append(time_step)
                ctx.save_for_backward(*saved_tensors)

            return domain.velocityResult

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, velocity_result_grad):
            # if not timeStepNorm: raise NotImplementedError("CorrectVelocity: Gradient expects timeStepNorm.")
            if version not in [0, 1]:
                raise NotImplementedError(
                    "CorrectVelocity: Gradient only implemented for Finite Difference pressure gradients (version 0 or 1)."
                )

            if any(ctx.needs_input_grad):
                with SAMPLE("CorrectVelocity-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("correct velocity backward")

                    time_step = ctx.saved_tensors[-1]
                    if ctx.saved_tensors_domain_dict is not None:
                        set_domain_tensors_from_flat(
                            domain, ctx.saved_tensors_domain_dict, ctx.saved_tensors
                        )

                    # domain.setVelocityResultGrad(velocity_result_grad)
                    _set_gradient_input(
                        velocity_result_grad,
                        domain.setVelocityResultGrad,
                        "CorrectVelocity",
                        "VELOCITY_RESULT_GRAD",
                    )
                    domain.CreateAGrad()
                    domain.CreatePressureGradOnBlocks()
                    domain.CreatePressureRHSGrad()
                    domain.UpdateDomainData()

                    PISOtorch.CorrectVelocityGrad(domain, time_step, timeStepNorm)
                    exclusion_list = _EXCLUDED_GRADIENTS["CorrectVelocity"][1]
                    if exclude_pressure_grad:
                        exclusion_list = exclusion_list | {"PRESSURE_GRAD"}
                    _, grad_tensors = flatten_domain(
                        domain,
                        grad_tensor_filter,
                        exclusion_list=exclusion_list,
                    )
            else:
                if _LOG_DEBUG:
                    _LOG.debug("correct velocity backward empty")
                _, grad_tensors = flatten_domain(domain, grad_tensor_filter, empty=True)

            return (*grad_tensors,)

    CorrectVelocityFunction.apply(*tracked_tensors)


# storing the output of a custom gradient method in the domain object seems to prevent memory from being freed.
# resetting all tensors of the domain (e.g. at the end of an optimization loop) fixes this and prevent a kind of memory leak.
def reset_domain(domain):
    domain.CreateVelocityOnBlocks()
    domain.CreatePressureOnBlocks()
    domain.CreatePassiveScalarOnBlocks()
    domain.PrepareSolve()


# This seems to be sufficient to allow the memory to be reclaimed.
# Run after each optimization step on the domain that was used for backprop.
# Maybe call gc.collect() after, esp. when testing/debugging the memory usage.
# Alternative: domain.DetachFwd()
def detach_domain_fwd(domain):
    # detach all forward tensors
    for block in domain.getBlocks():
        block.setVelocity(block.velocity.detach())
        block.setPressure(block.pressure.detach())
        block.setPassiveScalar(block.passiveScalar.detach())
    domain.setA(domain.A.detach())
    domain.C.setValue(domain.C.value.detach())
    domain.P.setValue(domain.P.value.detach())
    domain.setScalarRHS(domain.scalarRHS.detach())
    domain.setScalarResult(domain.scalarResult.detach())
    domain.setVelocityRHS(domain.velocityRHS.detach())
    domain.setVelocityResult(domain.velocityResult.detach())
    domain.setPressureRHS(domain.pressureRHS.detach())
    domain.setPressureRHSdiv(domain.pressureRHSdiv.detach())
    domain.setPressureResult(domain.pressureResult.detach())
    if domain.hasEpot():
        for block in domain.getBlocks():
            if block.hasEpot():
                block.setEpot(block.epot.detach())
        domain.setEpotRHS(domain.epotRHS.detach())
        domain.setEpotResult(domain.epotResult.detach())


def is_tensor_empty(tensor):
    return tensor.dim() == 1 and tensor.size(0) == 0


def detach_domain_grad(domain):
    # detach all forward tensors
    for block in domain.getBlocks():
        if not is_tensor_empty(block.velocityGrad):
            block.setVelocityGrad(block.velocityGrad.detach())
        if not is_tensor_empty(block.pressureGrad):
            block.setPressureGrad(block.pressureGrad.detach())
        if not is_tensor_empty(block.passiveScalarGrad):
            block.setPassiveScalarGrad(block.passiveScalarGrad.detach())
    if not is_tensor_empty(domain.AGrad):
        domain.setAGrad(domain.AGrad.detach())
    domain.CGrad.setValue(domain.CGrad.value.detach())
    if not is_tensor_empty(domain.scalarRHSGrad):
        domain.setScalarRHSGrad(domain.scalarRHSGrad.detach())
    if not is_tensor_empty(domain.scalarResultGrad):
        domain.setScalarResultGrad(domain.scalarResultGrad.detach())
    if not is_tensor_empty(domain.velocityRHSGrad):
        domain.setVelocityRHSGrad(domain.velocityRHSGrad.detach())
    if not is_tensor_empty(domain.velocityResultGrad):
        domain.setVelocityResultGrad(domain.velocityResultGrad.detach())
    if not is_tensor_empty(domain.pressureRHSGrad):
        domain.setPressureRHSGrad(domain.pressureRHSGrad.detach())
    if not is_tensor_empty(domain.pressureRHSdivGrad):
        domain.setPressureRHSdivGrad(domain.pressureRHSdivGrad.detach())
    if not is_tensor_empty(domain.pressureResultGrad):
        domain.setPressureResultGrad(domain.pressureResultGrad.detach())
    if domain.hasEpotResultGrad():
        for block in domain.getBlocks():
            if block.hasEpotGrad() and not is_tensor_empty(block.epotGrad):
                block.setEpotGrad(block.epotGrad.detach())
        domain.setEpotResultGrad(domain.epotResultGrad.detach())


def detach_domain(domain):
    detach_domain_fwd(domain)
    detach_domain_grad(domain)


def matmul(
    vectorMatrixA: torch.Tensor,
    vectorMatrixB: torch.Tensor,
    transposeA: bool = False,
    invertA: bool = False,
    transposeB: bool = False,
    invertB: bool = False,
    transposeOutput: bool = False,
    invertOutput: bool = False,
) -> torch.Tensor:
    """compute the product matrix/vector * matrix/vector of matrices/vectors in the channel dimension.
    inputs:
    - vectorMatrixA: NCDHW tensor with a vectors or flat row-major matrices in the C dimension.
    - vectorMatrixB: NCDHW tensor with a vectors or flat row-major matrices in the C dimension.
    - transpose: transpose the matrix after loading/before writing. only affects matrices.
    - invert: invert the matrix after loading/before writing. only affects matrices.
    outputs:
    - NCDHW tensor with a channel structure corresponding to the inputs:
        matrix if both A and B are matrices,
        vector if exacly one of A or B is a matrix,
        scalar if both A and B are vectors
    """

    class matmulFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, vectorMatrixA, vectorMatrixB):
            with SAMPLE("matmul-FWD"):
                if _LOG_DEBUG:
                    _LOG.debug("matmul forward")

                ctx.transposeA = transposeA
                ctx.invertA = invertA
                ctx.transposeB = transposeB
                ctx.invertB = invertB
                ctx.transposeOutput = transposeOutput
                ctx.invertOutput = invertOutput

                ctx.save_for_backward(vectorMatrixA, vectorMatrixB)

                result = PISOtorch.matmul(
                    vectorMatrixA,
                    vectorMatrixB,
                    transposeA,
                    invertA,
                    transposeB,
                    invertB,
                    transposeOutput,
                    invertOutput,
                )

            return result

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, result_grad):
            if any(ctx.needs_input_grad):
                with SAMPLE("matmul-BWD"):
                    if _LOG_DEBUG:
                        _LOG.debug("matmul backward")

                    if ctx.invertOutput:
                        raise NotImplementedError(
                            "Can not compute gradients for inverted matrices (output)."
                        )
                    if ctx.needs_input_grad[0] and ctx.invertA:
                        raise NotImplementedError(
                            "Can not compute gradients for inverted matrices (input A)."
                        )
                    if ctx.needs_input_grad[1] and ctx.invertB:
                        raise NotImplementedError(
                            "Can not compute gradients for inverted matrices (input B)."
                        )

                    vectorMatrixA, vectorMatrixB = ctx.saved_tensors

                    # returns 0 gradient if the matrix is inverted
                    vectorMatrixA_grad, vectorMatrixB_grad = PISOtorch.matmulGrad(
                        vectorMatrixA,
                        vectorMatrixB,
                        result_grad,
                        ctx.transposeA,
                        ctx.invertA,
                        ctx.transposeB,
                        ctx.invertB,
                        ctx.transposeOutput,
                        ctx.invertOutput,
                    )

                    grad_tensors = (
                        (vectorMatrixA_grad if ctx.needs_input_grad[0] else None),
                        (vectorMatrixB_grad if ctx.needs_input_grad[1] else None),
                    )
            else:
                if _LOG_DEBUG:
                    _LOG.debug("matmul backward empty")
                grad_tensors = (None, None)

            return (*grad_tensors,)

    return matmulFunction.apply(vectorMatrixA, vectorMatrixB)
