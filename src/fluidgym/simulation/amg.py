"""Algebraic-multigrid preconditioning for the Poisson-type solves."""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

__all__ = [
    "AMGHierarchy",
    "AMGSolveInfo",
    "amg_pcg_solve",
    "build_amg_hierarchy",
    "csr_to_scipy",
    "is_pyamg_available",
]

_LOG = logging.getLogger("fluidgym.AMG")

# Weighted-Jacobi damping based on torch-sla hardcoded vakze
_JACOBI_OMEGA = 2.0 / 3.0

# Diagonal entries below this are treated as 1.0 rather than inverted
_MIN_ABS_DIAG = 1e-30


def is_pyamg_available() -> bool:
    """Return whether the optional ``pyamg`` dependency can be imported."""
    try:
        import pyamg  # noqa: F401
    except ImportError:
        return False
    return True


def _load_pyamg():
    try:
        import pyamg
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "AMG preconditioning needs the 'pyamg' package for the setup phase. "
            "Install it with `pip install pyamg`."
        ) from exc
    return pyamg


# ---------------------------------------------------------------------------
# Matrix conversion
# ---------------------------------------------------------------------------


def csr_to_scipy(csr_mat: Any) -> Any:
    """Convert a ``PISOtorch.CSRmatrix`` to a host-side scipy CSR matrix."""
    import scipy.sparse as sp

    n = csr_mat.getRows()
    row = csr_mat.row.detach().cpu()
    index = csr_mat.index.detach().cpu()
    value = csr_mat.value.detach().cpu()

    keep = index >= 0
    n_dropped = int((~keep).sum())
    if n_dropped:
        _LOG.warning(
            "Dropping %d sentinel entries (column index < 0) while converting the "
            "matrix for the AMG setup.",
            n_dropped,
        )
        # Recount the entries per row before compacting, so the row pointers stay
        # consistent with the filtered value/index arrays
        row_of_entry = torch.repeat_interleave(
            torch.arange(n, dtype=torch.long), row.diff().long()
        )
        counts = torch.bincount(row_of_entry[keep], minlength=n)
        row = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(counts, dim=0)])
        index = index[keep]
        value = value[keep]

    return sp.csr_matrix(
        (
            value.to(torch.float64).numpy(),
            index.to(torch.int64).numpy(),
            row.to(torch.int64).numpy(),
        ),
        shape=(n, n),
    )


def _scipy_to_torch_csr(
    mat: Any, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Lift a scipy sparse matrix onto the device as a ``torch.sparse_csr`` tensor."""
    mat = mat.tocsr()
    mat.sort_indices()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Sparse CSR tensor support is in beta state"
        )
        return torch.sparse_csr_tensor(
            torch.as_tensor(mat.indptr, dtype=torch.int64, device=device),
            torch.as_tensor(mat.indices, dtype=torch.int64, device=device),
            torch.as_tensor(mat.data, dtype=dtype, device=device),
            size=mat.shape,
        )


def _spmv(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Sparse mat-vec that works for both 1-D vectors and (n, k) blocks."""
    if x.dim() == 1:
        return (A @ x.unsqueeze(1)).squeeze(1)
    return A @ x


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


@dataclass
class _Level:
    A: torch.Tensor  # sparse_csr, n x n
    D_inv: torch.Tensor  # dense, n
    R: torch.Tensor | None = None  # sparse_csr, n_coarse x n (None on coarsest)
    P: torch.Tensor | None = None  # sparse_csr, n x n_coarse (None on coarsest)


@dataclass
class AMGHierarchy:
    """A multigrid hierarchy, applied as a single V-cycle.

    Callable as ``z = hierarchy(r)``, i.e. it is a preconditioner in the form the
    Krylov solvers expect.
    """

    levels: list[_Level]
    coarse_pinv: torch.Tensor  # dense pseudo-inverse of the coarsest operator
    num_pre_smooth: int = 1
    num_post_smooth: int = 1
    project_constant: bool = False
    method: str = "ruge_stuben"
    setup_seconds: float = 0.0

    @property
    def device(self) -> torch.device:
        """Device the hierarchy's tensors live on."""
        return self.levels[0].D_inv.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the hierarchy's tensors."""
        return self.levels[0].D_inv.dtype

    @property
    def operator_complexity(self) -> float:
        """Total nnz across levels divided by the finest-level nnz."""
        nnz = [lvl.A._nnz() for lvl in self.levels]
        return sum(nnz) / nnz[0] if nnz[0] else math.nan

    def describe(self) -> str:
        """Return a human-readable summary of the level sizes and complexity."""
        rows = [
            f"  level {i}: n={lvl.A.shape[0]:>10d}  nnz={lvl.A._nnz():>11d}"
            for i, lvl in enumerate(self.levels)
        ]
        return (
            f"AMG hierarchy ({self.method}, {len(self.levels)} levels, "
            f"operator complexity {self.operator_complexity:.2f}, "
            f"setup {self.setup_seconds:.1f}s)\n" + "\n".join(rows)
        )

    # -- application --------------------------------------------------------

    def _smooth(
        self, level: _Level, x: torch.Tensor, b: torch.Tensor, sweeps: int
    ) -> torch.Tensor:
        for _ in range(sweeps):
            x = x + _JACOBI_OMEGA * level.D_inv * (b - _spmv(level.A, x))
        return x

    def _v_cycle(self, b: torch.Tensor, level_idx: int) -> torch.Tensor:
        level = self.levels[level_idx]
        if level.R is None:  # coarsest
            return self.coarse_pinv @ b

        x = self._smooth(level, torch.zeros_like(b), b, self.num_pre_smooth)
        residual = b - _spmv(level.A, x)
        coarse_correction = self._v_cycle(_spmv(level.R, residual), level_idx + 1)
        x = x + _spmv(level.P, coarse_correction)  # pyright: ignore[reportArgumentType]
        return self._smooth(level, x, b, self.num_post_smooth)

    def __call__(self, r: torch.Tensor) -> torch.Tensor:
        """Apply one V-cycle to the residual ``r``."""
        if self.project_constant:
            r = r - r.mean()
        z = self._v_cycle(r, 0)
        if self.project_constant:
            z = z - z.mean()
        return z


def build_amg_hierarchy(
    csr_mat: Any,
    *,
    method: str = "ruge_stuben",
    max_levels: int = 20,
    max_coarse: int = 64,
    strength: float = 0.25,
    num_pre_smooth: int = 1,
    num_post_smooth: int = 1,
    project_constant: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    **pyamg_kwargs: Any,
) -> AMGHierarchy:
    """Build a V-cycle hierarchy for ``csr_mat`` using pyamg on the host.

    Parameters
    ----------
    method: str
        ``"ruge_stuben"`` (default) or ``"smoothed_aggregation"``. Classical
        Ruge-Stuben is the default on the strength of measurements on the small
        control duct (6.7M unknowns, epot at rtol 3e-3), where it beat smoothed
        aggregation on every axis at identical Nu.

    max_coarse: int
        Size below which coarsening stops. The coarsest operator is inverted
        densely, so keep it small.

    project_constant: bool
        Set for a singular (pure-Neumann) system, e.g. the insulating duct's
        potential equation. See the module docstring.

    Notes
    -----
    The setup runs on the CPU and needs the matrix on the host, so this is a
    genuinely expensive one-off: budget for it, and call it once per matrix.
    """
    import time

    pyamg = _load_pyamg()

    device = device or csr_mat.value.device
    dtype = dtype or csr_mat.value.dtype

    start = time.perf_counter()
    A_host = csr_to_scipy(csr_mat)

    if method == "smoothed_aggregation":
        ml = pyamg.smoothed_aggregation_solver(
            A_host,
            max_levels=max_levels,
            max_coarse=max_coarse,
            strength=("symmetric", {"theta": strength}),  # pyright: ignore[reportArgumentType]
            **pyamg_kwargs,
        )
    elif method == "ruge_stuben":
        ml = pyamg.ruge_stuben_solver(
            A_host,
            max_levels=max_levels,
            max_coarse=max_coarse,
            strength=("classical", {"theta": strength}),
            **pyamg_kwargs,
        )
    else:
        raise ValueError(
            f"Unknown AMG method {method!r}; expected 'smoothed_aggregation' or "
            "'ruge_stuben'."
        )

    levels: list[_Level] = []
    for i, lvl in enumerate(ml.levels):
        A_dev = _scipy_to_torch_csr(lvl.A, device, dtype)  # pyright: ignore[reportArgumentType]

        diag = torch.as_tensor(lvl.A.diagonal().copy(), dtype=dtype, device=device)
        # A zero diagonal would make the smoother produce inf/nan; leaving those
        # rows unsmoothed is the safe behaviour
        d_inv = torch.where(
            diag.abs() > _MIN_ABS_DIAG, 1.0 / diag, torch.ones_like(diag)
        )

        is_coarsest = i == len(ml.levels) - 1
        levels.append(
            _Level(
                A=A_dev,
                D_inv=d_inv,
                R=None if is_coarsest else _scipy_to_torch_csr(lvl.R, device, dtype),  # pyright: ignore[reportArgumentType]
                P=None if is_coarsest else _scipy_to_torch_csr(lvl.P, device, dtype),  # pyright: ignore[reportArgumentType]
            )
        )

    # Pseudo-inverse rather than an LU: the coarse operator of a singular system
    # is itself singular, and this is small enough (<= max_coarse) that a dense
    # pinv is free. It also keeps the module clear of any LAPACK/cuSOLVER path
    coarse_dense = np.asarray(ml.levels[-1].A.todense(), dtype=np.float64)
    coarse_pinv = torch.as_tensor(
        np.linalg.pinv(coarse_dense), dtype=dtype, device=device
    )

    hierarchy = AMGHierarchy(
        levels=levels,
        coarse_pinv=coarse_pinv,
        num_pre_smooth=num_pre_smooth,
        num_post_smooth=num_post_smooth,
        project_constant=project_constant,
        method=method,
        setup_seconds=time.perf_counter() - start,
    )

    _LOG.debug("%s", hierarchy.describe())

    return hierarchy


# ---------------------------------------------------------------------------
# Preconditioned CG
# ---------------------------------------------------------------------------


@dataclass
class AMGSolveInfo:
    """Duck-type of ``PISOtorch.LinearSolverResultInfo``.

    The telemetry in ``fluidgym.simulation.solver_stats`` and the error handling
    in ``PISOtorch_diff._check_solver_return_infos`` both read these four fields
    off whatever the solver returned, so the AMG path reports exactly like the
    CUDA ones and needs no special-casing downstream.
    """

    finalResidual: float
    usedIterations: int
    converged: bool
    isFiniteResidual: bool

    def __str__(self) -> str:
        return (
            f"AMGSolveInfo(finalResidual={self.finalResidual:.5g}, "
            f"usedIterations={self.usedIterations:d}, "
            f"converged={int(self.converged):d}, "
            f"isFiniteResidual={int(self.isFiniteResidual):d})"
        )


def _criterion_residual(r: torch.Tensor) -> torch.Tensor:
    """``||r||_2 / sqrt(n)`` -- the NORM2_NORMALIZED criterion the kernels use."""
    return torch.linalg.vector_norm(r) / math.sqrt(r.numel())


def amg_pcg_solve(
    rhs: torch.Tensor,
    x: torch.Tensor,
    hierarchy: AMGHierarchy,
    *,
    tol: float,
    max_iter: int,
    return_best_result: bool = False,
) -> list[AMGSolveInfo]:
    """Solve ``A x = rhs`` in place with AMG-preconditioned CG.

    ``A`` is the finest level of ``hierarchy``, so the operator the Krylov
    iteration uses is by construction the same one the hierarchy was built for.

    The stopping test is ``||r||_2/sqrt(n) < tol``, matching ``NORM2_NORMALIZED``
    in the CUDA kernels, so a tolerance keeps its meaning when switching solver.

    ``rhs`` may hold several right-hand sides concatenated (as the CUDA solver
    allows); they are solved one after another. ``x`` is written in place.
    """
    A = hierarchy.levels[0].A
    n = A.shape[0]

    if rhs.numel() % n:
        raise ValueError(
            f"RHS size {rhs.numel()} is not a multiple of the matrix size {n}."
        )
    n_rhs = rhs.numel() // n

    rhs_flat = rhs.reshape(n_rhs, n)
    x_flat = x.reshape(n_rhs, n)

    infos: list[AMGSolveInfo] = []
    for i in range(n_rhs):
        info = _pcg_one(
            A,
            hierarchy,
            rhs_flat[i],
            x_flat[i],
            tol=tol,
            max_iter=max_iter,
            return_best_result=return_best_result,
        )
        infos.append(info)
    return infos


def _pcg_one(
    A: torch.Tensor,
    M: AMGHierarchy,
    b: torch.Tensor,
    x_out: torch.Tensor,
    *,
    tol: float,
    max_iter: int,
    return_best_result: bool,
) -> AMGSolveInfo:
    b = b.to(M.dtype)
    x = x_out.to(M.dtype).clone()

    if M.project_constant:
        b = b - b.mean()

    r = b - _spmv(A, x)
    if M.project_constant:
        r = r - r.mean()

    z = M(r)
    p = z.clone()
    rz = torch.dot(r, z)

    best_residual = float(_criterion_residual(r))
    best_x = x.clone() if return_best_result else None
    used_iterations = 0
    converged = best_residual < tol

    for iteration in range(max_iter):
        used_iterations = iteration
        residual = float(_criterion_residual(r))
        if not math.isfinite(residual):
            break
        if residual < best_residual:
            best_residual = residual
            if best_x is not None:
                best_x.copy_(x)
        if residual < tol:
            converged = True
            break

        Ap = _spmv(A, p)
        pAp = torch.dot(p, Ap)
        if float(pAp) == 0.0:
            break
        alpha = rz / pAp

        x = x + alpha * p
        r_new = r - alpha * Ap
        if M.project_constant:
            r_new = r_new - r_new.mean()
        z_new = M(r_new)

        # Polak-Ribiere beta: robust when the preconditioner is not exactly
        # constant between iterations, and it degenerates to Fletcher-Reeves
        # when it is
        beta = torch.dot(z_new, r_new - r) / rz
        p = z_new + beta * p
        r, z, rz = r_new, z_new, torch.dot(z_new, r_new)
    else:
        used_iterations = max_iter
        residual = float(_criterion_residual(r))
        if residual < best_residual:
            best_residual = residual
            if best_x is not None:
                best_x.copy_(x)
        converged = residual < tol

    final_residual = float(_criterion_residual(r))
    is_finite = math.isfinite(final_residual) and bool(torch.isfinite(x).all())

    if return_best_result and not converged and best_x is not None:
        x = best_x
        final_residual = best_residual

    if is_finite:
        x_out.copy_(x.to(x_out.dtype))
    else:
        # Match the CUDA solvers' contract: a non-finite result is zeroed rather
        # than propagated into the simulation state
        x_out.zero_()

    return AMGSolveInfo(
        finalResidual=final_residual,
        usedIterations=used_iterations,
        converged=bool(converged),
        isFiniteResidual=is_finite,
    )
