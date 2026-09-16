"""Scale-invariant tolerance specifications for the linear solvers.

The PISOtorch CG/BiCGStab kernels stop on ``NORM2_NORMALIZED``, i.e.

    ||r||_2 / sqrt(n) < tol

(see ``cg_solver_kernel.cu``; the relative-to-``||r0||`` variant is commented out
in ``bicgstab_solver_kernel.cu``). The ``sqrt(n)`` removes the dependence on the
*cell count*, but not on the *magnitude of the right-hand side*. A tolerance that
is well matched to a short duct therefore becomes a much tighter -- and much more
expensive, possibly unreachable -- demand in a longer one, where the flow is more
developed and ``||b||`` is larger.

:class:`SolverTolerance` expresses the tolerance relative to the RHS instead. With

    tol = rtol * ||b||_2 / sqrt(n)

the stop test becomes exactly ``||r||_2 < rtol * ||b||_2``: a true relative
residual whose meaning is independent of the domain size, the cell volume, ``dt``,
and the Hartmann/Reynolds numbers. That is what makes a tolerance chosen on the
small duct transferable to the large one.

Plain floats keep their current absolute meaning everywhere, so this is opt-in.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "SolverTolerance",
    "parse_tolerance",
    "resolve_tolerance",
    "rhs_scale",
]


# Residuals cannot be driven below the round-off level of the accumulation, so a
# relative tolerance must be floored. The factor mirrors the one already used by
# the torch-sla backend (``PISOtorch_sla._PRECISION_FLOOR_FACTOR``) so that the
# two backends agree on when a solve has stagnated rather than failed
PRECISION_FLOOR_FACTOR = 8.0


def rhs_scale(rhs: torch.Tensor) -> float:
    """Return ``||b||_2 / sqrt(n)``, the RHS in the solver's residual units.

    This is the same norm the ``NORM2_NORMALIZED`` criterion applies to the
    residual, so ``residual / rhs_scale`` is directly the relative residual.
    """
    n = rhs.numel()
    if n == 0:
        return 0.0
    # float64 accumulation: the norm of a large single-precision RHS is otherwise
    # itself inaccurate enough to matter at the tolerances used here
    norm = torch.linalg.vector_norm(rhs.detach().to(torch.float64))
    return float(norm.item()) / math.sqrt(n)


@dataclass(frozen=True)
class SolverTolerance:
    """A tolerance specified relative to the right-hand side.

    Parameters
    ----------
    rtol: float or None
        Relative residual target. The resolved absolute tolerance is
        ``rtol * ||b||_2 / sqrt(n)``, which makes the solver's stop test
        equivalent to ``||r||_2 < rtol * ||b||_2``.

    atol: float or None
        Absolute floor, in the same RMS units as the solver criterion. Prevents
        over-solving when the RHS is (near) zero -- e.g. the electric potential
        equation at rest, where ``u x e_b`` is uniform and its divergence
        vanishes. At least one of ``rtol``/``atol`` must be given.

    Notes
    -----
    The resolved tolerance is additionally floored at
    ``PRECISION_FLOOR_FACTOR * eps * ||b||_2 / sqrt(n)``, below which the solver
    can only stagnate.
    """

    rtol: float | None = None
    atol: float | None = None

    def __post_init__(self) -> None:
        if self.rtol is None and self.atol is None:
            raise ValueError("SolverTolerance needs at least one of rtol, atol.")
        for name in ("rtol", "atol"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, numbers.Real):
                raise TypeError(f"SolverTolerance.{name} must be a float or None.")
            if not float(value) > 0:
                raise ValueError(f"SolverTolerance.{name} must be positive.")

    def resolve(self, rhs: torch.Tensor) -> float:
        """Return the absolute tolerance to hand to the solver for this RHS."""
        scale = rhs_scale(rhs)
        tol = 0.0
        if self.rtol is not None:
            tol = self.rtol * scale
        if self.atol is not None:
            tol = max(tol, self.atol)

        floor = PRECISION_FLOOR_FACTOR * torch.finfo(rhs.dtype).eps * scale
        tol = max(tol, floor)

        if not tol > 0.0:
            # ``scale`` is zero (or denormal) and no atol was given. The solve is
            # trivially satisfied by x=0; any positive number does, so use the
            # dtype default rather than handing the kernel a zero tolerance it
            # can never meet
            from fluidgym.simulation.pict import PISOtorch_diff

            tol = PISOtorch_diff._get_solver_tolerance(dtype=rhs.dtype)
        return float(tol)

    def __str__(self) -> str:
        parts = []
        if self.rtol is not None:
            parts.append(f"rtol={self.rtol:.03e}")
        if self.atol is not None:
            parts.append(f"atol={self.atol:.03e}")
        return f"SolverTolerance({', '.join(parts)})"


def parse_tolerance(spec: Any) -> float | SolverTolerance | None:
    """Coerce a config value into a tolerance usable by the simulation.

    Accepts ``None``, a number (absolute tolerance, unchanged semantics), a
    :class:`SolverTolerance`, or a mapping with ``rtol``/``atol`` keys -- the
    latter is what a Hydra/OmegaConf config node looks like, e.g.::

        tol:
          potential: {rtol: 1.0e-6, atol: 1.0e-12}
          pressure: 1.0e-5
    """
    if spec is None or isinstance(spec, SolverTolerance):
        return spec
    if isinstance(spec, numbers.Real) and not isinstance(spec, bool):
        return float(spec)

    # OmegaConf DictConfig is a Mapping, but only after resolution; go through
    # the generic mapping protocol so no hard dependency on omegaconf is needed
    if hasattr(spec, "keys"):
        keys = set(spec.keys())
        unknown = keys - {"rtol", "atol"}
        if unknown:
            raise ValueError(
                f"Unknown tolerance keys {sorted(unknown)}, "
                "expected 'rtol' and/or 'atol'."
            )
        rtol = spec.get("rtol", None)
        atol = spec.get("atol", None)
        return SolverTolerance(
            rtol=None if rtol is None else float(rtol),
            atol=None if atol is None else float(atol),
        )

    raise TypeError(
        f"Cannot interpret {spec!r} as a solver tolerance; expected None, a float, "
        "a SolverTolerance, or a mapping with rtol/atol."
    )


def resolve_tolerance(tol: Any, rhs: torch.Tensor) -> Any:
    """Resolve ``tol`` against ``rhs``, passing non-relative values through.

    This is the single choke point that turns a :class:`SolverTolerance` into the
    absolute number the CUDA kernels expect. Floats, tensors and ``None`` are
    returned unchanged so existing behaviour is bit-identical.
    """
    if isinstance(tol, SolverTolerance):
        return tol.resolve(rhs)
    return tol
