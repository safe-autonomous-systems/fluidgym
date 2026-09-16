"""Telemetry for the linear solves.

Every solve in the codebase funnels through
``PISOtorch_diff._linear_solve_wrapper``, and the CUDA kernels already return a
``LinearSolverResultInfo`` per right-hand side with the final residual, the
iteration count and a converged flag. Until now that information was discarded
unless the solve failed, so there was no way to tell how hard the pressure or the
electric-potential solve was actually working -- or how close it was running to
``max_iter``.

This module adds an opt-in sink. When no recorder is active nothing is computed
and nothing is allocated, so the RL inner loop is unaffected.

The residuals are worth reading as physics, not just as solver bookkeeping:

* the pressure residual is the mass-conservation error of the projection;
* the electric-potential residual is the charge-conservation error. The
  face-based current density is built so that ``div J`` equals the epot Poisson
  residual (see ``mhd_simulation.add_lorentz_force``), so an under-converged
  potential solve feeds a spurious Lorentz force back into the momentum equation.

Usage
-----
>>> with SolverStatsRecorder() as rec:
...     sim.single_step()
>>> rec.summary()["solver/epot/iters_mean"]
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Probe",
    "SolverRecord",
    "SolverStatsRecorder",
    "is_active",
    "record",
    "saturation_warn_fraction",
    "set_saturation_warn_fraction",
    "want_timing",
]

_LOG = logging.getLogger("fluidgym.SolverStats")

# Active recorders, innermost last. A list rather than a single slot so that a
# short diagnostic recorder can be nested inside an episode-long one
_SINKS: list[SolverStatsRecorder] = []

_SATURATION_LOG_FRACTION = 0.8


def set_saturation_warn_fraction(fraction: float) -> None:
    """Set the ``used_iterations / max_iterations`` ratio that triggers a warning."""
    global _SATURATION_LOG_FRACTION
    if not 0.0 < fraction <= 1.0:
        raise ValueError("saturation warn fraction must be in (0, 1].")
    _SATURATION_LOG_FRACTION = float(fraction)


def saturation_warn_fraction() -> float:
    """Return the iteration fraction above which a solve is logged as saturated."""
    return _SATURATION_LOG_FRACTION


def is_active() -> bool:
    """Whether any recorder is currently collecting. Cheap; call before measuring."""
    return bool(_SINKS)


@dataclass
class SolverRecord:
    """One linear solve, as reported by the solver kernels."""

    tag: str
    n_rhs: int
    iterations: int
    max_iterations: int
    final_residual: float
    tolerance: float
    rhs_scale: float
    converged: bool
    is_finite: bool
    use_BiCG: bool = False
    transpose: bool = False
    is_fwd: bool | None = None
    wall_time: float = 0.0

    @property
    def relative_residual(self) -> float:
        """``||r|| / ||b||``. NaN when the RHS scale was not measured."""
        if self.rhs_scale > 0.0:
            return self.final_residual / self.rhs_scale
        return math.nan

    @property
    def saturation(self) -> float:
        """Fraction of ``max_iterations`` consumed."""
        if self.max_iterations > 0:
            return self.iterations / self.max_iterations
        return math.nan


def record(rec: SolverRecord) -> None:
    """Hand a record to every active recorder. No-op when none is active."""
    for sink in _SINKS:
        sink.add(rec)


@dataclass
class SolverStatsRecorder:
    """Collect :class:`SolverRecord` s over a scope.

    Parameters
    ----------
    time_solves: bool
        Measure wall time per solve. Requires a device synchronization around
        each solve, so it is only meaningful (and only paid for) while a recorder
        is active.

    keep_records: bool
        Retain every individual record. Off by default: the running aggregates
        are enough for per-step logging, and an episode of the large duct is
        hundreds of thousands of solves.
    """

    time_solves: bool = True
    keep_records: bool = False
    records: list[SolverRecord] = field(default_factory=list)
    _agg: dict[str, dict[str, float]] = field(default_factory=dict)

    # -- collection ---------------------------------------------------------

    def add(self, rec: SolverRecord) -> None:
        """Aggregate a single solver record."""
        if self.keep_records:
            self.records.append(rec)

        # Adjoint solves are aggregated separately: in a differentiable run they
        # would otherwise be averaged together with the forward solves, hiding
        # the fact that they can behave quite differently at the same tolerance
        key = rec.tag if rec.is_fwd is not False else rec.tag + ":bwd"

        agg = self._agg.setdefault(
            key,
            {
                "n_solves": 0.0,
                "n_rhs": 0.0,
                "iters_sum": 0.0,
                "iters_max": 0.0,
                "res_max": 0.0,
                "rel_res_max": 0.0,
                "tol_min": math.inf,
                "tol_max": 0.0,
                "rhs_scale_max": 0.0,
                "saturation_max": 0.0,
                "n_not_converged": 0.0,
                "n_not_finite": 0.0,
                "n_zero_iterations": 0.0,
                "time_total": 0.0,
            },
        )
        agg["n_solves"] += 1.0
        agg["n_rhs"] += rec.n_rhs
        agg["iters_sum"] += rec.iterations
        agg["iters_max"] = max(agg["iters_max"], rec.iterations)
        agg["res_max"] = max(agg["res_max"], rec.final_residual)
        rel = rec.relative_residual
        if not math.isnan(rel):
            agg["rel_res_max"] = max(agg["rel_res_max"], rel)
        agg["tol_min"] = min(agg["tol_min"], rec.tolerance)
        agg["tol_max"] = max(agg["tol_max"], rec.tolerance)
        agg["rhs_scale_max"] = max(agg["rhs_scale_max"], rec.rhs_scale)
        sat = rec.saturation
        if not math.isnan(sat):
            agg["saturation_max"] = max(agg["saturation_max"], sat)
        if not rec.converged:
            agg["n_not_converged"] += 1.0
        if not rec.is_finite:
            agg["n_not_finite"] += 1.0
        # A solve that met its tolerance before doing any work returns its initial
        # iterate untouched. Harmless for a warm-started forward solve, but an
        # adjoint solve starts from zero, so on the ``:bwd`` keys this counts
        # gradients that were silently discarded rather than computed
        if rec.iterations <= 0:
            agg["n_zero_iterations"] += 1.0
        agg["time_total"] += rec.wall_time

    def reset(self) -> None:
        """Drop all records and aggregates."""
        self.records.clear()
        self._agg.clear()

    # -- reporting ----------------------------------------------------------

    @property
    def tags(self) -> Iterable[str]:
        """The solver tags seen so far."""
        return self._agg.keys()

    def summary(self, prefix: str = "solver") -> dict[str, float]:
        """Flat ``{key: value}`` aggregates, ready for an info dict or a CSV row.

        Keys are ``<prefix>/<tag>/<stat>``. ``rel_res_max`` -- the largest
        relative residual ``||r||/||b||`` seen for that solve -- is the one to
        watch: unlike the iteration count it is comparable across domain sizes,
        and unlike the absolute residual it is comparable across tolerances.
        """
        out: dict[str, float] = {}
        for tag, agg in sorted(self._agg.items()):
            n = agg["n_solves"]
            base = f"{prefix}/{tag}"
            out[f"{base}/n_solves"] = n
            out[f"{base}/iters_mean"] = agg["iters_sum"] / n if n else 0.0
            out[f"{base}/iters_max"] = agg["iters_max"]
            out[f"{base}/res_max"] = agg["res_max"]
            out[f"{base}/rel_res_max"] = agg["rel_res_max"]
            out[f"{base}/tol_min"] = (
                0.0 if math.isinf(agg["tol_min"]) else agg["tol_min"]
            )
            out[f"{base}/tol_max"] = agg["tol_max"]
            out[f"{base}/rhs_scale_max"] = agg["rhs_scale_max"]
            out[f"{base}/saturation_max"] = agg["saturation_max"]
            out[f"{base}/n_not_converged"] = agg["n_not_converged"]
            out[f"{base}/not_converged_rate"] = agg["n_not_converged"] / n if n else 0.0
            out[f"{base}/n_not_finite"] = agg["n_not_finite"]
            out[f"{base}/n_zero_iterations"] = agg["n_zero_iterations"]
            out[f"{base}/zero_iteration_rate"] = (
                agg["n_zero_iterations"] / n if n else 0.0
            )
            if self.time_solves:
                out[f"{base}/time_total"] = agg["time_total"]
        return out

    def describe(self) -> str:
        """Return a human-readable per-tag summary of the collected solves."""
        lines = []
        for tag, agg in sorted(self._agg.items()):
            n = agg["n_solves"]
            iters_mean = agg["iters_sum"] / n if n else 0.0
            tol_min = 0.0 if math.isinf(agg["tol_min"]) else agg["tol_min"]
            lines.append(
                f"  {tag:<10s} n={int(n):<7d} "
                f"iters mean/max={iters_mean:.1f}/{int(agg['iters_max']):d}  "
                f"res_max={agg['res_max']:.03e}  "
                f"rel_res_max={agg['rel_res_max']:.03e}  "
                f"tol=[{tol_min:.03e}, {agg['tol_max']:.03e}]  "
                f"sat_max={agg['saturation_max']:.2f}  "
                f"not_conv={int(agg['n_not_converged']):d}  "
                f"zero_it={int(agg['n_zero_iterations']):d}  "
                f"t={agg['time_total']:.2f}s"
            )
        return "Solver stats:\n" + ("\n".join(lines) if lines else "  (no solves)")

    # -- context management -------------------------------------------------

    def __enter__(self) -> SolverStatsRecorder:
        _SINKS.append(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            _SINKS.remove(self)
        except ValueError:  # pragma: no cover - defensive
            pass


def want_timing() -> bool:
    """Whether any active recorder asked for wall-time measurement."""
    return any(sink.time_solves for sink in _SINKS)


class Probe:
    """Measures one solve and turns its ``LinearSolverResultInfo`` s into records.

    Instantiated only while a recorder is active (guard with :func:`is_active`),
    so a normal run pays neither the device synchronization for the timing nor
    the norm reduction for the RHS scale.
    """

    __slots__ = ("_rhs", "_tag", "_time", "_start", "_device")

    def __init__(self, rhs: Any, tag: str) -> None:
        self._rhs = rhs
        self._tag = tag
        self._time = want_timing()
        self._start = 0.0
        self._device = getattr(rhs, "device", None)

    def _sync(self) -> None:
        if self._device is not None and self._device.type == "cuda":
            import torch

            torch.cuda.synchronize(self._device)

    def start(self) -> None:
        """Start timing a solve."""
        if self._time:
            self._sync()
            self._start = time.perf_counter()

    def stop(
        self,
        solver_infos: Iterable[Any],
        tolerance: float,
        max_iterations: int,
        use_BiCG: bool = False,
        transpose: bool = False,
        is_fwd: bool | None = None,
    ) -> None:
        """Stop timing and record the results reported by ``solver_infos``."""
        elapsed = 0.0
        if self._time:
            self._sync()
            elapsed = time.perf_counter() - self._start

        from fluidgym.simulation.solver_tolerance import rhs_scale as _rhs_scale

        scale = _rhs_scale(self._rhs)

        infos = list(solver_infos)
        n_rhs = len(infos)
        for info in infos:
            rec = SolverRecord(
                tag=self._tag,
                n_rhs=n_rhs,
                iterations=int(info.usedIterations),
                max_iterations=int(max_iterations),
                final_residual=float(info.finalResidual),
                tolerance=float(tolerance),
                rhs_scale=scale,
                converged=bool(info.converged),
                is_finite=bool(info.isFiniteResidual),
                use_BiCG=use_BiCG,
                transpose=transpose,
                is_fwd=is_fwd,
                # Attribute the measured time to the first RHS only, so summing
                # ``time_total`` over a batched solve does not multiply-count it
                wall_time=elapsed if info is infos[0] else 0.0,
            )
            record(rec)

            if (
                rec.converged
                and rec.max_iterations > 0
                and rec.saturation > _SATURATION_LOG_FRACTION
            ):
                _LOG.debug(
                    "Solve '%s' converged but used %d/%d iterations (%.0f%% of the "
                    "limit) at tol %.03e (rel. residual %.03e).",
                    rec.tag,
                    rec.iterations,
                    rec.max_iterations,
                    100.0 * rec.saturation,
                    rec.tolerance,
                    rec.relative_residual,
                )
