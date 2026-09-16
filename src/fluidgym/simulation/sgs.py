"""Sub-grid-scale (LES) eddy-viscosity models shared by the FluidGym environments."""

from typing import Any, Literal

import torch

from fluidgym.simulation.extensions import (
    PISOtorch,  # type: ignore[import-untyped,import-not-found]
)
from fluidgym.simulation.pict.PISOtorch_simulation import append_prep_fn

SGSModel = Literal["none", "smagorinsky", "wale"]

#: Sensible defaults per model. Note the conventions differ, see
#: :func:`append_sgs_viscosity_prep_fn`.
DEFAULT_COEFFICIENTS: dict[str, float] = {
    "smagorinsky": 0.1**2,  # C_s**2, not C_s
    "wale": 0.325,  # textbook Cw, squared inside the extension
}


def append_sgs_viscosity_prep_fn(
    prep_fn: dict[str, Any],
    model: SGSModel,
    coefficient: float,
    ndims: int,
    dtype: torch.dtype,
    cuda_device: torch.device,
    cpu_device: torch.device | None = None,
    scale_sqr: list[torch.Tensor] | None = None,
) -> None:
    """Register a ``"PRE"`` hook setting ``block.viscosity = nu_SGS + nu`` each substep.

    The eddy viscosity is recomputed from the current velocity field before the
    momentum matrix is assembled, and added to the molecular viscosity per cell.
    It affects momentum only; passive scalars keep their molecular diffusivity.

    Parameters
    ----------
    prep_fn: dict[str, Any]
        The callback dict to append to. Modified in place.

    model: SGSModel
        The sub-grid-scale model. ``"none"`` registers nothing.

    coefficient: float
        The model coefficient. The conventions differ per model and are *not*
        interchangeable: ``"smagorinsky"`` expects ``C_s**2``, while ``"wale"``
        expects the textbook ``Cw`` (~0.325-0.5), which the extension squares
        internally.

    ndims: int
        Number of spatial dimensions. ``"wale"`` requires 3.

    dtype: torch.dtype
        Dtype of the domain; the coefficient tensor must match it.

    cuda_device: torch.device
        The CUDA device the simulation runs on.

    cpu_device: torch.device | None
        Device for the coefficient tensor, which the extension requires on the
        host. Defaults to CPU.

    scale_sqr: list[torch.Tensor] | None
        Optional per-block multiplicative damping field applied to nu_SGS before
        the molecular viscosity is added (e.g. squared van Driest for a channel).
    """
    if model == "none" or coefficient == 0:
        return
    if model not in ("smagorinsky", "wale"):
        raise ValueError(f"Unknown SGS model {model!r}.")
    if model == "wale" and ndims < 3:
        # Cayley-Hamilton makes g2 isotropic for a divergence-free 1D/2D field, so
        # the WALE operator is identically zero. Fail rather than run a null model
        raise ValueError(
            "The WALE operator vanishes identically for divergence-free 1D/2D "
            f"fields; it is only meaningful in 3D (got ndims={ndims})."
        )

    cpu_device = torch.device("cpu") if cpu_device is None else cpu_device
    t_coefficient = torch.tensor([coefficient], dtype=dtype, device=cpu_device)

    sgs_fn = (
        PISOtorch.SGSviscosityIncompressibleSmagorinsky
        if model == "smagorinsky"
        else PISOtorch.SGSviscosityIncompressibleWALE
    )

    def add_block_SGS_viscosity(domain, **kwargs):
        domain.UpdateDomainData()

        sgs_viscosities = sgs_fn(domain, t_coefficient)
        base_viscosity = domain.viscosity.to(cuda_device)

        for idx, (block, visc) in enumerate(
            zip(domain.getBlocks(), sgs_viscosities, strict=True)
        ):
            if scale_sqr is not None:
                visc = visc * scale_sqr[idx]
            block.setViscosity(visc + base_viscosity)

        domain.UpdateDomainData()

    append_prep_fn(prep_fn, "PRE", add_block_SGS_viscosity)
