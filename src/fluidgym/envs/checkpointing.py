"""Activation checkpointing for differentiable BPTT rollouts."""

from __future__ import annotations

import gc
import os
import shutil
import socket
import tempfile
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from phipict import _C

from fluidgym.logging import get_logger

logger = get_logger(__name__)


class _SpilledTensor:
    """One saved tensor living in a file instead of in RAM.

    The file is unlinked when this handle is collected, so disk is reclaimed on
    exactly the same schedule host memory would have been: autograd drops its
    reference to the saved tensor, the handle dies, the file goes.

    Bytes are written through a ``uint8`` view rather than via ``numpy`` dtypes for 
    better compatibility.
    """

    __slots__ = ("path", "shape", "dtype", "__weakref__")

    def __init__(self, path: str, tensor: torch.Tensor) -> None:
        self.path = path
        self.shape = tuple(tensor.shape)
        self.dtype = tensor.dtype
        flat = tensor.detach().contiguous().reshape(-1)
        flat.view(torch.uint8).numpy().tofile(path)

    def load(self) -> torch.Tensor:
        raw = np.fromfile(self.path, dtype=np.uint8)
        return torch.from_numpy(raw).view(self.dtype).reshape(self.shape)

    def __del__(self) -> None:
        try:
            os.unlink(self.path)
        except (OSError, TypeError):  # already gone, or interpreter teardown
            pass


# A piece of carried state: a zero-arg getter and a one-arg setter
Accessor = tuple[Callable[..., Any], Callable[..., None]]


def default_domain_accessors(domain: Any) -> list[Accessor]:
    """Accessors for the generic per-step evolving solver state of a domain.

    Per block: ``velocity`` / ``pressure`` / ``passiveScalar`` / ``epot`` /
    ``velocitySource``, plus every *varying* ``FixedBoundary`` velocity and
    passive scalar, plus ``domain.epotResult``. Absent fields are skipped, so the
    same list works for every env.

    Assembled matrices, RHS and intermediate result vectors are excluded: they are
    rebuilt every step, so carrying them would only cost memory.
    """
    acc: list[Accessor] = []
    for block in domain.getBlocks():
        acc.append((lambda b=block: b.velocity, lambda t, b=block: b.setVelocity(t)))
        acc.append((lambda b=block: b.pressure, lambda t, b=block: b.setPressure(t)))
        acc.append(
            (lambda b=block: b.passiveScalar, lambda t, b=block: b.setPassiveScalar(t))
        )
        acc.append((lambda b=block: b.epot, lambda t, b=block: b.setEpot(t)))
        acc.append(
            (
                lambda b=block: b.velocitySource,
                lambda t, b=block: b.setVelocitySource(t),
            )
        )
        for bidx in range(domain.getSpatialDims() * 2):
            bound = block.getBoundary(bidx)
            if not isinstance(bound, _C.FixedBoundary):
                continue
            # Static boundaries never change and their setters may reject writes
            if not bound.isVelocityStatic:
                acc.append(
                    (
                        lambda bd=bound: bd.velocity,
                        lambda t, bd=bound: bd.setVelocity(t),
                    )
                )
            if bound.hasPassiveScalar() and not bound.isPassiveScalarStatic():
                acc.append(
                    (
                        lambda bd=bound: bd.passiveScalar,
                        lambda t, bd=bound: bd.setPassiveScalar(t),
                    )
                )
    if domain.hasEpot():
        acc.append((lambda: domain.epotResult, lambda t: domain.setEpotResult(t)))
    return acc


def snapshot(
    accessors: Sequence[Accessor], clone: bool
) -> tuple[list[torch.Tensor], list[bool]]:
    """Capture present accessor tensors plus a presence mask.

    ``clone=True`` is required for a snapshot that must survive the in-place
    forward rollout.
    """
    tensors: list[torch.Tensor] = []
    mask: list[bool] = []
    for getter, _setter in accessors:
        t = getter()
        present = t is not None and torch.is_tensor(t) and t.numel() > 0
        mask.append(present)
        if present:
            assert t is not None
            tensors.append(t.clone() if clone else t)
    return tensors, mask


def restore(
    accessors: Sequence[Accessor],
    tensors: Sequence[torch.Tensor],
    mask: Sequence[bool],
    on_restored: Callable[[], None] | None = None,
) -> None:
    """Write a snapshot back through the accessor setters."""
    it = iter(tensors)
    for present, (_getter, setter) in zip(mask, accessors, strict=True):
        if present:
            setter(next(it))
    if on_restored is not None:
        on_restored()


def _flatten(obj: Any) -> tuple[list[torch.Tensor], Any]:
    """Split a nested result into its tensors and a rebuild spec."""
    if torch.is_tensor(obj):
        return [obj], _Leaf()
    if isinstance(obj, dict):
        tensors: list[torch.Tensor] = []
        dict_spec: dict[Any, Any] = {}
        for key, value in obj.items():
            sub_t, sub_s = _flatten(value)
            tensors += sub_t
            dict_spec[key] = sub_s
        return tensors, dict_spec
    if isinstance(obj, (list, tuple)):
        tensors = []
        seq_spec: list[Any] = []
        for value in obj:
            sub_t, sub_s = _flatten(value)
            tensors += sub_t
            seq_spec.append(sub_s)
        return tensors, (type(obj), seq_spec)
    return [], _Const(obj)


class _Leaf:
    pass


class _Const:
    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value


def _unflatten(spec: Any, it: Any) -> Any:
    if isinstance(spec, _Leaf):
        return next(it)
    if isinstance(spec, _Const):
        return spec.value
    if isinstance(spec, dict):
        return {k: _unflatten(v, it) for k, v in spec.items()}
    kind, items = spec
    return kind(_unflatten(v, it) for v in items)


class _CheckpointedStep(torch.autograd.Function):
    """One checkpointed environment step. See the module docstring."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        restore_fn,
        run_fn,
        snapshot_fn,
        release_fn,
        mask,
        n_grad_inputs,
        offload_device,
        spill_dir,
        spec_out,
        *args,
    ):
        grad_inputs = args[:n_grad_inputs]
        carry = args[n_grad_inputs:]

        ctx.restore_fn = restore_fn
        ctx.run_fn = run_fn
        ctx.snapshot_fn = snapshot_fn
        ctx.release_fn = release_fn
        ctx.mask = mask
        ctx.n_grad_inputs = n_grad_inputs
        ctx.offload_device = offload_device

        if offload_device is None:
            ctx.save_for_backward(*args)
            ctx.compute_device = None
        else:
            # Only the small grad inputs stay on the compute device; the carry is
            # parked off-GPU (or on disk) until this step's backward
            ctx.save_for_backward(*grad_inputs)
            ctx.compute_device = (
                carry[0].device
                if carry
                else (grad_inputs[0].device if grad_inputs else None)
            )
            if spill_dir is None:
                ctx.offloaded_carry = [t.to(offload_device) for t in carry]
            else:
                ctx.offloaded_carry = [
                    _SpilledTensor(
                        os.path.join(spill_dir, f"carry-{id(ctx):x}-{i:04d}.bin"),
                        t.to("cpu"),
                    )
                    for i, t in enumerate(carry)
                ]

        with torch.no_grad():
            restore_fn(list(carry), mask)
            result = run_fn(grad_inputs)
            outputs, spec = _flatten(result)
            spec_out.append(spec)
            new_carry, _ = snapshot_fn(clone=True)

        ctx.n_carry = len(new_carry)
        return (*(t.detach() for t in new_carry), *(t.detach() for t in outputs))

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        if ctx.offload_device is None:
            saved = ctx.saved_tensors
            grad_inputs = saved[: ctx.n_grad_inputs]
            carry = saved[ctx.n_grad_inputs :]
        else:
            grad_inputs = ctx.saved_tensors
            carry = [
                (t.load() if isinstance(t, _SpilledTensor) else t).to(
                    ctx.compute_device
                )
                for t in ctx.offloaded_carry
            ]

        # The replay below overwrites the live domain, and the engine runs segments
        # in reverse, so the last one replayed is the *first* of the rollout --
        # without restoring this, backward() would leave the simulation rewound to
        # the start of the horizon. Values only: every caller detaches at the
        # chunk boundary anyway
        with torch.no_grad():
            live_state, live_mask = ctx.snapshot_fn(clone=True)

        gi_leaves = [t.detach().requires_grad_(True) for t in grad_inputs]
        carry_leaves = [t.detach().requires_grad_(True) for t in carry]

        with torch.enable_grad():
            ctx.restore_fn(carry_leaves, ctx.mask)
            result = ctx.run_fn(gi_leaves)
            outputs, _ = _flatten(result)
            new_carry, _ = ctx.snapshot_fn(clone=True)
            all_outputs = (*new_carry, *outputs)

        # Outputs with no grad path (e.g. no_grad-updated boundaries) carry no
        # gradient anyway
        diff_out, diff_g = [], []
        for out, gout in zip(all_outputs, grad_outputs, strict=True):
            if out.requires_grad and gout is not None:
                diff_out.append(out)
                diff_g.append(gout)

        inputs = (*gi_leaves, *carry_leaves)

        if diff_out:
            grads = torch.autograd.grad(
                diff_out, inputs, grad_outputs=diff_g, allow_unused=True
            )
        else:
            grads = tuple(None for _ in inputs)

        # backward() is a pure gradient computation: leave the state as we found it.
        # This also drops the last references the replay left in the solver state,
        # the fields the segment wrote are graph-connected and would pin its tape.
        # The carry is only the part of that state a step reads; ``release_fn``
        # covers the rest, which the replay rebuilt just as graph-connected
        with torch.no_grad():
            if ctx.release_fn is not None:
                ctx.release_fn()
            ctx.restore_fn(live_state, live_mask)

        del result, outputs, new_carry, all_outputs, diff_out, diff_g
        del inputs, gi_leaves, carry_leaves, carry
        gc.collect()

        return (None, None, None, None, None, None, None, None, None, *grads)


_CARRY_PREFIX = "fluidgym-carry-"


def _spill_scope() -> str:
    """Directory name isolating this user's, host's and job's carries."""
    for var in ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID", "JOB_ID"):
        job = os.environ.get(var)
        if job:
            break
    else:
        job = f"pid{os.getpid()}"
    user = os.environ.get("USER") or "unknown"
    host = socket.gethostname().split(".")[0]
    return f"{user}-{host}-{job}"


def _scope_prefix() -> str:
    """The ``<user>-<host>-`` part of :func:`_spill_scope`, for sweeping."""
    return _spill_scope().rsplit("-", 1)[0] + "-"


def _carry_dir_pid(name: str) -> int | None:
    """The pid encoded in a carry directory name, or ``None`` if it has none."""
    if not name.startswith(_CARRY_PREFIX):
        return None
    rest = name[len(_CARRY_PREFIX) :]
    pid, _, _ = rest.partition("-")
    return int(pid) if pid.isdigit() else None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # someone else's pid: alive, just not ours to signal
        return True
    except OSError:
        return True
    return True


def sweep_stale_carry_dirs(root: str) -> None:
    """Delete carry directories left behind by dead processes of this user.

    A run killed outright (OOM, ``scancel``) never runs ``_SpilledTensor.__del__``,
    so its carries stay on disk forever. Only scopes tagged with this user and host
    are touched, so the pid check is meaningful and other users' files are never
    considered.
    """
    prefix = _scope_prefix()
    try:
        scopes = [e for e in os.listdir(root) if e.startswith(prefix)]
    except OSError:
        return

    for scope in scopes:
        scope_path = os.path.join(root, scope)
        try:
            entries = os.listdir(scope_path)
        except OSError:
            continue
        for entry in entries:
            pid = _carry_dir_pid(entry)
            if pid is None or pid == os.getpid() or _pid_alive(pid):
                continue
            logger.info("Removing stale BPTT carry dir %s", os.path.join(scope, entry))
            shutil.rmtree(os.path.join(scope_path, entry), ignore_errors=True)
        # An empty scope belongs to a finished job; drop it so root stays readable
        try:
            os.rmdir(scope_path)
        except OSError:
            pass


def make_carry_spill_dir(root: str) -> str:
    """A fresh directory for this process's spilled carries.

    Carries go to ``<root>/<user>-<host>-<job>/fluidgym-carry-<pid>-XXXXXXXX``, so
    concurrent runs -- ranks of one job, several jobs, several users on one node --
    never share a directory and never trip over each other's directory permissions.
    Leftovers from dead processes of this user are swept first.
    """
    os.makedirs(root, exist_ok=True)
    sweep_stale_carry_dirs(root)
    scope = os.path.join(root, _spill_scope())
    os.makedirs(scope, exist_ok=True)
    return tempfile.mkdtemp(prefix=f"{_CARRY_PREFIX}{os.getpid()}-", dir=scope)


def cleanup_carry_spill_dir(path: str | None) -> None:
    """Remove a directory from :func:`make_carry_spill_dir` and its scope if empty.

    Safe to call more than once, and safe to call while files are still in it: the
    handles that own them are gone by the time anything calls this.
    """
    if not path:
        return
    shutil.rmtree(path, ignore_errors=True)
    try:
        os.rmdir(os.path.dirname(path))
    except OSError:  # other ranks of this job are still running
        pass


def checkpointed_step(
    accessors: Sequence[Accessor],
    step_fn: Callable[..., Any],
    grad_inputs: Sequence[torch.Tensor] = (),
    on_restored: Callable[[], None] | None = None,
    release_fn: Callable[[], None] | None = None,
    offload_device: torch.device | str | None = None,
    spill_dir: str | None = None,
) -> Any:
    """Run ``step_fn`` as one checkpointed segment and return its result.

    ``step_fn(*grad_inputs)`` may return any nesting of tensors, dicts, lists and
    plain values; tensors come back connected to the outer graph, everything else
    is passed through from the rollout.

    It is called twice: Once under ``no_grad`` now and once under ``enable_grad``
    in backward. Thus, anything it mutates that also affects its own result must be
    reachable through ``accessors``, which are reset before each call.

    ``release_fn`` detaches the state a step writes but does not read back, which
    is therefore not part of the carry: assembled matrices, right-hand sides,
    intermediate result vectors. The replay in backward leaves those as
    graph-connected as the rollout did, and since they live in the solver rather
    than in the graph, they would keep that segment's tape alive for as long as
    the solver does. Backward calls this before it restores the carry.
    """
    if offload_device is not None and not isinstance(offload_device, torch.device):
        offload_device = torch.device(offload_device)
    if spill_dir is not None and offload_device is None:
        offload_device = torch.device("cpu")

    def restore_fn(tensors, mask):
        restore(accessors, tensors, mask, on_restored)

    def snapshot_fn(clone):
        return snapshot(accessors, clone=clone)

    def run_fn(gi):
        return step_fn(*gi)

    state, mask = snapshot_fn(clone=True)
    spec_out: list[Any] = []
    result = _CheckpointedStep.apply(
        restore_fn,
        run_fn,
        snapshot_fn,
        release_fn,
        mask,
        len(grad_inputs),
        offload_device,
        spill_dir,
        spec_out,
        *grad_inputs,
        *state,
    )

    # Re-derive the carry length from the live domain so a field that appears
    # mid-rollout is threaded from then on
    _, mask = snapshot_fn(clone=False)
    n_state = sum(mask)
    # Leave the live domain holding the grad-connected end state, so the next
    # step and the observation stay differentiable
    restore_fn(list(result[:n_state]), mask)  # type: ignore

    return _unflatten(spec_out[0], iter(result[n_state:]))  # type: ignore
