"""Vectorized FluidGym environment whose environments are spread over processes/GPUs.

Every worker process holds one batched environment (``FluidEnv(n_envs=k)``); the
workers' environments together form one vectorized environment with
``len(cuda_ids) * k`` environments and exactly the API of a vectorized
``FluidEnv``: a leading env dim on actions, observations, rewards, flags and infos
(``[E, n_agents, ...]`` in MARL mode). So batching within a GPU and parallelism
across GPUs are interchangeable, see :func:`fluidgym.make_vec`.

The workers are driven by a small request/reply protocol: every request is
answered, either with its result or with the worker's traceback, and a worker
that dies is detected through its process sentinel. A failure in a worker thus
raises :class:`WorkerError` in the parent instead of hanging it.
"""

import multiprocessing as mp
import signal
import traceback
import weakref
from collections.abc import Callable, Sequence
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from typing import Any

import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

from fluidgym.registry import make
from fluidgym.types import EnvMode, FluidEnvLike

# Seconds a worker gets to shut down after it was asked to, before it is terminated
_CLOSE_TIMEOUT = 60.0

# Requests the parent sends as ``(op, payload)``. Every request but ``_CLOSE`` is
# answered with ``(True, result)`` or ``(False, traceback)``
_CALL = "call"
_GETATTR = "getattr"
_CLOSE = "close"


class _Missing:
    """Reply to a ``_GETATTR`` request for an attribute the env does not have."""


class _Method:
    """Reply to a ``_GETATTR`` request for a method, which cannot be pickled."""


class WorkerError(RuntimeError):
    """A worker process of a :class:`ParallelFluidEnv` failed or died."""


def _map_tensors(value: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    """Apply ``fn`` to every tensor in (nested) tuples, lists and dicts."""
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, dict):
        return {k: _map_tensors(v, fn) for k, v in value.items()}
    # Exact types: a named tuple cannot be rebuilt from an iterable
    if type(value) in (tuple, list):
        return type(value)(_map_tensors(v, fn) for v in value)
    return value


def _cat(values: Sequence[Any]) -> Any:
    """Concatenate per-worker results along the env dim, on the CPU.

    Tensors, and (nested) dicts of them, are concatenated along dim 0.
    """
    first = values[0]
    if isinstance(first, dict):
        return {k: _cat([v[k] for v in values]) for k in first}
    if torch.is_tensor(first):
        return torch.cat([v.detach().cpu() for v in values], dim=0)
    raise TypeError(f"Unsupported result type: {type(first)}")


def _worker(
    rank: int,
    env_id: str,
    env_kwargs: dict[str, Any],
    cuda_id: int,
    n_envs: int,
    pipe: Connection,
) -> None:
    """Build a batched environment and serve the parent's requests on ``pipe``.

    The first message is the reply to the implicit construction request: the
    environment's spec, or the traceback of why it could not be built.
    """
    # Ctrl+C reaches the whole process group; the parent decides when to shut down
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    env = None
    try:
        try:
            torch.cuda.set_device(cuda_id)
            device = torch.device(f"cuda:{cuda_id}")
            env = make(id=env_id, **env_kwargs, n_envs=n_envs, cuda_device=device)
            env._env_index_offset = rank * n_envs
            spec = {
                "action_space": env.action_space,
                "observation_space": env.observation_space,
                "n_agents": env.n_agents,
                "metrics": env.metrics,
                "episode_length": env.episode_length,
                "use_marl": env.use_marl,
            }
        except Exception:
            pipe.send((False, traceback.format_exc()))
            return
        pipe.send((True, spec))

        def to_device(t: torch.Tensor) -> torch.Tensor:
            return t.to(device)

        def to_cpu(t: torch.Tensor) -> torch.Tensor:
            return t.detach().cpu()

        while True:
            try:
                op, payload = pipe.recv()
            except EOFError:
                # The parent is gone
                break
            if op == _CLOSE:
                break

            try:
                if op == _CALL:
                    name, args, kwargs = payload
                    result = getattr(env, name)(
                        *_map_tensors(args, to_device),
                        **_map_tensors(kwargs, to_device),
                    )
                elif op == _GETATTR:
                    result = getattr(env, payload, _Missing)
                    if callable(result) and not isinstance(result, type):
                        result = _Method
                else:
                    raise ValueError(f"Unknown request {op!r}.")
                # Results go through the pipe as CPU tensors
                reply = (True, _map_tensors(result, to_cpu))
            except Exception:
                reply = (False, traceback.format_exc())

            try:
                pipe.send(reply)
            except OSError:
                # The parent is gone
                break
            except Exception:
                # The result could not be pickled; the pipe is still in sync
                pipe.send((False, traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        torch.cuda.empty_cache()
        pipe.close()


def _shutdown(pipes: list[Connection], processes: list[BaseProcess]) -> None:
    """Ask every worker to exit, and terminate the ones that do not.

    A module-level function, so that the finalizer of a :class:`ParallelFluidEnv`
    does not keep the env alive.
    """
    for pipe in pipes:
        try:
            pipe.send((_CLOSE, None))
        except OSError:
            # Already dead
            pass

    for process in processes:
        process.join(_CLOSE_TIMEOUT)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(_CLOSE_TIMEOUT)
        if process.is_alive():
            process.kill()
            process.join()

    for pipe in pipes:
        pipe.close()


class ParallelFluidEnv(FluidEnvLike):
    """Vectorized FluidGym environment running its environments on several GPUs.

    Attributes not defined here (e.g. ``step_length``, ``ndims`` or ``mode``) are
    read from the environment of the first worker; methods not defined here are
    called on it.

    Parameters
    ----------
    env_id: str
        The environment identifier.

    cuda_ids: list[int]
        The CUDA device of every worker process. A device may occur several times
        (several workers on one GPU).

    n_envs_per_worker: int
        Number of environments every worker simulates together (batched on its
        GPU). Defaults to 1.

    env_kwargs: dict[str, Any]
        The keyword arguments to pass to the environment constructor.

    Raises
    ------
    WorkerError
        If a worker cannot build its environment.
    """

    def __init__(
        self,
        env_id: str,
        cuda_ids: list[int],
        n_envs_per_worker: int = 1,
        **env_kwargs: Any,
    ):
        if len(cuda_ids) == 0:
            raise ValueError("ParallelFluidEnv needs at least one CUDA device.")
        if n_envs_per_worker < 1:
            raise ValueError("n_envs_per_worker must be positive.")
        if "n_envs" in env_kwargs:
            raise ValueError(
                "Pass n_envs_per_worker to ParallelFluidEnv, or use "
                "fluidgym.make_vec(env_id, n_envs, devices)."
            )
        if env_kwargs.get("differentiable", False):
            raise ValueError(
                "ParallelFluidEnv does not support differentiable environments."
            )

        self.__n_workers = len(cuda_ids)
        self.__per_worker = n_envs_per_worker
        self.__n_envs = self.__n_workers * n_envs_per_worker
        self.__cuda_ids = list(cuda_ids)
        # Set when a request was interrupted, see __request
        self.__broken = False
        # For sample_action, seeded by seed() and reset(seed=...)
        self.__rng: torch.Generator | None = None

        # A local context: the global start method belongs to the application
        ctx = mp.get_context("spawn")
        self.__pipes: list[Connection] = []
        self.__processes: list[BaseProcess] = []
        # Shuts the workers down on close(), garbage collection and interpreter
        # exit. Holds the lists, so it also covers workers started before a failure
        # further down
        self.__finalizer = weakref.finalize(
            self, _shutdown, self.__pipes, self.__processes
        )
        for rank, cuda_id in enumerate(self.__cuda_ids):
            parent_conn, child_conn = ctx.Pipe()
            process = ctx.Process(
                target=_worker,
                args=(
                    rank,
                    env_id,
                    env_kwargs,
                    cuda_id,
                    n_envs_per_worker,
                    child_conn,
                ),
                name=f"ParallelFluidEnv-worker-{rank}",
            )
            process.start()
            # Only the worker may hold its end, or its death never reaches the
            # parent as EOF
            child_conn.close()
            self.__pipes.append(parent_conn)
            self.__processes.append(process)

        # Wait until every worker has built its environment
        try:
            specs = self.__collect(range(self.__n_workers), "construction")
        except BaseException:
            self.close()
            raise
        self.__spec: dict[str, Any] = specs[0]

    def __worker_name(self, rank: int) -> str:
        return f"worker {rank} (cuda:{self.__cuda_ids[rank]})"

    def __dead(self, rank: int) -> WorkerError:
        """The error for a worker that exited without replying."""
        self.__broken = True
        process = self.__processes[rank]
        process.join(1.0)
        return WorkerError(
            f"ParallelFluidEnv {self.__worker_name(rank)} died "
            f"(exit code {process.exitcode})."
        )

    def __check_usable(self) -> None:
        if not self.__finalizer.alive:
            raise RuntimeError("ParallelFluidEnv is closed.")
        if self.__broken:
            raise RuntimeError(
                "ParallelFluidEnv is out of sync with its workers after an "
                "interrupted request or a worker's death; close it and create a new "
                "one."
            )

    def __recv(self, rank: int) -> tuple[bool, Any]:
        """Wait for the reply of worker ``rank``, or for it to die."""
        pipe, process = self.__pipes[rank], self.__processes[rank]
        wait([pipe, process.sentinel])
        # A worker that replied and then died still has its reply in the pipe
        if pipe.poll():
            try:
                return pipe.recv()
            except EOFError:
                pass
        raise self.__dead(rank)

    def __collect(self, ranks: Sequence[int], what: str) -> list[Any]:
        """Receive one reply from each of ``ranks``, raising the first failure.

        Every reply is received before anything is raised, so the pipes stay in
        sync when a worker reports an error.
        """
        try:
            replies = [(rank, *self.__recv(rank)) for rank in ranks]
        except BaseException:
            # A reply left in a pipe would be read as the answer to the next request
            self.__broken = True
            raise
        for rank, ok, value in replies:
            if not ok:
                raise WorkerError(
                    f"ParallelFluidEnv {self.__worker_name(rank)} failed in "
                    f"{what}:\n\n{value}"
                )
        return [value for _, _, value in replies]

    def __request(self, op: str, payloads: dict[int, Any], what: str) -> dict[int, Any]:
        """Send ``payloads[rank]`` to every worker in it, then collect the replies."""
        self.__check_usable()
        sent = []
        try:
            for rank, payload in payloads.items():
                try:
                    self.__pipes[rank].send((op, payload))
                except OSError:
                    raise self.__dead(rank) from None
                sent.append(rank)
        except BaseException:
            self.__broken = True
            raise
        return dict(zip(sent, self.__collect(sent, what), strict=True))

    def __call(
        self,
        name: str,
        args: Sequence[tuple] | None = None,
        ranks: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Call ``env.<name>(*args[i], **kwargs)`` on the workers, in parallel.

        ``args`` holds one argument tuple per worker in ``ranks`` (all workers by
        default); without it, every worker is called with ``kwargs`` only.
        """
        ranks = list(range(self.__n_workers)) if ranks is None else list(ranks)
        per_worker: list[tuple] = [()] * len(ranks) if args is None else list(args)
        payloads = {
            rank: (name, tuple(a), kwargs)
            for rank, a in zip(ranks, per_worker, strict=True)
        }
        return list(self.__request(_CALL, payloads, f"{name}()").values())

    def __getattr__(self, name: str) -> Any:
        # Only called if normal attribute lookup fails on self
        if name.startswith("_ParallelFluidEnv__") or name.startswith("__"):
            raise AttributeError(name)
        value = self.__request(_GETATTR, {0: name}, f"getattr({name!r})")[0]
        if value is _Missing:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        if value is _Method:

            def method(*args: Any, **kwargs: Any) -> Any:
                return self.__call(name, [args], ranks=[0], **kwargs)[0]

            return method
        return value

    def __enter__(self) -> "ParallelFluidEnv":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __split(self, values: Any) -> list[Any]:
        """Split per-environment values (length ``n_envs``) into per-worker chunks."""
        k = self.__per_worker
        return [values[w * k : (w + 1) * k] for w in range(self.__n_workers)]

    @property
    def action_space(self) -> spaces.Box:
        """The action space of one environment (and agent)."""
        return self.__spec["action_space"]

    @property
    def observation_space(self) -> spaces.Dict:
        """The observation space of one environment (and agent)."""
        return self.__spec["observation_space"]

    @property
    def differentiable(self) -> bool:
        """Whether the environment is differentiable."""
        return False

    @property
    def n_agents(self) -> int:
        """The number of agents of one environment."""
        return self.__spec["n_agents"]

    @property
    def metrics(self) -> list[str]:
        """The list of metrics tracked by the environment."""
        return self.__spec["metrics"]

    @property
    def episode_length(self) -> int:
        """The number of steps per episode."""
        return self.__spec["episode_length"]

    @property
    def use_marl(self) -> bool:
        """Whether the environment is in multi-agent reinforcement learning mode."""
        return self.__spec["use_marl"]

    @property
    def n_envs(self) -> int:
        """The number of environments over all workers."""
        return self.__n_envs

    @property
    def num_envs(self) -> int:
        """Alias of :attr:`n_envs`, the name vectorized-env libraries use."""
        return self.__n_envs

    @property
    def vectorized(self) -> bool:
        """Always True: the API carries a leading env dim."""
        return True

    @property
    def cuda_device(self) -> torch.device:
        """The CUDA device of the first worker."""
        return torch.device(f"cuda:{self.__cuda_ids[0]}")

    def reset(
        self,
        seed: int | None = None,
        randomize: bool | None = None,
        domain_idx: int | Sequence[int] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Reset all environments, see :meth:`FluidEnv.reset`.

        Parameters
        ----------
        seed: int | None
            The seed to use for random number generation. If None, the current seed is
            used. Worker ``r`` is seeded with ``seed + r``.

        randomize: bool | None
            Whether to randomize the initial state. If None, the default behavior is
            used.

        domain_idx: int | Sequence[int] | None
            Index of the initial domain to load, one for all environments or one per
            environment. Defaults to None.

        Returns
        -------
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
            The initial observations ``[n_envs, ...]`` and an info dictionary.
        """
        if domain_idx is None or isinstance(domain_idx, (int, np.integer)):
            idxs: list[Any] = [domain_idx] * self.__n_workers
        else:
            if len(domain_idx) != self.__n_envs:
                raise ValueError(
                    f"domain_idx has {len(domain_idx)} entries for "
                    f"{self.__n_envs} environments."
                )
            idxs = [list(chunk) for chunk in self.__split(list(domain_idx))]

        args = [
            (None if seed is None else seed + rank, randomize, idx)
            for rank, idx in enumerate(idxs)
        ]
        results = self.__call("reset", args)
        if seed is not None:
            self.__rng = torch.Generator().manual_seed(seed)

        obs, infos = zip(*results, strict=True)
        return _cat(obs), _cat(infos)

    def step(
        self, action: torch.Tensor
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Step all environments, see :meth:`FluidEnv.step`.

        Parameters
        ----------
        action: torch.Tensor
            The actions ``[n_envs, ...]``.

        Returns
        -------
        tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor,
        dict[str, torch.Tensor]]
            Observations, rewards, terminated and truncated flags (bool ``[n_envs]``)
            and infos, all with a leading env dim (on the CPU).
        """
        if action.shape[0] != self.__n_envs:
            raise ValueError(
                f"Expected action batch size {self.__n_envs}, but got {action.shape[0]}"
            )

        # Actions go through the pipe as CPU tensors
        args = [(chunk.detach().cpu(),) for chunk in self.__split(action)]
        results = self.__call("step", args)

        obs, rewards, terms, truncs, infos = zip(*results, strict=True)
        return _cat(obs), _cat(rewards), _cat(terms), _cat(truncs), _cat(infos)

    def render(
        self,
        save: bool = False,
        render_3d: bool = False,
        filename: str | None = None,
        output_path: Any | None = None,
        env_ids: int | Sequence[int] | None = None,
    ) -> np.ndarray:
        """Render environments, see :meth:`FluidEnv.render`.

        Parameters
        ----------
        save: bool
            Whether to save the rendered frame as a PNG file. Defaults to False.

        render_3d: bool
            Whether to enable 3d rendering. Defaults to False.

        filename: str | None
            The filename of the saved PNG files. If None, a default name is used.
            Defaults to None.

        output_path: Path | None
            The output path to save the rendered files. If None, saves to the current
            directory. Defaults to None.

        env_ids: int | Sequence[int] | None
            One index for a single frame, a sequence for a stack of frames, None
            (the default) for all environments.

        Returns
        -------
        np.ndarray
            ``[H, W, 3]`` for a single index, else ``[len(env_ids), H, W, 3]``.
        """
        single = isinstance(env_ids, (int, np.integer))
        if env_ids is None:
            ids = list(range(self.__n_envs))
        elif single:
            ids = [int(env_ids)]  # type: ignore[arg-type]
        else:
            ids = [int(i) for i in env_ids]  # type: ignore[union-attr]
        for i in ids:
            if not 0 <= i < self.__n_envs:
                raise ValueError(
                    f"env_ids {i} out of range for {self.__n_envs} environments."
                )

        # The workers render their requested environments in parallel
        k = self.__per_worker
        per_worker: dict[int, list[int]] = {}
        for i in ids:
            per_worker.setdefault(i // k, []).append(i % k)
        # Positional in the order of FluidEnv.render: the env ids differ per worker
        args = [
            (save, render_3d, filename, output_path, local)
            for local in per_worker.values()
        ]
        frames = dict(
            zip(
                per_worker,
                self.__call("render", args, ranks=list(per_worker)),
                strict=True,
            )
        )

        taken = dict.fromkeys(per_worker, 0)
        out = []
        for i in ids:
            w = i // k
            out.append(frames[w][taken[w]])
            taken[w] += 1
        return out[0] if single else np.stack(out)

    def seed(self, seed: int) -> None:
        """Update the random seeds and seed the random number generators.

        Worker ``r`` is seeded with ``seed + r``, so the workers' environments
        differ when randomized.

        Parameters
        ----------
        seed: int
            The seed to set.
        """
        self.__call("seed", [(seed + rank,) for rank in range(self.__n_workers)])
        self.__rng = torch.Generator().manual_seed(seed)

    def train(self) -> None:
        """Set the environment to training mode."""
        self.__call("train")

    def val(self) -> None:
        """Set the environment to validation mode."""
        self.__call("val")

    def test(self) -> None:
        """Set the environment to test mode."""
        self.__call("test")

    def sample_action(self) -> torch.Tensor:
        """Sample a random action for every environment uniformly.

        Returns
        -------
        torch.Tensor
            Random actions ``[n_envs, ...]``, on :attr:`cuda_device`.
        """
        space = self.action_space
        if not isinstance(space, spaces.Box):
            raise TypeError("Only implemented for Box.")
        if self.__rng is None:
            raise RuntimeError("Environment must be seeded before sampling actions.")

        agents = (self.n_agents,) if self.use_marl else ()
        shape = (self.__n_envs, *agents, *space.shape)
        low = torch.as_tensor(space.low)
        high = torch.as_tensor(space.high)
        r = torch.rand(shape, generator=self.__rng, dtype=low.dtype)
        return (low + (high - low) * r).to(self.cuda_device)

    def get_state(self) -> Any:
        """Not implemented for ParallelFluidEnv."""
        raise NotImplementedError("get_state is not implemented for ParallelFluidEnv.")

    def set_state(self, state: Any) -> None:
        """Not implemented for ParallelFluidEnv."""
        raise NotImplementedError("set_state is not implemented for ParallelFluidEnv.")

    def load_initial_domain(self, idx: int, mode: EnvMode | None = None) -> None:
        """Load the initial domain ``idx`` into all environments.

        Parameters
        ----------
        idx: int
            Index of the initial domain to load.

        mode: EnvMode | None
            Environment mode ('train', 'val', 'test'). If None, uses the current mode.
            Defaults to None.
        """
        self.__call("load_initial_domain", idx=idx, mode=mode)

    def get_uncontrolled_episode_metrics(self) -> list[pd.DataFrame | None]:
        """The uncontrolled episode metrics of every environment.

        Returns
        -------
        list[pd.DataFrame | None]
            One entry per environment.
        """
        metrics: list[pd.DataFrame | None] = []
        for worker_metrics in self.__call("get_uncontrolled_episode_metrics"):
            metrics += worker_metrics
        return metrics

    def detach(self) -> None:
        """Not implemented for ParallelFluidEnv."""
        raise NotImplementedError("detach is not implemented for ParallelFluidEnv.")

    def close(self) -> None:
        """Shut down all worker processes. Safe to call more than once."""
        self.__finalizer()
