"""Registry for FluidGym environments."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from fluidgym.envs.fluid_env import FluidEnv

T = TypeVar("T")


@dataclass
class EnvSpec:
    """Specification for a FluidGym environment."""

    entry_point: Callable
    kwargs: dict[str, Any]


class EnvRegistry:
    """Registry for FluidGym environments."""

    def __init__(self) -> None:
        self.env_specs: dict[str, EnvSpec] = {}

    def register(
        self, id: str, entry_point: Callable, defaults: dict[str, Any], **kwargs: Any
    ) -> None:
        """Register an environment with the given ID and default kwargs.

        Parameters
        ----------
        id: str
            The unique identifier for the environment.

        entry_point: Callable
            The callable that creates an instance of the environment.

        defaults: dict[str, Any]
            Default keyword arguments for the environment constructor.

        kwargs: Any
            Additional keyword arguments to override the defaults.

        Raises
        ------
        ValueError
            If an environment with the given ID is already registered.
        """
        if id in self.env_specs:
            raise ValueError(f"Environment {id} is already registered.")
        kwargs = {**defaults, **kwargs}
        self.env_specs[id] = EnvSpec(entry_point=entry_point, kwargs=kwargs)

    def make(self, id: str, **kwargs: Any) -> FluidEnv:
        """Create an environment instance with the given ID and optional kwargs.

        Parameters
        ----------
        id: str
            The unique identifier of the environment to create.

        kwargs: Any
            Additional keyword arguments to pass to the environment constructor.

        Returns
        -------
        FluidEnv
            An instance of the requested environment.
        """
        if id not in self.env_specs:
            raise ValueError(f"Environment {id} not found. Did you register it?")
        spec = self.env_specs[id]
        _kwargs = {**spec.kwargs, **kwargs}
        env: FluidEnv = spec.entry_point(**_kwargs)

        return env

    @property
    def ids(self) -> list[str]:
        """Get a list of all registered environment IDs.

        Returns
        -------
        list[str]
            A list of registered environment identifiers.
        """
        return list(self.env_specs.keys())


registry = EnvRegistry()


def register(
    id: str, entry_point: Callable[..., T], defaults: dict[str, Any], **kwargs: Any
) -> None:
    """Register an environment with the given ID and default kwargs."""
    registry.register(id, entry_point, defaults, **kwargs)


def make(id: str, **kwargs: Any) -> FluidEnv:
    """Create an environment instance with the given ID and optional kwargs.

    Parameters
    ----------
    id: str
        The unique identifier of the environment to create.

    kwargs: Any
        Additional keyword arguments to pass to the environment constructor.

    Returns
    -------
    FluidEnv
    """
    return registry.make(id, **kwargs)


def make_vec(
    id: str,
    n_envs: int,
    devices: list[int] | None = None,
    **kwargs: Any,
) -> Any:
    """Create a vectorized environment of ``n_envs`` environments.

    Without ``devices``, the environments are batched in one simulation on one GPU
    (``make(id, n_envs=n_envs)``). With ``devices``, they are split evenly over
    one worker process per entry (a device may occur several times), each
    batching its share, see :class:`fluidgym.envs.parallel_env.ParallelFluidEnv`.
    Both have the same vectorized API.

    Parameters
    ----------
    id: str
        The unique identifier of the environment to create.

    n_envs: int
        The total number of environments.

    devices: list[int] | None
        CUDA device of every worker process, or None for a single process.
        Defaults to None.

    kwargs: Any
        Additional keyword arguments to pass to the environment constructor.

    Returns
    -------
    FluidEnv | ParallelFluidEnv
        The vectorized environment.
    """
    if n_envs < 1:
        raise ValueError(f"n_envs must be positive, got {n_envs}.")
    if devices is None:
        return registry.make(id, n_envs=n_envs, **kwargs)
    if n_envs % len(devices) != 0:
        raise ValueError(
            f"n_envs={n_envs} is not divisible by the {len(devices)} devices."
        )
    from fluidgym.envs.parallel_env import ParallelFluidEnv

    return ParallelFluidEnv(
        env_id=id,
        cuda_ids=list(devices),
        n_envs_per_worker=n_envs // len(devices),
        **kwargs,
    )
