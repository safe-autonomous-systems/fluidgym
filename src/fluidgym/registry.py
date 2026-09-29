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
