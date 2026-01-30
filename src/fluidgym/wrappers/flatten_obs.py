"""A wrapper that flattens the observation space."""

import torch
from gymnasium import spaces

from fluidgym.types import FluidEnvLike
from fluidgym.wrappers.fluid_wrapper import FluidWrapper
from fluidgym.wrappers.util import flatten_dict_space

DEFAULT_KEYS = ["temperature", "velocity"]


class FlattenObservation(FluidWrapper):
    """A wrapper that flattens the observation space.

    It flattens each observation tensor in the observation dictionary.

    Parameters
    ----------
    env: FluidEnvLike
        The environment to wrap.
    """

    def __init__(self, env: FluidEnvLike) -> None:
        super().__init__(env)
        if not isinstance(self._env.observation_space, spaces.Dict):
            raise ValueError(
                "FlattenObservation wrapper only supports Dict observation spaces."
            )

        self.__keys = [
            k for k in DEFAULT_KEYS if k in self._env.observation_space.spaces
        ]
        self.__observation_space = flatten_dict_space(
            space=self._env.observation_space,
            keys=self.__keys,
        )
        self.__flatten_start_dim = 1 if env.use_marl else 0

    @property
    def observation_space(self) -> spaces.Box:
        """The observation space of the environment."""
        return self.__observation_space

    def __flatten_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [obs[k].flatten(start_dim=self.__flatten_start_dim) for k in self.__keys],
            dim=self.__flatten_start_dim,
        )

    def reset(
        self,
        seed: int | None = None,
        randomize: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Resets the environment to an initial internal state, returning an initial
        observation and info.

        Parameters
        ----------
        seed: int | None
            The seed to use for random number generation. If None, the current seed is
            used.

        randomize: bool | None
            Whether to randomize the initial state. If None, the default behavior is
            used.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            A tuple containing the initial observation and an info dictionary.
        """
        obs, info = self._env.reset(seed=seed, randomize=randomize)
        return self.__flatten_obs(obs), info

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool, bool, dict[str, torch.Tensor]]:
        """Run one timestep of the environment's dynamics using the agent actions.

        When the end of an episode is reached (``terminated or truncated``), it is
        necessary to call :meth:`reset` to reset this environment's state for the next
        episode.

        Parameters
        ----------
        action: torch.Tensor
            The action to take.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, bool, bool, dict[str, torch.Tensor]]
            A tuple containing the flattened observation, reward, terminated flag,
            truncated flag, and info dictionary.
        """
        obs, reward, terminated, truncated, info = self._env.step(action)
        return self.__flatten_obs(obs), reward, terminated, truncated, info
