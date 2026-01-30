"""A wrapper that extracts specific observations from the observation dictionary."""

import torch
from gymnasium import spaces

from fluidgym.types import FluidEnvLike
from fluidgym.wrappers.fluid_wrapper import FluidWrapper


class ObsExtraction(FluidWrapper):
    """A wrapper that extracts specific observations from the observation dictionary.

    It extracts only the observations specified in the `keys` list.

    Parameters
    ----------
    env: FluidEnvLike
        The environment to wrap.

    keys: list[str] | None
        The list of keys to extract from the observation dictionary.
    """

    def __init__(self, env: FluidEnvLike, keys: list[str]) -> None:
        super().__init__(env)
        if len(keys) == 0:
            raise ValueError("Keys list must be non-empty or None.")

        if not isinstance(self._env.observation_space, spaces.Dict):
            raise ValueError(
                "ObsExtraction wrapper only supports Dict observation spaces."
            )

        for k in keys:
            if k not in self._env.observation_space.spaces:
                raise ValueError(f"Key '{k}' not found in observation space.")

        self.__keys = keys
        self.__observation_space = spaces.Dict(
            {k: self._env.observation_space.spaces[k] for k in keys}
        )

    @property
    def observation_space(self) -> spaces.Dict:
        """The observation space of the environment."""
        return self.__observation_space

    def __filter_obs(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.__keys is not None:
            obs = {k: obs[k] for k in self.__keys}
        return obs

    def reset(
        self,
        seed: int | None = None,
        randomize: bool | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
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
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
            A tuple containing the initial observation and an info dictionary.
        """
        obs, info = self._env.reset(seed=seed, randomize=randomize)
        obs = self.__filter_obs(obs)

        return obs, info

    def step(
        self, action: torch.Tensor
    ) -> tuple[
        dict[str, torch.Tensor], torch.Tensor, bool, bool, dict[str, torch.Tensor]
    ]:
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
        tuple[
        dict[str, torch.Tensor], torch.Tensor, bool, bool, dict[str, torch.Tensor]]
            A tuple containing the observation, reward, terminated flag, truncated flag,
            and info dictionary.
        """
        obs, reward, terminated, truncated, info = self._env.step(action)
        obs = self.__filter_obs(obs)

        return obs, reward, terminated, truncated, info
