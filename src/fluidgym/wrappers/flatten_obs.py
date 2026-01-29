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
    env: EnvLike
        The environment to wrap.
    """

    def __init__(
        self, env: FluidEnvLike, airfoil_3d_legacy_compatible: bool = False
    ) -> None:
        super().__init__(env)
        if not isinstance(self._env.observation_space, spaces.Dict):
            raise ValueError(
                "FlattenObservation wrapper only supports Dict observation spaces."
            )
        self.__observation_space = flatten_dict_space(self._env.observation_space)
        self.__airfoil_3d_legacy_compatible = airfoil_3d_legacy_compatible

    @property
    def observation_space(self) -> spaces.Box:
        """The observation space of the environment."""
        return self.__observation_space

    def __airfoil3d_legacy_compatibility(self, flat_obs: torch.Tensor) -> torch.Tensor:
        """
        Insert zero entries at fixed indices to expand the observation from
        shape [209, ...] to [216, ...].
        """
        idxs = [157, 165, 173, 181, 188, 196, 204]

        # Original and new sizes along dim 0
        old_n = flat_obs.size(0)
        new_n = old_n + len(idxs)

        # Allocate output (zeros by default)
        out = flat_obs.new_zeros((new_n, *flat_obs.shape[1:]))

        # Boolean mask of where original values should go
        mask = torch.ones(new_n, dtype=torch.bool, device=flat_obs.device)
        mask[idxs] = False

        # Fill non-insert positions with original data
        out[mask] = flat_obs

        return out

    def __flatten_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.__airfoil_3d_legacy_compatible:
            return torch.cat(
                [
                    self.__airfoil3d_legacy_compatibility(obs[k]).reshape(-1)
                    for k in obs.keys()
                    if k in DEFAULT_KEYS
                ],
                dim=0,
            )
        else:
            return torch.cat(
                [obs[k].reshape(-1) for k in obs.keys() if k in DEFAULT_KEYS], dim=0
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
