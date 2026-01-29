"""A wrapper that parallelizes the environment across multiple GPUs."""

from enum import Enum

import torch

from fluidgym.types import FluidEnvLike
from fluidgym.wrappers.fluid_wrapper import FluidWrapper


class Command(Enum):
    """Environment commands for inter-process communication."""

    STEP = "step"
    STEP_MARL = "step_marl"
    RESET = "reset"
    RESET_MARL = "reset_marl"
    SEED = "seed"
    CLOSE = "close"
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MultiGPU(FluidWrapper):
    """A wrapper that parallelizes the environment across multiple GPUs.

    Parameters
    ----------
    env: EnvT
        The environment to wrap.
    """

    def __init__(self, env: FluidEnvLike, GPU_ids: list[int]) -> None:
        super().__init__(env)
        self.__GPU_ids = GPU_ids
        self.__num_envs = len(GPU_ids)

    @property
    def n_agents(self) -> int:
        """The number of agents in the environment."""
        return self._env.n_agents * self.__num_envs

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
        # TODO implement
        obs, info = self._env.reset(seed=seed, randomize=randomize)
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
        # TODO implement
        obs, reward, terminated, truncated, info = self._env.step(action)
        return obs, reward, terminated, truncated, info
