"""Abstract base class for multi-agent FluidGym environments."""

from abc import ABC, abstractmethod

import numpy as np
import torch
from gymnasium import spaces

from fluidgym.envs.fluid_env import FluidEnv


class MultiAgentFluidEnv(FluidEnv, ABC):
    """Abstract base class for multi-agent FluidGym environments."""

    _local_reward_weight: float | None

    @property
    @abstractmethod
    def local_observation_space_shape(self) -> tuple[int, ...]:
        """The shape of the local observation space for each agent."""
        raise NotImplementedError

    @property
    @abstractmethod
    def n_agents(self) -> int:
        """The number of agents in the environment."""
        raise NotImplementedError

    def reset_marl(
        self,
        seed: int | None = None,
        randomize: bool | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Reset the environment to the initial state for multiple agents.

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
        tuple[torch.Tensor, dict]
            A tuple containing the initial observations for all agents and an info
            dictionary.
        """
        raise NotImplementedError

    def step_marl(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool, bool, dict[str, torch.Tensor]]:
        """Take a step in the environment using the given actions for multiple agents.

        Parameters
        ----------
        actions: torch.Tensor
            The actions to take for each agent.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, bool, bool, dict[str, torch.Tensor]]
            A tuple containing the observations, rewards, terminated flag, truncated
            flag, and info dictionary.
        """
        raise NotImplementedError

    @property
    def local_action_space(self) -> spaces.Box:
        """The action space for each agent."""
        low, high = self._action_range
        return spaces.Box(
            low=low,
            high=high,
            shape=self.action_space.shape[1:],
            dtype=np.float32,
        )

    @property
    def local_observation_space(self) -> spaces.Box:
        """The observation space for each agent."""
        low, high = self._observation_range
        return spaces.Box(
            low=low,
            high=high,
            shape=self.local_observation_space_shape,
            dtype=np.float32,
        )
