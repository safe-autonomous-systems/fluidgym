"""PettingZoo interface for FluidGym multi-agent environments."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium.spaces import Box
from pettingzoo.utils.env import ParallelEnv  # type: ignore

from fluidgym.types import FluidEnvLike


class PettingZooFluidEnv(ParallelEnv):
    """The PettingZoo interface for MultiAgentFluidEnv environments."""

    metadata = {"name": "fluidgym_pettingzoo_env"}

    action_spaces: dict[int, Box]
    observation_spaces: dict[int, Box]

    _observations: dict[int, np.ndarray] | None = None
    _rewards: dict[int, float] | None = None
    _dones: dict[int, bool] | None = None
    _infos: dict[int, dict[str, np.ndarray]] | None = None

    def __init__(self, env: FluidEnvLike):
        """Initialize the PettingZooFluidEnv.

        Parameters
        ----------
        env: MultiAgentFluidEnv
            The FluidGym multi-agent environment to wrap.
        """
        if not env.use_marl or env.n_agents <= 1:
            raise ValueError(
                "PettingZooFluidEnv requires a multi-agent FluidGym environment."
            )

        if not isinstance(env.observation_space, Box):
            raise ValueError("PettingZooFluidEnv only supports Box observation spaces.")

        super().__init__()

        self.__env = env
        self.possible_agents = list(range(env.n_agents))
        self.agents = self.possible_agents[:]
        self.action_spaces = dict.fromkeys(self.agents, env.action_space)
        self.observation_spaces = dict.fromkeys(self.agents, env.observation_space)

    def __to_np(self, data: torch.Tensor) -> np.ndarray:
        return data.detach().cpu().numpy()

    def __to_np_dict(self, data: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        return {key: value.detach().cpu().numpy() for key, value in data.items()}

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        randomize: bool | None = None,
    ) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]:
        """Resets the environment.

        Parameters
        ----------
        seed: int | None
            The seed to use for random number generation. If None, the current seed is
            used.

        options: dict | None
            Additional options for resetting the environment (not used here).

        randomize: bool | None
            Whether to randomize the initial state. If None, the default behavior is
            used.

        Returns
        -------
        tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]
            A tuple containing the initial observations for all agents and an info
            dictionary.
        """
        local_obs, _ = self.__env.reset(seed=seed, randomize=randomize)
        assert isinstance(local_obs, torch.Tensor)

        local_obs_np = self.__to_np(local_obs)

        self._observations = {
            agent: local_obs_np[agent] for agent in self.possible_agents
        }
        self._rewards = dict.fromkeys(self.agents, 0.0)
        self._dones = dict.fromkeys(self.agents, False)
        self._infos = {agent: {} for agent in self.agents}

        return self._observations, self._infos

    def step(
        self, actions: dict[int, np.ndarray]
    ) -> tuple[
        dict[int, np.ndarray],
        dict[int, float],
        dict[int, bool],
        dict[int, bool],
        dict[int, dict[str, np.ndarray]],
    ]:
        """Take a step in the environment using the given actions for all agents.

        Parameters
        ----------
        actions: dict[int, np.ndarray]
            The actions to take for each agent.

        Returns
        -------
        tuple[
            dict[int, np.ndarray],
            dict[int, float],
            dict[int, bool],
            dict[int, bool],
            dict[int, dict[str, np.ndarray]]
        ]
            A tuple containing the observations, rewards, terminated flags, truncated
            flags, and info dictionaries for all agents.
        """
        actions_tensor = torch.stack(
            [
                torch.as_tensor(
                    actions[agent],
                    device=self.__env.cuda_device,
                )
                for agent in self.agents
            ]
        )

        local_obs, agent_rewards, terminated, truncated, info = self.__env.step(
            actions_tensor
        )
        agent_rewards_np = self.__to_np(agent_rewards)
        assert isinstance(local_obs, torch.Tensor)
        local_obs_np = self.__to_np(local_obs)

        self._observations = {agent: local_obs_np[agent] for agent in self.agents}
        self._rewards = {agent: float(agent_rewards_np[agent]) for agent in self.agents}
        self._dones = dict.fromkeys(self.agents, terminated or truncated)
        self._infos = {agent: self.__to_np_dict(info) for agent in self.agents}

        return self._observations, self._rewards, self._dones, self._dones, self._infos

    def render(
        self,
        save: bool = False,
        render_3d: bool = False,
        filename: str | None = None,
        output_path: Path | None = None,
    ) -> np.ndarray:
        """Render the current state of the environment. For compatibility, this method
        returns the rendered frame as a numpy array in addition to the usual rendering
        behavior in FluidGym.

        Parameters
        ----------
        save: bool
            Whether to save the rendered frame as a PNG file. Defaults to False.

        render_3d: bool
            Whether to enable 3d rendering. Defaults to False.

        filename: str | None
            The filename to save the GIF file. If None, a default name is used.
            Defaults to None.

        output_path: Path | None
            The output path to save the rendered files. If None, saves to the current
            directory. Defaults to None.

        Returns
        -------
        np.ndarray
            The rendered frame as a numpy array.
        """
        return self.__env.render(
            save=save,
            render_3d=render_3d,
            filename=filename,
            output_path=output_path,
        )

    def close(self):
        """Closes the rendering window."""
        pass

    @property
    def unwrapped(self) -> FluidEnvLike:
        """Return the unwrapped FluidGym environment."""
        if hasattr(self.__env, "unwrapped"):
            return self.__env.unwrapped  # type: ignore[attr-defined]
        else:
            return self.__env

    def seed(self, seed: int) -> None:
        """Set the seed for the environment's random number generator."""
        self.__env.seed(seed)
