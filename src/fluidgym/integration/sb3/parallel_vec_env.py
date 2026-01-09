"""The StableBaselines3 VecEnv interface for ParallelFluidEnv environments."""

import logging
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import torch
from gymnasium.spaces import Box
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices

from fluidgym.envs.fluid_env import FluidEnv
from fluidgym.envs.multi_agent_fluid_env import MultiAgentFluidEnv
from fluidgym.envs.parallel_env import ParallelFluidEnv
from fluidgym.registry import make

logger = logging.getLogger("fluidgym.integration.sb3")


PLOT_MAX_ACTIONS = 64


class ParallelVecEnv(VecEnv):
    """The stable-baselines3 VecEnv interface for ParallelFluidEnv environments."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: ParallelFluidEnv, rl_mode: str, auto_reset: bool = True):
        self.__env = env
        self.__auto_reset = auto_reset

        self.observations = None

        if rl_mode not in ["sarl", "marl"]:
            raise ValueError(f"Invalid rl_mode: {rl_mode}. Must be 'sarl' or 'marl'.")

        self.__rl_mode = rl_mode
        self.__dummy_env = make(id=env.env_id)
        if self.__rl_mode == "marl" and not isinstance(
            self.__dummy_env, MultiAgentFluidEnv
        ):
            raise ValueError(
                f"Environment {env.env_id} is not a MultiAgentFluidEnv, "
                "but rl_mode is set to 'marl'."
            )

        if self.__rl_mode == "marl":
            assert isinstance(self.__dummy_env, MultiAgentFluidEnv)

            num_envs = env.num_envs * self.__dummy_env.n_agents
            obs_shape = self.__dummy_env.local_observation_space_shape
            action_shape = (self.__dummy_env.action_space_shape[1],)
        else:
            num_envs = env.num_envs
            obs_shape = self.__dummy_env.observation_space_shape
            action_shape = self.__dummy_env.action_space_shape

        observation_space = Box(
            low=self.__dummy_env._observation_range[0],
            high=self.__dummy_env._observation_range[1],
            shape=obs_shape,
            dtype=np.float32,
        )
        action_space = Box(
            low=self.__dummy_env._action_range[0],
            high=self.__dummy_env._action_range[1],
            shape=action_shape,
            dtype=np.float32,
        )

        super().__init__(
            num_envs=num_envs,
            observation_space=observation_space,
            action_space=action_space,
        )

    @property
    def is_marl(self) -> bool:
        """Whether the environment is multi-agent."""
        return self.__rl_mode == "marl"

    @property
    def n_agents(self) -> int:
        """Number of agents in the environment."""
        if not self.is_marl:
            raise ValueError("n_agents is only available for multi-agent environments.")

        assert isinstance(self.__dummy_env, MultiAgentFluidEnv)
        return self.__dummy_env.n_agents

    def __to_np(self, data: list[torch.Tensor] | torch.Tensor) -> np.ndarray:
        if isinstance(data, list):
            return np.stack([d.detach().cpu().numpy() for d in data], axis=0)

        return data.detach().cpu().numpy()

    def __to_np_dict(self, data: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        return {key: value.detach().cpu().numpy() for key, value in data.items()}

    def reset(self, randomize: bool | None = None) -> np.ndarray:
        """Reset the environment and return initial observations for all agents.

        Parameters
        ----------
        randomize: bool | None
            Whether to randomize the initial state. If None, the default behavior is
            used. Defaults to None.

        Returns
        -------
        np.ndarray
            The initial observations for all agents.
        """
        if self.is_marl:
            obs, _ = self.__env.reset_marl(randomize=randomize)
        else:
            obs, _ = self.__env.reset(randomize=randomize)
        return self.__to_np(obs)

    def step_async(self, actions: np.ndarray) -> None:
        """Tell all the environments to start taking a step with the given actions.
        Call step_wait() to get the results of the step. You should not call this if a
        step_async run is already pending.

        Parameters
        ----------
        actions: np.ndarray
            The actions to take for all agents.

        Note
        ----
        This method just stores the actions to be taken, the actual step is performed
        in step_wait().
        """
        self._actions = torch.as_tensor(actions, dtype=self.__dummy_env._dtype)

    def step_wait(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Wait for the step taken with step_async().

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]
            A tuple containing the observations, rewards, done flags, and info
            dictionaries for all agents.
        """
        if self.is_marl:
            result = self.__env.step_marl(self._actions)
        else:
            result = self.__env.step(self._actions)
        (obs, rewards, terms, truncs, infos) = result
        obs_np = self.__to_np(obs)
        rewards_np = self.__to_np(rewards)
        dones = np.full(len(obs), terms[0] or truncs[0], dtype=bool)
        infos_list = [self.__to_np_dict(info) for info in infos]

        if self.is_marl:
            assert isinstance(self.__dummy_env, MultiAgentFluidEnv)

            new_infos_list: list[dict[str, np.ndarray]] = []

            n_agents = self.__dummy_env.n_agents
            for i in range(self.num_envs):
                env_idx = i // n_agents

                # Copy the info dict
                info_dict = {
                    k: v.copy() if isinstance(v, np.ndarray) else v
                    for k, v in infos_list[env_idx].items()
                }
                new_infos_list.append(info_dict)
            infos_list = new_infos_list

        # Auto-reset
        if np.any(dones) and self.__auto_reset:
            for i in range(len(infos_list)):
                infos_list[i]["terminated_observation"] = obs_np[i]

            obs_np = self.reset()

        return obs_np, rewards_np, dones, infos_list

    def seed(self, seed: int) -> None:
        """Set the seed for all environments."""
        self.__env.seed(seed)

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        """Get an attribute of the environment.

        Parameters
        ----------
        attr_name: str
            The name of the attribute to get.

        indices: VecEnvIndices | None
            The indices of the environments to get the attribute from.

        Returns
        -------
        list[Any]
            A list of attribute values for each environment.
        """
        return [getattr(self.__env, attr_name)] * self.num_envs

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        """Set an attribute of the environment.

        Parameters
        ----------
        attr_name: str
            The name of the attribute to set.

        value: Any
            The value to set the attribute to.

        indices: VecEnvIndices | None
            The indices of the environments to set the attribute for.
        """
        setattr(self.__env, attr_name, value)

    def env_is_wrapped(
        self, wrapper_class: type[gymnasium.Wrapper], indices: VecEnvIndices = None
    ) -> list[bool]:
        """Check if the environment is wrapped with a specific wrapper class.
        Only required for compatibility with StableBaselines3.
        """
        return [False] * self.num_envs

    def render(  # type: ignore[override]
        self,
        mode: str = "human",
        save: bool = False,
        render_3d: bool = False,
        filename: str | None = None,
        output_path: Path | None = None,
    ) -> None:
        """Render the current state of the environment. Not implemented for
        ParallelVecEnv.

        Parameters
        ----------
        mode: str
            The mode to render with. Currently only 'human' is supported. Defaults to
            'human'.

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
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close the environment."""
        pass

    def env_method(
        self,
        method_name: str,
        *method_args: list[Any],
        indices: VecEnvIndices | None = None,
        **method_kwargs: dict[str, Any],
    ):
        """Call a method of the environment. Not implemented for ParallelVecEnv, only
        required for compatibility with StableBaselines3.

        Parameters
        ----------
        method_name: str
            The name of the method to call.

        *method_args
            The positional arguments to pass to the method.

        indices
            The indices of the environments to call the method on.

        **method_kwargs
            The keyword arguments to pass to the method.
        """
        raise NotImplementedError

    @property
    def id(self):
        """Unique identifier for the environment."""
        return self.__dummy_env.id

    @property
    def unwrapped(self) -> FluidEnv:  # type: ignore[override]
        """Return the unwrapped FluidGym environment."""
        return self.__dummy_env

    def train(self) -> None:
        """Set the environment to training mode."""
        self.__env.train()

    def val(self) -> None:
        """Set the environment to validation mode."""
        self.__env.val()

    def test(self) -> None:
        """Set the environment to test mode."""
        self.__env.test()

    @property
    def num_actions(self) -> int:
        """Return the number of agents (actions) in the environment."""
        return self.__dummy_env.action_space_shape[0]
