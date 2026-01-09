"""StableBaselines3 VecEnv interface for MultiAgentFluidEnv environments."""

from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import torch
from gymnasium.spaces import Box
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices

from fluidgym.envs.fluid_env import EnvMode
from fluidgym.envs.multi_agent_fluid_env import MultiAgentFluidEnv


class MultiAgentVecEnv(VecEnv):
    """The stable-baselines3 VecEnv interface for MultiAgentFluidEnv environments."""

    metadata = {"render_modes": ["rbg_array"]}

    def __init__(self, env: MultiAgentFluidEnv, auto_reset: bool = True):
        self.__env = env
        self.__agents = list(range(env.n_agents))
        self.__auto_reset = auto_reset

        if len(self.__agents) != self.__env.action_space_shape[0]:
            raise ValueError(
                f"Number of agents ({len(self.__agents)}) does not match the action"
                f"space shape {self.__env.action_space_shape}."
            )

        self.observations = None
        super().__init__(
            num_envs=len(self.__agents),
            observation_space=self.get_observation_space(),
            action_space=self.get_action_space(),
        )

    def __to_np(self, data: torch.Tensor) -> np.ndarray:
        return data.detach().cpu().numpy()

    def __to_np_dict(self, data: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        return {key: value.detach().cpu().numpy() for key, value in data.items()}

    def get_observation_space(self) -> Box:
        """Get the observation space for a single agent."""
        return Box(
            low=self.__env._observation_range[0],
            high=self.__env._observation_range[1],
            shape=self.__env.local_observation_space_shape,
            dtype=np.float32,
        )

    def get_action_space(self) -> Box:
        """Get the action space for a single agent."""
        return Box(
            low=self.__env._action_range[0],
            high=self.__env._action_range[1],
            shape=(self.__env.action_space_shape[1],),
            dtype=np.float32,
        )

    def reset(
        self, seed: int | None = None, randomize: bool | None = None
    ) -> np.ndarray:
        """Reset the environment and return initial observations for all agents.

        Parameters
        ----------
        The seed to use for random number generation. If None, the current seed is
        used.

        randomize: bool | None
            Whether to randomize the initial state. If None, the default behavior is
            used. Defaults to None.

        Returns
        -------
        np.ndarray
            The initial observations for all agents.
        """
        local_obs, _ = self.__env.reset_marl(randomize=randomize)
        local_obs_arr = self.__to_np(local_obs)
        return local_obs_arr

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
        self._actions = torch.as_tensor(
            actions,
            dtype=self.__env._dtype,
            device=self.__env._cuda_device,
        ).squeeze()

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
        local_obs, agent_rewards, term, trunc, info = self.__env.step_marl(
            self._actions
        )

        obs = self.__to_np(local_obs)
        rewards = self.__to_np(agent_rewards)

        done = term or trunc
        dones = np.full(len(self.__agents), done, dtype=bool)

        info = self.__to_np_dict(info)
        infos = [info for _ in self.__agents]

        # Auto-reset
        if done and self.__auto_reset:
            for i in range(len(self.__agents)):
                infos[i]["terminated_observation"] = obs[i]
            obs = self.reset()

        return obs, rewards, dones, infos

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
        return [getattr(self.__env, attr_name)] * len(self.__agents)

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
        """Whether the environment is wrapped with a specific wrapper class. This is
        only required for compatibility with StableBaselines3.

        Parameters
        ----------
        wrapper_class: type[gymnasium.Wrapper]
            The wrapper class to check for.

        indices: VecEnvIndices | None
            The indices of the environments to check. Not used.

        Returns
        -------
        list[bool]
            A list of booleans indicating whether each environment is wrapped with the
            specified wrapper class. This always returns False.
        """
        return [False] * len(self.__agents)

    def render(  # type: ignore[override]
        self,
        mode: str = "human",
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

        Returns
        -------
        np.ndarray
            The rendered frame as a numpy array.
        """
        self.__env.render(
            save=save,
            render_3d=render_3d,
            filename=filename,
            output_path=output_path,
        )
        render_data = self.__env._get_render_data(
            render_3d=render_3d, output_path=output_path
        )
        return render_data[list(render_data.keys())[0]]

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
        return self.__env.id

    @property
    def unwrapped(self) -> MultiAgentFluidEnv:  # type: ignore[override]
        """Return the unwrapped FluidGym environment."""
        return self.__env

    @property
    def mode(self) -> EnvMode:
        """The current mode of the environment ('train', 'val', or 'test')."""
        return self.__env.mode

    @mode.setter
    def mode(self, mode: EnvMode) -> None:
        self.__env.mode = mode

    def train(self) -> None:
        """Set the environment to training mode."""
        self.__env.train()

    def val(self) -> None:
        """Set the environment to validation mode."""
        self.__env.val()

    def test(self) -> None:
        """Set the environment to test mode."""
        self.__env.test()

    def save_gif(self, filename: str, output_path: Path | None = None) -> None:
        """Save the rendered frames as a GIF file.

        Parameters
        ----------
        filename: str
            The name of the file to save the GIF to.

        output_path: Path | None
            The output path to save the GIF file. If None, saves to the current
            directory. Defaults to None.
        """
        self.__env.save_gif(filename=filename, output_path=output_path)

    def seed(self, seed: int) -> None:  # type: ignore[override]
        """Update the random seeds and seed the random number generators.

        Parameters
        ----------
        seed: int
            The seed to set for the environment's random number generator.
        """
        self.__env.seed(seed)

    def init(self) -> None:
        """Initialize the environment."""
        self.__env.init()

    @property
    def num_actions(self) -> int:
        """Return the number of agents (actions) in the environment."""
        return self.__env.n_agents
