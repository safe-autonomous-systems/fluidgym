"""StableBaselines3 VecEnv interface for MultiAgentFluidEnv environments."""

from pathlib import Path
from typing import Any, cast

import gymnasium
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv as SB3VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices

from fluidgym.envs import FluidEnv
from fluidgym.types import FluidEnvLike


class VecFluidEnv(SB3VecEnv):
    """The stable-baselines3 VecEnv interface for MARL fluid environments."""

    metadata = {"render_modes": ["rbg_array"]}

    def __init__(self, env: FluidEnvLike, auto_reset: bool = True):
        self.__env = env
        self.__agents = list(range(env.n_agents))
        self.__auto_reset = auto_reset

        if not env.use_marl or env.n_agents <= 1:
            raise ValueError(
                "MultiAgentVecEnv can only be used with MARL fluid "
                "environments with multiple agents."
            )

        self.observations = None
        super().__init__(
            num_envs=len(self.__agents),
            observation_space=env.observation_space,
            action_space=env.action_space,
        )

    def __to_np(self, data: torch.Tensor) -> np.ndarray:
        return data.detach().cpu().numpy()

    def __to_np_dict(self, data: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        return {key: value.detach().cpu().numpy() for key, value in data.items()}

    def reset(
        self, seed: int | None = None, randomize: bool | None = None
    ) -> np.ndarray | dict[str, np.ndarray]:
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
        np.ndarray | dict[str, np.ndarray]:
            The initial observations for all agents.
        """
        local_obs, _ = self.__env.reset(seed=seed, randomize=randomize)
        if isinstance(local_obs, dict):
            return self.__to_np_dict(local_obs)
        else:
            return self.__to_np(local_obs)

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
            device=self.__env.cuda_device,
        )
        if self._actions.ndim > 2:
            self._actions = self._actions.unsqueeze(-1)

    def step_wait(
        self,
    ) -> tuple[
        np.ndarray | dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]
    ]:
        """Wait for the step taken with step_async().

        Returns
        -------
        tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]
            A tuple containing the observations, rewards, done flags, and info
            dictionaries for all agents.
        """
        local_obs, agent_rewards, term, trunc, info = self.__env.step(self._actions)

        local_obs_np: np.ndarray | dict[str, np.ndarray]
        if isinstance(local_obs, dict):
            local_obs_np = self.__to_np_dict(local_obs)
        else:
            local_obs_np = self.__to_np(local_obs)
        rewards = self.__to_np(agent_rewards)

        done = term or trunc
        dones = np.full(len(self.__agents), done, dtype=bool)

        info_np: dict[str, Any] = self.__to_np_dict(info)
        infos = [info_np for _ in self.__agents]

        # Auto-reset
        if done and self.__auto_reset:
            for i in range(len(self.__agents)):
                if isinstance(local_obs_np, dict):
                    infos[i]["terminated_observation"] = {
                        key: local_obs_np[key][i] for key in local_obs_np.keys()
                    }
                else:
                    infos[i]["terminated_observation"] = local_obs_np[i]
            local_obs_np = self.reset()

        return local_obs_np, rewards, dones, infos

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
        mode: str = "rbg_array",
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
            The mode to render with. Currently only 'rgb_array' is supported. Defaults
            to 'rgb_array'.

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
    def unwrapped(self) -> FluidEnv:  # type: ignore[override]
        """Return the unwrapped FluidGym environment."""
        if hasattr(self.__env, "unwrapped"):
            return self.__env.unwrapped  # type: ignore[return-value]
        else:
            return cast(FluidEnv, self.__env)

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

    @property
    def num_actions(self) -> int:
        """Return the number of agents (actions) in the environment."""
        return self.__env.n_agents
