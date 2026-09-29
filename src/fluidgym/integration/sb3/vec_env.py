"""StableBaselines3 VecEnv interface for vectorized and multi-agent fluid envs."""

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
    """The stable-baselines3 VecEnv interface for vectorized and MARL fluid envs.

    Every agent of every environment is one SB3 environment: ``num_envs`` is
    ``n_envs * n_agents``, ordered environment-major (all agents of environment 0
    first). A vectorized single-agent environment has one SB3 environment per
    environment, a single MARL environment one per agent (as before).

    Parameters
    ----------
    env: FluidEnvLike
        A vectorized (``n_envs`` given) or multi-agent environment.

    auto_reset: bool
        Whether to reset all environments when their shared episode ends.
        Defaults to True.
    """

    metadata = {"render_modes": ["rbg_array"]}

    def __init__(self, env: FluidEnvLike, auto_reset: bool = True):
        self.__env = env
        self.__auto_reset = auto_reset

        self.__vectorized = bool(getattr(env, "vectorized", False))
        self.__n_envs = int(getattr(env, "n_envs", 1))
        self.__n_agents = env.n_agents if env.use_marl else 1
        # the layout the env acts in: [E?, A?, *action_shape]
        self.__lead_shape = ((self.__n_envs,) if self.__vectorized else ()) + (
            (self.__n_agents,) if env.use_marl else ()
        )

        if not self.__vectorized and (not env.use_marl or env.n_agents <= 1):
            raise ValueError(
                "VecFluidEnv can only be used with vectorized fluid environments "
                "(n_envs given) or MARL fluid environments with multiple agents. "
                "Wrap a single environment with GymFluidEnv instead."
            )

        self.observations = None
        super().__init__(
            num_envs=self.__n_envs * self.__n_agents,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )

    def __to_np(self, data: torch.Tensor) -> np.ndarray:
        # [E?, A?, ...] -> [E * A, ...]
        lead = len(self.__lead_shape)
        return data.detach().reshape(-1, *data.shape[lead:]).cpu().numpy()

    def __to_np_dict(self, data: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        return {key: self.__to_np(value) for key, value in data.items()}

    def __per_sb3_env(self, flags: Any) -> np.ndarray:
        """Per-environment (or shared) flags, one per SB3 environment."""
        flags_np = np.asarray(
            flags.cpu().numpy() if torch.is_tensor(flags) else flags, dtype=bool
        ).reshape(-1)
        return np.broadcast_to(
            flags_np.reshape(-1, 1), (self.__n_envs, self.__n_agents)
        ).reshape(-1)

    def __infos(self, info: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
        """The info of every SB3 environment: that of its environment."""
        infos = []
        for env_idx in range(self.__n_envs):
            env_info = {
                key: (value[env_idx] if self.__vectorized else value)
                .detach()
                .cpu()
                .numpy()
                for key, value in info.items()
            }
            # one dict per agent: auto-reset writes per-agent entries into it
            infos += [dict(env_info) for _ in range(self.__n_agents)]
        return infos

    def reset(
        self,
        seed: int | None = None,
        randomize: bool | None = None,
        domain_idx: int | list[int] | None = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Reset the environment and return initial observations for all agents.

        Parameters
        ----------
        The seed to use for random number generation. If None, the current seed is
        used.

        randomize: bool | None
            Whether to randomize the initial state. If None, the default behavior is
            used. Defaults to None.

        domain_idx: int | list[int] | None
            Index of the initial domain to load, one for all environments or one per
            environment. If None, the default behavior is used. Defaults to None.

        Returns
        -------
        np.ndarray | dict[str, np.ndarray]:
            The initial observations for all agents, ``[num_envs, ...]``.
        """
        local_obs, _ = self.__env.reset(
            seed=seed, randomize=randomize, domain_idx=domain_idx
        )
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
            The actions to take for all agents, ``[num_envs, ...]``.

        Note
        ----
        This method just stores the actions to be taken, the actual step is performed
        in step_wait().
        """
        self._actions = torch.as_tensor(
            actions,
            device=self.__env.cuda_device,
        ).reshape(*self.__lead_shape, *(self.action_space.shape or ()))

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
        rewards = agent_rewards.detach().reshape(-1).cpu().numpy()

        dones = self.__per_sb3_env(term) | self.__per_sb3_env(trunc)
        infos = self.__infos(info)

        # Auto-reset: all environments share the episode, so they are done together
        if bool(dones.all()) and self.__auto_reset:
            for i in range(self.num_envs):
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
        return [False] * self.num_envs

    def render(  # type: ignore[override]
        self,
        mode: str = "rbg_array",
        save: bool = False,
        render_3d: bool = False,
        filename: str | None = None,
        output_path: Path | None = None,
        env_ids: int | list[int] | None = None,
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
            The filename of the saved PNG files. If None, a default name is used.
            Defaults to None.

        output_path: Path | None
            The output path to save the rendered files. If None, saves to the current
            directory. Defaults to None.

        env_ids: int | list[int] | None
            The environments of a vectorized environment to render (all if None).
            Defaults to None.

        Returns
        -------
        np.ndarray
            The rendered frame ``[H, W, 3]``, or ``[N, H, W, 3]`` for several
            environments of a vectorized environment.
        """
        kwargs: dict[str, Any] = {} if env_ids is None else {"env_ids": env_ids}
        return self.__env.render(
            save=save,
            render_3d=render_3d,
            filename=filename,
            output_path=output_path,
            **kwargs,
        )

    def close(self) -> None:
        """Close the environment."""
        self.__env.close()

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
        """Return the number of agents (actions) of one environment."""
        return self.__env.n_agents
