"""Gymnasium interface for FluidGym environments."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from gymnasium import Env, spaces
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from fluidgym.envs.fluid_env import FluidEnv
from fluidgym.types import FluidEnvLike


class GymFluidEnv(Env):
    """Base class for FluidGym Gymnasium environments."""

    metadata = {"render_modes": ["rbg_array"], "render_fps": 24}
    action_space: spaces.Box
    observation_space: spaces.Space

    def __init__(self, env: FluidEnvLike, render_mode: str | None = None):
        super().__init__()

        if env.use_marl:
            raise ValueError(
                "GymFluidEnv does not support multi-agent environments. "
                "Please use a single-agent environment."
            )

        if getattr(env, "vectorized", False):
            raise ValueError(
                "GymFluidEnv wraps a single environment; use GymVecFluidEnv for a "
                "vectorized one (n_envs given)."
            )

        if render_mode is not None and render_mode != "rgb_array":
            raise ValueError(
                f"Unsupported render mode: {render_mode}. "
                f"Only 'rgb_array' is supported."
            )
        self.render_mode = render_mode

        self.__env = env
        self.action_space = self.__env.action_space
        self.observation_space = self.__env.observation_space

    def __to_np(
        self, data: torch.Tensor | dict[str, torch.Tensor]
    ) -> np.ndarray | dict[str, np.ndarray]:
        def t_to_np(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().numpy()

        if isinstance(data, torch.Tensor):
            return t_to_np(data)
        elif isinstance(data, dict):
            return {k: t_to_np(v) for k, v in data.items()}
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def step(
        self, action: np.ndarray
    ) -> tuple[
        np.ndarray | dict[str, np.ndarray], float, bool, bool, dict[str, np.ndarray]
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        When the end of an episode is reached (``terminated or truncated``), it is
        necessary to call :meth:`reset` to reset this environment's state for the next
        episode.

        Parameters
        ----------
        action: np.ndarray
            The action to take.

        Returns
        -------
        tuple[
        np.ndarray | dict[str, np.ndarray], float, bool, bool, dict[str, np.ndarray]]
            A tuple containing the observation, reward, terminated flag, truncated flag,
            and info dictionary.
        """
        obs, reward, terminated, truncated, info = self.__env.step(
            torch.tensor(action, device=self.__env.cuda_device)
        )
        info_np = {k: np.array(self.__to_np(v)) for k, v in info.items()}

        return (
            self.__to_np(obs),
            float(reward),
            bool(terminated),
            bool(truncated),
            info_np,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        randomize: bool | None = None,
        domain_idx: int | None = None,
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict[str, np.ndarray]]:
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

        domain_idx: int | None
            Index of the initial domain to load. If None, the default behavior is
            used. Defaults to None.

        Returns
        -------
        tuple[np.ndarray | dict[str, np.ndarray], dict[str, np.ndarray]]
            A tuple containing the initial observation and an info dictionary.
        """
        obs, info = self.__env.reset(
            seed=seed, randomize=randomize, domain_idx=domain_idx
        )
        info_np = {k: np.array(self.__to_np(v)) for k, v in info.items()}

        return self.__to_np(obs), info_np

    def render(
        self,
        save: bool = False,
        render_3d: bool = False,
        filename: str | None = None,
        output_path: Path | None = None,
    ) -> np.ndarray | None:
        """Render the current state of the environment.

        Parameters
        ----------
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

        Returns
        -------
        np.ndarray | None
            The rendered frame as a numpy array if the render mode is "rgb_array",
            otherwise None.
        """
        frame = self.__env.render(
            save=save,
            render_3d=render_3d,
            filename=filename,
            output_path=output_path,
        )
        return frame if self.render_mode == "rgb_array" else None

    def close(self):
        """After the user has finished using the environment, close contains the code
        necessary to "clean up" the environment.
        """
        self.__env.close()

    @property
    def unwrapped(self) -> FluidEnv:  # type: ignore[override]
        """Returns the base non-wrapped environment."""
        if hasattr(self.__env, "unwrapped"):
            return self.__env.unwrapped  # type: ignore
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

    def seed(self, seed: int) -> None:
        """Set the seed for the environment.

        Parameters
        ----------
        seed: int
            The seed to use for random number generation.
        """
        self.__env.seed(seed)

    @property
    def num_actions(self) -> int:
        """Return the number of actions in the environment."""
        return int(np.prod(self.action_space.shape))


class GymVecFluidEnv(VectorEnv):
    """Gymnasium ``VectorEnv`` interface for vectorized FluidGym environments.

    Wraps a FluidGym environment made with ``n_envs``; the environments are
    simulated together (on one GPU, or across GPUs with ``ParallelFluidEnv``).
    They share the episode: all are truncated at the same step and never
    terminate early, so with ``auto_reset`` all of them are reset in that same
    step (gymnasium's ``SAME_STEP`` autoreset mode), and the last observations
    are in ``info["final_obs"]`` (``info["final_info"]`` for the infos).

    Parameters
    ----------
    env: FluidEnvLike
        A vectorized single-agent FluidGym environment.

    render_mode: str | None
        None or "rgb_array". Defaults to None.

    auto_reset: bool
        Whether to reset all environments when their episode ends. Defaults to
        True.
    """

    def __init__(
        self,
        env: FluidEnvLike,
        render_mode: str | None = None,
        auto_reset: bool = True,
    ):
        if not getattr(env, "vectorized", False):
            raise ValueError(
                "GymVecFluidEnv requires a vectorized environment (n_envs given); "
                "use GymFluidEnv for a single one."
            )
        if env.use_marl:
            raise ValueError(
                "GymVecFluidEnv does not support multi-agent environments; use "
                "TorchRLFluidEnv or the SB3 VecFluidEnv."
            )
        if render_mode is not None and render_mode != "rgb_array":
            raise ValueError(
                f"Unsupported render mode: {render_mode}. "
                f"Only 'rgb_array' is supported."
            )

        self.__env = env
        self.__auto_reset = auto_reset
        self.render_mode = render_mode
        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": 24,
            "autoreset_mode": (
                AutoresetMode.SAME_STEP if auto_reset else AutoresetMode.DISABLED
            ),
        }

        self.num_envs = env.n_envs
        self.single_action_space = env.action_space
        self.single_observation_space = env.observation_space
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )

    @staticmethod
    def __to_np(data: Any) -> Any:
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        if isinstance(data, dict):
            return {k: GymVecFluidEnv.__to_np(v) for k, v in data.items()}
        raise TypeError(f"Unsupported data type: {type(data)}")

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset all environments.

        Parameters
        ----------
        seed: int | None
            The seed to use for random number generation. If None, the current seed is
            used. One seed seeds all environments (they share the random generators).

        options: dict[str, Any] | None
            ``randomize`` (bool) and ``domain_idx`` (one index, or one per
            environment), see ``FluidEnv.reset``.

        Returns
        -------
        tuple[Any, dict[str, Any]]
            The initial observations ``[num_envs, ...]`` and an info dictionary.
        """
        if isinstance(seed, list):
            if len(set(seed)) > 1:
                raise ValueError(
                    "The environments share their random generators: pass one seed."
                )
            seed = seed[0] if seed else None
        options = options or {}
        obs, info = self.__env.reset(
            seed=seed,
            randomize=options.get("randomize"),
            domain_idx=options.get("domain_idx"),
        )
        return self.__to_np(obs), self.__to_np(info)

    def step(self, actions: np.ndarray) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """Step all environments.

        Parameters
        ----------
        actions: np.ndarray
            The actions ``[num_envs, ...]``.

        Returns
        -------
        tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]
            Observations, rewards, terminated and truncated flags (``[num_envs]``
            each) and the infos (arrays ``[num_envs, ...]``).
        """
        obs, reward, terminated, truncated, info = self.__env.step(
            torch.as_tensor(actions, device=self.__env.cuda_device)
        )
        obs_np = self.__to_np(obs)
        info_np = self.__to_np(info)
        terminated_np = self.__to_np(terminated)
        truncated_np = self.__to_np(truncated)
        done = terminated_np | truncated_np

        if self.__auto_reset and bool(done.any()):
            final_obs = np.empty(self.num_envs, dtype=object)
            final_info = np.empty(self.num_envs, dtype=object)
            for i in range(self.num_envs):
                final_obs[i] = (
                    {k: v[i] for k, v in obs_np.items()}
                    if isinstance(obs_np, dict)
                    else obs_np[i]
                )
                final_info[i] = {k: v[i] for k, v in info_np.items()}
            obs_np, reset_info = self.reset()
            info_np = {
                **info_np,
                **reset_info,
                "final_obs": final_obs,
                "_final_obs": done,
                "final_info": final_info,
                "_final_info": done,
            }

        return (
            obs_np,
            self.__to_np(reward).astype(np.float64),
            terminated_np,
            truncated_np,
            info_np,
        )

    def render(self) -> tuple[Any, ...] | None:
        """The frames of all environments, as gymnasium vector envs return them.

        Returns
        -------
        tuple[np.ndarray, ...] | None
            One ``[H, W, 3]`` frame per environment if the render mode is
            "rgb_array", otherwise None.
        """
        if self.render_mode != "rgb_array":
            return None
        return tuple(self.__env.render())

    @property
    def unwrapped(self) -> FluidEnv:  # type: ignore[override]
        """Returns the base non-wrapped environment."""
        if hasattr(self.__env, "unwrapped"):
            return self.__env.unwrapped  # type: ignore
        return cast(FluidEnv, self.__env)

    def close_extras(self, **kwargs: Any) -> None:
        """Close the wrapped environment, called by :meth:`close`."""
        self.__env.close()

    def train(self) -> None:
        """Set the environment to training mode."""
        self.__env.train()

    def val(self) -> None:
        """Set the environment to validation mode."""
        self.__env.val()

    def test(self) -> None:
        """Set the environment to test mode."""
        self.__env.test()

    def seed(self, seed: int) -> None:
        """Set the seed for the environment.

        Parameters
        ----------
        seed: int
            The seed to use for random number generation.
        """
        self.__env.seed(seed)
