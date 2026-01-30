"""Gymnasium interface for FluidGym environments."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from gymnasium import Env, spaces

from fluidgym.envs.fluid_env import FluidEnv
from fluidgym.types import FluidEnvLike


class GymFluidEnv(Env):
    """Base class for FluidGym Gymnasium environments."""

    metadata = {"render_modes": ["rbg_array"], "render_fps": 24}
    action_space: spaces.Box
    observation_space: spaces.Space

    def __init__(self, env: FluidEnvLike):
        super().__init__()

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

        Returns
        -------
        tuple[np.ndarray | dict[str, np.ndarray], dict[str, np.ndarray]]
            A tuple containing the initial observation and an info dictionary.
        """
        obs, info = self.__env.reset(seed=seed, randomize=randomize)
        info_np = {k: np.array(self.__to_np(v)) for k, v in info.items()}

        return self.__to_np(obs), info_np

    def render(
        self,
        save: bool = False,
        render_3d: bool = False,
        filename: str | None = None,
        output_path: Path | None = None,
    ) -> None:
        """Render the current state of the environment.

        Parameters
        ----------
        save: bool
            Whether to save the rendered frame as a PNG file. Defaults to False.

        render_3d: bool
            Whether to enable 3d rendering. Defaults to False.

        filename: str | None
            The filename to save the GIF file. If None, a default name is used. Defaults
            to None.

        output_path: Path | None
            The output path to save the rendered files. If None, saves to the current
            directory. Defaults to None.
        """
        self.__env.render(
            save=save,
            render_3d=render_3d,
            filename=filename,
            output_path=output_path,
        )

    def save_gif(
        self, filename: str = "fluidgym.gif", output_path: Path | None = None
    ) -> None:
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

    def close(self):
        """After the user has finished using the environment, close contains the code
        necessary to "clean up" the environment.
        """
        pass

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
