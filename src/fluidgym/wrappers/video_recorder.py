"""A wrapper that records and saves rendered gifs."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fluidgym.logging import get_logger
from fluidgym.types import FluidEnvLike
from fluidgym.util.video import save_gif
from fluidgym.wrappers.fluid_wrapper import FluidWrapper

logger = get_logger(__name__)


class VideoRecorder(FluidWrapper):
    """Render and save every episode of the environment as gif.

    The frames returned by :meth:`render` (the environment's default render slice)
    are kept by the wrapper and written when the next episode starts or the
    environment is closed. For a vectorized environment, including a
    :class:`ParallelFluidEnv`, each recorded environment gets its own gif.

    Parameters
    ----------
    env: FluidEnvLike
        The environment to wrap.

    filename: str
        The base name of the gifs. Episode ``k`` is saved as
        ``<filename>_ep<k>.gif``.

    output_path: Path | None
        The directory to save the gifs to. If None, the current directory is used.
        Defaults to None.

    render_kwargs: dict[str, Any] | None
        Keyword arguments of every :meth:`render` call. Defaults to None.

    env_ids: int | Sequence[int] | None
        The environments of a vectorized environment to record, each into its own
        gif (``<filename>_env<i>_ep<k>.gif``). None (the default) records all of
        them. Must be None for a single environment.

    fps: int
        The frames per second of the gifs. Defaults to 24.
    """

    def __init__(
        self,
        env: FluidEnvLike,
        filename: str,
        output_path: Path | None = None,
        render_kwargs: dict[str, Any] | None = None,
        env_ids: int | Sequence[int] | None = None,
        fps: int = 24,
    ) -> None:
        super().__init__(env)
        if env_ids is not None and not env.vectorized:
            raise ValueError("env_ids is only supported for a vectorized environment.")

        self.__filename = filename
        self.__output_path = Path(".") if output_path is None else output_path
        self.__fps = fps
        self.__render_kwargs = dict(render_kwargs or {})
        self.__render_kwargs["env_ids"] = env_ids

        # One frame list per recorded environment, None for a single environment
        self.__ids: list[int | None]
        if not env.vectorized:
            self.__ids = [None]
        elif env_ids is None:
            self.__ids = list(range(env.n_envs))
        elif isinstance(env_ids, (int, np.integer)):
            self.__ids = [int(env_ids)]
        else:
            self.__ids = [int(i) for i in env_ids]
        self.__frames: list[list[np.ndarray]] = [[] for _ in self.__ids]

        self.__ep_steps = 0
        self.__ep_idx = 0

    def __render(self) -> None:
        frames = self._env.render(**self.__render_kwargs)
        if len(self.__ids) == 1 and frames.ndim == 3:
            frames = frames[None]
        for buffer, frame in zip(self.__frames, frames, strict=True):
            buffer.append(frame)

    def __save_episode(self) -> None:
        """Save the recorded frames of the current episode, then clear them."""
        if self.__ep_steps > 0:
            for env_idx, frames in zip(self.__ids, self.__frames, strict=True):
                suffix = "" if env_idx is None else f"_env{env_idx}"
                name = f"{self.__filename}{suffix}_ep{self.__ep_idx}.gif"
                path = save_gif(frames, self.__output_path / name, fps=self.__fps)
                logger.info(f"GIF saved to {path}")

        self.__frames = [[] for _ in self.__ids]
        self.__ep_steps = 0

    def step(
        self, action: torch.Tensor
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        bool | torch.Tensor,
        bool | torch.Tensor,
        dict[str, torch.Tensor],
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
        step_tuple = self._env.step(action)
        self.__ep_steps += 1

        self.__render()

        return step_tuple

    def reset(
        self,
        seed: int | None = None,
        randomize: bool | None = None,
        domain_idx: int | Sequence[int] | None = None,
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

        domain_idx: int | Sequence[int] | None
            Index of the initial domain to load, one for all environments or one per
            environment of a vectorized environment. If None, the default behavior
            is used. Defaults to None.

        Returns
        -------
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
            A tuple containing the initial observation and an info dictionary.
        """
        self.__save_episode()

        obs_tuple = self._env.reset(
            seed=seed, randomize=randomize, domain_idx=domain_idx
        )
        self.__ep_idx += 1

        self.__render()

        return obs_tuple

    def close(self) -> None:
        """Save the recorded frames of the current episode and close the
        environment.
        """
        self.__save_episode()
        self._env.close()
