"""A wrapper that records and saves rendered gifs."""

import torch
from typing import Any
from pathlib import Path

from fluidgym.types import FluidEnvLike
from fluidgym.wrappers.fluid_wrapper import FluidWrapper


class VideoRecorder(FluidWrapper):
    """Render and save every episode of the environment as gif.

    Parameters
    ----------
    env: FluidEnvLike
        The environment to wrap.

    output_path: Path
    """

    def __init__(
        self,
        env: FluidEnvLike,
        filename: str,
        output_path: Path | None = None,
        render_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(env)
        self.__filename = filename
        self.__output_path = output_path
        self.__render_kwargs = render_kwargs or {}
        self.__ep_steps = 0
        self.__ep_idx = 0

    def __render(self) -> None:
        self._env.render(**self.__render_kwargs)

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
        step_tuple = self._env.step(action)
        self.__ep_steps += 1

        self.__render()

        return step_tuple

    def reset(
        self,
        seed: int | None = None,
        randomize: bool | None = None,
        domain_idx: int | None = None,
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

        domain_idx: int | None
            Index of the initial domain to load. If None, the default behavior is
            used. Defaults to None.

        Returns
        -------
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
            A tuple containing the initial observation and an info dictionary.
        """
        if self.__ep_steps > 0:
            self._env.save_gif(
                filename=self.__filename + f"_ep{self.__ep_idx}",
                output_path=self.__output_path
            )

        obs_tuple = self._env.reset(
            seed=seed, randomize=randomize, domain_idx=domain_idx
        )
        self.__ep_steps = 0
        self.__ep_idx += 1
        
        self.__render()

        return obs_tuple
