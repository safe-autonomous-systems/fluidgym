"""Type definitions and protocols for FluidGym environments."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import pandas as pd
import torch
from gymnasium import spaces


class EnvMode(Enum):
    """Environment modes for training, validation, and testing."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@runtime_checkable
class FluidEnvLike(Protocol):
    """Protocol for FluidGym environment-like objects, i.e., FluidEnv and wrappers."""

    @property
    def action_space(self) -> spaces.Box:
        """The action space of the environment."""
        ...

    @property
    def observation_space(self) -> spaces.Box | spaces.Dict:
        """The observation space of the environment."""
        ...

    @property
    def differentiable(self) -> bool:
        """Whether the environment is differentiable."""
        ...

    @property
    def use_marl(self) -> bool:
        """Whether the environment is in multi-agent reinforcement learning mode."""
        ...

    @property
    def metrics(self) -> list[str]:
        """The list of metrics tracked by the environment."""
        ...

    @property
    def episode_length(self) -> int:
        """The number of steps per episode."""
        ...

    @property
    def n_agents(self) -> int:
        """The number of agents in the environment."""
        ...

    @property
    def cuda_device(self) -> torch.device:
        """The CUDA device used by the environment."""
        ...

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
        ...

    def reset(
        self, seed: int | None = None, randomize: bool | None = None
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

        Returns
        -------
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
            A tuple containing the initial observation and an info dictionary.
        """
        ...

    def seed(self, seed: int) -> None:
        """Update the random seeds and seed the random number generators.

        Parameters
        ----------
        seed: int
            The seed to set. If None, the current seed is used.
        """
        ...

    def render(
        self,
        save: bool = False,
        render_3d: bool = False,
        filename: str | None = None,
        output_path: Any | None = None,
    ) -> np.ndarray:
        """Render the current state of the environment.

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
        ...

    def sample_action(self) -> torch.Tensor:
        """Sample a random action uniformly from the action space.

        Returns
        -------
        torch.Tensor
            A random action.
        """
        ...

    def get_state(self) -> Any:
        """Get the current state of the environment.

        Returns
        -------
        EnvState
            The current state of the environment.
        """
        ...

    def set_state(self, state: Any) -> None:
        """Set the current state of the environment.

        Parameters
        ----------
        state: EnvState
            The state to set the environment to.
        """
        ...

    def train(self) -> None:
        """Set the environment to training mode."""
        ...

    def val(self) -> None:
        """Set the environment to evaluation mode."""
        ...

    def test(self) -> None:
        """Set the environment to test mode."""
        ...

    def save_gif(self, filename: str, output_path: Path | None = None) -> None:
        """Save the rendered frames as a GIF file.

        Parameters
        ----------
        filename: str
            The filename for the GIF file.

        output_path: Path | None
            The output path to save the GIF file. If None, saves to the current
            directory. Defaults to None.
        """
        ...

    def load_initial_domain(self, idx: int, mode: EnvMode | None = None) -> None:
        """Public method to load the initial domain from disk
        using the current mode.

        Parameters
        ----------
        idx: int
            Index of the initial domain to load.

        mode: EnvMode | None
            Environment mode ('train', 'val', 'test'). If None, uses the current mode.
            Defaults to None.
        """
        ...

    def get_uncontrolled_episode_metrics(self) -> pd.DataFrame | None:
        """Get the uncontrolled episode metrics for the current domain.

        Note: This method returns the metrics for the currently loaded
        (non-randomized) initial domain. If the environment has been reset
        with randomization, the metrics may not correspond to the current state.

        Returns
        -------
        pd.DataFrame | None
            The uncontrolled episode metrics, or None if not available.
        """
        ...

    def detach(self) -> None:
        """Detach all tensors in the simulation from the computation graph."""
        ...


EnvT = TypeVar("EnvT", bound=FluidEnvLike)
