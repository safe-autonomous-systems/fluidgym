"""A base wrapper class for FluidEnv environments."""

from pathlib import Path
from typing import Any, Generic, cast

import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

from fluidgym.envs.fluid_env import EnvState, FluidEnv
from fluidgym.types import EnvMode, EnvT, FluidEnvLike


class FluidWrapper(FluidEnvLike, Generic[EnvT]):  # type: ignore[misc]
    """A base wrapper class for FluidEnv environments.

    Parameters
    ----------
    env: FluidEnvLike
        The environment to wrap.
    """

    def __init__(self, env: FluidEnvLike) -> None:
        self._env = env

    def __getattr__(self, name: str) -> Any:
        # Only called if normal attribute lookup fails on self.
        return getattr(self._env, name)

    @property
    def unwrapped(self) -> FluidEnv:
        """The base environment, removing all wrappers."""
        e = self._env
        while isinstance(e, FluidWrapper):
            e = e._env
        return cast(FluidEnv, e)

    @property
    def use_marl(self) -> bool:
        """Whether the environment is in multi-agent reinforcement learning mode."""
        return self._env.use_marl

    @property
    def n_agents(self) -> int:
        """The number of agents in the environment."""
        return self._env.n_agents

    @property
    def episode_length(self) -> int:
        """The number of steps per episode."""
        return self._env.episode_length

    @property
    def metrics(self) -> list[str]:
        """The list of metrics tracked by the environment."""
        return self._env.metrics

    @property
    def cuda_device(self) -> torch.device:
        """The CUDA device used by the environment."""
        return self._env.cuda_device

    @property
    def differentiable(self) -> bool:
        """Whether the environment is differentiable."""
        return self._env.differentiable

    def train(self) -> None:
        """Set the environment to training mode."""
        self._env.train()

    def val(self) -> None:
        """Set the environment to evaluation mode."""
        self._env.val()

    def test(self) -> None:
        """Set the environment to test mode."""
        self._env.test()

    def sample_action(self) -> torch.Tensor:
        """Sample a random action uniformly from the action space.

        Returns
        -------
        torch.Tensor
            A random action.
        """
        return self._env.sample_action()

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
        return self._env.step(action)

    def seed(self, seed: int) -> None:
        """Update the random seeds and seed the random number generators.

        Parameters
        ----------
        seed: int
            The seed to set. If None, the current seed is used.
        """
        self._env.seed(seed)

    def reset(
        self,
        seed: int | None = None,
        randomize: bool | None = None,
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
        return self._env.reset(seed=seed, randomize=randomize)

    def render(
        self,
        save: bool = False,
        render_3d: bool = False,
        filename: str | None = None,
        output_path: Path | None = None,
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
        return self._env.render(
            save=save,
            render_3d=render_3d,
            filename=filename,
            output_path=output_path,
        )

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
        self._env.save_gif(filename=filename, output_path=output_path)

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
        self._env.load_initial_domain(idx=idx, mode=mode)

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
        return self._env.get_uncontrolled_episode_metrics()

    def detach(self) -> None:
        """Detach all tensors in the simulation from the computation graph."""
        self._env.detach()

    @property
    def action_space(self) -> spaces.Box:
        """The action space of the environment."""
        return self._env.action_space

    @property
    def observation_space(self) -> spaces.Box | spaces.Dict:
        """The observation space of the environment."""
        return self._env.observation_space

    def get_state(self) -> EnvState:
        """Get the current state of the environment.

        Returns
        -------
        EnvState
            The current state of the environment.
        """
        return self._env.get_state()

    def set_state(self, state: EnvState) -> None:
        """Set the current state of the environment.

        Parameters
        ----------
        state: EnvState
            The state to set the environment to.
        """
        return self._env.set_state(state)
