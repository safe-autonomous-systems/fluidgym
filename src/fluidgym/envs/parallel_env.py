"""Parallel FluidGym environment running multiple environments on separate GPUs."""

import multiprocessing as mp
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

from fluidgym.registry import make
from fluidgym.types import EnvMode, FluidEnvLike


class Command(Enum):
    """Environment commands for inter-process communication."""

    STEP = "step"
    RESET = "reset"
    SEED = "seed"
    CLOSE = "close"
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    LOAD_INITIAL_DOMAIN = "load_initial_domain"


class ParallelFluidEnv(FluidEnvLike):
    """Parallel FluidGym environment running multiple environments on separate GPUs.

    Parameters
    ----------
    env_id: str
        The environment identifier.

    cuda_ids: list[int]
        The list of CUDA device IDs to use for the parallel environments.

    env_kwargs: dict[str, Any]
        The keyword arguments to pass to the environment constructor.
    """

    def __init__(self, env_id: str, cuda_ids: list[int], **env_kwargs: Any):
        self.__n_envs = len(cuda_ids)
        self.__env_kwargs = env_kwargs
        self.__cuda_ids = cuda_ids
        self.__env_id = env_id

        if env_kwargs.get("differentiable", False):
            raise ValueError(
                "ParallelFluidEnv does not support differentiable environments."
            )

        self.__dummy_env = make(
            id=env_id, **env_kwargs, cuda_device=torch.device(f"cuda:{cuda_ids[0]}")
        )

        mp.set_start_method("spawn", force=True)
        self.__pipes, self.__processes = self.__make_vec_gpu_envs()

    def __getattr__(self, name: str) -> Any:
        # Only called if normal attribute lookup fails on self.
        return getattr(self._env, name)

    @property
    def action_space(self) -> spaces.Box:
        """The action space of the environment."""
        return self.__dummy_env.action_space

    @property
    def observation_space(self) -> spaces.Dict:
        """The observation space of the environment."""
        return self.__dummy_env.observation_space

    @property
    def differentiable(self) -> bool:
        """Whether the environment is differentiable."""
        return False

    @property
    def n_agents(self) -> int:
        """The number of agents in the environment."""
        return self.__n_envs * self.__dummy_env.n_agents

    @property
    def metrics(self) -> list[str]:
        """The list of metrics tracked by the environment."""
        return self.__dummy_env.metrics

    @property
    def episode_length(self) -> int:
        """The number of steps per episode."""
        return self.__dummy_env.episode_length

    @property
    def use_marl(self) -> bool:
        """Whether the environment is in multi-agent reinforcement learning mode."""
        return self.__dummy_env.use_marl

    @property
    def num_envs(self) -> int:
        """The number of parallel environments."""
        return self.__n_envs

    @property
    def cuda_device(self) -> torch.device:
        """The CUDA device used by the environment."""
        return self.__dummy_env.cuda_device

    @staticmethod
    def _worker(
        process_id: int, env_id: str, env_kwargs: dict[str, Any], cuda_id: int, pipe
    ):
        torch.cuda.set_device(cuda_id)

        cuda_device = torch.device(f"cuda:{cuda_id}")

        env = make(id=env_id, **env_kwargs, cuda_device=cuda_device)

        while True:
            try:
                cmd, data = pipe.recv()
            except EOFError:
                break

            if cmd == Command.STEP:
                action = data.to(cuda_device)
                pipe.send(env.step(action))

            elif cmd == Command.RESET:
                seed, randomize = data
                pipe.send(env.reset(seed=seed, randomize=randomize))

            elif cmd == Command.SEED:
                seed = data
                env.seed(seed)

            elif cmd == Command.TRAIN:
                env.train()

            elif cmd == Command.VAL:
                env.val()

            elif cmd == Command.TEST:
                env.test()

            elif cmd == Command.LOAD_INITIAL_DOMAIN:
                idx, mode = data
                env.load_initial_domain(idx, mode)

            elif cmd == Command.CLOSE:
                break

        torch.cuda.empty_cache()
        pipe.close()

    def __make_vec_gpu_envs(self) -> tuple[list[Any], list[mp.Process]]:
        parent_pipes, processes = [], []

        for rank, gpu in enumerate(self.__cuda_ids):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(
                target=self._worker,
                args=(rank, self.__env_id, self.__env_kwargs, gpu, child_conn),
            )
            p.start()
            parent_pipes.append(parent_conn)
            processes.append(p)

        return parent_pipes, processes

    def __aggregate_obs(
        self, obs: tuple[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Aggregate observations from multiple environments into a single batch.

        Parameters
        ----------
        obs: list[dict[str, torch.Tensor]]
            A list of observations from each environment.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing the aggregated observations.
        """
        aggregated_obs = {}
        for key in obs[0].keys():
            if self.__dummy_env.use_marl:
                aggregated_obs[key] = torch.concatenate(
                    [o[key].cpu() for o in obs], dim=0
                )
            else:
                aggregated_obs[key] = torch.stack([o[key].cpu() for o in obs], dim=0)
        return aggregated_obs

    def reset(
        self, seed: int | None = None, randomize: bool | None = None
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
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
        data = (seed, randomize)
        for pipe in self.__pipes:
            pipe.send((Command.RESET, data))
        results = [pipe.recv() for pipe in self.__pipes]
        obs, infos = zip(*results, strict=True)

        infos_list = [{k: v.cpu() for k, v in info.items()} for info in infos]

        return self.__aggregate_obs(obs), infos_list

    def step(
        self, action: torch.Tensor
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        list[bool],
        list[bool],
        list[dict[str, torch.Tensor]],
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
        tuple[dict[str, torch.Tensor], torch.Tensor, list[bool], list[bool],
        dict[str, torch.Tensor]]
            A tuple containing the observation, reward, terminated flag, truncated flag,
            and info dictionary.
        """
        if action.shape[0] != self.__n_envs:
            raise ValueError(
                f"Expected action batch size {self.__n_envs}, but got {action.shape[0]}"
            )

        action_chunked = list(torch.chunk(action, self.num_envs, dim=0))

        for pipe, action in zip(self.__pipes, action_chunked, strict=True):
            if not self.__dummy_env.use_marl:
                assert action.shape[0] == 1, (
                    "Expected action shape (1, action_dim) for single-agent mode."
                )
                action = action.squeeze(0)
            pipe.send((Command.STEP, action))

        results = [pipe.recv() for pipe in self.__pipes]
        obs, rewards, terms, truncs, infos = zip(*results, strict=True)

        rewards_list = [r.cpu() for r in rewards]
        infos_list = [{k: v.detach().cpu() for k, v in info.items()} for info in infos]

        return (
            self.__aggregate_obs(obs),
            torch.stack(rewards_list),
            list(terms),
            list(truncs),
            list(infos_list),
        )

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
        raise NotImplementedError("Rendering is not implemented for ParallelFluidEnv.")

    def seed(self, seed: int) -> None:
        """Update the random seeds and seed the random number generators.

        Parameters
        ----------
        seed: int
            The seed to set. If None, the current seed is used.
        """
        self.__dummy_env.seed(seed)
        for pipe in self.__pipes:
            pipe.send((Command.SEED, seed))

    def train(self) -> None:
        """Set the environment to training mode."""
        for pipe in self.__pipes:
            pipe.send((Command.TRAIN, None))

    def val(self) -> None:
        """Set the environment to validation mode."""
        for pipe in self.__pipes:
            pipe.send((Command.VAL, None))

    def test(self) -> None:
        """Set the environment to test mode."""
        for pipe in self.__pipes:
            pipe.send((Command.TEST, None))

    def sample_action(self) -> torch.Tensor:
        """Sample a random action uniformly from the action space.

        Returns
        -------
        torch.Tensor
            A random action.
        """
        if self.__dummy_env.use_marl:
            return torch.concatenate(
                [self.__dummy_env.sample_action() for _ in range(self.__n_envs)], dim=0
            )
        else:
            return torch.stack(
                [self.__dummy_env.sample_action() for _ in range(self.__n_envs)]
            )

    def get_state(self) -> Any:
        """Get the current state of the environment.

        Returns
        -------
        EnvState
            The current state of the environment.
        """
        raise NotImplementedError("get_state is not implemented for ParallelFluidEnv.")

    def set_state(self, state: Any) -> None:
        """Set the current state of the environment.

        Parameters
        ----------
        state: EnvState
            The state to set the environment to.
        """
        raise NotImplementedError("set_state is not implemented for ParallelFluidEnv.")

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
        raise NotImplementedError("save_gif is not implemented for ParallelFluidEnv.")

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
        for pipe in self.__pipes:
            pipe.send((Command.LOAD_INITIAL_DOMAIN, (idx, mode)))

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
        raise NotImplementedError(
            "get_uncontrolled_episode_metrics is not implemented for ParallelFluidEnv."
        )

    def detach(self) -> None:
        """Detach all tensors in the simulation from the computation graph."""
        raise NotImplementedError("detach is not implemented for ParallelFluidEnv.")

    def close(self):
        """Close all environment processes."""
        for pipe in self.__pipes:
            pipe.send((Command.CLOSE, None))
            pipe.close()

        for p in self.__processes:
            p.join()

        torch.cuda.empty_cache()
