"""Parallel FluidGym environment running multiple environments on separate GPUs."""

import multiprocessing as mp
from enum import Enum
from typing import Any

import torch

from fluidgym.envs.multi_agent_fluid_env import MultiAgentFluidEnv
from fluidgym.registry import make


class Command(Enum):
    """Environment commands for inter-process communication."""

    STEP = "step"
    STEP_MARL = "step_marl"
    RESET = "reset"
    RESET_MARL = "reset_marl"
    SEED = "seed"
    CLOSE = "close"
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class ParallelFluidEnv:
    """Parallel FluidGym environment running multiple environments on separate GPUs.

    Parameters
    ----------
    env_id: str
        The environment identifier.

    env_kwargs: dict[str, Any]
        The keyword arguments to pass to the environment constructor.

    cuda_ids: list[int]
        The list of CUDA device IDs to use for the parallel environments.
    """

    def __init__(self, env_id: str, env_kwargs: dict[str, Any], cuda_ids: list[int]):
        self._n_envs = len(cuda_ids)
        self._env_id = env_id
        self._env_kwargs = env_kwargs
        self._cuda_ids = cuda_ids

        mp.set_start_method("spawn", force=True)
        self._pipes, self._processes = self._make_vec_gpu_envs()

    @property
    def num_envs(self) -> int:
        """The number of parallel environments."""
        return self._n_envs

    @property
    def env_id(self) -> str:
        """The environment identifier of the current environment."""
        return self._env_id

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

            elif cmd == Command.STEP_MARL:
                actions = data.to(cuda_device)
                if not isinstance(env, MultiAgentFluidEnv):
                    raise ValueError(
                        "step_marl() can only be called on MultiAgentFluidEnv "
                        "environments."
                    )
                pipe.send(env.step_marl(actions))

            elif cmd == Command.RESET_MARL:
                seed, randomize = data
                if not isinstance(env, MultiAgentFluidEnv):
                    raise ValueError(
                        "reset_marl() can only be called on MultiAgentFluidEnv "
                        "environments."
                    )
                pipe.send(env.reset_marl(seed=seed, randomize=randomize))

            elif cmd == Command.SEED:
                seed = data
                env.seed(seed)

            elif cmd == Command.TRAIN:
                env.train()

            elif cmd == Command.VAL:
                env.val()

            elif cmd == Command.TEST:
                env.test()

            elif cmd == Command.CLOSE:
                break
        torch.cuda.empty_cache()
        pipe.close()

    def _make_vec_gpu_envs(self) -> tuple[list[Any], list[mp.Process]]:
        parent_pipes, processes = [], []

        for rank, gpu in enumerate(self._cuda_ids):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(
                target=self._worker,
                args=(rank, self._env_id, self._env_kwargs, gpu, child_conn),
            )
            p.start()
            parent_pipes.append(parent_conn)
            processes.append(p)

        return parent_pipes, processes

    def reset(
        self,
        seed: int | None = None,
        randomize: bool | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
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
        tuple[torch.Tensor, list[dict[str, torch.Tensor]]]
            A tuple containing the initial observations and a list of info dictionaries.
        """
        data = (seed, randomize)
        for pipe in self._pipes:
            pipe.send((Command.RESET, data))
        results = [pipe.recv() for pipe in self._pipes]
        obs, infos = zip(*results, strict=True)

        # Bring tensors to CPU
        obs_list = [o.cpu() for o in obs]
        infos_list = [{k: v.detach().cpu() for k, v in info.items()} for info in infos]

        return torch.stack(obs_list), infos_list

    def reset_marl(
        self,
        seed: int | None = None,
        randomize: bool | None = None,
    ):
        """Reset the environment to the initial state for multiple agents.

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
        tuple[torch.Tensor, dict]
            A tuple containing the initial observations for all agents and a list of
            info dictionaries.
        """
        data = (seed, randomize)
        for pipe in self._pipes:
            pipe.send((Command.RESET_MARL, data))
        results = [pipe.recv() for pipe in self._pipes]
        obs, infos = zip(*results, strict=True)

        # Bring tensors to CPU
        obs_list = [o.cpu() for o in obs]
        infos_list = [{k: v.detach().cpu() for k, v in info.items()} for info in infos]

        return torch.concatenate(obs_list), infos_list

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[bool], list[bool], list[dict]]:
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
        tuple[torch.Tensor, torch.Tensor, bool, bool, dict[str, torch.Tensor]]
            A tuple containing the observation, reward, terminated flag, truncated flag,
            and info dictionary.
        """
        assert len(actions) == len(self._pipes), "One action per environment required."

        for pipe, action in zip(self._pipes, actions, strict=True):
            pipe.send((Command.STEP, action))

        results = [pipe.recv() for pipe in self._pipes]
        obs, rewards, terms, truncs, infos = zip(*results, strict=True)

        obs_list = [o.cpu() for o in obs]
        rewards_list = [r.cpu() for r in rewards]
        infos_list = [{k: v.detach().cpu() for k, v in info.items()} for info in infos]

        return (
            torch.stack(obs_list),
            torch.stack(rewards_list),
            list(terms),
            list(truncs),
            list(infos_list),
        )

    def step_marl(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[bool], list[bool], list[dict]]:
        """Take a step in the environment using the given actions for multiple agents.

        Parameters
        ----------
        actions: torch.Tensor
            The actions to take for each agent.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, bool, bool, dict[str, torch.Tensor]]
            A tuple containing the observations, rewards, terminated flags, truncated
            flags, and a list of info dictionaries.
        """
        actions_chunked = list(torch.chunk(actions, self.num_envs, dim=0))

        for pipe, action in zip(self._pipes, actions_chunked, strict=True):
            pipe.send((Command.STEP_MARL, action))

        results = [pipe.recv() for pipe in self._pipes]
        obs, rewards, terms, truncs, infos = zip(*results, strict=True)

        obs_list = [o.cpu() for o in obs]
        rewards_list = [r.cpu() for r in rewards]
        infos_list = [{k: v.detach().cpu() for k, v in info.items()} for info in infos]

        return (
            torch.concatenate(obs_list),
            torch.concatenate(rewards_list),
            list(terms),
            list(truncs),
            list(infos_list),
        )

    def seed(self, seed: int) -> None:
        """Set the seed for all environments.

        Parameters
        ----------
        seed: int
            The seed to set.
        """
        for pipe in self._pipes:
            pipe.send((Command.SEED, seed))

    def train(self) -> None:
        """Set all environments to training mode."""
        for pipe in self._pipes:
            pipe.send((Command.TRAIN, None))

    def val(self) -> None:
        """Set all environments to validation mode."""
        for pipe in self._pipes:
            pipe.send((Command.VAL, None))

    def test(self) -> None:
        """Set all environments to test mode."""
        for pipe in self._pipes:
            pipe.send((Command.TEST, None))

    def close(self):
        """Close all environment processes."""
        for pipe in self._pipes:
            pipe.send((Command.CLOSE, None))
            pipe.close()

        for p in self._processes:
            p.join()

        torch.cuda.empty_cache()
