"""TorchRL interface for FluidGym environments."""

import numpy as np
import torch
from gymnasium import spaces
from tensordict import TensorDict
from torchrl.data.tensor_specs import Bounded, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from fluidgym.types import FluidEnvLike


def _box_to_bounded(space: spaces.Box, device: torch.device) -> Bounded | Unbounded:
    """Convert a gymnasium Box space to a TorchRL Bounded spec.

    Parameters
    ----------
    space: spaces.Box
        The Box space to convert.

    device: torch.device
        The device on which to create the spec tensors.

    Returns
    -------
    Bounded | Unbounded
        A TorchRL Bounded spec if the Box has finite bounds, or an Unbounded spec
        if the Box has infinite bounds.
    """
    low = torch.tensor(space.low, dtype=torch.float32, device=device)
    high = torch.tensor(space.high, dtype=torch.float32, device=device)
    if np.all(np.isinf(space.low)) and np.all(np.isinf(space.high)):
        return Unbounded(
            shape=torch.Size(space.shape), dtype=torch.float32, device=device
        )
    return Bounded(
        low=low,
        high=high,
        shape=torch.Size(space.shape),
        dtype=torch.float32,
        device=device,
    )


def _obs_space_to_composite(
    obs_space: spaces.Box | spaces.Dict,
    device: torch.device,
    batch_shape: torch.Size,
) -> Composite:
    """Convert a gymnasium observation space to a TorchRL Composite spec.

    Parameters
    ----------
    obs_space: spaces.Box | spaces.Dict
        The observation space to convert.

    device: torch.device
        The device on which to create the spec tensors.

    batch_shape: torch.Size
        The batch shape to use for the spec tensors. For single-agent environments,
        this should be torch.Size([]). For multi-agent environments with N agents,
        this should be torch.Size([N]).

    Returns
    -------
    Composite
        A TorchRL Composite spec representing the observation space.
    """
    if isinstance(obs_space, spaces.Dict):
        space = {}
        for key, subspace in obs_space.spaces.items():
            assert isinstance(subspace, spaces.Box), (
                f"Only Box subspaces are supported, but '{key}' is {type(subspace)}"
            )
            space[key] = _box_to_bounded(subspace, device)
        return Composite(
            space,
            shape=batch_shape,
        )
    return Composite(
        observation=_box_to_bounded(obs_space, device),
        shape=batch_shape,
    )


class TorchRLFluidEnv(EnvBase):
    """TorchRL interface for FluidGym environments.

    For single-agent environments (``use_marl=False``) the batch size is ``()``.
    For multi-agent environments (``use_marl=True``) the N agents are exposed as
    a virtual batch of size ``(N,)``, so TorchRL sees N independent environments.

    Parameters
    ----------
    env: FluidEnvLike
        The FluidGym environment to wrap.
    """

    def __init__(self, env: FluidEnvLike):
        n_agents = env.n_agents if env.use_marl else None
        batch_size = torch.Size([n_agents]) if n_agents is not None else torch.Size([])

        super().__init__(device=env.cuda_device, batch_size=batch_size)
        self.__env = env
        self._n_agents = n_agents
        self._make_spec()

    # ------------------------------------------------------------------
    # Spec construction
    # ------------------------------------------------------------------

    def _make_spec(self) -> None:
        device = self.device
        batch_shape = self.batch_size  # () for single-agent, (N,) for multi-agent

        self.observation_spec = _obs_space_to_composite(
            self.__env.observation_space, device, batch_shape
        )
        self.state_spec = self.observation_spec.clone()

        action_space = self.__env.action_space
        if self._n_agents is not None:
            n = self._n_agents
            low = (
                torch.tensor(action_space.low, dtype=torch.float32, device=device)
                .expand(n, *action_space.shape)
                .clone()
            )
            high = (
                torch.tensor(action_space.high, dtype=torch.float32, device=device)
                .expand(n, *action_space.shape)
                .clone()
            )
            action_shape = (n, *action_space.shape)
        else:
            low = torch.tensor(action_space.low, dtype=torch.float32, device=device)
            high = torch.tensor(action_space.high, dtype=torch.float32, device=device)
            action_shape = action_space.shape

        self.action_spec = Bounded(
            low=low,
            high=high,
            shape=torch.Size(action_shape),
            dtype=torch.float32,
            device=device,
        )
        reward_shape = (*batch_shape, 1)
        self.reward_spec = Unbounded(
            shape=torch.Size(reward_shape), dtype=torch.float32, device=device
        )

        self.done_spec = Composite(
            done=Categorical(
                n=2,
                shape=torch.Size((*batch_shape, 1)),
                dtype=torch.bool,
                device=device,
            ),
            terminated=Categorical(
                n=2,
                shape=torch.Size((*batch_shape, 1)),
                dtype=torch.bool,
                device=device,
            ),
            truncated=Categorical(
                n=2,
                shape=torch.Size((*batch_shape, 1)),
                dtype=torch.bool,
                device=device,
            ),
        )

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Take a step in the environment using the action from tensordict.

        Parameters
        ----------
        tensordict: TensorDict
            A TensorDict containing an "action" key with shape (*batch_shape,
            action_dim).

        Returns
        -------
        TensorDict
            A TensorDict containing the next observation, reward, and done flags.
        """
        if not self.__env.differentiable:
            with torch.no_grad():
                obs, reward, term, trunc, _ = self.__env.step(tensordict["action"])
        else:
            obs, reward, term, trunc, _ = self.__env.step(tensordict["action"])

        if not isinstance(obs, dict):
            obs = {"observation": obs}

        done = term | trunc

        def _flag_tensor(flag: bool) -> torch.Tensor:
            """Broadcast a scalar bool flag to (*batch_shape, 1)."""
            t = torch.tensor([[flag]], dtype=torch.bool, device=self.device)
            if self._n_agents is not None:
                t = t.expand(self._n_agents, 1)
            return t

        def _to_spec_shape(t: torch.Tensor) -> torch.Tensor:
            """Ensure a reward tensor matches (*batch_shape, 1)."""
            if self._n_agents is not None and t.ndim == 1:
                t = t.unsqueeze(-1)
            elif t.ndim == 0:
                t = t.unsqueeze(-1)
            return t

        td = TensorDict(
            {
                **obs,
                "reward": _to_spec_shape(reward).detach(),
                "done": _flag_tensor(done),
                "terminated": _flag_tensor(term),
                "truncated": _flag_tensor(trunc),
            },
            batch_size=self.batch_size,
        )

        return td

    def _reset(self, tensordict: TensorDict | None, **kwargs) -> TensorDict:
        """Reset the environment and return the initial observation as a TensorDict.

        Parameters
        ----------
        tensordict: TensorDict | None
            Ignored. Included for compatibility with TorchRL's EnvBase interface.

        Returns
        -------
        TensorDict
            A TensorDict containing the initial observation.
        """
        obs, _ = self.__env.reset()
        if not isinstance(obs, dict):
            obs = {"observation": obs}

        return TensorDict(obs, batch_size=self.batch_size)

    def _set_seed(self, seed: int) -> None:
        """Sets the random seed for the environment.

        Parameters
        ----------
        seed: int
            The random seed to set.
        """
        self.__env.seed(seed)
