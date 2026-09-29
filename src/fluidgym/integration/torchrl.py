"""TorchRL interface for FluidGym environments."""

import numpy as np
import torch
from gymnasium import spaces
from tensordict import TensorDict
from torchrl.data.tensor_specs import Bounded, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from fluidgym.types import FluidEnvLike


def _box_to_bounded(
    space: spaces.Box,
    device: torch.device,
    batch_shape: torch.Size | None = None,
) -> Bounded | Unbounded:
    """Convert a gymnasium Box space to a TorchRL Bounded spec.

    Parameters
    ----------
    space: spaces.Box
        The Box space to convert.

    device: torch.device
        The device on which to create the spec tensors.

    batch_shape: torch.Size | None
        Leading batch dims of the spec (the same bounds for every entry).
        Defaults to None (no batch dims).

    Returns
    -------
    Bounded | Unbounded
        A TorchRL Bounded spec if the Box has finite bounds, or an Unbounded spec
        if the Box has infinite bounds.
    """
    shape = torch.Size((*(batch_shape or ()), *space.shape))
    if np.all(np.isinf(space.low)) and np.all(np.isinf(space.high)):
        return Unbounded(shape=shape, dtype=torch.float32, device=device)
    low = torch.tensor(space.low, dtype=torch.float32, device=device)
    high = torch.tensor(space.high, dtype=torch.float32, device=device)
    return Bounded(
        low=low.expand(shape).clone(),
        high=high.expand(shape).clone(),
        shape=shape,
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
            space[key] = _box_to_bounded(subspace, device, batch_shape)
        return Composite(
            space,
            shape=batch_shape,
        )
    return Composite(
        observation=_box_to_bounded(obs_space, device, batch_shape),
        shape=batch_shape,
    )


class TorchRLFluidEnv(EnvBase):
    """TorchRL interface for FluidGym environments.

    The batch size is that of the env's leading dims: ``()`` for a single
    environment, ``(E,)`` for a vectorized one (``n_envs=E``). In MARL mode the N
    agents are exposed as a (further) virtual batch dim: ``(N,)``, or ``(E, N)``
    for a vectorized environment, so TorchRL sees independent environments.

    Parameters
    ----------
    env: FluidEnvLike
        The FluidGym environment to wrap.

    from_pixels: bool
        Whether to add the rendered frames to every step and reset, as ``pixels``
        (uint8, ``[*batch, H, W, 3]``), e.g. for TorchRL's ``VideoRecorder``.
        Not supported in MARL mode. Defaults to False.
    """

    def __init__(self, env: FluidEnvLike, from_pixels: bool = False):
        n_agents = env.n_agents if env.use_marl else None
        vectorized = bool(getattr(env, "vectorized", False))
        batch_size = torch.Size(
            ([env.n_envs] if vectorized else [])
            + ([n_agents] if n_agents is not None else [])
        )
        if from_pixels and env.use_marl:
            raise ValueError("from_pixels is not supported in MARL mode.")

        super().__init__(device=env.cuda_device, batch_size=batch_size)
        self.__env = env
        self._n_agents = n_agents
        self._from_pixels = from_pixels

        # The specs are float32 while the simulation may run in float64. The
        # casts in _step/_reset are differentiable, so a gradient crosses back into
        # the solver's precision on its way into the simulation
        unwrapped = getattr(env, "unwrapped", env)
        self._env_dtype = getattr(unwrapped, "_dtype", torch.float32)

        self._make_spec()

    @property
    def fluid_env(self) -> FluidEnvLike:
        """The wrapped FluidGym environment."""
        return self.__env

    # ------------------------------------------------------------------
    # Spec construction
    # ------------------------------------------------------------------

    def _make_spec(self) -> None:
        device = self.device
        batch_shape = self.batch_size  # (E?, N?)

        self.observation_spec = _obs_space_to_composite(
            self.__env.observation_space, device, batch_shape
        )
        if self._from_pixels:
            frame = np.asarray(self.__env.render())
            self.observation_spec["pixels"] = Bounded(
                low=0,
                high=255,
                shape=torch.Size((*batch_shape, *frame.shape[-3:])),
                dtype=torch.uint8,
                device=device,
            )
        self.state_spec = self.observation_spec.clone()

        action_space = self.__env.action_space
        action_shape = (*batch_shape, *action_space.shape)
        low = (
            torch.tensor(action_space.low, dtype=torch.float32, device=device)
            .expand(action_shape)
            .clone()
        )
        high = (
            torch.tensor(action_space.high, dtype=torch.float32, device=device)
            .expand(action_shape)
            .clone()
        )

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
            shape=batch_shape,
        )

    def _flag_tensor(self, flag: bool | torch.Tensor) -> torch.Tensor:
        """A (per-environment) flag broadcast to ``(*batch_shape, 1)``."""
        t = torch.as_tensor(flag, dtype=torch.bool, device=self.device)
        # [E] per environment: shared by the agents of an environment
        t = t.reshape(*t.shape, *[1] * (len(self.batch_size) + 1 - t.ndim))
        return t.expand(*self.batch_size, 1)

    def _pixels(self) -> dict[str, torch.Tensor]:
        """The rendered frames, if requested."""
        if not self._from_pixels:
            return {}
        frames = np.ascontiguousarray(self.__env.render())
        return {"pixels": torch.as_tensor(frames, device=self.device)}

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
        action = tensordict["action"].to(dtype=self._env_dtype)
        if not self.__env.differentiable:
            with torch.no_grad():
                obs, reward, term, trunc, _ = self.__env.step(action)
        else:
            with torch.enable_grad():
                obs, reward, term, trunc, _ = self.__env.step(action)

        if not isinstance(obs, dict):
            obs = {"observation": obs}
        obs = {k: v.to(dtype=torch.float32) for k, v in obs.items()}

        # In differentiable mode the reward keeps its graph, which is what an analytic
        # policy gradient differentiates
        reward = torch.as_tensor(reward, device=self.device).to(dtype=torch.float32)
        if not self.__env.differentiable:
            reward = reward.detach()

        term_t = self._flag_tensor(term)
        trunc_t = self._flag_tensor(trunc)

        td = TensorDict(
            {
                **obs,
                **self._pixels(),
                "reward": reward.reshape(*self.batch_size, 1),
                "done": term_t | trunc_t,
                "terminated": term_t,
                "truncated": trunc_t,
            },
            batch_size=self.batch_size,
        )

        return td

    def _reset(self, tensordict: TensorDict | None, **kwargs) -> TensorDict:
        """Reset the environment and return the initial observation as a TensorDict.

        All environments of a vectorized environment share the episode, so a
        partial reset (``_reset`` flags) resets all of them.

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
        obs = {k: v.to(dtype=torch.float32) for k, v in obs.items()}

        return TensorDict({**obs, **self._pixels()}, batch_size=self.batch_size)

    def _set_seed(self, seed: int) -> None:
        """Sets the random seed for the environment.

        Parameters
        ----------
        seed: int
            The random seed to set.
        """
        self.__env.seed(seed)
