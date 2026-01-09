"""TorchRL interface for FluidEnv environments."""

from tensordict import TensorDict, TensorDictBase
from torchrl.envs.common import EnvBase

from fluidgym.envs.fluid_env import FluidEnv


class TorchRLFluidEnv(EnvBase):
    """The TorchRL interface for FluidEnv environments."""

    def __init__(
        self,
        env: FluidEnv,
    ):
        """Initialize the TorchRLFluidEnv.

        Parameters
        ----------
        env: FluidEnv
            The FluidGym environment to wrap.
        """
        if not isinstance(env, FluidEnv):
            raise ValueError("env must be an instance of FluidEnv.")

        super().__init__(
            device=env.cuda_device,
        )
        self.__env = env

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        obs, reward, term, trunc, info = self.__env.step(tensordict["action"])
        out = TensorDict(
            {
                "obs": obs,
                "reward": reward,
                "term": term,
                "trunc": trunc,
                **info,
            },
            tensordict.shape,
        )
        return out

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        obs, info = self.__env.reset()
        out = TensorDict(
            {
                "obs": obs,
                **info,
            },
            tensordict.shape,
        )
        return out

    def _set_seed(self, seed: int) -> None:
        self.__env.seed(seed)
