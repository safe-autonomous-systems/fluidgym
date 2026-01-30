"""A wrapper that adds Gaussian noise to the sensor observations."""

import torch

from fluidgym.types import FluidEnvLike
from fluidgym.wrappers.fluid_wrapper import FluidWrapper


class SensorNoise(FluidWrapper):
    """A wrapper that adds Gaussian noise to the actions.

    It adds Gaussian noise with standard deviation `sigma` to the actions.

    Parameters
    ----------
    env: FluidEnvLike
        The environment to wrap.

    sigma: float
        The standard deviation of the Gaussian noise.

    seed: int
        The random seed for the noise generator.
    """

    def __init__(self, env: FluidEnvLike, sigma: float, seed: int) -> None:
        super().__init__(env)
        self.__sigma = sigma
        self.__torch_rng = torch.Generator(device=self._env.cuda_device).manual_seed(
            seed
        )

    def __add_noise(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            k: v
            + torch.randn(
                size=v.shape,
                generator=self.__torch_rng,
                device=self._env.cuda_device,
                dtype=v.dtype,
            )
            * self.__sigma
            for k, v in obs.items()
        }

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
        obs, info = self._env.reset(seed=seed, randomize=randomize)
        obs = self.__add_noise(obs)

        return obs, info

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
        obs, reward, terminated, truncated, info = self._env.step(action)
        obs = self.__add_noise(obs)

        return obs, reward, terminated, truncated, info
