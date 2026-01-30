"""A wrapper that adds Gaussian noise to the actions."""

import torch

from fluidgym.types import FluidEnvLike
from fluidgym.wrappers.fluid_wrapper import FluidWrapper


class ActionNoise(FluidWrapper):
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
        noisy_action = (
            action
            + torch.randn(
                size=action.shape,
                generator=self.__torch_rng,
                device=self._env.cuda_device,
                dtype=action.dtype,
            )
            * self.__sigma
        )

        return self._env.step(noisy_action)
