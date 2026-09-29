"""A wrapper that charges the reward for the power the actuation draws."""

import torch

from fluidgym.types import FluidEnvLike
from fluidgym.wrappers.fluid_wrapper import FluidWrapper


class PowerPenalty(FluidWrapper):
    """Charge the reward for actuation power as ``reward - penalty * ||action||^2``.

    Parameters
    ----------
    env: FluidEnvLike
        The environment to wrap.

    penalty: float
        The coefficient scaling the squared L2 norm of the action.
    """

    def __init__(self, env: FluidEnvLike, penalty: float) -> None:
        super().__init__(env)
        self.__penalty = penalty

    def step(
        self, action: torch.Tensor
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        bool | torch.Tensor,
        bool | torch.Tensor,
        dict[str, torch.Tensor],
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

        # one penalty per env (vectorized) and agent (MARL), like the reward
        lead = int(getattr(self._env, "vectorized", False)) + int(self._env.use_marl)
        action_power = (
            torch.linalg.vector_norm(action.flatten(start_dim=lead), dim=-1) ** 2
        )

        action_power = action_power.to(device=reward.device, dtype=reward.dtype)
        reward = reward - self.__penalty * action_power

        return obs, reward, terminated, truncated, info
